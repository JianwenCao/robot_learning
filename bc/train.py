"""Train the goal-conditioned BC policy.

Two modes:
  --overfit N        : take ONE batch and train on it for N steps. Gate: loss < 0.01.
  --epochs E         : full training on all train samples for E epochs.

Saves to ``bc/runs/<timestamp>/`` containing:
  best.pt        — best-val-loss checkpoint
  last.pt        — final-epoch checkpoint
  stats.json     — normalization stats used (so deploy can load both together)
  train_log.csv  — per-step train loss + per-epoch val loss
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import CHUNK_K, RUNS_DIR
from .dataset import Eval1BCDataset
from .model import GoalCondBCPolicy
from .normalize import Stats


# ----------------------------------------------------------- training utils ---

@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    base_lr: float = 3e-4
    backbone_lr_mult: float = 0.1
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    num_workers: int = 4
    seed: int = 0
    overfit_steps: int = 0           # > 0 enables overfit-one-batch mode


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
              dim_weights: torch.Tensor | None = None) -> torch.Tensor:
    """L1 over (B, k, D) actions weighted by mask (B, k) and optional per-dim
    weights (D,). The denominator divides by the (weighted) number of valid
    (sample, chunk-step, dim) tuples so the loss magnitude is comparable
    across runs with different ``dim_weights``."""
    err = (pred - target).abs()                              # (B, k, D)
    m = mask.unsqueeze(-1)                                   # (B, k, 1)
    if dim_weights is not None:
        w = dim_weights.view(1, 1, -1)                       # (1, 1, D)
        num = (err * m * w).sum()
        den = (m * w).sum().clamp(min=1.0)
    else:
        num = (err * m).sum()
        den = m.sum().clamp(min=1.0) * pred.shape[-1]
    return num / den


def cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


# ---------------------------------------------------------------- main loop ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overfit", type=int, default=0,
                    help="If > 0, take ONE batch and run that many steps on it.")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--gripper-weight", type=float, default=1.0,
                    help="Per-dim L1 weight for the gripper output. v1 used "
                         "1.0; v2 uses 5.0 to combat the 'never opens' "
                         "failure mode (gripper transitions are ~1.2% of frames).")
    ap.add_argument("--gripper-transition-upsample", type=int, default=0,
                    help="Repeat frames near gripper transitions this many "
                         "extra times in the dataloader index.")
    ap.add_argument("--aug-strength", type=str, default="v1", choices=["v1", "v3"],
                    help="Image augmentation strength. v3 = stronger color DR.")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = RUNS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {out_dir}")

    # ---- data ----
    train_raw = Eval1BCDataset(split="train")
    print(f"Fitting stats on {len(train_raw)} train samples...")
    stats = Stats.fit(train_raw)
    stats.save(out_dir / "stats.json")
    del train_raw

    train_ds = Eval1BCDataset(
        split="train", stats=stats,
        gripper_transition_upsample=args.gripper_transition_upsample,
    )
    val_ds = Eval1BCDataset(split="val", stats=stats)  # eval on raw frames
    print(f"train: {train_ds.num_episodes} eps / {len(train_ds)} samples "
          f"(transition-window frames: {train_ds._n_trans_frames})")
    print(f"val:   {val_ds.num_episodes} eps / {len(val_ds)} samples")

    dim_weights = torch.tensor(
        [1.0] * 5 + [args.gripper_weight], dtype=torch.float32
    ).to(device)
    print(f"per-dim L1 weights: {dim_weights.cpu().tolist()}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )

    # ---- model ----
    model = GoalCondBCPolicy(k=CHUNK_K, aug_strength=args.aug_strength).to(device)
    optim = torch.optim.AdamW(
        model.param_groups(args.lr, backbone_lr_mult=0.1),
        weight_decay=1e-4,
    )

    log_path = out_dir / "train_log.csv"
    log_f = log_path.open("w", newline="")
    log = csv.writer(log_f)
    log.writerow(["phase", "step", "epoch", "train_l1", "val_l1", "lr"])

    # ====================================================================
    # MODE A: overfit one batch
    # ====================================================================
    if args.overfit > 0:
        batch = next(iter(train_loader))
        img = batch["img"].to(device, non_blocking=True)
        proprio = batch["proprio"].to(device, non_blocking=True)
        bowl = batch["bowl"].to(device, non_blocking=True)
        a_tgt = batch["action_chunk"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        model.train()
        # Disable image augmentation for the overfit test — otherwise the loss
        # floor is set by aug noise, not model capacity. We still use train()
        # mode for BN. (Pretend not to augment by replacing the module.)
        import torch.nn as nn
        model._train_aug = nn.Identity()

        target_loss = 0.01
        t0 = time.time()
        last_print = 0
        for step in range(1, args.overfit + 1):
            pred = model(img, proprio, bowl)
            loss = masked_l1(pred, a_tgt, mask, dim_weights=dim_weights)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if step <= 5 or step - last_print >= 25 or step == args.overfit:
                print(f"[overfit] step {step:4d}  loss={loss.item():.5f}")
                last_print = step
            log.writerow(["overfit", step, 0, f"{loss.item():.6f}", "", f"{args.lr:.6e}"])
            if loss.item() < target_loss:
                print(f"\nOK: reached loss<{target_loss} in {step} steps "
                      f"({time.time()-t0:.1f}s).")
                log_f.close()
                return
        print(f"\nFAIL: did not reach loss<{target_loss} in {args.overfit} steps "
              f"(final loss={loss.item():.5f}).")
        log_f.close()
        sys.exit(1)

    # ====================================================================
    # MODE B: full training
    # ====================================================================
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup = min(1000, total_steps // 10)

    best_val = float("inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0; ep_n = 0; t0 = time.time()
        for batch in train_loader:
            global_step += 1
            lr_scale = cosine_with_warmup(global_step, warmup, total_steps)
            for g in optim.param_groups:
                # g['lr'] was set to base_lr * (backbone_mult or 1) at init.
                # Multiply by schedule scale each step.
                g["lr"] = g.get("initial_lr", g["lr"])  # noqa: F841
            # apply scale explicitly so we keep the per-group ratio
            optim.param_groups[0]["lr"] = args.lr * 0.1 * lr_scale
            optim.param_groups[1]["lr"] = args.lr * lr_scale

            img = batch["img"].to(device, non_blocking=True)
            proprio = batch["proprio"].to(device, non_blocking=True)
            bowl = batch["bowl"].to(device, non_blocking=True)
            a_tgt = batch["action_chunk"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            pred = model(img, proprio, bowl)
            loss = masked_l1(pred, a_tgt, mask, dim_weights=dim_weights)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

            bsz = img.shape[0]
            ep_loss += loss.item() * bsz; ep_n += bsz
            log.writerow(["train", global_step, epoch,
                         f"{loss.item():.6f}", "", f"{optim.param_groups[1]['lr']:.3e}"])

        train_avg = ep_loss / max(1, ep_n)

        # validation
        model.eval()
        v_loss = 0.0; v_n = 0
        with torch.no_grad():
            for batch in val_loader:
                img = batch["img"].to(device, non_blocking=True)
                proprio = batch["proprio"].to(device, non_blocking=True)
                bowl = batch["bowl"].to(device, non_blocking=True)
                a_tgt = batch["action_chunk"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                pred = model(img, proprio, bowl)
                l = masked_l1(pred, a_tgt, mask)
                bsz = img.shape[0]
                v_loss += l.item() * bsz; v_n += bsz
        val_avg = v_loss / max(1, v_n)
        log.writerow(["val", global_step, epoch, f"{train_avg:.6f}",
                     f"{val_avg:.6f}", f"{optim.param_groups[1]['lr']:.3e}"])
        log_f.flush()

        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{args.epochs}  train_l1={train_avg:.4f}  "
              f"val_l1={val_avg:.4f}  lr={optim.param_groups[1]['lr']:.2e}  ({dt:.1f}s)")

        # checkpoints
        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_l1": val_avg,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_avg < best_val:
            best_val = val_avg
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  → new best (val_l1={val_avg:.4f}) → best.pt")

    log_f.close()
    print(f"\nDone. best val_l1={best_val:.4f}. run dir: {out_dir}")


if __name__ == "__main__":
    main()
