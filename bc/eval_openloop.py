"""Open-loop eval: predict actions on held-out val episodes, report per-joint
L1 error in *raw degree units*. Gate from BC_EVAL1_PLAN §1.4 / §5.1:
median per-joint arm L1 < 5°.

This is a *teacher-forced* eval — we feed the dataset's true proprio at every
step and just measure the model's action prediction error against the demo
action. It does NOT measure compounding error; that's closed-loop eval (step 8).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ACTION_DIM, CHUNK_K, RUNS_DIR
from .dataset import Eval1BCDataset
from .model import GoalCondBCPolicy
from .normalize import Stats

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="bc_eval1_v1",
                    help="Run name under bc/runs/")
    ap.add_argument("--ckpt", type=str, default="best.pt")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    run_dir = RUNS_DIR / args.run
    stats = Stats.load(run_dir / "stats.json")
    ck = torch.load(run_dir / args.ckpt, map_location="cpu", weights_only=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GoalCondBCPolicy(k=CHUNK_K).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Loaded ckpt epoch={ck['epoch']} val_l1={ck['val_l1']:.4f}")

    val = Eval1BCDataset(split="val", stats=stats)
    loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"val: {val.num_episodes} eps / {len(val)} samples")

    # Accumulate per-step-in-chunk per-joint absolute errors (in raw deg).
    sums = np.zeros((CHUNK_K, ACTION_DIM), dtype=np.float64)
    counts = np.zeros((CHUNK_K, ACTION_DIM), dtype=np.float64)

    # Also accumulate first-step errors per joint for the gate metric.
    first_step_errs: list[np.ndarray] = []  # (n_samples, ACTION_DIM)

    with torch.no_grad():
        for batch in loader:
            img = batch["img"].to(device, non_blocking=True)
            proprio = batch["proprio"].to(device, non_blocking=True)
            bowl = batch["bowl"].to(device, non_blocking=True)
            mask = batch["mask"].numpy().astype(bool)            # (B, k)
            a_tgt_norm = batch["action_chunk"]                   # (B, k, 6) normalized

            pred_norm = model(img, proprio, bowl).cpu()          # (B, k, 6) normalized

            # Denormalize → raw degrees / gripper-units
            pred_raw = stats.denormalize("action", pred_norm.numpy())
            tgt_raw = stats.denormalize("action", a_tgt_norm.numpy())
            err = np.abs(pred_raw - tgt_raw)                     # (B, k, 6)

            for k in range(CHUNK_K):
                m_k = mask[:, k]                                 # (B,)
                if not m_k.any():
                    continue
                sums[k] += err[m_k, k].sum(axis=0)
                counts[k] += m_k.sum()

            # First-step (k=0) errors — used for the gate.
            m0 = mask[:, 0]
            first_step_errs.append(err[m0, 0])

    mean_err = sums / np.maximum(counts, 1.0)                    # (k, 6)
    fs = np.concatenate(first_step_errs, axis=0)                 # (N, 6)
    median_fs = np.median(fs, axis=0)                            # (6,)

    # ---- print -----------------------------------------------------------
    print("\nPer-chunk-step mean |err| (raw units):")
    header = "step | " + " | ".join(f"{n:>14s}" for n in JOINT_NAMES)
    print(header)
    print("-" * len(header))
    for k in range(CHUNK_K):
        row = " | ".join(f"{mean_err[k, j]:14.3f}" for j in range(ACTION_DIM))
        print(f"{k:4d} | {row}")

    print("\nFirst-step (k=0) per-joint stats on val (raw units):")
    print(f"  {'joint':<14}  {'mean':>8}  {'median':>8}  {'p90':>8}  {'p99':>8}")
    for j, name in enumerate(JOINT_NAMES):
        print(f"  {name:<14}  {fs[:, j].mean():8.3f}  "
              f"{np.median(fs[:, j]):8.3f}  {np.percentile(fs[:, j], 90):8.3f}  "
              f"{np.percentile(fs[:, j], 99):8.3f}")

    arm_median = median_fs[:5]
    grip_median = median_fs[5]
    arm_median_overall = float(np.median(arm_median))
    print(f"\nArm-joint medians (deg): {[f'{x:.2f}' for x in arm_median]}  "
          f"→ median-of-medians = {arm_median_overall:.2f}°")
    print(f"Gripper median: {grip_median:.2f} units")

    gate_arm = arm_median_overall < 5.0
    gate_grip = grip_median < 10.0
    print(f"\nGate: median arm L1 < 5°  → {'PASS' if gate_arm else 'FAIL'} ({arm_median_overall:.2f}°)")
    print(f"Gate: median grip L1 < 10 → {'PASS' if gate_grip else 'FAIL'} ({grip_median:.2f})")


if __name__ == "__main__":
    main()
