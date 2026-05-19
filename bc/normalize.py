"""Per-dimension normalization stats for proprio / action / bowl.

Usage:
  stats = Stats.fit(train_dataset)
  stats.save(path)
  stats = Stats.load(path)
  x_n = stats.normalize("proprio", x)         # x: (..., 6)
  x   = stats.denormalize("action", x_n)

Stats are computed in float64 for numerical stability, then stored as float32.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


@dataclass
class _DimStat:
    mean: list[float]
    std: list[float]


@dataclass
class Stats:
    proprio: _DimStat
    action: _DimStat
    bowl: _DimStat

    KEYS = ("proprio", "action", "bowl")

    # --------------------------------------------------------------- fit ---
    @classmethod
    def fit(cls, dataset, batch_size: int = 1024) -> "Stats":
        """Fit on a dataset that yields the unnormalized sample dict.

        We use Welford-like accumulators: compute mean and variance in
        float64. For 'action' we collapse the chunk axis (k, 6) → (k*6,)
        per-dim, treating each chunk step as a sample.
        """
        from torch.utils.data import DataLoader

        if getattr(dataset, "stats", None) is not None:
            raise ValueError("Refusing to fit on a dataset that is already normalized")

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, drop_last=False)

        accum: dict[str, dict] = {
            "proprio": {"n": 0, "sum": None, "sq": None, "d": None},
            "action":  {"n": 0, "sum": None, "sq": None, "d": None},
            "bowl":    {"n": 0, "sum": None, "sq": None, "d": None},
        }

        def _update(name: str, x: np.ndarray, weights: np.ndarray | None = None):
            # x: (N, D). weights: (N,) or None.
            a = accum[name]
            if weights is None:
                n_add = x.shape[0]
                s = x.sum(axis=0)
                sq = (x * x).sum(axis=0)
            else:
                w = weights.astype(np.float64)
                n_add = float(w.sum())
                s = (x * w[:, None]).sum(axis=0)
                sq = (x * x * w[:, None]).sum(axis=0)
            if a["sum"] is None:
                a["sum"] = s.copy(); a["sq"] = sq.copy(); a["d"] = x.shape[1]
            else:
                a["sum"] += s; a["sq"] += sq
            a["n"] += n_add

        for batch in loader:
            proprio = batch["proprio"].numpy().astype(np.float64)        # (B, 6)
            action = batch["action_chunk"].numpy().astype(np.float64)    # (B, k, 6)
            mask = batch["mask"].numpy().astype(np.float64)              # (B, k)
            bowl = batch["bowl"].numpy().astype(np.float64)              # (B, 3)

            _update("proprio", proprio)
            _update("bowl", bowl)
            B, K, D = action.shape
            _update("action", action.reshape(B * K, D), mask.reshape(B * K))

        def _finalize(a) -> _DimStat:
            n = a["n"]; mean = a["sum"] / n
            var = a["sq"] / n - mean * mean
            std = np.sqrt(np.clip(var, 1e-12, None))
            # Clamp tiny std (e.g. bowl z is constant in the demos) to 1.0
            # so the normalization becomes a no-op for that dim instead of
            # blowing up to ±inf or producing NaNs.
            std = np.where(std < 1e-4, 1.0, std)
            return _DimStat(mean=mean.tolist(), std=std.tolist())

        return cls(
            proprio=_finalize(accum["proprio"]),
            action=_finalize(accum["action"]),
            bowl=_finalize(accum["bowl"]),
        )

    # ------------------------------------------------------ persistence ---
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({k: asdict(getattr(self, k)) for k in self.KEYS}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Stats":
        raw = json.loads(Path(path).read_text())
        return cls(**{k: _DimStat(**raw[k]) for k in cls.KEYS})

    # -------------------------------------------------------- transforms ---
    def _arr(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        d = getattr(self, name)
        return np.asarray(d.mean, dtype=np.float32), np.asarray(d.std, dtype=np.float32)

    def normalize(self, name: str, x):
        mean, std = self._arr(name)
        if isinstance(x, torch.Tensor):
            mean_t = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
            std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype)
            return (x - mean_t) / std_t
        return ((x - mean) / std).astype(np.float32)

    def denormalize(self, name: str, x):
        mean, std = self._arr(name)
        if isinstance(x, torch.Tensor):
            mean_t = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
            std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype)
            return x * std_t + mean_t
        return (x * std + mean).astype(np.float32)


# --------------------------------------------------------------- smoke test ---

def _smoke_test() -> None:
    from .dataset import Eval1BCDataset

    train = Eval1BCDataset(split="train")
    print(f"Fitting stats on {len(train)} samples...")
    stats = Stats.fit(train)

    for name in Stats.KEYS:
        d = getattr(stats, name)
        print(f"  {name:8s} mean={[f'{m:+.3f}' for m in d.mean]} "
              f"std={[f'{s:.3f}' for s in d.std]}")

    # Save / load round-trip
    out = Path("bc/runs/_stats_smoke.json")
    stats.save(out)
    loaded = Stats.load(out)
    for name in Stats.KEYS:
        a, b = getattr(stats, name), getattr(loaded, name)
        assert a.mean == b.mean and a.std == b.std, name
    print("Save/load round-trip: OK")

    # Verify normalization on train set produces ~mean 0, ~std 1 (on TRAIN split only).
    train_n = Eval1BCDataset(split="train", stats=loaded)
    from torch.utils.data import DataLoader
    accum_p, accum_a, accum_b = [], [], []
    for batch in DataLoader(train_n, batch_size=512, num_workers=0):
        accum_p.append(batch["proprio"].numpy())
        # weight chunk dims by mask
        m = batch["mask"].numpy().astype(bool)
        a = batch["action_chunk"].numpy()
        accum_a.append(a[m])
        accum_b.append(batch["bowl"].numpy())
    p = np.concatenate(accum_p); a = np.concatenate(accum_a); b = np.concatenate(accum_b)
    print(f"\nPost-normalization (train):")
    print(f"  proprio mean={p.mean(0).round(3)}  std={p.std(0).round(3)}")
    print(f"  action  mean={a.mean(0).round(3)}  std={a.std(0).round(3)}")
    print(f"  bowl    mean={b.mean(0).round(3)}  std={b.std(0).round(3)}")

    assert np.abs(p.mean(0)).max() < 1e-3, "proprio mean not ~0 after normalize"
    assert np.abs(p.std(0) - 1.0).max() < 1e-2, "proprio std not ~1 after normalize"
    assert np.abs(a.mean(0)).max() < 1e-3, "action mean not ~0 after normalize"
    assert np.abs(a.std(0) - 1.0).max() < 1e-2, "action std not ~1 after normalize"

    # Round-trip on a raw sample
    raw = Eval1BCDataset(split="train")[0]
    norm = loaded.normalize("action", raw["action_chunk"].numpy())
    back = loaded.denormalize("action", norm)
    err = float(np.abs(back - raw["action_chunk"].numpy()).max())
    print(f"\nRound-trip max abs err (action_chunk): {err:.3e}")
    assert err < 1e-4, "denormalize(normalize(x)) != x"

    out.unlink(missing_ok=True)
    print("\nOK: normalize smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
