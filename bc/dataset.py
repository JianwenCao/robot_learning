"""LeRobot v3.0 → torch Dataset for Eval-1 BC.

Each sample yields a dict:
  img            (3, 72, 128) uint8 — wrist sim-render
  proprio        (6,) float32       — current joint positions, degrees
  bowl           (3,) float32       — target bowl xyz in robot base frame, meters
  action_chunk   (k, 6) float32     — next k actions (absolute joint targets, degrees)
  mask           (k,) float32       — 1.0 for valid chunk positions, 0.0 for padded

Episodes are split by `episode key` into train/val (80/20) deterministically.
If `stats` is provided, proprio/action/bowl are returned normalized. Image is
kept as uint8 (CHW) and converted/augmented downstream in the model pipeline.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import (
    BOWL_DIM,
    CHUNK_K,
    DEMOS_ROOT,
    IMG_C,
    IMG_H,
    IMG_W,
    PILOTS,
    PROPRIO_DIM,
    ACTION_DIM,
)


@dataclass(frozen=True)
class Episode:
    pilot: str
    ep_idx: int
    parquet_start: int     # row index in pilot parquet
    length: int
    npy_start: int         # frame index in pilot sim_renders npy
    bowl_xyz: np.ndarray   # (3,) float32

    @property
    def key(self) -> str:
        return f"{self.pilot}#{self.ep_idx}"


def _load_episode_targets(meta_csv: Path) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    with meta_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row["episode_index"])
            out[ep] = np.array(
                [float(row["target_x"]), float(row["target_y"]), float(row["target_z"])],
                dtype=np.float32,
            )
    return out


def _build_episodes(pilot_root: Path) -> tuple[list[Episode], pd.DataFrame, np.ndarray]:
    """For one pilot: load parquet + sim_renders npy + episode_targets.

    Returns (episodes, parquet_df, wrist_npy_mmap). The parquet is assumed
    to be in (episode_index, frame_index) sorted order (verified upstream).
    """
    pilot = pilot_root.name

    parquet_files = sorted((pilot_root / "data" / "chunk-000").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet under {pilot_root}/data/chunk-000")
    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)

    # Verify monotonic order — assumption the alignment depends on.
    if not (df["episode_index"].diff().fillna(0) >= 0).all():
        raise AssertionError(f"{pilot}: episode_index not monotonic in parquet")

    npy_path = DEMOS_ROOT / "sim_renders" / pilot / "wrist_images.npy"
    offsets_path = DEMOS_ROOT / "sim_renders" / pilot / "episode_offsets.npy"
    wrist = np.load(npy_path, mmap_mode="r")        # (T, 3, 72, 128) uint8
    offsets = np.load(offsets_path)                 # (N, 2) int

    targets = _load_episode_targets(pilot_root / "meta" / "episode_targets.csv")

    # Cross-check: parquet group lengths must match offsets[:, 1]
    grp = df.groupby("episode_index").size().sort_index().values
    if len(grp) != len(offsets):
        raise AssertionError(f"{pilot}: parquet eps {len(grp)} != offsets eps {len(offsets)}")
    if not np.array_equal(grp, offsets[:, 1]):
        raise AssertionError(f"{pilot}: parquet ep lengths != npy ep lengths")
    if int(offsets[:, 1].sum()) != wrist.shape[0]:
        raise AssertionError(f"{pilot}: offsets sum != npy total frames")

    # Build episode descriptors. parquet_start = cumulative sum of group sizes,
    # which (because parquet is sorted) equals the first row index of that ep.
    parquet_starts = np.concatenate([[0], np.cumsum(grp)[:-1]])
    episodes: list[Episode] = []
    for i, (start_p, length) in enumerate(zip(parquet_starts, grp)):
        if i not in targets:
            raise KeyError(f"{pilot}: missing bowl target for episode {i}")
        episodes.append(
            Episode(
                pilot=pilot,
                ep_idx=i,
                parquet_start=int(start_p),
                length=int(length),
                npy_start=int(offsets[i, 0]),
                bowl_xyz=targets[i],
            )
        )
    return episodes, df, wrist


def _split_episodes(eps: list[Episode], split: str, seed: int = 0) -> list[Episode]:
    """Deterministic 80/20 by-episode split, pooled across pilots."""
    keys = sorted(e.key for e in eps)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(keys))
    n_val = max(1, int(round(0.2 * len(keys))))
    val_keys = {keys[i] for i in perm[:n_val]}
    if split == "train":
        return [e for e in eps if e.key not in val_keys]
    if split == "val":
        return [e for e in eps if e.key in val_keys]
    if split == "all":
        return list(eps)
    raise ValueError(f"unknown split {split!r}")


class Eval1BCDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        k: int = CHUNK_K,
        pilots: Optional[list[str]] = None,
        stats: Optional["Stats"] = None,   # type: ignore[name-defined]
        seed: int = 0,
        gripper_transition_upsample: int = 0,
        gripper_transition_threshold: float = 5.0,
        gripper_transition_window: int = 2,
    ):
        """If ``gripper_transition_upsample > 0``, frames within ±window of a
        gripper transition (|Δaction[5]| > threshold) are repeated that many
        extra times in the index. Combats class imbalance — only ~1.2 % of
        frames are gripper transitions, so the v1 model learned to predict
        "gripper closed" as a safe default and never opened to release.
        """
        self.split = split
        self.k = int(k)
        self.stats = stats

        pilots = pilots or PILOTS

        self._pilot_df: dict[str, pd.DataFrame] = {}
        self._pilot_wrist: dict[str, np.ndarray] = {}
        all_eps: list[Episode] = []
        for pname in pilots:
            eps, df, wrist = _build_episodes(DEMOS_ROOT / pname)
            self._pilot_df[pname] = df
            self._pilot_wrist[pname] = wrist
            all_eps.extend(eps)

        self.episodes = _split_episodes(all_eps, split=split, seed=seed)

        # Precompute per-pilot float arrays of action + state for the SELECTED
        # episodes, indexed by parquet row. This avoids per-getitem object
        # extraction from the parquet rows (which is the bottleneck).
        self._action_arr: dict[str, np.ndarray] = {}
        self._state_arr: dict[str, np.ndarray] = {}
        for pname, df in self._pilot_df.items():
            self._action_arr[pname] = np.stack(df["action"].values).astype(np.float32)
            self._state_arr[pname] = np.stack(df["observation.state"].values).astype(np.float32)

        # Flatten (ep, frame_in_ep) → global sample index.
        # Note: every frame within an episode is a valid sample START; chunk
        # tails are padded if they run past the episode boundary.
        self._index: list[tuple[Episode, int]] = []
        n_trans_frames = 0
        for ep in self.episodes:
            grip = self._action_arr[ep.pilot][
                ep.parquet_start : ep.parquet_start + ep.length, 5
            ]  # gripper action across this episode
            # Mark frames in ±window of any |delta| > threshold transition.
            delta_big = np.abs(np.diff(grip)) > gripper_transition_threshold
            in_window = np.zeros(ep.length, dtype=bool)
            for t in np.where(delta_big)[0]:
                lo = max(0, t - gripper_transition_window)
                hi = min(ep.length, t + 1 + gripper_transition_window)
                in_window[lo:hi] = True
            for f in range(ep.length):
                self._index.append((ep, f))
                if gripper_transition_upsample > 0 and in_window[f]:
                    n_trans_frames += 1
                    for _ in range(gripper_transition_upsample):
                        self._index.append((ep, f))
        self._n_trans_frames = n_trans_frames

    # -------------------------------------------------------------- public ---
    def __len__(self) -> int:
        return len(self._index)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def num_frames(self) -> int:
        return sum(e.length for e in self.episodes)

    def __getitem__(self, i: int):
        ep, f_in_ep = self._index[i]
        pname = ep.pilot

        # --- image ---
        global_npy_idx = ep.npy_start + f_in_ep
        # mmap returns a read-only view — copy so torch.from_numpy is writable.
        img = np.array(self._pilot_wrist[pname][global_npy_idx], copy=True)  # (3,H,W) uint8

        # --- proprio + action chunk ---
        parquet_row = ep.parquet_start + f_in_ep
        proprio = self._state_arr[pname][parquet_row].copy()             # (6,)

        # Future k actions, padding the tail with the last in-episode action.
        end_row = ep.parquet_start + ep.length
        chunk_end = min(parquet_row + self.k, end_row)
        chunk_len = chunk_end - parquet_row
        action_chunk = np.empty((self.k, ACTION_DIM), dtype=np.float32)
        action_chunk[:chunk_len] = self._action_arr[pname][parquet_row:chunk_end]
        if chunk_len < self.k:
            action_chunk[chunk_len:] = self._action_arr[pname][chunk_end - 1]
        mask = np.zeros(self.k, dtype=np.float32)
        mask[:chunk_len] = 1.0

        # --- bowl ---
        bowl = ep.bowl_xyz.copy()                                        # (3,)

        # --- optional normalize ---
        if self.stats is not None:
            proprio = self.stats.normalize("proprio", proprio)
            action_chunk = self.stats.normalize("action", action_chunk)
            bowl = self.stats.normalize("bowl", bowl)

        return {
            "img": torch.from_numpy(img),                                # uint8 (3,H,W)
            "proprio": torch.from_numpy(proprio),                        # f32 (6,)
            "bowl": torch.from_numpy(bowl),                              # f32 (3,)
            "action_chunk": torch.from_numpy(action_chunk),              # f32 (k,6)
            "mask": torch.from_numpy(mask),                              # f32 (k,)
        }


# ---------------------------------------------------------------- smoke test ---

def _smoke_test() -> None:
    print(f"DEMOS_ROOT = {DEMOS_ROOT}")
    train = Eval1BCDataset(split="train")
    val = Eval1BCDataset(split="val")
    print(f"train: {train.num_episodes} eps, {train.num_frames} frames, "
          f"{len(train)} samples")
    print(f"val:   {val.num_episodes} eps, {val.num_frames} frames, "
          f"{len(val)} samples")
    print(f"train ep keys: {[e.key for e in train.episodes]}")
    print(f"val   ep keys: {[e.key for e in val.episodes]}")

    s = train[0]
    print("\nSample 0:")
    for k, v in s.items():
        print(f"  {k:14s} shape={tuple(v.shape)} dtype={v.dtype} "
              f"min={float(v.min()):.4f} max={float(v.max()):.4f}")

    # Sanity check: shapes match config
    assert s["img"].shape == (IMG_C, IMG_H, IMG_W), s["img"].shape
    assert s["proprio"].shape == (PROPRIO_DIM,)
    assert s["bowl"].shape == (BOWL_DIM,)
    assert s["action_chunk"].shape == (CHUNK_K, ACTION_DIM)
    assert s["mask"].shape == (CHUNK_K,)
    assert s["img"].dtype == torch.uint8

    # Late-in-episode sample to verify chunk padding kicks in.
    ep = train.episodes[0]
    late_f = ep.length - 3       # 3 frames before episode end → chunk pads tail
    # find global i for (ep, late_f)
    i_late = next(i for i, (e, f) in enumerate(train._index) if e.key == ep.key and f == late_f)
    s_late = train[i_late]
    n_valid = int(s_late["mask"].sum().item())
    print(f"\nLate sample (ep len {ep.length}, frame {late_f}): "
          f"valid chunk steps = {n_valid}/{CHUNK_K}")
    assert n_valid == min(CHUNK_K, ep.length - late_f)

    # Action range sanity (degrees).
    a = s["action_chunk"][0]
    assert a.abs().max() < 200.0, "actions out of expected deg range — unit bug?"
    print("\nOK: dataset smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
