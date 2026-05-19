# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the Bonus-B singulation task.

The headline signal is ``min_pairwise_xy_distance`` over the active set.
The policy gets dense credit for increasing this minimum (≡ spreading
the cluster) and a sparse +1 indicator when every pair is ≥ a success
threshold (≡ "individually graspable"). Plus a small per-cube-on-table
bonus that switches on once the policy has unstacked a vertical column
(reading cube z and rewarding all-active-cubes-on-table).

A "no-fling" penalty discourages high-velocity hits that shoot a cube
off the workspace: ``cube_off_table`` termination already catches the
worst, but a continuous penalty smooths the gradient near the boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from .events import COLOR_NAMES, NUM_COLORS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_cube_pos_w(env: "ManagerBasedRLEnv", cube_prefix: str = "cube_") -> torch.Tensor:
    parts = []
    for name in COLOR_NAMES:
        cube: RigidObject = env.scene[f"{cube_prefix}{name}"]
        parts.append(cube.data.root_pos_w[:, :3])
    return torch.stack(parts, dim=1)


def _all_cube_lin_vel_w(env: "ManagerBasedRLEnv", cube_prefix: str = "cube_") -> torch.Tensor:
    parts = []
    for name in COLOR_NAMES:
        cube: RigidObject = env.scene[f"{cube_prefix}{name}"]
        parts.append(cube.data.root_lin_vel_w)
    return torch.stack(parts, dim=1)


def _active_pairwise_xy_distances(
    env: "ManagerBasedRLEnv", cube_prefix: str = "cube_"
) -> torch.Tensor:
    """Return ``(N, P)`` pairwise xy distances over the active subset, with
    distances involving any non-active cube replaced by ``+inf``.

    Same shape every call (``P = NUM_COLORS * (NUM_COLORS - 1) / 2 = 15``)
    so it composes cleanly with reduction ops downstream; non-active
    pair distances become inactive via the +inf substitution and don't
    affect ``.min()`` / ``.mean()`` (use ``masked_select`` for mean to
    avoid mixing in +inf).
    """
    pos = _all_cube_pos_w(env, cube_prefix)[:, :, :2]  # (N, 6, 2)
    active = env._singulation_active_mask  # (N, 6) bool
    N, K, _ = pos.shape
    # Pairwise diff: (N, K, K, 2)
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)  # (N, K, K)
    # Mask: both endpoints must be active.
    pair_active = active.unsqueeze(2) & active.unsqueeze(1)  # (N, K, K)
    # Upper triangle indices (i < j) so we get each pair once.
    iu, ju = torch.triu_indices(K, K, offset=1, device=pos.device)
    pair_d = dist[:, iu, ju]            # (N, P=15)
    pair_a = pair_active[:, iu, ju]     # (N, P) bool
    pair_d = torch.where(pair_a, pair_d, torch.full_like(pair_d, float("inf")))
    return pair_d


# ---------------------------------------------------------------------------
# Dense — increase the minimum pairwise xy distance.
# ---------------------------------------------------------------------------


def min_pairwise_xy(
    env: "ManagerBasedRLEnv",
    cap: float = 0.10,
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """``(N,)`` reward = clamp(min_pairwise_xy, 0, cap) / cap → [0, 1].

    Caps at ``cap`` (default 10 cm) so the reward doesn't keep growing
    indefinitely once the cubes are well-separated — the policy gets
    no benefit from flinging them to the table edges. Returns 0 when
    the active set is empty (shouldn't happen) due to inf-min behavior.
    """
    d = _active_pairwise_xy_distances(env, cube_prefix)  # (N, P) with inf for inactive
    min_d = d.min(dim=1).values
    # Replace +inf (all-inactive case) with 0.
    min_d = torch.where(torch.isfinite(min_d), min_d, torch.zeros_like(min_d))
    return (min_d.clamp(min=0.0, max=cap) / cap)


def mean_pairwise_xy(
    env: "ManagerBasedRLEnv",
    cap: float = 0.10,
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """``(N,)`` reward = clamp(mean_pairwise_xy_active, 0, cap) / cap.

    Complements :func:`min_pairwise_xy` — the policy can get partial
    credit by spreading some pairs while still working on the worst.
    """
    d = _active_pairwise_xy_distances(env, cube_prefix)
    finite_mask = torch.isfinite(d)
    safe_d = torch.where(finite_mask, d, torch.zeros_like(d))
    count = finite_mask.float().sum(dim=1).clamp(min=1.0)
    mean_d = safe_d.sum(dim=1) / count
    return (mean_d.clamp(min=0.0, max=cap) / cap)


# ---------------------------------------------------------------------------
# Dense — get every cube down off the stack.
# ---------------------------------------------------------------------------


def all_cubes_on_table(
    env: "ManagerBasedRLEnv",
    height_threshold: float = 0.05,
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """1.0 when every active cube has z below ``height_threshold``.

    Specifically targets the *stacked* arrangement — when cubes are
    initially piled, this term goes from 0 (top cubes high) to 1 once
    the policy has unstacked them. ``height_threshold = 0.05`` lets a
    single cube sitting on the table (z ≈ 0.01) and a cube held briefly
    at gripper height (~0.04) both count as "on table"; an actual stack
    has the top cube at ~0.05+ depending on count.
    """
    pos = _all_cube_pos_w(env, cube_prefix)  # (N, 6, 3)
    z = pos[:, :, 2]
    active = env._singulation_active_mask
    # For inactive cubes, treat them as "on table" so they don't fail
    # the all-clause.
    on_table = (z < height_threshold) | (~active)
    return on_table.all(dim=1).float()


# ---------------------------------------------------------------------------
# Sparse — success indicator (all active pairwise xy ≥ threshold).
# ---------------------------------------------------------------------------


def singulation_success(
    env: "ManagerBasedRLEnv",
    min_separation: float = 0.05,
    on_table_height: float = 0.05,
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """1.0 if every active pair has xy dist ≥ ``min_separation`` AND every
    active cube is on the table.

    Latches ``env._singulation_success_latch`` (idempotent OR-update) so
    a curriculum metric can report per-episode success rate at reset.
    """
    d = _active_pairwise_xy_distances(env, cube_prefix)
    # Smallest finite pairwise — non-active pairs are +inf, ignored.
    min_d = d.min(dim=1).values
    sep_ok = torch.where(torch.isfinite(min_d), min_d, torch.full_like(min_d, float("inf"))) >= min_separation

    pos = _all_cube_pos_w(env, cube_prefix)
    z = pos[:, :, 2]
    active = env._singulation_active_mask
    on_table_all = ((z < on_table_height) | (~active)).all(dim=1)

    indicator = sep_ok & on_table_all
    if not hasattr(env, "_singulation_success_latch"):
        env._singulation_success_latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._singulation_success_latch |= indicator
    return indicator.float()


# ---------------------------------------------------------------------------
# No-fling penalty — cap on cube speeds.
# ---------------------------------------------------------------------------


def cube_overspeed_penalty(
    env: "ManagerBasedRLEnv",
    speed_cap: float = 0.30,
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """Penalty proportional to max over-cap speed across active cubes.

    Returns ``max(0, speed - cap)`` summed over active cubes, clipped at
    1 per cube so a single bad fling doesn't dominate. Composed with a
    negative weight in cfg.
    """
    vel = _all_cube_lin_vel_w(env, cube_prefix)  # (N, 6, 3)
    speed = torch.norm(vel, dim=-1)  # (N, 6)
    active = env._singulation_active_mask
    over = (speed - speed_cap).clamp(min=0.0) / speed_cap
    over = over.clamp(max=1.0) * active.float()
    return over.sum(dim=1)


# ---------------------------------------------------------------------------
# Reach proxy — pull the EE toward the closest-pair midpoint so the policy
# at least *engages* with the cluster (vs sitting at home pose).
# ---------------------------------------------------------------------------


def reach_closest_pair(
    env: "ManagerBasedRLEnv",
    std: float = 0.10,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """Dense reward pulling EE toward the midpoint of the *closest active
    pair* — the pair the policy most needs to separate.

    Returns ``1 - tanh(d_ee_midpoint / std)``. The midpoint shifts as
    the policy makes progress: once the originally-closest pair is
    separated, a different pair becomes the new closest, and the EE
    attractor moves to that one.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    pos = _all_cube_pos_w(env, cube_prefix)[:, :, :2]  # (N, 6, 2)
    active = env._singulation_active_mask
    K = pos.shape[1]
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)
    pair_active = active.unsqueeze(2) & active.unsqueeze(1)
    iu, ju = torch.triu_indices(K, K, offset=1, device=pos.device)
    pair_d = dist[:, iu, ju]                # (N, P)
    pair_a = pair_active[:, iu, ju]
    pair_d = torch.where(pair_a, pair_d, torch.full_like(pair_d, float("inf")))
    # Closest pair indices per env.
    closest = pair_d.argmin(dim=1)          # (N,) long
    i_idx = iu[closest]
    j_idx = ju[closest]
    pi = pos.gather(1, i_idx.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
    pj = pos.gather(1, j_idx.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)
    midpoint_xy = (pi + pj) * 0.5

    d_ee = torch.norm(ee_w[:, :2] - midpoint_xy, dim=1)
    return 1.0 - torch.tanh(d_ee / std)


# ---------------------------------------------------------------------------
# Curriculum metric — TB log of success rate.
# ---------------------------------------------------------------------------


def log_singulation_metrics(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
) -> dict[str, float]:
    latch = getattr(env, "_singulation_success_latch", None)
    if latch is None or env_ids is None:
        return {}
    if isinstance(env_ids, torch.Tensor):
        if env_ids.numel() == 0:
            return {}
    elif len(env_ids) == 0:
        return {}
    outcomes = latch[env_ids].float()
    metrics = {
        "success_rate": outcomes.mean().item(),
        "n_episodes_ended": float(outcomes.numel()),
    }
    # Split by arrangement (stacked vs clustered) for diagnosis.
    arr = env._singulation_arrangement[env_ids]
    n_stack = (arr == 0).sum().item()
    n_cluster = (arr == 1).sum().item()
    if n_stack > 0:
        metrics["success_stacked"] = outcomes[arr == 0].mean().item()
    if n_cluster > 0:
        metrics["success_clustered"] = outcomes[arr == 1].mean().item()
    # Split by n_active (3 vs 4).
    n_active = env._singulation_n_active[env_ids]
    n3 = (n_active == 3).sum().item()
    n4 = (n_active == 4).sum().item()
    if n3 > 0:
        metrics["success_n3"] = outcomes[n_active == 3].mean().item()
    if n4 > 0:
        metrics["success_n4"] = outcomes[n_active == 4].mean().item()
    latch[env_ids] = False
    return metrics
