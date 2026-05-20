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

from isaac_so_arm101.tasks.pickplace.mdp.observations import gripper_state

from .events import ARRANGEMENT_NAMES, COLOR_NAMES, NUM_ARRANGEMENTS, NUM_COLORS

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
# Intermediate — grasp-and-place bias (sim2real-friendly behaviour).
# ---------------------------------------------------------------------------


def lift_then_place(
    env: "ManagerBasedRLEnv",
    z_lo: float = 0.07,
    z_hi: float = 0.20,
    gripper_closed_threshold: float = 0.25,
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """1.0 when any active cube is lifted into ``[z_lo, z_hi]`` AND the
    gripper is closed.

    Biases the emergent strategy toward grasp-and-place over pure pushing.
    Real Feetech grip force is low — sustained sliding contact is
    unreliable on hardware, so we want the sim policy to prefer firmly
    grasping a cube, lifting it clear of the cluster, and placing it
    aside. The z-band excludes both "cube still on table" (z < 0.07,
    nothing happened) and "cube being flung overhead" (z > 0.20).

    Gripper closure is read from the joint position directly (matches the
    real-robot signal). Threshold of 0.25 splits the open=0.5 vs
    close=0.0 binary command cleanly.
    """
    pos = _all_cube_pos_w(env, cube_prefix)
    z = pos[:, :, 2]
    active = env._singulation_active_mask
    in_band = (z > z_lo) & (z < z_hi) & active
    any_lifted = in_band.any(dim=1)

    g = gripper_state(env).squeeze(-1)
    closed = g < gripper_closed_threshold
    return (any_lifted & closed).float()


# ---------------------------------------------------------------------------
# Bowl avoidance — keep singulated cubes out of the bowl xy so the
# chained P2 (Eval-3 pick-and-place) handoff isn't confused by a cube
# already sitting in its release zone.
# ---------------------------------------------------------------------------


def bowl_avoidance(
    env: "ManagerBasedRLEnv",
    bowl_command_name: str = "bowl_pose",
    near_threshold: float = 0.06,
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """1.0 if any active cube's xy is within ``near_threshold`` of the
    bowl xy AND that cube is on or near the table (z < 0.06).

    Composed with a negative weight in cfg. Doesn't trigger while a cube
    is being transported overhead (z > 0.06) — only catches the bad
    end-state "cube ended up in the bowl spot."
    """
    cmd = env.command_manager.get_command(bowl_command_name)
    bowl_xy = cmd[:, :2]  # (N, 2) in robot frame

    # Cube xy in robot frame.
    from isaaclab.utils.math import subtract_frame_transforms
    robot = env.scene["robot"]
    pos_w = _all_cube_pos_w(env, cube_prefix)  # (N, 6, 3)
    n, k, _ = pos_w.shape
    root_pos = robot.data.root_state_w[:, :3].unsqueeze(1).expand(-1, k, -1)
    root_quat = robot.data.root_state_w[:, 3:7].unsqueeze(1).expand(-1, k, -1)
    pos_b, _ = subtract_frame_transforms(
        root_pos.reshape(-1, 3), root_quat.reshape(-1, 4), pos_w.reshape(-1, 3),
    )
    pos_b = pos_b.reshape(n, k, 3)

    active = env._singulation_active_mask  # (N, 6) bool
    d_xy = torch.norm(pos_b[:, :, :2] - bowl_xy.unsqueeze(1), dim=-1)  # (N, 6)
    near_bowl = (d_xy < near_threshold) & active & (pos_b[:, :, 2] < 0.06)
    return near_bowl.any(dim=1).float()


# ---------------------------------------------------------------------------
# Curriculum metric — TB log of success rate (split by arrangement).
# ---------------------------------------------------------------------------


def log_singulation_metrics(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
) -> dict[str, float]:
    """Per-reset metrics. Splits success rate by arrangement family and
    by n_active so per-family progress is visible in TensorBoard.
    """
    latch = getattr(env, "_singulation_success_latch", None)
    if latch is None or env_ids is None:
        return {}
    if isinstance(env_ids, torch.Tensor):
        if env_ids.numel() == 0:
            return {}
    elif len(env_ids) == 0:
        return {}
    outcomes = latch[env_ids].float()
    metrics: dict[str, float] = {
        "success_rate": outcomes.mean().item(),
        "n_episodes_ended": float(outcomes.numel()),
    }

    # Split by arrangement family (11-way).
    arr_idx = env._singulation_arrangement_idx[env_ids]
    for k, name in enumerate(ARRANGEMENT_NAMES):
        mask = arr_idx == k
        n_k = mask.sum().item()
        if n_k > 0:
            metrics[f"success_{name.lower()}"] = outcomes[mask].mean().item()

    # Split by n_active (3 vs 4).
    n_active = env._singulation_n_active[env_ids]
    for nv in (3, 4):
        mask = n_active == nv
        n_k = mask.sum().item()
        if n_k > 0:
            metrics[f"success_n{nv}"] = outcomes[mask].mean().item()

    latch[env_ids] = False
    return metrics
