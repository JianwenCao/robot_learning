# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Placement events for the Eval-3 sequential pick-and-place task.

Differs from Eval-2's :func:`clutterpickplace.mdp.events.place_clutter_blocks`
in two ways:

* Four cubes go into the workspace per episode (not two), spread out via
  rejection sampling so they're individually graspable from the start.
* The bowl positions sampled by :class:`SequentialGoalCommand` are
  rejection-sampled against the cube positions here (in two stages: we
  place cubes first, then the command sampler reads cube positions when
  deciding bowl xy).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ---------------------------------------------------------------------------
# Palette (duplicated from clutterpickplace to keep tasks independent —
# small enough that DRY isn't worth a cross-task import).
# ---------------------------------------------------------------------------

COLOR_NAMES: tuple[str, ...] = ("blue", "yellow", "purple", "orange", "green", "red")
NUM_COLORS = len(COLOR_NAMES)

BLOCK_COLORS: dict[str, tuple[float, float, float]] = {
    "blue":   (0.17, 0.27, 0.62),
    "yellow": (0.95, 0.77, 0.06),
    "purple": (0.42, 0.22, 0.51),
    "orange": (0.90, 0.35, 0.16),
    "green":  (0.15, 0.49, 0.28),
    "red":    (0.78, 0.14, 0.17),
}

HIDDEN_PARK_XY: tuple[tuple[float, float], ...] = tuple(
    (-0.60, -0.25 + 0.10 * i) for i in range(NUM_COLORS)
)

# Eval-3 task constants
N_ACTIVE_BLOCKS = 4
N_GOAL_STEPS = 3


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def place_seq_blocks(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    block_x: tuple[float, float] = (0.13, 0.22),
    block_y: tuple[float, float] = (-0.12, 0.12),
    min_block_separation: float = 0.05,
    table_z: float = 0.01,
    max_attempts: int = 20,
    bowl_x: tuple[float, float] = (0.15, 0.28),
    bowl_y: tuple[float, float] = (-0.12, 0.12),
    min_bowl_separation: float = 0.10,
    distinct_bowls: bool = True,
    command_name: str = "seq_goal",
    cube_prefix: str = "cube_",
) -> None:
    """Sample the full per-episode schedule, then place all six cubes.

    Per-episode sampling is done here (not in
    :class:`SequentialGoalCommand`) because Isaac Lab's reset pipeline
    runs the event manager before the command manager. The command is
    a passive view onto the env buffers we write here:

    * ``env._seq_active_indices``    ``(N, 4)`` long — palette indices of
      the 4 active cubes per env.
    * ``env._seq_goal_color_pos``    ``(N, 3)`` long ∈ [0, 4) — for each
      of the 3 steps, which slot inside ``active_indices`` is the target.
    * ``env._seq_goal_bowl_idx``     ``(N, 3)`` long ∈ [0, 3) — which of
      the 3 bowls is targeted at each step.
    * ``env._seq_bowl_positions``    ``(N, 3, 2)`` float — bowl xy
      positions in robot root frame, rejection-sampled to be ≥
      ``min_bowl_separation`` apart.
    * ``env._target_cube_idx_per_step`` ``(N, 3)`` long — derived
      ``active[step_color_pos[step]]`` per step, cached for hot-path
      reward access.

    Cube placement: rejection-samples each active cube's xy so pairwise
    distance ≥ ``min_block_separation`` (5 cm by default). Up to
    ``max_attempts`` redraws per env; if no valid layout is found the
    last sample is accepted.
    """
    del command_name  # The command is passive; we own all the state.
    if isinstance(env_ids, torch.Tensor):
        if env_ids.numel() == 0:
            return
        env_ids_t = env_ids.long()
    else:
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    n = env_ids_t.numel()
    device = env.device

    # -----------------------------------------------------------------
    # Sample active cubes + 3-step schedule + bowl positions.
    # -----------------------------------------------------------------
    # 4 distinct palette indices.
    perms = torch.argsort(torch.rand((n, NUM_COLORS), device=device), dim=1)
    active = perms[:, :N_ACTIVE_BLOCKS]  # (n, 4)

    # 3-step goal sequence: which slot-in-active is the target at each step.
    goal_color_pos = torch.randint(0, N_ACTIVE_BLOCKS, (n, N_GOAL_STEPS), device=device)

    # 3-step bowl indices.
    if distinct_bowls:
        goal_bowl_idx = torch.argsort(torch.rand((n, N_GOAL_STEPS), device=device), dim=1)
    else:
        goal_bowl_idx = torch.randint(0, N_GOAL_STEPS, (n, N_GOAL_STEPS), device=device)

    # 3 bowl positions, rejection-sampled.
    bowl_positions = torch.zeros((n, N_GOAL_STEPS, 2), device=device)
    for slot in range(N_GOAL_STEPS):
        cand = torch.stack(
            [
                torch.empty(n, device=device).uniform_(*bowl_x),
                torch.empty(n, device=device).uniform_(*bowl_y),
            ],
            dim=1,
        )
        if slot == 0:
            bowl_positions[:, 0] = cand
            continue
        for _ in range(max_attempts):
            placed = bowl_positions[:, :slot]
            d = torch.norm(cand.unsqueeze(1) - placed, dim=2)
            bad = d.min(dim=1).values < min_bowl_separation
            if not bad.any():
                break
            n_bad = int(bad.sum())
            cand_new = torch.stack(
                [
                    torch.empty(n_bad, device=device).uniform_(*bowl_x),
                    torch.empty(n_bad, device=device).uniform_(*bowl_y),
                ],
                dim=1,
            )
            cand[bad] = cand_new
        bowl_positions[:, slot] = cand

    # Write schedule into env buffers (allocated eagerly by the command's
    # __init__; lazy-init here as a safety net).
    for buf_name, default in (
        ("_seq_active_indices", torch.zeros((env.num_envs, N_ACTIVE_BLOCKS), dtype=torch.long, device=device)),
        ("_seq_goal_color_pos", torch.zeros((env.num_envs, N_GOAL_STEPS), dtype=torch.long, device=device)),
        ("_seq_goal_bowl_idx",  torch.zeros((env.num_envs, N_GOAL_STEPS), dtype=torch.long, device=device)),
        ("_seq_bowl_positions", torch.zeros((env.num_envs, N_GOAL_STEPS, 2), dtype=torch.float32, device=device)),
    ):
        if not hasattr(env, buf_name):
            setattr(env, buf_name, default)
    env._seq_active_indices[env_ids_t] = active
    env._seq_goal_color_pos[env_ids_t] = goal_color_pos
    env._seq_goal_bowl_idx[env_ids_t] = goal_bowl_idx
    env._seq_bowl_positions[env_ids_t] = bowl_positions
    # Reset step counter for the resetting envs.
    if hasattr(env, "_seq_step_idx"):
        env._seq_step_idx[env_ids_t] = 0

    robot = env.scene["robot"]
    root_xy_w = robot.data.root_pos_w[env_ids_t, :2]

    # Rejection-sampled spread layout per env.
    pair_local_xy = torch.zeros((n, N_ACTIVE_BLOCKS, 2), device=device)
    for slot in range(N_ACTIVE_BLOCKS):
        good = torch.zeros(n, dtype=torch.bool, device=device)
        cand = torch.zeros((n, 2), device=device)
        for _ in range(max_attempts):
            need = ~good
            n_need = int(need.sum().item())
            if n_need == 0:
                break
            cand_new = torch.stack(
                [
                    torch.empty(n_need, device=device).uniform_(*block_x),
                    torch.empty(n_need, device=device).uniform_(*block_y),
                ],
                dim=1,
            )
            cand[need] = cand_new
            # Check distance against already-placed slots [0, slot).
            if slot == 0:
                good = good | need  # always accept first
            else:
                placed = pair_local_xy[:, :slot, :]  # (n, slot, 2)
                d = torch.norm(cand.unsqueeze(1) - placed, dim=2)  # (n, slot)
                ok = (d.min(dim=1).values >= min_block_separation)
                good = good | (need & ok)
        pair_local_xy[:, slot] = cand

    quat = torch.zeros((n, 4), device=device)
    quat[:, 0] = 1.0
    zero_vel = torch.zeros((n, 6), device=device)

    for k, name in enumerate(COLOR_NAMES):
        cube = env.scene[f"{cube_prefix}{name}"]
        # Is this cube active in this env? Find its slot (or None).
        # active is (n, 4); we want the slot index in [0, 4) where this k appears.
        slot_mask = (active == k)  # (n, 4) bool
        is_active = slot_mask.any(dim=1)
        # slot_idx: 0..3 for active envs, anything for inactive (we'll
        # gate writes below)
        slot_idx = slot_mask.float().argmax(dim=1)  # (n,) long-ish

        target_xy = torch.empty((n, 2), device=device)
        park_x, park_y = HIDDEN_PARK_XY[k]
        target_xy[:, 0] = park_x
        target_xy[:, 1] = park_y
        if is_active.any():
            active_xy = pair_local_xy.gather(
                1, slot_idx.view(-1, 1, 1).expand(-1, 1, 2)
            ).squeeze(1)
            target_xy[is_active] = active_xy[is_active]

        z = torch.where(
            is_active,
            torch.full_like(target_xy[:, 0], table_z),
            torch.full_like(target_xy[:, 0], 0.05),
        )
        pos_local = torch.stack([target_xy[:, 0], target_xy[:, 1], z], dim=1)
        pos_w = pos_local.clone()
        pos_w[:, :2] += root_xy_w
        pose = torch.cat([pos_w, quat], dim=1)
        cube.write_root_pose_to_sim(pose, env_ids=env_ids_t)
        cube.write_root_velocity_to_sim(zero_vel, env_ids=env_ids_t)

    # Cache the per-step palette target lookup (hot path for rewards).
    if not hasattr(env, "_active_cube_indices"):
        env._active_cube_indices = torch.zeros(
            (env.num_envs, N_ACTIVE_BLOCKS), dtype=torch.long, device=device
        )
    if not hasattr(env, "_target_cube_idx_per_step"):
        env._target_cube_idx_per_step = torch.zeros(
            (env.num_envs, N_GOAL_STEPS), dtype=torch.long, device=device
        )
    env._active_cube_indices[env_ids_t] = active
    env._target_cube_idx_per_step[env_ids_t] = active.gather(1, goal_color_pos)


def reset_seq_latches(env: "ManagerBasedEnv", env_ids: torch.Tensor) -> None:
    """Clear per-episode latches at reset (sequential variant).

    Independent of clutterpickplace's latches — we maintain
    ``env._seq_step_idx`` plus per-step ``_was_grasped`` / ``_was_over_high``
    latches sized ``(N, N_GOAL_STEPS)`` so each sub-goal has its own gate.
    """
    if isinstance(env_ids, torch.Tensor):
        if env_ids.numel() == 0:
            return
        ids = env_ids.long()
    else:
        if len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    if hasattr(env, "_seq_step_idx"):
        env._seq_step_idx[ids] = 0
    if hasattr(env, "_seq_was_grasped"):
        env._seq_was_grasped[ids] = False
    if hasattr(env, "_seq_was_over_bowl_above_rim"):
        env._seq_was_over_bowl_above_rim[ids] = False
    if hasattr(env, "_seq_success_per_step_latch"):
        env._seq_success_per_step_latch[ids] = False
    if hasattr(env, "_seq_step_release_indicator"):
        env._seq_step_release_indicator[ids] = False
