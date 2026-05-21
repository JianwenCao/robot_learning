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

# Eval-3 sequence constants: four cubes are active on the table, three
# distinct cubes are picked in order, and the remaining active cube is a
# distractor.
N_ACTIVE_BLOCKS = 4
N_GOAL_STEPS = 3
N_BOWLS = 1  # Single bowl per rollout.


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def reset_robot_to_default(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset_name: str = "robot",
    reset_joint_targets: bool = False,
) -> None:
    """Reset only the robot articulation, leaving cubes untouched."""
    if isinstance(env_ids, torch.Tensor):
        if env_ids.numel() == 0:
            return
        ids = env_ids.long()
    else:
        if len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    robot = env.scene[asset_name]
    root_state = robot.data.default_root_state[ids].clone()
    root_state[:, 0:3] += env.scene.env_origins[ids]
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=ids)

    joint_pos = robot.data.default_joint_pos[ids].clone()
    joint_vel = robot.data.default_joint_vel[ids].clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ids)
    if reset_joint_targets:
        robot.set_joint_position_target(joint_pos, env_ids=ids)
        robot.set_joint_velocity_target(joint_vel, env_ids=ids)


def place_seq_blocks(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    block_x: tuple[float, float] = (0.13, 0.28),
    block_y: tuple[float, float] = (-0.15, 0.15),
    min_block_separation: float = 0.12,
    table_z: float = 0.01,
    max_attempts: int = 80,
    active_count_choices: tuple[int, ...] = (4,),
    bowl_x: tuple[float, float] = (0.15, 0.28),
    bowl_y: tuple[float, float] = (-0.12, 0.12),
    min_bowl_block_separation: float = 0.10,
    command_name: str = "seq_goal",
    cube_prefix: str = "cube_",
) -> None:
    """Sample a 3-target sequence, then place 4 cubes + 1 bowl.

    Per-episode sampling is done here (not in
    :class:`SequentialGoalCommand`) because Isaac Lab's reset pipeline
    runs the event manager before the command manager. The command is
    a passive view onto the env buffers we write here:

    * ``env._seq_active_indices``    ``(N, 4)`` long — palette indices of
      up to 4 active cubes per env.
    * ``env._seq_active_count``      ``(N,)`` long, normally all 4.
    * ``env._seq_goal_color_pos``    ``(N, 3)`` long ∈ [0, active_count)
      — ordered target slots inside ``active_indices``. The fourth active
      cube is a distractor.
    * ``env._seq_bowl_positions``    ``(N, 1, 2)`` float — bowl xy in
      robot frame, rejection-sampled against the 4 placed cubes
      (≥ ``min_bowl_block_separation``).
    * ``env._target_cube_idx_per_step`` ``(N, 3)`` long — derived
      ``active[step_color_pos[step]]`` per step, cached for hot-path
      reward access.

    Cube placement: active cubes placed independently in the workspace
    with sequential rejection sampling — each new cube must be ≥
    ``min_block_separation`` from all previously-placed cubes. Default
    8 cm gives ~6 cm edge gap for 2 cm cubes — wider than the SO-ARM
    gripper finger span so the policy can pick any cube without
    contacting a neighbour. The bowl is sampled FIRST so cube placement
    can reject against it.
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
    # Sample active cubes + target sequence + bowl position.
    # -----------------------------------------------------------------
    choices = torch.tensor(active_count_choices, device=device, dtype=torch.long)
    choices = choices[(choices >= N_GOAL_STEPS) & (choices <= N_ACTIVE_BLOCKS)]
    if choices.numel() == 0:
        raise ValueError(
            f"active_count_choices must include values in [{N_GOAL_STEPS}, {N_ACTIVE_BLOCKS}], "
            f"got {active_count_choices}."
        )
    active_count = choices[torch.randint(0, choices.numel(), (n,), device=device)]

    # Up to 4 distinct palette indices. Only slots < active_count are
    # placed on the table; later slots stay parked.
    perms = torch.argsort(torch.rand((n, NUM_COLORS), device=device), dim=1)
    active = perms[:, :N_ACTIVE_BLOCKS]  # (n, 4)

    # Three distinct target slots, sampled in order from the active cubes.
    # Slots not active in a given env get a high score and cannot enter
    # the first N_GOAL_STEPS entries after argsort.
    slot_ids = torch.arange(N_ACTIVE_BLOCKS, device=device).view(1, -1)
    scores = torch.rand((n, N_ACTIVE_BLOCKS), device=device)
    scores = torch.where(slot_ids < active_count.view(-1, 1), scores, torch.full_like(scores, 2.0))
    goal_color_pos = torch.argsort(scores, dim=1)[:, :N_GOAL_STEPS]

    # Single bowl per rollout — buffer kept ``(N, N_BOWLS=1, 2)`` for
    # uniformity with the existing observation/reward plumbing.
    bowl_positions = torch.zeros((n, N_BOWLS, 2), device=device)

    # Write schedule into env buffers (allocated eagerly by the command's
    # __init__; lazy-init here as a safety net).
    for buf_name, default in (
        ("_seq_active_indices", torch.zeros((env.num_envs, N_ACTIVE_BLOCKS), dtype=torch.long, device=device)),
        ("_seq_active_count", torch.full((env.num_envs,), N_ACTIVE_BLOCKS, dtype=torch.long, device=device)),
        ("_seq_goal_color_pos", torch.zeros((env.num_envs, N_GOAL_STEPS), dtype=torch.long, device=device)),
        ("_seq_bowl_positions", torch.zeros((env.num_envs, N_BOWLS, 2), dtype=torch.float32, device=device)),
    ):
        if not hasattr(env, buf_name):
            setattr(env, buf_name, default)
    env._seq_active_indices[env_ids_t] = active
    env._seq_active_count[env_ids_t] = active_count
    env._seq_goal_color_pos[env_ids_t] = goal_color_pos
    # NOTE: ``_seq_bowl_positions`` is written below, AFTER bowl sampling
    # (which depends on the cube layout sampled in this function).
    # Reset step counter for the resetting envs.
    if hasattr(env, "_seq_step_idx"):
        env._seq_step_idx[env_ids_t] = 0

    robot = env.scene["robot"]
    root_xy_w = robot.data.root_pos_w[env_ids_t, :2]

    # -----------------------------------------------------------------
    # Step 1: sample the single bowl position FIRST.
    # -----------------------------------------------------------------
    # Bowl is sampled first so the block placement loop below can reject
    # against it.
    bowl_positions[:, 0] = torch.stack(
        [
            torch.empty(n, device=device).uniform_(*bowl_x),
            torch.empty(n, device=device).uniform_(*bowl_y),
        ],
        dim=1,
    )
    env._seq_bowl_positions[env_ids_t] = bowl_positions

    # -----------------------------------------------------------------
    # Step 2: rejection-sampled spread layout for up to 4 active cubes,
    # checking each new cube against (a) all previously-placed cubes
    # (≥ ``min_block_separation``) AND (b) the bowl
    # (≥ ``min_bowl_block_separation``). Cubes thus always spawn
    # *slightly away* from the bowl position — no need to rely on
    # post-hoc bowl-rejection to enforce separation.
    # -----------------------------------------------------------------
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
            # vs bowls (always check — bowls are placed already)
            d_bowl = torch.norm(cand.unsqueeze(1) - bowl_positions, dim=2)
            ok_bowl = d_bowl.min(dim=1).values >= min_bowl_block_separation
            # vs previously-placed cubes
            slot_active = slot < active_count
            if slot == 0:
                ok_blk = torch.ones(n, dtype=torch.bool, device=device)
            else:
                placed = pair_local_xy[:, :slot, :]
                d_blk = torch.norm(cand.unsqueeze(1) - placed, dim=2)
                ok_blk = d_blk.min(dim=1).values >= min_block_separation
            ok = (ok_bowl & ok_blk) | (~slot_active)
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
        slot_is_present = slot_mask.any(dim=1)
        # slot_idx: 0..3 for active envs, anything for inactive (we'll
        # gate writes below)
        slot_idx = slot_mask.float().argmax(dim=1)  # (n,) long-ish

        target_xy = torch.empty((n, 2), device=device)
        park_x, park_y = HIDDEN_PARK_XY[k]
        target_xy[:, 0] = park_x
        target_xy[:, 1] = park_y
        is_active = torch.zeros(n, dtype=torch.bool, device=device)
        if slot_is_present.any():
            active_xy = pair_local_xy.gather(
                1, slot_idx.view(-1, 1, 1).expand(-1, 1, 2)
            ).squeeze(1)
            is_active = slot_is_present & (slot_idx < active_count)
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


def place_seq_blocks_once(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    block_x: tuple[float, float] = (0.13, 0.28),
    block_y: tuple[float, float] = (-0.15, 0.15),
    min_block_separation: float = 0.12,
    table_z: float = 0.01,
    max_attempts: int = 80,
    active_count_choices: tuple[int, ...] = (4,),
    bowl_x: tuple[float, float] = (0.15, 0.28),
    bowl_y: tuple[float, float] = (-0.12, 0.12),
    min_bowl_block_separation: float = 0.10,
    command_name: str = "seq_goal",
    cube_prefix: str = "cube_",
) -> None:
    """Place cubes only on the first reset for each env.

    Later episode resets preserve the live cube poses while the robot
    reset/latch events still prepare the next rollout.
    """
    if isinstance(env_ids, torch.Tensor):
        if env_ids.numel() == 0:
            return
        ids = env_ids.long()
    else:
        if len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    if not hasattr(env, "_seq_blocks_initialized"):
        env._seq_blocks_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    init_ids = ids[~env._seq_blocks_initialized[ids]]
    if init_ids.numel() == 0:
        return

    place_seq_blocks(
        env,
        init_ids,
        block_x=block_x,
        block_y=block_y,
        min_block_separation=min_block_separation,
        table_z=table_z,
        max_attempts=max_attempts,
        active_count_choices=active_count_choices,
        bowl_x=bowl_x,
        bowl_y=bowl_y,
        min_bowl_block_separation=min_bowl_block_separation,
        command_name=command_name,
        cube_prefix=cube_prefix,
    )
    env._seq_blocks_initialized[init_ids] = True


def randomize_robot_initial_joint_pos(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    joint_delta_ranges: dict[str, tuple[float, float]] | None = None,
    gripper_open_pos: float = 0.5,
    asset_name: str = "robot",
) -> None:
    """Reset the arm near home with joint-space domain randomization.

    The sampled state is cached so the command term can hard-reset the arm
    to the same per-episode initial pose after a successful release.
    """
    if isinstance(env_ids, torch.Tensor):
        if env_ids.numel() == 0:
            return
        ids = env_ids.long()
    else:
        if len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    if joint_delta_ranges is None:
        joint_delta_ranges = {
            "shoulder_pan": (-0.15, 0.15),
            "shoulder_lift": (-0.12, 0.12),
            "elbow_flex": (-0.12, 0.12),
            "wrist_flex": (-0.15, 0.15),
            "wrist_roll": (-0.20, 0.20),
        }

    robot = env.scene[asset_name]
    q = robot.data.default_joint_pos[ids].clone()
    qd = torch.zeros_like(q)

    for joint_name, (lo, hi) in joint_delta_ranges.items():
        joint_idx = robot.find_joints(joint_name)[0][0]
        q[:, joint_idx] += torch.empty(ids.numel(), device=env.device).uniform_(lo, hi)

    gripper_idx = robot.find_joints("gripper")[0][0]
    q[:, gripper_idx] = gripper_open_pos

    if not hasattr(env, "_seq_initial_joint_pos"):
        env._seq_initial_joint_pos = robot.data.default_joint_pos.clone()
    if not hasattr(env, "_seq_initial_joint_vel"):
        env._seq_initial_joint_vel = torch.zeros_like(robot.data.default_joint_pos)
    env._seq_initial_joint_pos[ids] = q
    env._seq_initial_joint_vel[ids] = qd

    robot.write_joint_state_to_sim(q, qd, env_ids=ids)


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
    if hasattr(env, "_seq_sub_step_count"):
        env._seq_sub_step_count[ids] = 0
    if hasattr(env, "_seq_was_grasped"):
        env._seq_was_grasped[ids] = False
    if hasattr(env, "_seq_was_over_bowl_above_rim"):
        env._seq_was_over_bowl_above_rim[ids] = False
    if hasattr(env, "_seq_success_per_step_latch"):
        env._seq_success_per_step_latch[ids] = False
    if hasattr(env, "_seq_success_per_step_latch_strict"):
        env._seq_success_per_step_latch_strict[ids] = False
    if hasattr(env, "_seq_step_release_indicator"):
        env._seq_step_release_indicator[ids] = False
