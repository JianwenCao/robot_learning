# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-cube placement events for the clutter pick-and-place task.

The scene spawns six fixed-color cubes (one per palette entry). On every
episode reset we sample two of them to be **active** in the workspace
(placed adjacent to form a flat 2-cube cluster) and **park** the other four
outside the wrist-cam FOV (env-local ``x = -0.6 m``, behind the robot,
where the table footprint doesn't extend and the gripper-mounted camera
can't see at any joint configuration the policy is going to discover).

The active pair and the target index inside that pair are sampled by the
:class:`TargetColorCommand` *before* this event runs, so we just read its
buffers (``cmd.active_indices``, ``cmd.target_idx_in_pair``) here. The
ordering is enforced by Isaac Lab's reset pipeline: the command manager
calls ``_resample_command`` for env_ids first, then the event manager
applies its ``mode="reset"`` terms.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ---------------------------------------------------------------------------
# Palette — fixed across all multi-cube tasks (Eval 2 / 3 / Singulation B).
# RGB values are normalized to [0, 1] from the project spec
# (#2C469D, #F1C40F, #6A3982, #E65A28, #257C48, #C7242C).
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

# Env-local xy for parking inactive cubes — outside the table footprint
# (table extends x∈[-0.05, 0.55], y∈[-0.5, 0.5]). At x=-0.6 the cube is
# behind the robot base; the wrist camera, parented to the gripper, never
# looks back that far. Six distinct y slots keep the parked cubes from
# stacking on top of one another (1-cm spacing → ≥ 8 mm gap edge-to-edge
# for a 2 cm cube). z=0.05 lets them fall to the ground plane (-1.05),
# where they settle out of view.
HIDDEN_PARK_XY: tuple[tuple[float, float], ...] = tuple(
    (-0.60, -0.25 + 0.10 * i) for i in range(NUM_COLORS)
)


# ---------------------------------------------------------------------------
# Per-episode block placement
# ---------------------------------------------------------------------------


def place_clutter_blocks(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    cluster_center_x: tuple[float, float] = (0.15, 0.22),
    cluster_center_y: tuple[float, float] = (-0.10, 0.10),
    half_separation: tuple[float, float] = (0.0125, 0.030),
    table_z: float = 0.01,
    command_name: str = "target_color",
    cube_prefix: str = "cube_",
) -> None:
    """Sample the active pair + target, then place all cubes accordingly.

    All per-episode sampling happens here (NOT in the
    :class:`TargetColorCommand`) because Isaac Lab's reset pipeline runs
    event_manager BEFORE command_manager.reset — sampling in the command
    would be one episode behind. The :class:`TargetColorCommand` is a
    passive view onto the env buffers we write here.

    Writes (and lazy-allocates if needed):

    * ``env._active_cube_indices`` ``(N, 2)`` long — two distinct palette
      indices ∈ [0, NUM_COLORS) drawn uniformly without replacement.
    * ``env._target_idx_in_pair``  ``(N,)``  long ∈ {0, 1}.
    * ``env._target_cube_idx``     ``(N,)``  long ∈ [0, NUM_COLORS).

    Geometry of the active pair:

    * cluster center ``c = (cx, cy)`` is sampled uniformly from
      ``cluster_center_x × cluster_center_y``.
    * random in-plane axis angle ``θ ~ U[0, 2π)``.
    * per-env half-separation ``hs ~ U(*half_separation)``.
    * cube ``A`` (slot 0) at ``c + hs * (cos θ, sin θ)``.
    * cube ``B`` (slot 1) at ``c - hs * (cos θ, sin θ)``.

    Default ``half_separation=(0.0125, 0.030)`` puts cube centers
    2.5–6.0 cm apart per episode — for 2 cm cubes that's 0.5–4 cm of
    edge-to-edge margin. The Eval-2 spec says "adjacent (flat cluster)",
    which in practice (human-placed evaluation) means visibly separated
    but close — not touching. Sampling a range covers placement noise
    and trains a margin-robust policy.
    """
    del command_name  # The command is now passive; we own all the state.
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

    # Sample per-env (active pair, target-in-pair) — same logic that
    # used to live in TargetColorCommand._resample_command. Two distinct
    # palette indices via random-argsort top-2; coin flip for the target.
    perms = torch.argsort(torch.rand((n, NUM_COLORS), device=device), dim=1)
    active = perms[:, :2]                                          # (n, 2) long
    target_in_pair = torch.randint(0, 2, (n,), device=device)       # (n,) long
    target_palette_idx = active.gather(1, target_in_pair.view(-1, 1)).squeeze(1)

    # Allocate buffers lazily (TargetColorCommand.__init__ usually beats
    # us to it, but stay robust).
    if not hasattr(env, "_active_cube_indices"):
        env._active_cube_indices = torch.zeros((env.num_envs, 2), dtype=torch.long, device=device)
    if not hasattr(env, "_target_idx_in_pair"):
        env._target_idx_in_pair = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    if not hasattr(env, "_target_cube_idx"):
        env._target_cube_idx = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    env._active_cube_indices[env_ids_t] = active
    env._target_idx_in_pair[env_ids_t] = target_in_pair
    env._target_cube_idx[env_ids_t] = target_palette_idx

    # Robot root xy in WORLD frame for the resetting envs — env_local
    # workspace coords are added to this so each env's cubes land in its
    # own tile.
    robot = env.scene["robot"]
    root_xy_w = robot.data.root_pos_w[env_ids_t, :2]  # (n, 2)

    # Sample cluster center (env-local, robot frame) + axis + per-env
    # half-separation. Margin is sampled per episode so the policy sees
    # the full distribution from touching-ish to clearly-separated.
    cx = torch.empty(n, device=device).uniform_(*cluster_center_x)
    cy = torch.empty(n, device=device).uniform_(*cluster_center_y)
    theta = torch.empty(n, device=device).uniform_(0.0, 2.0 * math.pi)
    hs_lo, hs_hi = float(half_separation[0]), float(half_separation[1])
    hs = torch.empty(n, device=device).uniform_(hs_lo, hs_hi)
    dx = hs * torch.cos(theta)
    dy = hs * torch.sin(theta)

    cube_a_local_xy = torch.stack([cx + dx, cy + dy], dim=1)
    cube_b_local_xy = torch.stack([cx - dx, cy - dy], dim=1)
    pair_local_xy = torch.stack([cube_a_local_xy, cube_b_local_xy], dim=1)  # (n, 2, 2)

    # Identity quaternion (w, x, y, z) — cubes are spawn-identical so the
    # initial orientation doesn't matter; let physics relax them.
    quat = torch.zeros((n, 4), device=device)
    quat[:, 0] = 1.0
    zero_vel = torch.zeros((n, 6), device=device)

    # Write poses for every cube in the palette.
    for k, name in enumerate(COLOR_NAMES):
        cube = env.scene[f"{cube_prefix}{name}"]
        # Decide per-env whether THIS cube is active and which slot it
        # occupies (0 or 1), else it goes to the parked slot.
        is_slot0 = active[:, 0] == k
        is_slot1 = active[:, 1] == k
        is_active = is_slot0 | is_slot1

        target_xy = torch.empty((n, 2), device=device)
        # default = parked slot for this cube (fixed env-local xy)
        park_x, park_y = HIDDEN_PARK_XY[k]
        target_xy[:, 0] = park_x
        target_xy[:, 1] = park_y
        # overwrite for envs where this cube is active
        if is_slot0.any():
            target_xy[is_slot0] = pair_local_xy[is_slot0, 0]
        if is_slot1.any():
            target_xy[is_slot1] = pair_local_xy[is_slot1, 1]

        # Active z = table_z (rest on table). Parked z = 0.05 (will fall
        # to ground at -1.04; out of any plausible wrist-cam FOV).
        z = torch.where(is_active, torch.full_like(target_xy[:, 0], table_z), torch.full_like(target_xy[:, 0], 0.05))
        pos_local = torch.stack([target_xy[:, 0], target_xy[:, 1], z], dim=1)  # (n, 3)

        # Convert env-local (robot frame) to world: add the robot's root
        # xy (the robot is fixed-base so its root quat is identity and
        # z=0 wrt the env origin).
        pos_w = pos_local.clone()
        pos_w[:, :2] += root_xy_w

        pose = torch.cat([pos_w, quat], dim=1)
        cube.write_root_pose_to_sim(pose, env_ids=env_ids_t)
        cube.write_root_velocity_to_sim(zero_vel, env_ids=env_ids_t)

    # (Sampling + buffer writes already done at the top of this function.)


# ---------------------------------------------------------------------------
# Latch resets (separate from those in pickplace.mdp.events because they
# act on env._target_was_grasped / env._target_was_over_bowl_above_rim —
# the *target-aware* latches maintained in clutterpickplace.mdp.rewards).
# ---------------------------------------------------------------------------


def reset_target_latches(env: "ManagerBasedEnv", env_ids: torch.Tensor) -> None:
    """Clear the target-aware per-episode latches at reset.

    Specifically:

    * ``env._target_was_grasped`` — lift-once latch on the *target* cube.
    * ``env._target_was_over_bowl_above_rim`` — "approached from above"
      latch on the target.
    * ``env._target_task_success_latch`` — release-into-bowl latch, read
      by the curriculum metrics term.

    Idempotent — silent no-op if a latch buffer hasn't been allocated yet
    (first call before any reward term touched it).
    """
    if isinstance(env_ids, torch.Tensor):
        if env_ids.numel() == 0:
            return
    elif len(env_ids) == 0:
        return
    for name in ("_target_was_grasped", "_target_was_over_bowl_above_rim", "_target_task_success_latch"):
        flag = getattr(env, name, None)
        if flag is not None:
            flag[env_ids] = False
