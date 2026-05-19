# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the clutter pick-and-place task.

The success predicate matches :func:`mdp.rewards.release_target_in_bowl`
exactly so the TB metrics and the (optional) success termination report
the same outcome. ``block_off_table_any`` extends Eval-1's
:func:`pickplace.mdp.terminations.block_off_table` to fire if *any* of
the six palette cubes goes off the table — primarily useful for the
target, but also catches the rare case where the policy flings the
distractor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from .events import COLOR_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_task_success(
    env: "ManagerBasedRLEnv",
    r_safe: float = 0.06,
    bowl_height: float = 0.06,
    gripper_open_threshold: float = 0.2,
    block_speed_threshold: float = 0.05,
    command_name: str = "bowl_pose",
    gripper_joint_name: str = "gripper",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """Bool: target cube placed in bowl + released + settled, with both
    safety latches firing. Mirror of :func:`release_target_in_bowl`.

    Not used as a termination by default (matches Eval-1's behavior — see
    :class:`TerminationsCfg`). Kept for completeness so a downstream cfg
    can enable it cheaply.
    """
    robot: Articulation = env.scene[robot_cfg.name]

    parts = []
    for name in COLOR_NAMES:
        cube: RigidObject = env.scene[f"{cube_prefix}{name}"]
        parts.append(cube.data.root_pos_w[:, :3])
    all_pos = torch.stack(parts, dim=1)
    target_w = all_pos.gather(
        1, env._target_cube_idx.view(-1, 1, 1).expand(-1, 1, 3)
    ).squeeze(1)

    bowl_b = env.command_manager.get_command(command_name)[:, :2]
    bowl_w = robot.data.root_pos_w[:, :2] + bowl_b
    in_xy = torch.norm(target_w[:, :2] - bowl_w, dim=1) < r_safe
    low = target_w[:, 2] < bowl_height

    gripper_idx = robot.find_joints(gripper_joint_name)[0][0]
    gripper_q = robot.data.joint_pos[:, gripper_idx]
    opened = gripper_q > gripper_open_threshold

    vel_parts = []
    for name in COLOR_NAMES:
        cube: RigidObject = env.scene[f"{cube_prefix}{name}"]
        vel_parts.append(cube.data.root_lin_vel_w)
    all_vel = torch.stack(vel_parts, dim=1)
    target_v = all_vel.gather(
        1, env._target_cube_idx.view(-1, 1, 1).expand(-1, 1, 3)
    ).squeeze(1)
    settled = torch.norm(target_v, dim=1) < block_speed_threshold

    was_lifted = getattr(env, "_target_was_grasped", None)
    if was_lifted is None:
        was_lifted = torch.zeros_like(in_xy, dtype=torch.bool)
    was_over_high = getattr(env, "_target_was_over_bowl_above_rim", None)
    if was_over_high is None:
        was_over_high = torch.zeros_like(in_xy, dtype=torch.bool)

    return in_xy & low & opened & settled & was_lifted & was_over_high


def block_off_table_any(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    z_min_w: float = -0.05,
    xy_max_b: float = 0.6,
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """Terminate if **any** cube falls off the table (xy too far from robot,
    or z below the table).

    Per-cube xy is checked in the robot frame (invariant to per-env tile
    offset). z is checked in world frame against ``z_min_w`` since the
    table top is at z=0 globally. Returns ``(N,)`` bool — OR-aggregate
    over the 6 palette cubes.

    NOTE: parked inactive cubes naturally fail this check (they sit at
    z = -1.04 on the ground plane). We MUST gate the check to active
    cubes only — read ``env._active_cube_indices``.
    """
    robot: Articulation = env.scene[robot_cfg.name]

    active = env._active_cube_indices  # (N, 2) long
    # build per-env, per-active-slot mask
    out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for slot in range(2):
        # gather (N, 3) of the cube currently in this slot
        # easiest: loop over palette cubes and OR-in
        for k, name in enumerate(COLOR_NAMES):
            cube: RigidObject = env.scene[f"{cube_prefix}{name}"]
            mask = active[:, slot] == k  # (N,)
            if not mask.any():
                continue
            pos_w = cube.data.root_pos_w[mask]
            pos_b, _ = subtract_frame_transforms(
                robot.data.root_state_w[mask, :3],
                robot.data.root_state_w[mask, 3:7],
                pos_w,
            )
            too_low = pos_w[:, 2] < z_min_w
            too_far_xy = torch.norm(pos_b[:, :2], dim=1) > xy_max_b
            out[mask] = out[mask] | (too_low | too_far_xy)
    return out
