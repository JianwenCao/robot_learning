# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the SO-ARM101 pick-and-place task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_success(
    env: ManagerBasedRLEnv,
    r_safe: float = 0.06,
    bowl_height: float = 0.06,
    gripper_open_threshold: float = 0.2,
    block_speed_threshold: float = 0.05,
    command_name: str = "bowl_pose",
    gripper_joint_name: str = "gripper",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End the episode positively the first frame the release condition holds.

    Mirrors :func:`mdp.rewards.release_in_bowl` but returns a bool tensor
    expected by ``TerminationTermCfg``. Keeping the two in sync is important
    — if the success criterion drifts apart from the release reward, the
    policy may learn to dance around success without ever triggering it.

    Now also ANDs in the two per-episode latches maintained by the reward
    module so this predicate matches the tightened ``release_in_bowl``:

    * ``env._was_grasped`` — lifted ≥ 0.07 m at some prior step (closes
      drag-on-table exploit).
    * ``env._was_over_bowl_above_rim`` — cube above rim height AND over
      bowl xy at some prior step (closes lateral-slide-into-bowl exploit;
      forces over-the-top descent for real-rig deploy safety).

    Note: ``SceneEntityCfg`` defaults aren't auto-resolved when used as
    function defaults (only when explicitly passed via ``params``), so we
    look up the gripper joint index dynamically each call.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    block_xy = obj.data.root_pos_w[:, :2]
    block_z = obj.data.root_pos_w[:, 2]

    bowl_b = env.command_manager.get_command(command_name)[:, :2]
    bowl_w = robot.data.root_pos_w[:, :2] + bowl_b
    in_xy = torch.norm(block_xy - bowl_w, dim=1) < r_safe
    low = block_z < bowl_height

    gripper_idx = robot.find_joints(gripper_joint_name)[0][0]
    gripper_q = robot.data.joint_pos[:, gripper_idx]
    opened = gripper_q > gripper_open_threshold

    settled = torch.norm(obj.data.root_lin_vel_w, dim=1) < block_speed_threshold

    was_lifted = getattr(env, "_was_grasped", None)
    if was_lifted is None:
        was_lifted = torch.zeros_like(in_xy, dtype=torch.bool)
    was_over_high = getattr(env, "_was_over_bowl_above_rim", None)
    if was_over_high is None:
        was_over_high = torch.zeros_like(in_xy, dtype=torch.bool)

    return in_xy & low & opened & settled & was_lifted & was_over_high


def block_off_table(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    z_min_w: float = -0.05,
    xy_max_b: float = 0.6,
) -> torch.Tensor:
    """Terminate (failure) if the block falls off the table or is flung away.

    The xy check is done in the **robot root frame** so it's invariant to
    the per-env tile offset that ``env_spacing`` introduces in world
    coordinates (each robot's root sits at ``(env_idx_x*spacing, env_idx_y*spacing, 0)``).
    Without this conversion the check fires on every step for every env
    except env_0 — see commit notes for the chase that diagnosed it.

    The z check stays in world frame: the table's top is at z=0 globally
    so ``z_min_w=-0.05`` means "5 cm below the table", which is the same
    threshold for every env regardless of its xy tile.
    """
    from isaaclab.utils.math import subtract_frame_transforms

    obj: RigidObject = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    pos_w = obj.data.root_pos_w

    pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], pos_w
    )
    too_low = pos_w[:, 2] < z_min_w
    too_far_xy = torch.norm(pos_b[:, :2], dim=1) > xy_max_b
    return too_low | too_far_xy
