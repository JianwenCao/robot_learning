# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the SO-ARM101 pick-and-place task.

These functions are bound as ``ObsTerm``s in :mod:`pickplace_env_cfg`. They
are intentionally lightweight so the same definitions can be used by both
the deployable *policy* observation group and the privileged *critic*
observation group used during PPO training.

All quantities are expressed in the **robot root frame** unless stated
otherwise — this keeps the policy's input distribution invariant to the
simulation's world origin and matches how observations are constructed on
the real robot at deploy time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Block / object position
# ---------------------------------------------------------------------------


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Block xyz expressed in the robot root frame.

    Used by the **critic** (privileged) — the policy never sees this directly,
    it must infer block location from the wrist camera.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    pos_w = obj.data.root_pos_w[:, :3]
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], pos_w
    )
    return pos_b


# ---------------------------------------------------------------------------
# End-effector pose (FK on joints, expressed in robot frame)
# ---------------------------------------------------------------------------


def ee_xyz_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Full 3-D end-effector position in the robot root frame.

    Uses the ``ee_frame`` ``FrameTransformer`` configured in the scene.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_w
    )
    return ee_b


def ee_proj_xy(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End-effector projected onto the table plane (xy in robot frame).

    Recommended by the TA: gives the policy a 2-D Cartesian feature so it
    doesn't have to learn forward kinematics through its MLP. Easy to
    replicate on the real robot via the same FK chain on host.
    """
    return ee_xyz_in_robot_root_frame(env, ee_frame_cfg, robot_cfg)[:, :2]


# ---------------------------------------------------------------------------
# Bowl as a goal — read from the command manager
# ---------------------------------------------------------------------------


def bowl_xy(
    env: ManagerBasedRLEnv, command_name: str = "bowl_pose"
) -> torch.Tensor:
    """Bowl (x, y) goal in the robot frame, as set by the command manager."""
    return env.command_manager.get_command(command_name)[:, :2]


def bowl_xyz(
    env: ManagerBasedRLEnv, command_name: str = "bowl_pose"
) -> torch.Tensor:
    """Full (x, y, z) bowl goal in the robot frame."""
    return env.command_manager.get_command(command_name)[:, :3]


def ee_to_bowl_xy(
    env: ManagerBasedRLEnv,
    command_name: str = "bowl_pose",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Vector from the ee projection to the bowl, in the table plane.

    Redundant given ``ee_proj_xy`` and ``bowl_xy``, but the explicit
    subtraction is a known shortcut that accelerates reach-stage learning.
    """
    return bowl_xy(env, command_name) - ee_proj_xy(env, ee_frame_cfg, robot_cfg)


def block_to_bowl_xy(
    env: ManagerBasedRLEnv,
    command_name: str = "bowl_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Vector from the block to the bowl in the table plane (privileged)."""
    block = object_position_in_robot_root_frame(env, robot_cfg, object_cfg)[:, :2]
    return bowl_xy(env, command_name) - block


def gripper_to_block(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """3-D vector from the end-effector to the block (privileged)."""
    ee_b = ee_xyz_in_robot_root_frame(env, ee_frame_cfg, robot_cfg)
    blk_b = object_position_in_robot_root_frame(env, robot_cfg, object_cfg)
    return blk_b - ee_b


# ---------------------------------------------------------------------------
# Gripper state
# ---------------------------------------------------------------------------


def gripper_state(
    env: ManagerBasedRLEnv,
    gripper_joint_name: str = "gripper",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Current gripper joint position (single scalar per env).

    Mirrors the real-robot signal ``feetech.read_present_position(gripper_id)``.
    Joint resolved by name each call — see :func:`is_grasped` rationale.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    idx = asset.find_joints(gripper_joint_name)[0][0]
    return asset.data.joint_pos[:, idx : idx + 1]


# ---------------------------------------------------------------------------
# Grasped flag (privileged — derived from kinematics + height)
# ---------------------------------------------------------------------------


def is_grasped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    grasp_distance: float = 0.04,
    minimal_height: float = 0.025,
) -> torch.Tensor:
    """Heuristic ``is_grasped`` flag without contact sensors.

    The SO-ARM101 articulation cfg currently has ``activate_contact_sensors=False``
    (waiting on capsule support), so we approximate by:

    * block lifted above ``minimal_height``, AND
    * end-effector within ``grasp_distance`` of the block.

    Returns a float tensor of shape ``(num_envs, 1)`` so it slots into obs
    concatenation for the critic.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    block_pos_w = obj.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    dist = torch.norm(block_pos_w - ee_w, dim=1)
    lifted = block_pos_w[:, 2] > minimal_height
    close = dist < grasp_distance
    return (lifted & close).float().unsqueeze(-1)
