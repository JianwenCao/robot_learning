# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the SO-ARM101 pick-and-place task.

Stage gating is critical: without it the dense reach term dominates and the
policy never moves on to grasp. Each reward returns ``(num_envs,)``; the
weights are set in ``pickplace_env_cfg.RewardsCfg``.

Stages:
    1. ``reach_block``   — dense, ``(1 - is_grasped) * tanh(...)`` until grasp.
    2. ``grasp_event``   — sparse one-shot bonus the first time the block is
       picked up (latched per episode).
    3. ``transport``     — dense, gated on ``is_grasped``: gripper-xy → bowl-xy.
    4. ``place``         — sparse, block xy near bowl AND block low.
    5. ``release``       — terminal, place-condition AND gripper opened AND
       block roughly stationary.

Penalties:
    * action L2 (cheap regularizer)
    * action-rate L2 (smoothness — important for sim-to-real)
    * drop after grasp
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Helpers (kept private so they don't get auto-exported into mdp.*)
# ---------------------------------------------------------------------------


def _grasped_mask(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    grasp_distance: float,
    minimal_height: float,
) -> torch.Tensor:
    """Bool tensor (num_envs,) — same heuristic as ``observations.is_grasped``.

    Duplicated locally so reward functions don't take a hard dep on the obs
    module's signatures (and so they can be tweaked independently).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    block_pos_w = obj.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    dist = torch.norm(block_pos_w - ee_w, dim=1)
    return (block_pos_w[:, 2] > minimal_height) & (dist < grasp_distance)


def _bowl_xy_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Bowl xy in the *world* frame (matches the block's world-frame xy).

    The command lives in the robot root frame; we transform it to world by
    adding the robot's root xy. (For a fixed-base arm this is just an offset.)
    """
    robot: Articulation = env.scene["robot"]
    bowl_b = env.command_manager.get_command(command_name)[:, :2]
    return robot.data.root_pos_w[:, :2] + bowl_b


# ---------------------------------------------------------------------------
# Stage 1 — reach
# ---------------------------------------------------------------------------


def reach_block(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    grasp_distance: float = 0.04,
    minimal_height: float = 0.025,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Dense reach reward, *gated off* once the block is grasped.

    Without the gate this term dominates throughout the episode and the
    policy hovers above the block instead of advancing to grasp.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    dist = torch.norm(obj.data.root_pos_w - ee_frame.data.target_pos_w[..., 0, :], dim=1)
    grasped = _grasped_mask(env, object_cfg, ee_frame_cfg, grasp_distance, minimal_height)
    return (1.0 - grasped.float()) * (1.0 - torch.tanh(dist / std))


# ---------------------------------------------------------------------------
# Stage 2 — grasp event (one-shot per episode)
# ---------------------------------------------------------------------------


def grasp_event(
    env: ManagerBasedRLEnv,
    grasp_distance: float = 0.04,
    minimal_height: float = 0.025,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Indicator (0/1) that the block is currently grasped.

    Note: this is *not* one-shot — it pays out every step the block is held.
    Combined with ``transport`` (also gated on grasp) this works well in
    practice and avoids the bookkeeping of an episode-state buffer that
    Isaac Lab reward terms don't carry by default.
    """
    grasped = _grasped_mask(env, object_cfg, ee_frame_cfg, grasp_distance, minimal_height)
    return grasped.float()


# ---------------------------------------------------------------------------
# Stage 3 — transport (gripper xy → bowl xy, gated on grasp)
# ---------------------------------------------------------------------------


def transport_to_bowl(
    env: ManagerBasedRLEnv,
    std: float = 0.15,
    grasp_distance: float = 0.04,
    minimal_height: float = 0.025,
    command_name: str = "bowl_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Dense gripper-xy → bowl-xy reward, only active while holding the block.

    Gating on ``is_grasped`` is what stops the policy from learning to fly
    the empty gripper over the bowl.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_xy_w = ee_frame.data.target_pos_w[..., 0, :2]
    bowl_w = _bowl_xy_w(env, command_name)
    dist = torch.norm(ee_xy_w - bowl_w, dim=1)
    grasped = _grasped_mask(env, object_cfg, ee_frame_cfg, grasp_distance, minimal_height)
    return grasped.float() * (1.0 - torch.tanh(dist / std))


# ---------------------------------------------------------------------------
# Stage 4 — place (block over bowl AND low)
# ---------------------------------------------------------------------------


def place_in_bowl(
    env: ManagerBasedRLEnv,
    r_safe: float = 0.06,
    bowl_height: float = 0.06,
    grasp_distance: float = 0.04,
    minimal_height: float = 0.025,
    command_name: str = "bowl_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for *positioning* the held block over the bowl.

    Gated on ``is_grasped`` — without that gate, a block sitting on the
    table that happened to spawn near the bowl xy pays out every step
    (observed empirically in the 100-iter smoke run: place reward 1.2/episode
    with grasp reward ~0). This matches the lift-task pattern of
    ``object_goal_distance`` multiplied by ``(object.z > minimal_height)``.

    See §3.3 of EVAL1_PLAN — the bowl is not modeled as a mesh, so the
    geometric check is "xy near goal AND block roughly at table height".
    """
    obj: RigidObject = env.scene[object_cfg.name]
    block_xy = obj.data.root_pos_w[:, :2]
    block_z = obj.data.root_pos_w[:, 2]
    bowl_w = _bowl_xy_w(env, command_name)
    in_xy = torch.norm(block_xy - bowl_w, dim=1) < r_safe
    low = block_z < bowl_height
    grasped = _grasped_mask(env, object_cfg, ee_frame_cfg, grasp_distance, minimal_height)
    return (in_xy & low & grasped).float()


# ---------------------------------------------------------------------------
# Stage 5 — release (terminal: place + gripper open + block stationary)
# ---------------------------------------------------------------------------


def release_in_bowl(
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
    """Terminal release reward — fires when the policy actually lets go.

    All three conditions must hold simultaneously:

    * place condition (block xy near bowl, block low),
    * gripper joint position above ``gripper_open_threshold`` (i.e. open),
    * block linear speed below ``block_speed_threshold`` m/s (settled).

    See :func:`mdp.terminations.task_success` for why the gripper joint is
    resolved by name inside the function rather than via ``SceneEntityCfg``.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # place
    block_xy = obj.data.root_pos_w[:, :2]
    block_z = obj.data.root_pos_w[:, 2]
    bowl_w = _bowl_xy_w(env, command_name)
    in_xy = torch.norm(block_xy - bowl_w, dim=1) < r_safe
    low = block_z < bowl_height

    # gripper open (gripper joint is in [0, 1.7] roughly; >0.2 means opened)
    gripper_idx = robot.find_joints(gripper_joint_name)[0][0]
    gripper_q = robot.data.joint_pos[:, gripper_idx]
    opened = gripper_q > gripper_open_threshold

    # block roughly stationary
    settled = torch.norm(obj.data.root_lin_vel_w, dim=1) < block_speed_threshold

    return (in_xy & low & opened & settled).float()


# ---------------------------------------------------------------------------
# Penalties
# ---------------------------------------------------------------------------


def block_dropped(
    env: ManagerBasedRLEnv,
    drop_height: float = 0.005,
    r_safe: float = 0.06,
    command_name: str = "bowl_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalty for dropping the block on the table away from the bowl.

    Returns ``1.0`` when the block is on the table (z < drop_height) AND
    not within the bowl radius. Multiplied by a negative weight in cfg.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    block_z = obj.data.root_pos_w[:, 2]
    bowl_w = _bowl_xy_w(env, command_name)
    far_from_bowl = torch.norm(obj.data.root_pos_w[:, :2] - bowl_w, dim=1) >= r_safe
    on_table = block_z < drop_height
    return (on_table & far_from_bowl).float()
