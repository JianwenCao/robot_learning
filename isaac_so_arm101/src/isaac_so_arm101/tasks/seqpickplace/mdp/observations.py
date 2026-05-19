# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for Eval-3 (sequential pick-and-place).

The policy obs is the same shape concept as Eval-2 but the goal-conditioning
vector encodes the *current* sub-goal: target color one-hot + current bowl
xy + step one-hot. Privileged critic obs additionally exposes the full
3-step schedule, the four active cube positions, and the current target
cube's position.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import FrameTransformer, TiledCamera
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from isaac_so_arm101.tasks.pickplace.mdp.observations import (
    _normalize_rgb,
    apply_color_jitter,
)

from .events import COLOR_NAMES, NUM_COLORS, N_ACTIVE_BLOCKS, N_GOAL_STEPS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Cube position helpers
# ---------------------------------------------------------------------------


def _all_cube_pos_w(env: "ManagerBasedRLEnv", cube_prefix: str = "cube_") -> torch.Tensor:
    parts = []
    for name in COLOR_NAMES:
        cube: RigidObject = env.scene[f"{cube_prefix}{name}"]
        parts.append(cube.data.root_pos_w[:, :3])
    return torch.stack(parts, dim=1)


def _current_target_palette_idx(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """``(N,)`` palette index of the current step's target cube."""
    step = env._seq_step_idx.clamp(max=N_GOAL_STEPS - 1)
    return env._target_cube_idx_per_step.gather(1, step.view(-1, 1)).squeeze(1)


# ---------------------------------------------------------------------------
# Policy goal-conditioning input
# ---------------------------------------------------------------------------


def seq_goal_vector(env: "ManagerBasedRLEnv", command_name: str = "seq_goal") -> torch.Tensor:
    """``(N, 11)`` — target color one-hot + bowl xy + step one-hot.

    Identical to ``env.command_manager.get_command(command_name)`` but
    exposed under the obs API for consistency.
    """
    return env.command_manager.get_command(command_name)


def current_target_color_onehot(
    env: "ManagerBasedRLEnv", command_name: str = "seq_goal"
) -> torch.Tensor:
    """``(N, 6)`` one-hot of the current step's target color."""
    return env.command_manager.get_command(command_name)[:, :NUM_COLORS]


def current_target_bowl_xy(
    env: "ManagerBasedRLEnv", command_name: str = "seq_goal"
) -> torch.Tensor:
    """``(N, 2)`` current step's bowl xy in robot frame."""
    return env.command_manager.get_command(command_name)[:, NUM_COLORS:NUM_COLORS + 2]


def current_step_onehot(
    env: "ManagerBasedRLEnv", command_name: str = "seq_goal"
) -> torch.Tensor:
    """``(N, 3)`` one-hot of the current step idx (clamped to last)."""
    return env.command_manager.get_command(command_name)[:, NUM_COLORS + 2:]


# ---------------------------------------------------------------------------
# Critic privileged obs
# ---------------------------------------------------------------------------


def all_active_block_positions(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """``(N, 12)`` — xyz of the 4 active cubes, concatenated in active-set order.

    Privileged: gives the critic full geometric awareness of the layout
    so it can credit the actor for navigating around the 3 non-current-
    target cubes correctly. Concatenation order matches
    ``cmd.active_indices`` so the critic can correlate each slot's
    position with which palette color sits there.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    all_pos = _all_cube_pos_w(env, cube_prefix)  # (N, 6, 3)
    active = env._active_cube_indices  # (N, 4) long
    active_w = all_pos.gather(1, active.view(-1, N_ACTIVE_BLOCKS, 1).expand(-1, -1, 3))
    # to robot frame
    root_pos = robot.data.root_state_w[:, :3].unsqueeze(1)
    root_quat = robot.data.root_state_w[:, 3:7].unsqueeze(1).expand(-1, N_ACTIVE_BLOCKS, -1)
    # subtract_frame_transforms is batched on the leading dim; reshape
    pos_w_flat = active_w.reshape(-1, 3)
    pos_root = root_pos.expand(-1, N_ACTIVE_BLOCKS, -1).reshape(-1, 3)
    quat_root = root_quat.reshape(-1, 4)
    pos_b_flat, _ = subtract_frame_transforms(pos_root, quat_root, pos_w_flat)
    pos_b = pos_b_flat.reshape(-1, N_ACTIVE_BLOCKS * 3)
    return pos_b


def current_target_block_position(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """``(N, 3)`` xyz of the current target cube in robot root frame. Privileged."""
    robot: Articulation = env.scene[robot_cfg.name]
    palette_idx = _current_target_palette_idx(env)
    all_pos = _all_cube_pos_w(env, cube_prefix)
    target_w = all_pos.gather(1, palette_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], target_w
    )
    return pos_b


def current_target_gripper_to_block(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """``(N, 3)`` EE→current_target vector in robot root frame. Privileged."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_w
    )
    blk_b = current_target_block_position(env, robot_cfg, cube_prefix)
    return blk_b - ee_b


def current_target_block_to_bowl_xy(
    env: "ManagerBasedRLEnv",
    command_name: str = "seq_goal",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """``(N, 2)`` current_target → current_bowl in robot frame. Privileged."""
    bowl_xy = current_target_bowl_xy(env, command_name)
    target = current_target_block_position(env, robot_cfg, cube_prefix)[:, :2]
    return bowl_xy - target


# ---------------------------------------------------------------------------
# Wrist RGB (DR-applied; no seg)
# ---------------------------------------------------------------------------


def wrist_rgb_dr(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam"),
    corrupt: bool = True,
    rgb_brightness_jitter: float = 0.15,
    rgb_noise_std: float = 5.0 / 255.0,
    hue_microjitter_deg: float = 3.0,
) -> torch.Tensor:
    """See :func:`clutterpickplace.mdp.observations.wrist_rgb_dr` for full DR semantics."""
    cam: TiledCamera = env.scene.sensors[sensor_cfg.name]
    rgb = _normalize_rgb(cam.data.output["rgb"])
    n = rgb.shape[0]

    dr = getattr(env, "_wrist_image_dr", None)
    if dr is not None:
        scale = dr[:, :3].view(-1, 3, 1, 1)
        bright = dr[:, 3].view(-1, 1, 1, 1)
        rgb = (rgb * scale + bright).clamp_(0.0, 1.0)

    hsv_dr = getattr(env, "_wrist_hsv_dr", None)
    if hsv_dr is not None:
        rgb = apply_color_jitter(rgb, hsv_dr[:, 0], hsv_dr[:, 1], hsv_dr[:, 2])

    if corrupt and rgb_brightness_jitter > 0.0:
        bscale = 1.0 + (torch.rand(n, 1, 1, 1, device=rgb.device) * 2 - 1) * rgb_brightness_jitter
        rgb = (rgb * bscale).clamp_(0.0, 1.0)
    if corrupt and hue_microjitter_deg > 0.0:
        import math as _math
        micro = (torch.rand(n, device=rgb.device) * 2 - 1) * hue_microjitter_deg * (_math.pi / 180.0)
        rgb = apply_color_jitter(
            rgb, micro,
            torch.ones(n, device=rgb.device),
            torch.ones(n, device=rgb.device),
        )
    if corrupt and rgb_noise_std > 0.0:
        rgb = (rgb + torch.randn_like(rgb) * rgb_noise_std).clamp_(0.0, 1.0)
    return rgb
