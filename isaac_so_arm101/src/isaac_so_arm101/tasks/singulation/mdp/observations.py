# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the Bonus-B singulation task.

Policy obs:

* State: joint pos / vel, gripper, ee xy, last action.
* Goal-conditioning: ``n_active_onehot`` (3 vs 4) + ``arrangement_onehot``
  (stacked vs clustered). Lets the policy adapt its strategy to the
  initial config (e.g. clear the top cube first for stacks).
* Wrist RGB.

Critic obs additionally exposes all six cube xyz positions (palette
order) plus the active mask, so the value function can attend to the
cubes the policy is actually trying to separate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCamera
from isaaclab.utils.math import subtract_frame_transforms

from isaac_so_arm101.tasks.pickplace.mdp.observations import (
    _normalize_rgb,
    apply_color_jitter,
)

from .events import COLOR_NAMES, NUM_COLORS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _all_cube_pos_w(env: "ManagerBasedRLEnv", cube_prefix: str = "cube_") -> torch.Tensor:
    parts = []
    for name in COLOR_NAMES:
        cube: RigidObject = env.scene[f"{cube_prefix}{name}"]
        parts.append(cube.data.root_pos_w[:, :3])
    return torch.stack(parts, dim=1)


def n_active_onehot(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """``(N, 2)`` — one-hot of n_active ∈ {3, 4}."""
    n_active = env._singulation_n_active  # values are 3 or 4
    out = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
    out[:, 0] = (n_active == 3).float()
    out[:, 1] = (n_active == 4).float()
    return out


def arrangement_onehot(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """``(N, 2)`` — one-hot of arrangement (0=stacked, 1=clustered)."""
    arr = env._singulation_arrangement
    out = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
    out[:, 0] = (arr == 0).float()
    out[:, 1] = (arr == 1).float()
    return out


def active_block_mask(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """``(N, 6)`` — palette-aligned active mask (privileged)."""
    return env._singulation_active_mask.float()


def all_cube_positions_robot_frame(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """``(N, 18)`` — palette-ordered xyz of all six cubes in robot frame. Privileged.

    Parked cubes have z ≈ -1.04 (resting on ground plane below the
    table). The critic learns to ignore them via the
    :func:`active_block_mask` signal.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    pos_w = _all_cube_pos_w(env, cube_prefix)  # (N, 6, 3)
    root_pos = robot.data.root_state_w[:, :3].unsqueeze(1).expand(-1, NUM_COLORS, -1)
    root_quat = robot.data.root_state_w[:, 3:7].unsqueeze(1).expand(-1, NUM_COLORS, -1)
    pos_b_flat, _ = subtract_frame_transforms(
        root_pos.reshape(-1, 3), root_quat.reshape(-1, 4), pos_w.reshape(-1, 3)
    )
    return pos_b_flat.reshape(-1, NUM_COLORS * 3)


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
