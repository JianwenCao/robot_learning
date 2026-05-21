# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for Eval-3 (sequential pick-and-place)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from .events import COLOR_NAMES, N_ACTIVE_BLOCKS, N_GOAL_STEPS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def all_steps_done(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Terminate (positive) when the single training target has completed."""
    return env._seq_step_idx >= N_GOAL_STEPS


def active_block_off_table(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    z_min_w: float = -0.05,
    xy_max_b: float = 0.6,
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """Terminate (failure) if any actually active cube falls off the table."""
    robot: Articulation = env.scene[robot_cfg.name]
    active = env._active_cube_indices  # (N, 4)
    active_count = getattr(env, "_seq_active_count", None)
    out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for slot in range(N_ACTIVE_BLOCKS):
        for k, name in enumerate(COLOR_NAMES):
            cube: RigidObject = env.scene[f"{cube_prefix}{name}"]
            mask = active[:, slot] == k
            if active_count is not None:
                mask = mask & (slot < active_count)
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
