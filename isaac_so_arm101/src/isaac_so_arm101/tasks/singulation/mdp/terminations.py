# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the singulation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from .events import COLOR_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def active_cube_off_table(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    z_min_w: float = -0.05,
    xy_max_b: float = 0.6,
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """Terminate (failure) if any *active* cube falls off the table."""
    robot: Articulation = env.scene[robot_cfg.name]
    active = env._singulation_active_mask  # (N, 6)
    out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for k, name in enumerate(COLOR_NAMES):
        cube: RigidObject = env.scene[f"{cube_prefix}{name}"]
        mask = active[:, k]
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


def singulation_done(
    env: "ManagerBasedRLEnv",
    min_separation: float = 0.05,
    on_table_height: float = 0.05,
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """Positive termination when the success indicator latch fires.

    Reads the latch maintained by
    :func:`mdp.rewards.singulation_success`. Optional — comment this
    out of TerminationsCfg if you want the policy to keep the cubes
    spread for the rest of the episode instead of ending early.
    """
    latch = getattr(env, "_singulation_success_latch", None)
    if latch is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return latch
