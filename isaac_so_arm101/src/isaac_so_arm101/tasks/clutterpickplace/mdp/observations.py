# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Target-aware observation terms for the clutter pick-and-place task.

The "object" of the task is the *target* cube whose palette index is
written to ``env._target_cube_idx`` by :func:`mdp.events.place_clutter_blocks`.
These observation terms gather positions from all 6 cube assets and index
into them per env.

The deployable **policy** group sees only:

* The target color one-hot (6 dims) — the goal-conditioning input.
* Standard state + wrist image. **No** privileged cube positions.

The privileged **critic** group additionally sees the target and
distractor positions plus the standard distance-to-bowl features. Same
asymmetric A-C pattern as Eval-1.
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

from .events import COLOR_NAMES, NUM_COLORS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Internal helper — gather all cube positions into a single tensor and pick
# out the target / distractor per env.
# ---------------------------------------------------------------------------


def _all_cube_pos_w(env: "ManagerBasedRLEnv", cube_prefix: str = "cube_") -> torch.Tensor:
    """Stack world-frame positions of all palette cubes: ``(N, NUM_COLORS, 3)``."""
    parts = []
    for name in COLOR_NAMES:
        cube: RigidObject = env.scene[f"{cube_prefix}{name}"]
        parts.append(cube.data.root_pos_w[:, :3])
    return torch.stack(parts, dim=1)


def _gather_by_index(all_pos: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """``all_pos`` is ``(N, K, 3)``; ``idx`` is ``(N,)`` long ∈ [0, K). Returns ``(N, 3)``."""
    return all_pos.gather(1, idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)


# ---------------------------------------------------------------------------
# Policy goal-conditioning input
# ---------------------------------------------------------------------------


def target_color_onehot(
    env: "ManagerBasedRLEnv", command_name: str = "target_color"
) -> torch.Tensor:
    """One-hot of the target color, ``(N, NUM_COLORS=6)``.

    Read straight from :class:`TargetColorCommand` (which exposes the
    one-hot as its ``command`` tensor). Goal-conditioning input for the
    policy. At eval time on the real arm, you pass in the same 6-dim
    one-hot that corresponds to the human-specified target color.
    """
    return env.command_manager.get_command(command_name)


# ---------------------------------------------------------------------------
# Critic privileged observations (target / distractor positions, distances)
# ---------------------------------------------------------------------------


def target_block_position(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """Target cube xyz in the robot root frame, ``(N, 3)``. Privileged."""
    robot: Articulation = env.scene[robot_cfg.name]
    target_w = _gather_by_index(_all_cube_pos_w(env, cube_prefix), env._target_cube_idx)
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], target_w
    )
    return pos_b


def distractor_block_position(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """Distractor (the *other* active cube) xyz in the robot root frame.

    Privileged — the policy doesn't see this, but it lets the critic
    estimate value better when the distractor is in the way.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    # distractor = active[:, 1 - target_idx_in_pair]
    cmd = env.command_manager.get_term("target_color")
    distractor_idx = cmd.active_indices.gather(
        1, (1 - cmd.target_idx_in_pair).view(-1, 1)
    ).squeeze(1)
    distractor_w = _gather_by_index(_all_cube_pos_w(env, cube_prefix), distractor_idx)
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], distractor_w
    )
    return pos_b


def target_block_to_bowl_xy(
    env: "ManagerBasedRLEnv",
    command_name: str = "bowl_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """``bowl_xy - target_xy`` in the robot root frame, ``(N, 2)``. Privileged."""
    bowl_b = env.command_manager.get_command(command_name)[:, :2]
    target_b = target_block_position(env, robot_cfg, cube_prefix)[:, :2]
    return bowl_b - target_b


def target_gripper_to_block(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """3-D vector from end-effector to *target* cube. Privileged."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_w
    )
    blk_b = target_block_position(env, robot_cfg, cube_prefix)
    return blk_b - ee_b


def wrist_rgb_dr(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam"),
    corrupt: bool = True,
    rgb_brightness_jitter: float = 0.15,
    rgb_noise_std: float = 5.0 / 255.0,
    hue_microjitter_deg: float = 3.0,
) -> torch.Tensor:
    """3-channel wrist RGB with full color DR for sim2real robustness.

    Layers (in order):

    1. **Per-channel tint** from ``env._wrist_image_dr`` (Eval-1 legacy
       linear tint, sampled at reset).
    2. **Per-episode HSV jitter** from ``env._wrist_hsv_dr`` — applies a
       hue rotation (around the gray axis), saturation scale, and value
       scale. This is the key add for Eval-2/3 color-conditioned tasks:
       a linear RGB tint alone can't shift hue, so under realistic
       lighting the trained policy would mis-classify colors. Sampled
       per-episode by :func:`randomize_wrist_hsv_dr`.
    3. **Per-step jitter** (gated by ``corrupt``):
       - Per-env brightness scale ±``rgb_brightness_jitter``.
       - Tiny per-step hue micro-jitter ±``hue_microjitter_deg`` —
         DrQ-style frame-to-frame augmentation that prevents
         memorization of any specific calibrated WB.
       - Gaussian RGB noise σ=``rgb_noise_std``.

    Returns ``(N, 3, H, W)`` in [0, 1].
    """
    cam: TiledCamera = env.scene.sensors[sensor_cfg.name]
    rgb = _normalize_rgb(cam.data.output["rgb"])
    n = rgb.shape[0]

    # 1) Per-channel tint.
    dr = getattr(env, "_wrist_image_dr", None)
    if dr is not None:
        scale = dr[:, :3].view(-1, 3, 1, 1)
        bright = dr[:, 3].view(-1, 1, 1, 1)
        rgb = (rgb * scale + bright).clamp_(0.0, 1.0)

    # 2) Per-episode HSV.
    hsv_dr = getattr(env, "_wrist_hsv_dr", None)
    if hsv_dr is not None:
        rgb = apply_color_jitter(rgb, hsv_dr[:, 0], hsv_dr[:, 1], hsv_dr[:, 2])

    # 3) Per-step jitter.
    if corrupt and rgb_brightness_jitter > 0.0:
        bscale = 1.0 + (torch.rand(n, 1, 1, 1, device=rgb.device) * 2 - 1) * rgb_brightness_jitter
        rgb = (rgb * bscale).clamp_(0.0, 1.0)
    if corrupt and hue_microjitter_deg > 0.0:
        import math as _math
        micro = (torch.rand(n, device=rgb.device) * 2 - 1) * hue_microjitter_deg * (_math.pi / 180.0)
        rgb = apply_color_jitter(
            rgb,
            micro,
            torch.ones(n, device=rgb.device),
            torch.ones(n, device=rgb.device),
        )
    if corrupt and rgb_noise_std > 0.0:
        rgb = (rgb + torch.randn_like(rgb) * rgb_noise_std).clamp_(0.0, 1.0)
    return rgb


def target_is_grasped(
    env: "ManagerBasedRLEnv",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_prefix: str = "cube_",
    grasp_distance: float = 0.04,
    minimal_height: float = 0.025,
) -> torch.Tensor:
    """Heuristic grasp flag against the *target* cube. Privileged.

    Same form as Eval-1's :func:`mdp.is_grasped` but indexed at the
    target cube. Returns ``(N, 1)`` float.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    target_w = _gather_by_index(_all_cube_pos_w(env, cube_prefix), env._target_cube_idx)
    dist = torch.norm(target_w - ee_w, dim=1)
    lifted = target_w[:, 2] > minimal_height
    close = dist < grasp_distance
    return (lifted & close).float().unsqueeze(-1)
