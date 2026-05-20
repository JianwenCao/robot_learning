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
import torch.nn.functional as F
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


# ---------------------------------------------------------------------------
# Per-color semantic-seg → target-keyed instance mask + Florence-mimic DR
# ---------------------------------------------------------------------------


def _resolve_color_class_ids(cam: TiledCamera) -> torch.Tensor | None:
    """Look up the per-color semantic class IDs and cache on ``cam``.

    Returns ``(NUM_COLORS,)`` long tensor mapping palette idx → seg class
    id, or ``None`` if the info dict isn't populated yet (first call
    before the camera has rendered). Once resolved, the tensor is cached
    as ``cam._cube_class_ids`` so subsequent calls are O(1).
    """
    cached = getattr(cam, "_cube_class_ids", None)
    if cached is not None:
        return cached
    info = cam.data.info.get("semantic_segmentation", {}) or {}
    id_map = info.get("idToLabels", {}) if isinstance(info, dict) else {}
    if not id_map:
        return None
    # Build a name → id dict by parsing each label entry.
    name_to_id: dict[str, int] = {}
    for k, v in id_map.items():
        try:
            id_int = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, dict):
            cls = v.get("class")
            if isinstance(cls, str):
                name_to_id[cls] = id_int
    ids = []
    for name in COLOR_NAMES:
        cid = name_to_id.get(f"cube_{name}")
        if cid is None:
            # Info dict is partial — wait for next call.
            return None
        ids.append(cid)
    tensor = torch.tensor(ids, dtype=torch.long, device=cam.data.output["rgb"].device)
    cam._cube_class_ids = tensor
    return tensor


def _binary_mask_for_palette_idx(
    seg: torch.Tensor, class_ids: torch.Tensor, palette_idx: torch.Tensor
) -> torch.Tensor:
    """Per-env binary mask for the palette idx in ``palette_idx``.

    Args:
        seg: ``(N, H, W)`` long seg-id image.
        class_ids: ``(NUM_COLORS,)`` long, palette idx → seg class id.
        palette_idx: ``(N,)`` long ∈ [0, NUM_COLORS).
    Returns: ``(N, 1, H, W)`` float in {0, 1}.
    """
    per_env_id = class_ids[palette_idx].view(-1, 1, 1)  # (N, 1, 1)
    return (seg == per_env_id).float().unsqueeze(1)


def _morph_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Erode (radius < 0) or dilate (radius > 0) a (N, 1, H, W) binary mask.

    Uses max-pool: dilate = max-pool, erode = -max-pool(-mask). Radius 0
    is identity. Kernel size = 2|r|+1, stride=1, padding=|r|.
    """
    if radius == 0:
        return mask
    k = 2 * abs(radius) + 1
    pad = abs(radius)
    if radius > 0:
        return F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=pad)


def wrist_rgb_mask_dr(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam"),
    corrupt: bool = True,
    rgb_brightness_jitter: float = 0.15,
    rgb_noise_std: float = 5.0 / 255.0,
    hue_microjitter_deg: float = 3.0,
    mask_dropout_prob: float = 0.10,
    mask_wrong_swap_prob: float = 0.03,
    mask_morph_max_radius: int = 2,
    mask_min_pixel_area: int = 8,
) -> torch.Tensor:
    """4-channel wrist obs: ``[R, G, B, target_mask]``.

    Channels 0–2 are RGB with the same DR pipeline as :func:`wrist_rgb_dr`
    (per-episode tint + HSV jitter, per-step brightness/hue/noise). Channel
    3 is a binary instance mask of the **target cube only**, filtered from
    the TiledCamera's ``semantic_segmentation`` output via per-color class
    IDs (each :class:`CuboidCfg` is tagged ``class:cube_<color>`` in
    :mod:`joint_pos_env_cfg`).

    The mask channel is corrupted to **mimic Florence-2's failure modes
    at deploy** (gated by ``corrupt`` — Play cfgs disable):

    * **Mask-area dropout** (``mask_min_pixel_area``): if the GT mask has
      fewer pixels than this threshold, zero it out. Models Florence-2's
      "cube too small in frame → no detection" behaviour the user
      observed at the wrist-cam working distance.
    * **Morphological jitter** (``mask_morph_max_radius``): per-env
      erode-or-dilate by a radius uniformly sampled in
      ``[-R, R]``. Models edge-pixel noise in the detector output.
    * **Full-frame dropout** (``mask_dropout_prob``): per-env Bernoulli
      probability the mask is entirely zeroed for this step. Models
      Florence-2's occasional total miss under bad lighting.
    * **Wrong-color swap** (``mask_wrong_swap_prob``): per-env Bernoulli
      probability the mask is the *distractor* cube's mask instead of the
      target's. This is the load-bearing term for sim2real robustness on
      multi-cube scenes — a single Florence misfire that masks the wrong
      colour is otherwise a catastrophic single-step failure.

    Notes:
      * The per-color class IDs aren't known until the first render
        populates the info dict; until then the mask channel is zeros
        (one transient frame at startup; identical pattern to Eval-1).
      * ``corrupt=False`` (the Play cfg path) returns the GT target mask
        with no DR — useful for eyeballing the channel in a viewer.
    """
    cam: TiledCamera = env.scene.sensors[sensor_cfg.name]
    out = cam.data.output

    # ---- RGB (verbatim from wrist_rgb_dr) ----------------------------------
    rgb = _normalize_rgb(out["rgb"])  # (N, 3, H, W) in [0, 1]
    n, _, h, w = rgb.shape

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

    # ---- Target instance mask ---------------------------------------------
    class_ids = _resolve_color_class_ids(cam)
    if class_ids is None:
        # First-frame transient — info dict not populated yet.
        mask = torch.zeros((n, 1, h, w), device=rgb.device, dtype=rgb.dtype)
        return torch.cat([rgb, mask], dim=1)

    seg = out["semantic_segmentation"]
    if seg.dim() == 4 and seg.shape[-1] == 1:
        seg = seg.squeeze(-1)
    seg = seg.long()

    target_palette = env._target_cube_idx
    mask = _binary_mask_for_palette_idx(seg, class_ids, target_palette)  # (N, 1, H, W)

    if corrupt:
        # 1) Wrong-color swap — replace target mask with the distractor's.
        if mask_wrong_swap_prob > 0.0:
            cmd = env.command_manager.get_term("target_color")
            distractor_palette = cmd.active_indices.gather(
                1, (1 - cmd.target_idx_in_pair).view(-1, 1)
            ).squeeze(1)
            distractor_mask = _binary_mask_for_palette_idx(seg, class_ids, distractor_palette)
            swap = (torch.rand(n, device=rgb.device) < mask_wrong_swap_prob).view(-1, 1, 1, 1)
            mask = torch.where(swap, distractor_mask, mask)

        # 2) Morphological jitter — per-env erode or dilate by [-R, R].
        if mask_morph_max_radius > 0:
            radii = torch.randint(
                -mask_morph_max_radius, mask_morph_max_radius + 1,
                (n,), device=rgb.device,
            )
            # Apply per radius value (small radius set, so loop is cheap).
            for r in torch.unique(radii).tolist():
                r_int = int(r)
                if r_int == 0:
                    continue
                sel = (radii == r_int)
                if not sel.any():
                    continue
                mask[sel] = _morph_mask(mask[sel], r_int)

        # 3) Small-area dropout — model "cube too small for Florence."
        if mask_min_pixel_area > 0:
            area = mask.sum(dim=(1, 2, 3))  # (N,)
            small = (area < float(mask_min_pixel_area)).view(-1, 1, 1, 1)
            mask = torch.where(small, torch.zeros_like(mask), mask)

        # 4) Full-frame dropout — model occasional total miss.
        if mask_dropout_prob > 0.0:
            drop = (torch.rand(n, device=rgb.device) < mask_dropout_prob).view(-1, 1, 1, 1)
            mask = torch.where(drop, torch.zeros_like(mask), mask)

    return torch.cat([rgb, mask.to(rgb.dtype)], dim=1)


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
