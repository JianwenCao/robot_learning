# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the Bonus-B singulation task.

Policy obs (deployable):
* State: joint pos / vel, gripper, ee xy, last action.
* Goal-conditioning: ``n_active_onehot`` (3 vs 4) + ``arrangement_onehot``
  (11-way) so the policy can adapt its strategy to the initial config.
* ``bowl_xy``: 2-D target the chained P2 (pick-and-place) reads after
  handoff; P1 only uses it to keep cubes out of the bowl.
* Wrist image: 4-channel RGB + union active-cube mask.

Critic obs additionally exposes all six cube xyz positions (palette
order), the active mask, and pairwise-distance / on-table stats so the
value function can attend to the cubes the policy is actually trying to
separate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCamera
from isaaclab.utils.math import subtract_frame_transforms

from isaac_so_arm101.tasks.pickplace.mdp.observations import (
    _normalize_rgb,
    apply_color_jitter,
)
from isaac_so_arm101.tasks.clutterpickplace.mdp.observations import (
    _morph_mask,
    _resolve_color_class_ids,
)

from .events import (
    ARRANGEMENT_NAMES,
    COLOR_NAMES,
    NUM_ARRANGEMENTS,
    NUM_COLORS,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_cube_pos_w(env: "ManagerBasedRLEnv", cube_prefix: str = "cube_") -> torch.Tensor:
    parts = []
    for name in COLOR_NAMES:
        cube: RigidObject = env.scene[f"{cube_prefix}{name}"]
        parts.append(cube.data.root_pos_w[:, :3])
    return torch.stack(parts, dim=1)


def _active_pairwise_xy_distances(
    env: "ManagerBasedRLEnv", cube_prefix: str = "cube_"
) -> torch.Tensor:
    """``(N, P)`` upper-triangular pairwise xy distances with inactive
    pairs replaced by ``+inf``.

    Mirrors the helper in ``rewards.py`` so we can expose summary stats
    as critic observations without re-implementing the masking logic.
    """
    pos = _all_cube_pos_w(env, cube_prefix)[:, :, :2]
    active = env._singulation_active_mask
    K = pos.shape[1]
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)
    pair_active = active.unsqueeze(2) & active.unsqueeze(1)
    iu, ju = torch.triu_indices(K, K, offset=1, device=pos.device)
    pair_d = dist[:, iu, ju]
    pair_a = pair_active[:, iu, ju]
    return torch.where(pair_a, pair_d, torch.full_like(pair_d, float("inf")))


# ---------------------------------------------------------------------------
# Policy obs — goal conditioning + state
# ---------------------------------------------------------------------------


def n_active_onehot(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """``(N, 2)`` — one-hot of n_active ∈ {3, 4}."""
    n_active = env._singulation_n_active
    out = torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)
    out[:, 0] = (n_active == 3).float()
    out[:, 1] = (n_active == 4).float()
    return out


def arrangement_onehot(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """``(N, NUM_ARRANGEMENTS=11)`` — one-hot of arrangement family.

    Index order matches ``ARRANGEMENT_NAMES`` from ``events.py``. At
    deploy this is populated from a CLI flag rather than the privileged
    ``_singulation_arrangement_idx`` (operator-known at eval time — see
    BONUS_B_PLAN.md §4).
    """
    idx = env._singulation_arrangement_idx
    out = torch.zeros((env.num_envs, NUM_ARRANGEMENTS), device=env.device, dtype=torch.float32)
    out.scatter_(1, idx.view(-1, 1), 1.0)
    return out


def bowl_xy(
    env: "ManagerBasedRLEnv", command_name: str = "bowl_pose"
) -> torch.Tensor:
    """``(N, 2)`` — bowl xy in robot root frame (from BowlPoseCommand).

    P1 reads this for the ``bowl_avoidance`` reward and so its wrist-image
    distribution sees the world conditioned on bowl_xy (same schema P2
    needs after handoff).
    """
    cmd = env.command_manager.get_command(command_name)
    return cmd[:, :2]


# ---------------------------------------------------------------------------
# Critic privileged obs
# ---------------------------------------------------------------------------


def active_block_mask(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """``(N, 6)`` — palette-aligned active mask (privileged)."""
    return env._singulation_active_mask.float()


def all_cube_positions_robot_frame(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """``(N, 18)`` — palette-ordered xyz of all six cubes in robot frame.

    Parked cubes have z ≈ -1.04 (resting on the ground plane below the
    table). The critic learns to ignore them via :func:`active_block_mask`.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    pos_w = _all_cube_pos_w(env, cube_prefix)
    root_pos = robot.data.root_state_w[:, :3].unsqueeze(1).expand(-1, NUM_COLORS, -1)
    root_quat = robot.data.root_state_w[:, 3:7].unsqueeze(1).expand(-1, NUM_COLORS, -1)
    pos_b_flat, _ = subtract_frame_transforms(
        root_pos.reshape(-1, 3), root_quat.reshape(-1, 4), pos_w.reshape(-1, 3)
    )
    return pos_b_flat.reshape(-1, NUM_COLORS * 3)


def min_pairwise_xy_active(env: "ManagerBasedRLEnv", cube_prefix: str = "cube_") -> torch.Tensor:
    """``(N, 1)`` — min pairwise xy distance over the active subset.

    Useful diagnostic for the critic; saturates at ~0.10 m for spread layouts.
    Returns 0 for envs where the active set has fewer than 2 cubes
    (shouldn't happen — n_active is always 3 or 4).
    """
    d = _active_pairwise_xy_distances(env, cube_prefix)
    min_d = d.min(dim=1).values
    min_d = torch.where(torch.isfinite(min_d), min_d, torch.zeros_like(min_d))
    return min_d.unsqueeze(-1)


def mean_pairwise_xy_active(env: "ManagerBasedRLEnv", cube_prefix: str = "cube_") -> torch.Tensor:
    """``(N, 1)`` — mean pairwise xy distance over active pairs."""
    d = _active_pairwise_xy_distances(env, cube_prefix)
    finite_mask = torch.isfinite(d)
    safe_d = torch.where(finite_mask, d, torch.zeros_like(d))
    count = finite_mask.float().sum(dim=1).clamp(min=1.0)
    return (safe_d.sum(dim=1) / count).unsqueeze(-1)


def n_cubes_off_table(
    env: "ManagerBasedRLEnv",
    height_threshold: float = 0.05,
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """``(N, 1)`` — number of active cubes with z above ``height_threshold``.

    Lift-status diagnostic for the critic; ranges 0 .. n_active. The
    threshold matches the `all_cubes_on_table` reward predicate.
    """
    pos = _all_cube_pos_w(env, cube_prefix)
    z = pos[:, :, 2]
    active = env._singulation_active_mask
    above = (z > height_threshold) & active
    return above.float().sum(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Wrist image — 3-ch RGB only (kept for teacher-fast / debug use)
# ---------------------------------------------------------------------------


def wrist_rgb_dr(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam"),
    corrupt: bool = True,
    rgb_brightness_jitter: float = 0.15,
    rgb_noise_std: float = 5.0 / 255.0,
    hue_microjitter_deg: float = 3.0,
) -> torch.Tensor:
    """3-channel wrist RGB. See ``clutterpickplace.mdp.observations.wrist_rgb_dr``
    for full DR semantics. Kept for teacher-fast and debug rollouts; the
    deployable singulation vision policy uses :func:`wrist_rgb_union_mask_dr`
    (4 channels)."""
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


# ---------------------------------------------------------------------------
# Wrist image — 4-ch RGB + union active-cube mask (deployable obs)
# ---------------------------------------------------------------------------


def wrist_rgb_union_mask_dr(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam"),
    corrupt: bool = True,
    rgb_brightness_jitter: float = 0.15,
    rgb_noise_std: float = 5.0 / 255.0,
    hue_microjitter_deg: float = 3.0,
    mask_dropout_prob: float = 0.10,
    mask_morph_max_radius: int = 2,
    mask_min_pixel_area: int = 8,
) -> torch.Tensor:
    """4-channel wrist obs: ``[R, G, B, union_mask]``.

    Channels 0-2 are RGB with the same per-episode tint + HSV DR + per-step
    brightness/hue/noise pipeline as
    :func:`clutterpickplace.mdp.observations.wrist_rgb_mask_dr`.

    Channel 3 is a binary mask of **all six palette cubes** (union over
    ``class:cube_*`` IDs), not just the target — singulation is colour-
    agnostic and only needs "cube vs not cube". Parked cubes don't appear
    in the wrist view (they're at x=-0.60, well outside the workspace
    band) so the union mask naturally only catches active cubes.

    Mask-channel DR matches Eval-2 §5 minus the wrong-colour-swap term
    (N/A for the union mask):

    * **Small-area dropout** (``mask_min_pixel_area``) — zero the mask
      when fewer than this many pixels are set. Models Florence-2's "cube
      too small in frame → no detection" behaviour.
    * **Morphological jitter** (``mask_morph_max_radius``) — per-env
      erode-or-dilate by a uniform radius in ``[-R, R]``. Models edge
      noise in the detector output.
    * **Full-frame dropout** (``mask_dropout_prob``) — per-env Bernoulli
      probability the mask is entirely zeroed. Forces the policy to keep
      working from RGB alone when Florence misses; critical insurance.

    All DR is gated by ``corrupt`` (Play cfgs disable).
    """
    cam: TiledCamera = env.scene.sensors[sensor_cfg.name]
    out = cam.data.output

    # ---- RGB (verbatim from wrist_rgb_dr) ----
    rgb = _normalize_rgb(out["rgb"])
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

    # ---- Union semantic-seg mask ----
    class_ids = _resolve_color_class_ids(cam)
    if class_ids is None:
        # First-frame transient — info dict not populated yet.
        mask = torch.zeros((n, 1, h, w), device=rgb.device, dtype=rgb.dtype)
        return torch.cat([rgb, mask], dim=1)

    seg = out["semantic_segmentation"]
    if seg.dim() == 4 and seg.shape[-1] == 1:
        seg = seg.squeeze(-1)
    seg = seg.long()  # (N, H, W)

    # Union over all 6 palette IDs. `torch.isin` is the clean way; on
    # older torch versions fall back to per-id OR.
    if hasattr(torch, "isin"):
        union = torch.isin(seg, class_ids).float().unsqueeze(1)  # (N, 1, H, W)
    else:
        union = torch.zeros((n, 1, h, w), device=rgb.device, dtype=rgb.dtype)
        for cid in class_ids.tolist():
            union = union + (seg == cid).float().unsqueeze(1)
        union = union.clamp_(0.0, 1.0)
    mask = union

    if corrupt:
        # Morphological jitter.
        if mask_morph_max_radius > 0:
            radii = torch.randint(
                -mask_morph_max_radius, mask_morph_max_radius + 1,
                (n,), device=rgb.device,
            )
            for r in torch.unique(radii).tolist():
                r_int = int(r)
                if r_int == 0:
                    continue
                sel = (radii == r_int)
                if not sel.any():
                    continue
                mask[sel] = _morph_mask(mask[sel], r_int)

        # Small-area dropout.
        if mask_min_pixel_area > 0:
            area = mask.sum(dim=(1, 2, 3))
            drop = (area < mask_min_pixel_area).view(-1, 1, 1, 1)
            mask = torch.where(drop, torch.zeros_like(mask), mask)

        # Full-frame Bernoulli dropout.
        if mask_dropout_prob > 0.0:
            keep = (torch.rand(n, device=rgb.device) >= mask_dropout_prob).view(-1, 1, 1, 1)
            mask = mask * keep.float()

    return torch.cat([rgb, mask], dim=1)
