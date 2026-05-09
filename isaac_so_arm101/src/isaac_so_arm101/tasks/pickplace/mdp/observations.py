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

import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, TiledCamera
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


# ---------------------------------------------------------------------------
# Wrist camera — RGB observation + real-cam intrinsics loader
# ---------------------------------------------------------------------------


# Project root containing ``camera_intrinsics.yaml``. Walk up from this file:
#  parents[0] mdp/
#  parents[1] pickplace/
#  parents[2] tasks/
#  parents[3] isaac_so_arm101/         (inner package)
#  parents[4] src/
#  parents[5] isaac_so_arm101/         (outer extension dir)
#  parents[6] project3/                (project root — has camera_intrinsics.yaml)
_PROJECT_ROOT = Path(__file__).resolve().parents[6]


def load_wrist_cam_intrinsics(
    yaml_path: str | os.PathLike | None = None,
    horizontal_aperture_cm: float = 20.955,
) -> dict:
    """Load real wrist-cam intrinsics and convert to Isaac ``CameraCfg`` kwargs.

    Reads ``camera_intrinsics.yaml`` (the calibration produced by
    ``cv2.calibrateCamera`` on the real wrist USB camera), extracts ``fx`` and
    image dimensions, and converts to USD pinhole-camera parameters.

    The conversion formula is

    .. code-block:: text

        focal_length = fx * horizontal_aperture / image_width

    which makes the simulated camera's horizontal FOV match the real cam's
    horizontal FOV exactly. ``horizontal_aperture`` is a free choice — Isaac's
    default ``20.955`` is what we use, so the ratio is what matters. We leave
    ``vertical_aperture`` as ``None`` (Isaac auto-derives it from the render
    aspect ratio for square pixels) — sanity check ``fy ≈ fx`` first since
    that assumption breaks otherwise.

    Args:
        yaml_path: Path to the intrinsics YAML. Defaults to ``camera_intrinsics.yaml``
            at the project root.
        horizontal_aperture_cm: USD horizontal aperture (Isaac convention is
            "in cm" per the ``PinholeCameraCfg`` docstring). Default matches
            Isaac's stock value.

    Returns:
        Dict with keys ``focal_length`` (cm), ``horizontal_aperture`` (cm),
        ``image_width`` (px), ``image_height`` (px), ``fx``, ``fy``, ``cx``, ``cy``,
        ``distortion`` (5-element list, plumb_bob k1,k2,p1,p2,k3).

    Note:
        Isaac's pinhole camera has **no distortion** — the deploy-side
        preprocess must run ``cv2.undistort`` with the returned ``distortion``
        coefficients so real frames match the perfect-pinhole sim render.
    """
    if yaml_path is None:
        yaml_path = _PROJECT_ROOT / "camera_intrinsics.yaml"
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"camera_intrinsics.yaml not found at {yaml_path}. "
            "Provide an explicit path or run wrist-cam calibration first."
        )

    # The YAML is the standard ROS camera_calibration format. The
    # ``projection_matrix`` field embeds ``!!python/object/apply`` numpy
    # scalar pickles that fail to load on newer numpy (the ``numpy._core``
    # path doesn't exist anymore). We only read ``camera_matrix`` and
    # ``distortion_coefficients`` — both plain list scalars — so we install
    # a permissive Loader that yields ``None`` for any unknown Python-tagged
    # node instead of raising.
    class _IgnorePythonTagsLoader(yaml.SafeLoader):
        pass

    def _ignore_python_object_apply(loader, tag_suffix, node):
        return None

    _IgnorePythonTagsLoader.add_multi_constructor(
        "tag:yaml.org,2002:python/object/apply:", _ignore_python_object_apply
    )
    _IgnorePythonTagsLoader.add_multi_constructor(
        "tag:yaml.org,2002:python/object/new:", _ignore_python_object_apply
    )
    _IgnorePythonTagsLoader.add_multi_constructor(
        "tag:yaml.org,2002:python/name:", _ignore_python_object_apply
    )

    with open(yaml_path, "r") as f:
        data = yaml.load(f, Loader=_IgnorePythonTagsLoader)

    K = data["camera_matrix"]["data"]  # row-major 9-element list
    fx, fy = float(K[0]), float(K[4])
    cx, cy = float(K[2]), float(K[5])
    W = int(data["image_width"])
    H = int(data["image_height"])
    dist = [float(d) for d in data["distortion_coefficients"]["data"]]

    if abs(fy - fx) / fx > 0.05:
        # Square-pixel assumption broken; Isaac's auto-derived
        # vertical_aperture won't match. Caller should set vertical_aperture
        # explicitly: vertical_aperture = fy * horizontal_aperture / fx ratio.
        import warnings

        warnings.warn(
            f"Wrist cam fx={fx:.3f}, fy={fy:.3f} differ by >5% — Isaac's "
            "auto-derived vertical_aperture (square pixels) will be off."
        )

    focal_length_cm = fx * horizontal_aperture_cm / W
    return {
        "focal_length": focal_length_cm,
        "horizontal_aperture": horizontal_aperture_cm,
        "image_width": W,
        "image_height": H,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "distortion": dist,
    }


def _normalize_rgb(img: torch.Tensor) -> torch.Tensor:
    """Convert a TiledCamera RGB tensor to ``(N, 3, H, W)`` float in ``[0,1]``."""
    if img.dtype == torch.uint8:
        img = img.float() / 255.0
    else:
        img = img.float()
        if img.max() > 1.5:
            img = img / 255.0
    return img.permute(0, 3, 1, 2).contiguous()


def wrist_rgb(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam"),
) -> torch.Tensor:
    """Legacy 3-channel RGB obs. Kept for backward compat — the live env
    uses :func:`wrist_image` (5 channels: RGB + depth + mask).
    """
    cam: TiledCamera = env.scene.sensors[sensor_cfg.name]
    return _normalize_rgb(cam.data.output["rgb"])


def wrist_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam"),
    depth_clip_m: float = 0.5,
    corrupt: bool = True,
    rgb_brightness_jitter: float = 0.15,
    rgb_noise_std: float = 5.0 / 255.0,
    depth_scale_jitter: float = 0.05,
    depth_noise_std: float = 0.01,
) -> torch.Tensor:
    """5-channel wrist-camera observation: ``[R, G, B, depth, mask]``.

    Returns ``(N, 5, H, W)`` floats in ``[0, 1]``. Sim and real-side deploy
    feed the same shape into the same CNN; on the real side, channels 3
    and 4 come from Depth Anything 3 and HSV-thresholding respectively
    (see ``DEPLOY.md`` and ``EVAL1_PLAN.md`` §8).

    Channels:

    * 0–2 RGB — TiledCamera ``rgb`` output. Optional per-step brightness
      jitter (``±rgb_brightness_jitter``) and additive Gaussian noise
      (σ=``rgb_noise_std``) when ``corrupt=True``.
    * 3 Depth — TiledCamera ``distance_to_camera`` clipped to
      ``[0, depth_clip_m]`` then divided by ``depth_clip_m`` so the
      channel sits in ``[0, 1]``. With ``corrupt=True`` we add a per-env
      multiplicative scale jitter (``±depth_scale_jitter``) and additive
      Gaussian noise (σ=``depth_noise_std``) to *mimic the artifacts a
      monocular metric depth model (DA3) produces on the real side*.
      Without this, the policy locks onto sim's perfect geometric depth
      and fails to transfer (the very gap §8 of EVAL1_PLAN warns about).
    * 4 Mask — binary cube mask from ``semantic_segmentation``. Any
      pixel labelled with the ``("class", "block")`` tag is 1.0; the
      rest is 0.0. On real, replicate via
      ``cv2.inRange(hsv, low, high)`` calibrated once on a wood-tone
      block on the gray table.

    ``corrupt`` is gated by a param the play config can override
    (``params={"corrupt": False}``) so visual-DR doesn't pollute eval.
    """
    cam: TiledCamera = env.scene.sensors[sensor_cfg.name]
    out = cam.data.output

    # ---- RGB ---------------------------------------------------------------
    rgb = _normalize_rgb(out["rgb"])  # (N, 3, H, W) in [0, 1]
    # Per-episode tint (constant within an episode; sampled at reset by
    # :func:`mdp.events.randomize_wrist_image_tint`). Substitutes for
    # material-level cube/table color DR — see that function's docstring
    # for why we do it here instead of via Isaac's Replicator path.
    dr = getattr(env, "_wrist_image_dr", None)
    if dr is not None:
        scale = dr[:, :3].view(-1, 3, 1, 1)
        bright = dr[:, 3].view(-1, 1, 1, 1)
        rgb = (rgb * scale + bright).clamp_(0.0, 1.0)
    if corrupt and rgb_brightness_jitter > 0.0:
        # Per-env scalar in [1-j, 1+j]. Broadcasts over C/H/W.
        n = rgb.shape[0]
        scale = 1.0 + (torch.rand(n, 1, 1, 1, device=rgb.device) * 2 - 1) * rgb_brightness_jitter
        rgb = (rgb * scale).clamp_(0.0, 1.0)
    if corrupt and rgb_noise_std > 0.0:
        rgb = (rgb + torch.randn_like(rgb) * rgb_noise_std).clamp_(0.0, 1.0)

    # ---- Depth -------------------------------------------------------------
    # ``distance_to_camera`` is (N, H, W, 1) in metres. Inf/NaN at sky/far →
    # treat as far-clip value, then clip to the workspace range.
    depth = out["distance_to_camera"].float()
    if depth.dim() == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    depth = torch.nan_to_num(depth, nan=depth_clip_m, posinf=depth_clip_m, neginf=0.0)
    depth = depth.clamp_(0.0, depth_clip_m) / depth_clip_m  # (N, H, W) in [0, 1]
    if corrupt:
        if depth_scale_jitter > 0.0:
            n = depth.shape[0]
            scale = 1.0 + (torch.rand(n, 1, 1, device=depth.device) * 2 - 1) * depth_scale_jitter
            depth = (depth * scale).clamp_(0.0, 1.0)
        if depth_noise_std > 0.0:
            depth = (depth + torch.randn_like(depth) * depth_noise_std).clamp_(0.0, 1.0)
    depth = depth.unsqueeze(1)  # (N, 1, H, W)

    # ---- Semantic mask -----------------------------------------------------
    # ``semantic_segmentation`` with ``colorize_semantic_segmentation=False``
    # returns (N, H, W, 1) int class IDs. Even with
    # ``semantic_filter="class:block"`` set on the camera, Isaac assigns
    # **three** IDs: 0=BACKGROUND, 1=UNLABELLED (all prims without the
    # block class — i.e. table, robot, ground), 2=block. So
    # ``(seg > 0)`` would mask the entire scene; we need to match the
    # block ID specifically. We look it up from ``info["idToLabels"]``
    # at first call and cache on the camera object (the mapping is
    # static after scene composition).
    seg = out["semantic_segmentation"]
    if seg.dim() == 4 and seg.shape[-1] == 1:
        seg = seg.squeeze(-1)
    # Cache the block ID, but ONLY after successfully matching it from
    # the info dict — never fall back to "max ID" speculatively, since
    # that would freeze the wrong ID forever (smoke test caught this:
    # info dict was empty on the first call, max-ID heuristic returned
    # 1 = UNLABELLED, mask covered 91% of the frame).
    block_id = getattr(cam, "_block_class_id", None)
    if block_id is None:
        info = cam.data.info.get("semantic_segmentation", {}) or {}
        id_map = info.get("idToLabels", {}) if isinstance(info, dict) else {}
        block_id = next(
            (int(k) for k, v in id_map.items() if isinstance(v, dict) and v.get("class") == "block"),
            None,
        )
        if block_id is not None:
            cam._block_class_id = block_id
    if block_id is not None:
        mask = (seg == block_id).float().unsqueeze(1)
    else:
        # Info dict not yet populated — this only happens on the very
        # first call before any render has completed. Return zeros for
        # this one frame; the next step will see the cached ID.
        mask = torch.zeros(seg.shape[0], 1, *seg.shape[1:], device=seg.device, dtype=torch.float32)

    return torch.cat([rgb, depth, mask], dim=1)  # (N, 5, H, W)
