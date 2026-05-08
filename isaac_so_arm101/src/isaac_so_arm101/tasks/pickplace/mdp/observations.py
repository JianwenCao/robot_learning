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


def wrist_rgb(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam"),
) -> torch.Tensor:
    """Wrist-camera RGB image, normalized to ``[0, 1]`` floats in ``(N, C, H, W)``.

    The TiledCamera returns ``(N, H, W, 3)`` uint8 (or float in [0,255]); we
    permute to channel-first and divide by 255 to match standard CNN input
    conventions. The shape ``(N, 3, H, W)`` matches what the real-side deploy
    preprocess produces after ``cv2.undistort`` + resize, so the policy sees
    the same tensor layout in both worlds.

    No further normalization (mean/std subtraction) is applied — keeping the
    raw [0,1] range means the encoder can learn its own statistics, and
    sim/real preprocessing stays as simple as possible.
    """
    cam: TiledCamera = env.scene.sensors[sensor_cfg.name]
    img = cam.data.output["rgb"]  # (N, H, W, 3) — TiledCamera convention
    # Some Isaac builds emit uint8, others float in [0,255]; normalize either.
    if img.dtype == torch.uint8:
        img = img.float() / 255.0
    else:
        img = img.float()
        if img.max() > 1.5:  # heuristic — assume [0,255] range
            img = img / 255.0
    return img.permute(0, 3, 1, 2).contiguous()  # (N, 3, H, W)
