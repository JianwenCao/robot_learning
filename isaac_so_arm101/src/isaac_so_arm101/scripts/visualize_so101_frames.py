# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize SO-ARM101 joint frames and wrist camera frame in Isaac Sim.

The script parses the SO-ARM101 URDF and runs forward kinematics in the
robot root frame. It then draws classic RGB XYZ frame markers at every
joint origin, plus the wrist camera offset used by the IsaacLab camera cfg.

Run from ``robot-learning/isaac_so_arm101``:

    uv run python src/isaac_so_arm101/scripts/visualize_so101_frames.py

Useful options:

    --headless
    --print-only
    --joint-pos wrist_flex=1.57 --joint-pos gripper=0.02
    --camera-parent gripper_link
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

DEFAULT_URDF = (
    Path(__file__).resolve().parents[1]
    / "robots"
    / "trs_so101"
    / "urdf"
    / "so_arm101.urdf"
)
DEFAULT_JOINT_POS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}
HOME_JOINT_POS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 1.57,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}
CAMERA_OFFSET_POS = np.array((-0.001, 0.1, -0.04), dtype=float)
CAMERA_OFFSET_QUAT_WXYZ = np.array((-0.404379, -0.912179, -0.0451242, 0.0486914), dtype=float)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="SO-ARM101 URDF path.")
parser.add_argument("--robot-prim", default="/World/Robot", help="Prim path for the spawned robot.")
parser.add_argument("--frame-scale", type=float, default=0.12, help="Scale of XYZ frame markers.")
parser.add_argument("--root-name", default="base_link", help="URDF link used as the robot root frame.")
parser.add_argument("--camera-parent", default="gripper_link", help="URDF link that owns the wrist camera offset.")
parser.add_argument("--camera-name", default="wrist_camera", help="Name used in the printed table.")
parser.add_argument(
    "--joint-pos",
    action="append",
    default=[],
    metavar="NAME=RAD",
    help="Override a joint position in radians. Can be passed multiple times.",
)
parser.add_argument("--home-pose", action="store_true", help="Use the SO_ARM101_CFG home joint positions for FK.")
parser.add_argument(
    "--include-fixed-joints",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Also visualize fixed joints, for example gripper_frame_joint.",
)
parser.add_argument(
    "--spawn-robot",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Spawn the URDF robot mesh/articulation in the stage.",
)
parser.add_argument("--print-only", action="store_true", help="Print FK table and exit after creating no markers.")
parser.add_argument(
    "--run-seconds",
    type=float,
    default=None,
    help="Exit automatically after this many seconds. Useful for headless smoke tests.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
import isaacsim.core.utils.stage as stage_utils  # noqa: E402

from isaac_so_arm101.robots.trs_so101.so_arm101 import SO_ARM101_CFG  # noqa: E402
from isaaclab.markers import VisualizationMarkers  # noqa: E402
from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402


@dataclass(frozen=True)
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


def _parse_vector(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if value is None:
        return np.array(default, dtype=float)
    return np.array([float(v) for v in value.split()], dtype=float)


def parse_urdf(urdf_path: Path) -> list[Joint]:
    root = ET.parse(urdf_path).getroot()
    joints: list[Joint] = []
    for joint_el in root.findall("joint"):
        origin_el = joint_el.find("origin")
        parent_el = joint_el.find("parent")
        child_el = joint_el.find("child")
        axis_el = joint_el.find("axis")
        if parent_el is None or child_el is None:
            continue
        joints.append(
            Joint(
                name=joint_el.attrib["name"],
                joint_type=joint_el.attrib.get("type", "fixed"),
                parent=parent_el.attrib["link"],
                child=child_el.attrib["link"],
                xyz=_parse_vector(origin_el.attrib.get("xyz") if origin_el is not None else None, (0.0, 0.0, 0.0)),
                rpy=_parse_vector(origin_el.attrib.get("rpy") if origin_el is not None else None, (0.0, 0.0, 0.0)),
                axis=_parse_vector(axis_el.attrib.get("xyz") if axis_el is not None else None, (0.0, 0.0, 1.0)),
            )
        )
    return joints


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm < 1.0e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ]
    )


def transform(xyz: np.ndarray, rot: np.ndarray) -> np.ndarray:
    tf = np.eye(4)
    tf[:3, :3] = rot
    tf[:3, 3] = xyz
    return tf


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    trace = np.trace(rot)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array([0.25 * s, (rot[2, 1] - rot[1, 2]) / s, (rot[0, 2] - rot[2, 0]) / s, (rot[1, 0] - rot[0, 1]) / s])
    if rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        return np.array([(rot[2, 1] - rot[1, 2]) / s, 0.25 * s, (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s])
    if rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        return np.array([(rot[0, 2] - rot[2, 0]) / s, (rot[0, 1] + rot[1, 0]) / s, 0.25 * s, (rot[1, 2] + rot[2, 1]) / s])
    s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
    return np.array([(rot[1, 0] - rot[0, 1]) / s, (rot[0, 2] + rot[2, 0]) / s, (rot[1, 2] + rot[2, 1]) / s, 0.25 * s])


def parse_joint_overrides(overrides: list[str], use_home_pose: bool) -> dict[str, float]:
    joint_pos = dict(HOME_JOINT_POS if use_home_pose else DEFAULT_JOINT_POS)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Expected --joint-pos NAME=RAD, got {item!r}")
        name, value = item.split("=", 1)
        joint_pos[name] = float(value)
    return joint_pos


def compute_fk(
    joints: list[Joint], root_name: str, joint_pos: dict[str, float]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    children_by_parent: dict[str, list[Joint]] = {}
    for joint in joints:
        children_by_parent.setdefault(joint.parent, []).append(joint)

    link_tf = {root_name: np.eye(4)}
    joint_tf: dict[str, np.ndarray] = {}
    stack = [root_name]
    while stack:
        parent = stack.pop()
        parent_tf = link_tf[parent]
        for joint in children_by_parent.get(parent, []):
            origin_tf = parent_tf @ transform(joint.xyz, rpy_matrix(joint.rpy))
            joint_tf[joint.name] = origin_tf
            if joint.joint_type in ("revolute", "continuous"):
                motion_tf = transform(np.zeros(3), axis_angle_matrix(joint.axis, joint_pos.get(joint.name, 0.0)))
            elif joint.joint_type == "prismatic":
                motion_tf = transform(joint.axis * joint_pos.get(joint.name, 0.0), np.eye(3))
            else:
                motion_tf = np.eye(4)
            link_tf[joint.child] = origin_tf @ motion_tf
            stack.append(joint.child)
    return link_tf, joint_tf


def print_pose_table(poses: list[tuple[str, np.ndarray]]) -> None:
    out = sys.__stdout__
    out.write("\nName                       root_x      root_y      root_z\n")
    out.write("---------------------------------------------------------\n")
    for name, tf in poses:
        pos = tf[:3, 3]
        out.write(f"{name:<24} {pos[0]:>9.5f} {pos[1]:>9.5f} {pos[2]:>9.5f}\n")
    out.flush()


def spawn_robot() -> None:
    enable_extension("isaacsim.asset.importer.urdf")
    SO_ARM101_CFG.spawn.func(args_cli.robot_prim, SO_ARM101_CFG.spawn)
    stage_utils.update_stage()


def main() -> None:
    joints = parse_urdf(args_cli.urdf)
    joint_pos = parse_joint_overrides(args_cli.joint_pos, args_cli.home_pose)
    link_tf, joint_tf = compute_fk(joints, args_cli.root_name, joint_pos)

    if args_cli.camera_parent not in link_tf:
        known = ", ".join(sorted(link_tf))
        raise ValueError(f"Unknown --camera-parent {args_cli.camera_parent!r}. Known links: {known}")

    camera_tf = link_tf[args_cli.camera_parent] @ transform(CAMERA_OFFSET_POS, quat_wxyz_to_matrix(CAMERA_OFFSET_QUAT_WXYZ))

    visible_joint_items = [
        (joint.name, joint_tf[joint.name])
        for joint in joints
        if args_cli.include_fixed_joints or joint.joint_type != "fixed"
    ]
    poses = [(args_cli.root_name, np.eye(4)), *visible_joint_items, (args_cli.camera_name, camera_tf)]
    print_pose_table(poses)

    if args_cli.print_only:
        return

    if args_cli.spawn_robot:
        spawn_robot()
        if args_cli.home_pose or args_cli.joint_pos:
            print(
                "[visualize] Note: FK markers use the requested joint positions, "
                "while the imported URDF mesh remains at its authored zero pose.",
                flush=True,
            )
    else:
        stage_utils.update_stage()

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.prim_path = "/Visuals/SO101JointAndCameraFrames"
    marker_cfg.markers["frame"].scale = (args_cli.frame_scale, args_cli.frame_scale, args_cli.frame_scale)
    markers = VisualizationMarkers(marker_cfg)

    translations = np.stack([tf[:3, 3] for _, tf in poses], axis=0).astype(np.float32)
    orientations = np.stack([matrix_to_quat_wxyz(tf[:3, :3]) for _, tf in poses], axis=0).astype(np.float32)
    marker_indices = np.zeros(len(poses), dtype=np.int64)
    markers.visualize(translations=translations, orientations=orientations, marker_indices=marker_indices)
    stage_utils.update_stage()

    print("\n[visualize] Red=X, green=Y, blue=Z. Close the Isaac Sim window to exit.", flush=True)
    if args_cli.headless and args_cli.run_seconds is not None:
        return
    start_time = time.monotonic()
    while simulation_app.is_running():
        if args_cli.run_seconds is not None and time.monotonic() - start_time >= args_cli.run_seconds:
            break
        simulation_app.update()


if __name__ == "__main__":
    main()
    simulation_app.close()
