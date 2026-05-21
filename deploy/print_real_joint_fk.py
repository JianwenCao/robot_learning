"""Print real SO-ARM101 joint frames from URDF forward kinematics.

The script reads the current LeRobot SO-ARM101 joint positions, applies them
to the URDF kinematic tree, and prints each joint frame relative to
``base_link``. It does not start Isaac Sim.
"""
from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GRIPPER_JAW_RAD_MIN = -0.174533
GRIPPER_JAW_RAD_MAX = 1.74533
GRIPPER_JAW_RAD_SPAN = GRIPPER_JAW_RAD_MAX - GRIPPER_JAW_RAD_MIN

DEFAULT_URDF = (
    PROJECT_ROOT
    / "isaac_so_arm101"
    / "src"
    / "isaac_so_arm101"
    / "robots"
    / "trs_so101"
    / "urdf"
    / "so_arm101.urdf"
)
CAMERA_PARENT_LINK = "gripper_link"
CAMERA_NAME = "wrist_cam_theory"
# Matches ``CameraCfg.OffsetCfg`` in
# isaac_so_arm101/tasks/pickplace/joint_pos_env_cfg.py. The camera is attached
# to ``Robot/gripper_link/wrist_cam`` with convention="ros".
CAMERA_OFFSET_POS = np.array([-0.001, 0.1, -0.04], dtype=np.float64)
CAMERA_OFFSET_QUAT_WXYZ = np.array([-0.404379, -0.912179, -0.0451242, 0.0486914], dtype=np.float64)


def _gripper_sim_rad_from_pct(pct: float) -> float:
    return pct / 100.0 * GRIPPER_JAW_RAD_SPAN + GRIPPER_JAW_RAD_MIN


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
        return np.array(default, dtype=np.float64)
    return np.array([float(v) for v in value.split()], dtype=np.float64)


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
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x + z * s - y * x * c, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def transform(xyz: np.ndarray, rot: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = xyz
    return out


def matrix_to_rpy(rot: np.ndarray) -> np.ndarray:
    sy = math.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0])
    singular = sy < 1.0e-9
    if not singular:
        roll = math.atan2(rot[2, 1], rot[2, 2])
        pitch = math.atan2(-rot[2, 0], sy)
        yaw = math.atan2(rot[1, 0], rot[0, 0])
    else:
        roll = math.atan2(-rot[1, 2], rot[1, 1])
        pitch = math.atan2(-rot[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    trace = np.trace(rot)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array(
            [0.25 * s, (rot[2, 1] - rot[1, 2]) / s, (rot[0, 2] - rot[2, 0]) / s, (rot[1, 0] - rot[0, 1]) / s],
            dtype=np.float64,
        )
    if rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        return np.array(
            [(rot[2, 1] - rot[1, 2]) / s, 0.25 * s, (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s],
            dtype=np.float64,
        )
    if rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        return np.array(
            [(rot[0, 2] - rot[2, 0]) / s, (rot[0, 1] + rot[1, 0]) / s, 0.25 * s, (rot[1, 2] + rot[2, 1]) / s],
            dtype=np.float64,
        )
    s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
    return np.array(
        [(rot[1, 0] - rot[0, 1]) / s, (rot[0, 2] + rot[2, 0]) / s, (rot[1, 2] + rot[2, 1]) / s, 0.25 * s],
        dtype=np.float64,
    )


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def compute_fk(
    joints: list[Joint],
    *,
    root_link: str,
    joint_pos_rad: dict[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    children_by_parent: dict[str, list[Joint]] = {}
    for joint in joints:
        children_by_parent.setdefault(joint.parent, []).append(joint)

    link_tf: dict[str, np.ndarray] = {root_link: np.eye(4, dtype=np.float64)}
    joint_tf: dict[str, np.ndarray] = {}
    stack = [root_link]
    while stack:
        parent = stack.pop()
        parent_tf = link_tf[parent]
        for joint in children_by_parent.get(parent, []):
            origin_tf = parent_tf @ transform(joint.xyz, rpy_matrix(joint.rpy))
            joint_tf[joint.name] = origin_tf
            q = float(joint_pos_rad.get(joint.name, 0.0))
            if joint.joint_type in ("revolute", "continuous"):
                motion_tf = transform(np.zeros(3, dtype=np.float64), axis_angle_matrix(joint.axis, q))
            elif joint.joint_type == "prismatic":
                motion_tf = transform(joint.axis * q, np.eye(3, dtype=np.float64))
            else:
                motion_tf = np.eye(4, dtype=np.float64)
            link_tf[joint.child] = origin_tf @ motion_tf
            stack.append(joint.child)
    return link_tf, joint_tf


def read_real_joint_pos_rad(port: str, robot_id: str, calibrate: bool, disable_torque: bool) -> dict[str, float]:
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    cfg = SO101FollowerConfig(port=port, id=robot_id)
    robot = SO101Follower(cfg)
    try:
        robot.connect(calibrate=calibrate)
        if disable_torque:
            robot.bus.disable_torque()
        obs = robot.get_observation()
        q: dict[str, float] = {}
        for name in JOINT_NAMES[:5]:
            q[name] = float(obs[f"{name}.pos"]) * (math.pi / 180.0)
        q["gripper"] = float(_gripper_sim_rad_from_pct(float(obs["gripper.pos"])))
        return q
    finally:
        robot.disconnect()


def parse_joint_overrides(items: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=RAD, got {item!r}")
        name, value = item.split("=", 1)
        out[name] = float(value)
    return out


def print_joint_frames(
    joints: list[Joint],
    link_tf: dict[str, np.ndarray],
    joint_pos: dict[str, float],
    include_fixed: bool,
) -> None:
    if include_fixed:
        ordered_joints = joints
    else:
        by_name = {joint.name: joint for joint in joints}
        ordered_joints = [by_name[name] for name in JOINT_NAMES if name in by_name]

    for joint in ordered_joints:
        if joint.child not in link_tf:
            continue
        if not include_fixed and joint.joint_type == "fixed":
            continue
        tf = link_tf[joint.child]
        pos = tf[:3, 3]
        rot = tf[:3, :3]
        rpy = matrix_to_rpy(rot)
        quat = matrix_to_quat_wxyz(rot)
        print(f"{joint.name}  ({joint.joint_type}, {joint.parent} -> {joint.child})")
        print(f"  q_rad:          {joint_pos.get(joint.name, 0.0): .6f}")
        print(f"  frame:          {joint.child} after joint motion, relative to base_link")
        print(f"  xyz_base_m:     [{pos[0]: .6f}, {pos[1]: .6f}, {pos[2]: .6f}]")
        print(f"  rpy_base_rad:   [{rpy[0]: .6f}, {rpy[1]: .6f}, {rpy[2]: .6f}]")
        print(f"  quat_wxyz:      [{quat[0]: .6f}, {quat[1]: .6f}, {quat[2]: .6f}, {quat[3]: .6f}]")
        print("  rotation_matrix:")
        for row in rot:
            print(f"    [{row[0]: .6f}, {row[1]: .6f}, {row[2]: .6f}]")
        print()


def print_named_frame(name: str, tf: np.ndarray, note: str) -> None:
    pos = tf[:3, 3]
    rot = tf[:3, :3]
    rpy = matrix_to_rpy(rot)
    quat = matrix_to_quat_wxyz(rot)
    print(f"{name}")
    print(f"  frame:          {note}")
    print(f"  xyz_base_m:     [{pos[0]: .6f}, {pos[1]: .6f}, {pos[2]: .6f}]")
    print(f"  rpy_base_rad:   [{rpy[0]: .6f}, {rpy[1]: .6f}, {rpy[2]: .6f}]")
    print(f"  quat_wxyz:      [{quat[0]: .6f}, {quat[1]: .6f}, {quat[2]: .6f}, {quat[3]: .6f}]")
    print("  rotation_matrix:")
    for row in rot:
        print(f"    [{row[0]: .6f}, {row[1]: .6f}, {row[2]: .6f}]")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Print SO-ARM101 real joint frames relative to base_link using URDF FK.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--root-link", type=str, default="base_link")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--robot-id", type=str, default="eva-follower")
    parser.add_argument("--calibrate", action="store_true", help="Allow LeRobot calibration if needed.")
    parser.add_argument("--disable-torque", action="store_true", help="Disable torque before reading the pose.")
    parser.add_argument("--no-hardware", action="store_true", help="Do not connect to LeRobot; use --joint-pos values only.")
    parser.add_argument(
        "--joint-pos",
        action="append",
        default=[],
        metavar="NAME=RAD",
        help="Override a joint position in radians. Can be repeated.",
    )
    parser.add_argument(
        "--include-fixed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also print fixed URDF joints. Default prints only movable joints.",
    )
    parser.add_argument(
        "--print-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the theoretical sim wrist camera frame attached to gripper_link.",
    )
    args = parser.parse_args()

    joints = parse_urdf(args.urdf)
    if args.no_hardware:
        joint_pos = {name: 0.0 for name in JOINT_NAMES}
    else:
        joint_pos = read_real_joint_pos_rad(args.port, args.robot_id, args.calibrate, args.disable_torque)
    joint_pos.update(parse_joint_overrides(args.joint_pos))

    print(f"URDF: {args.urdf}")
    print(f"root_link: {args.root_link}")
    print("joint_positions_rad:")
    for name in JOINT_NAMES:
        print(f"  {name}: {joint_pos.get(name, 0.0): .6f}")
    print()

    link_tf, _ = compute_fk(joints, root_link=args.root_link, joint_pos_rad=joint_pos)
    print_joint_frames(joints, link_tf, joint_pos, include_fixed=args.include_fixed)
    if args.print_camera:
        if CAMERA_PARENT_LINK not in link_tf:
            raise RuntimeError(f"camera parent link {CAMERA_PARENT_LINK!r} not found in FK tree")
        cam_tf = link_tf[CAMERA_PARENT_LINK] @ transform(CAMERA_OFFSET_POS, quat_wxyz_to_matrix(CAMERA_OFFSET_QUAT_WXYZ))
        print("sim_training_camera_offset:")
        print(f"  parent_link:    {CAMERA_PARENT_LINK}")
        print(f"  offset_xyz:     {CAMERA_OFFSET_POS.tolist()}")
        print(f"  offset_quat_wxyz: {CAMERA_OFFSET_QUAT_WXYZ.tolist()}")
        print(f"  convention:     ros")
        print_named_frame(
            CAMERA_NAME,
            cam_tf,
            f"{CAMERA_PARENT_LINK} @ CameraCfg.OffsetCfg(pos, rot, convention='ros')",
        )


if __name__ == "__main__":
    main()
