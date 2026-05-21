"""Verify AprilTag -> base_link transform with live SO101 FK and hand-eye.

This script is an end-to-end sanity check for:

    T_base_tag = T_base_gripper @ T_gripper_camera @ T_camera_tag

Use it after camera intrinsic calibration and ChArUco hand-eye calibration.
The tag must be a real AprilTag. The default tag size is 1.4 cm.

Typical use:
    python -m deploy.verify_apriltag_base_transform --camera /dev/video1 --tag-size 0.014
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from deploy.driver import EE_LOCAL_OFFSET, FK, INTRINSICS_YAML, JOINT_NAMES, URDF_PATH, _gripper_sim_rad_from_pct


DEPLOY_DIR = Path(__file__).resolve().parent
DEFAULT_HAND_EYE_YAML = DEPLOY_DIR / "hand_eye.yaml"


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise SystemExit("ERROR: cv2 is required. Activate the eva environment.") from e
    return cv2


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("ERROR: PyYAML is required.") from e

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_multi_constructor("tag:yaml.org,2002:python/object/apply:", lambda l, t, n: None)
    Loader.add_multi_constructor("tag:yaml.org,2002:python/object/new:", lambda l, t, n: None)
    Loader.add_multi_constructor("tag:yaml.org,2002:python/name:", lambda l, t, n: None)
    with open(path, "r") as f:
        return yaml.load(f, Loader=Loader)


def _load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, int | None, int | None]:
    if not path.exists():
        raise SystemExit(f"ERROR: camera intrinsics not found: {path}")
    data = _load_yaml(path)
    K = np.asarray(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(data["distortion_coefficients"]["data"], dtype=np.float64).reshape(-1)
    return K, dist, data.get("image_width"), data.get("image_height")


def _load_hand_eye(path: Path) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"ERROR: hand-eye calibration not found: {path}")
    data = _load_yaml(path)
    key = "T_gripper_camera" if "T_gripper_camera" in data else "T_ee_cam"
    return np.asarray(data[key], dtype=np.float64).reshape(4, 4)


def fk_T_base_gripper(fk: FK, joint_pos_rad: np.ndarray) -> np.ndarray:
    arm_vals = {n: float(v) for n, v in zip(JOINT_NAMES[:5], joint_pos_rad[:5])}
    th = [arm_vals[n] for n in fk.chain.get_joint_parameter_names()]
    T = fk.chain.forward_kinematics(th)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = T.rot_mat
    out[:3, 3] = np.asarray(T.pos, dtype=np.float64) + T.rot_mat @ EE_LOCAL_OFFSET
    return out


def _read_q_sim_rad(robot: Any) -> np.ndarray:
    obs = robot.get_observation()
    q = np.empty(6, dtype=np.float32)
    for i, name in enumerate(JOINT_NAMES[:5]):
        q[i] = float(obs[f"{name}.pos"]) * (np.pi / 180.0)
    q[5] = _gripper_sim_rad_from_pct(float(obs["gripper.pos"]))
    return q


def _open_camera(cv2: Any, args: argparse.Namespace):
    source = int(args.camera) if str(args.camera).isdigit() else str(args.camera)
    backend = cv2.CAP_V4L2 if args.backend == "v4l2" else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, backend)
    if args.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc.upper()))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: failed to open camera {args.camera!r}")
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[verify] opened camera {args.camera!r}: {actual_w}x{actual_h}")
    return cap


def _detect_apriltags(cv2: Any, detector: Any, bgr: np.ndarray, K: np.ndarray, dist: np.ndarray, tag_size: float):
    undist = cv2.undistort(bgr, K, dist)
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=tag_size,
    )
    out = []
    for det in detections:
        T_camera_tag = np.eye(4, dtype=np.float64)
        T_camera_tag[:3, :3] = np.asarray(det.pose_R, dtype=np.float64)
        T_camera_tag[:3, 3] = np.asarray(det.pose_t, dtype=np.float64).reshape(3)
        out.append(
            {
                "id": int(det.tag_id),
                "margin": float(det.decision_margin),
                "corners": np.asarray(det.corners, dtype=np.float64),
                "center": np.asarray(det.center, dtype=np.float64),
                "T_camera_tag": T_camera_tag,
            }
        )
    return undist, out


def _choose_detection(detections: list[dict], tag_id: int | None) -> dict | None:
    if tag_id is not None:
        return next((d for d in detections if d["id"] == tag_id), None)
    if not detections:
        return None
    return max(detections, key=lambda d: d["margin"])


def _draw(cv2: Any, bgr: np.ndarray, detections: list[dict], target: dict | None, base_xyz: np.ndarray | None, stats: str) -> np.ndarray:
    vis = bgr.copy()
    for det in detections:
        pts = det["corners"].astype(np.int32)
        color = (0, 255, 0) if target is not None and det["id"] == target["id"] else (0, 180, 255)
        cv2.polylines(vis, [pts], True, color, 2)
        c = tuple(det["center"].astype(int))
        cv2.putText(vis, f"id={det['id']} m={det['margin']:.0f}", c, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    if base_xyz is None:
        line1 = "target: not detected"
    else:
        line1 = f"base tag xyz m: {base_xyz[0]:+.3f} {base_xyz[1]:+.3f} {base_xyz[2]:+.3f}"
    cv2.putText(vis, line1, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, stats, (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "r=reset stats  q/ESC=quit", (16, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify AprilTag pose transformed into SO101 base_link.")
    p.add_argument("--camera", default="/dev/video1")
    p.add_argument("--backend", choices=("any", "v4l2"), default="v4l2")
    p.add_argument("--fourcc", default="MJPG")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--robot-id", default="eva-follower")
    p.add_argument("--calibrate", action="store_true", help="Allow LeRobot calibration if needed.")
    p.add_argument(
        "--disable-torque",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable servo torque after connecting so the arm can be moved by hand.",
    )
    p.add_argument("--intrinsics", type=Path, default=INTRINSICS_YAML)
    p.add_argument("--hand-eye", type=Path, default=DEFAULT_HAND_EYE_YAML)
    p.add_argument("--family", default="tagStandard41h12")
    p.add_argument("--tag-id", type=int, default=None, help="Target tag id. If omitted, use the detected tag with highest margin.")
    p.add_argument("--tag-size", type=float, default=0.014, help="AprilTag black/white border edge length in metres.")
    p.add_argument("--window", type=int, default=120, help="Rolling statistics window in valid detections.")
    p.add_argument("--print-every", type=float, default=0.5, help="Seconds between terminal prints.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cv2 = _require_cv2()
    K, dist, intr_w, intr_h = _load_intrinsics(args.intrinsics)
    if intr_w != args.width or intr_h != args.height:
        print(
            f"[verify] WARNING: intrinsics are {intr_w}x{intr_h}, capture requested {args.width}x{args.height}",
            file=sys.stderr,
        )
    T_gripper_camera = _load_hand_eye(args.hand_eye)

    try:
        from pupil_apriltags import Detector
    except ImportError as e:
        raise SystemExit("ERROR: pupil-apriltags is required. Install/run deploy setup first.") from e
    detector = Detector(families=args.family)

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    print(f"[verify] connecting robot {args.robot_id} on {args.port}")
    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
    robot.connect(calibrate=args.calibrate)
    if args.disable_torque:
        robot.bus.disable_torque()
        print("[verify] torque disabled; you can move the arm by hand")
    cap = _open_camera(cv2, args)
    fk = FK(URDF_PATH)

    positions = deque(maxlen=args.window)
    last_print = 0.0
    valid_count = 0

    print(
        f"[verify] AprilTag family={args.family}, tag_size={args.tag_size*1000:.1f}mm, "
        f"target_id={args.tag_id if args.tag_id is not None else 'best-margin'}"
    )
    print("[verify] Keep the printed tag fixed. Move the wrist camera; base xyz should stay nearly constant.")

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                raise SystemExit("ERROR: camera returned no frame")

            q = _read_q_sim_rad(robot)
            T_base_gripper = fk_T_base_gripper(fk, q)
            undist, detections = _detect_apriltags(cv2, detector, bgr, K, dist, args.tag_size)
            target = _choose_detection(detections, args.tag_id)

            base_xyz = None
            if target is not None:
                T_base_tag = T_base_gripper @ T_gripper_camera @ target["T_camera_tag"]
                base_xyz = T_base_tag[:3, 3].copy()
                positions.append(base_xyz)
                valid_count += 1

            if positions:
                arr = np.stack(list(positions), axis=0)
                mean = arr.mean(axis=0)
                std = arr.std(axis=0)
                stats = f"n={len(positions)} mean mm=({mean[0]*1000:+.0f},{mean[1]*1000:+.0f},{mean[2]*1000:+.0f}) std mm=({std[0]*1000:.1f},{std[1]*1000:.1f},{std[2]*1000:.1f})"
            else:
                stats = "n=0"

            now = time.monotonic()
            if now - last_print >= args.print_every:
                last_print = now
                if base_xyz is None:
                    ids = [d["id"] for d in detections]
                    print(f"[verify] no target detection; visible_ids={ids}")
                else:
                    print(
                        f"[verify] id={target['id']} margin={target['margin']:.1f} "
                        f"base_xyz_m=({base_xyz[0]:+.4f}, {base_xyz[1]:+.4f}, {base_xyz[2]:+.4f}) "
                        f"{stats}"
                    )

            vis = _draw(cv2, undist, detections, target, base_xyz, stats)
            cv2.imshow("apriltag base_link verification", vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                positions.clear()
                valid_count = 0
                print("[verify] reset rolling statistics")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot.disconnect()

    if positions:
        arr = np.stack(list(positions), axis=0)
        std = arr.std(axis=0) * 1000.0
        print(f"[verify] final rolling std xyz mm: [{std[0]:.2f}, {std[1]:.2f}, {std[2]:.2f}]")
    print(f"[verify] valid detections: {valid_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
