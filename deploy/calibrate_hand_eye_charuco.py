"""Interactive ChArUco hand-eye calibration for the SO-ARM101 wrist camera.

The ChArUco board must be fixed in the workspace. Move the arm-mounted camera
to varied poses, keep the board visible, and press SPACE to capture each pose.
The script solves ``T_gripper_camera`` and writes ``deploy/hand_eye.yaml``.

Typical use:
    python -m deploy.calibrate_hand_eye_charuco --camera /dev/video1 --port /dev/ttyACM0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from deploy.driver import EE_LOCAL_OFFSET, FK, INTRINSICS_YAML, JOINT_NAMES, URDF_PATH, _gripper_sim_rad_from_pct


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = Path(__file__).resolve().parent
DEFAULT_HAND_EYE_YAML = DEPLOY_DIR / "hand_eye.yaml"
DEFAULT_SESSION_DIR = DEPLOY_DIR / "hand_eye_charuco_session"

DICT_NAMES = {
    "DICT_4X4_50": 0,
    "DICT_4X4_100": 1,
    "DICT_4X4_250": 2,
    "DICT_4X4_1000": 3,
    "DICT_5X5_50": 4,
    "DICT_5X5_100": 5,
    "DICT_5X5_250": 6,
    "DICT_5X5_1000": 7,
    "DICT_6X6_50": 8,
    "DICT_6X6_100": 9,
    "DICT_6X6_250": 10,
    "DICT_6X6_1000": 11,
    "DICT_7X7_50": 12,
    "DICT_7X7_100": 13,
    "DICT_7X7_250": 14,
    "DICT_7X7_1000": 15,
    "DICT_ARUCO_ORIGINAL": 16,
}


@dataclass
class CharucoPose:
    T_camera_board: np.ndarray
    marker_count: int
    corner_count: int
    rms_reprojection_error_px: float
    charuco_corners: np.ndarray
    charuco_ids: np.ndarray


@dataclass
class Sample:
    q_sim_rad: np.ndarray
    T_base_gripper: np.ndarray
    T_camera_board: np.ndarray
    image_path: str
    marker_count: int
    corner_count: int
    reprojection_error_px: float


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise SystemExit("ERROR: cv2 is required. Activate the eva environment or install opencv-contrib-python.") from e
    if not hasattr(cv2, "aruco"):
        raise SystemExit("ERROR: cv2.aruco is missing. Install opencv-contrib-python.")
    return cv2


def _load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("ERROR: PyYAML is required.") from e
    if not path.exists():
        raise SystemExit(f"ERROR: camera intrinsics not found: {path}")

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_multi_constructor("tag:yaml.org,2002:python/object/apply:", lambda l, t, n: None)
    Loader.add_multi_constructor("tag:yaml.org,2002:python/object/new:", lambda l, t, n: None)
    Loader.add_multi_constructor("tag:yaml.org,2002:python/name:", lambda l, t, n: None)
    with open(path, "r") as f:
        data = yaml.load(f, Loader=Loader)
    K = np.asarray(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(data["distortion_coefficients"]["data"], dtype=np.float64).reshape(-1)
    return K, dist, data


def _dictionary_id(name_or_id: str) -> int:
    if name_or_id.isdigit():
        return int(name_or_id)
    key = name_or_id.upper()
    if key not in DICT_NAMES:
        valid = ", ".join(sorted(DICT_NAMES))
        raise SystemExit(f"Unknown dictionary {name_or_id!r}. Valid names: {valid}")
    return DICT_NAMES[key]


def _make_board(cv2: Any, squares_x: int, squares_y: int, square_m: float, marker_m: float, dictionary: Any):
    aruco = cv2.aruco
    if hasattr(aruco, "CharucoBoard_create"):
        return aruco.CharucoBoard_create(squares_x, squares_y, square_m, marker_m, dictionary)
    return aruco.CharucoBoard((squares_x, squares_y), square_m, marker_m, dictionary)


def _make_detector_params(cv2: Any):
    aruco = cv2.aruco
    if hasattr(aruco, "DetectorParameters_create"):
        params = aruco.DetectorParameters_create()
    else:
        params = aruco.DetectorParameters()
    if hasattr(params, "cornerRefinementMethod"):
        params.cornerRefinementMethod = getattr(aruco, "CORNER_REFINE_SUBPIX", 1)
    return params


def _make_charuco_detector(cv2: Any, board: Any, detector_params: Any):
    aruco = cv2.aruco
    if not hasattr(aruco, "CharucoDetector"):
        return None
    detector = aruco.CharucoDetector(board)
    if hasattr(detector, "setDetectorParameters"):
        detector.setDetectorParameters(detector_params)
    return detector


def _detect_charuco_pose(
    cv2: Any,
    bgr: np.ndarray,
    board: Any,
    dictionary: Any,
    detector_params: Any,
    charuco_detector: Any,
    K: np.ndarray,
    dist: np.ndarray,
    min_corners: int,
) -> CharucoPose | None:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    aruco = cv2.aruco

    if charuco_detector is not None:
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
    else:
        marker_corners, marker_ids, _ = aruco.detectMarkers(gray, dictionary, parameters=detector_params)
        if marker_ids is None or len(marker_ids) == 0:
            return None
        _n, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)

    marker_count = 0 if marker_ids is None else int(len(marker_ids))
    corner_count = 0 if charuco_ids is None else int(len(charuco_ids))
    if charuco_corners is None or charuco_ids is None or corner_count < min_corners:
        return None

    object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    err = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))

    R, _ = cv2.Rodrigues(rvec)
    T_camera_board = np.eye(4, dtype=np.float64)
    T_camera_board[:3, :3] = R
    T_camera_board[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return CharucoPose(
        T_camera_board=T_camera_board,
        marker_count=marker_count,
        corner_count=corner_count,
        rms_reprojection_error_px=rms,
        charuco_corners=np.asarray(charuco_corners, dtype=np.float32),
        charuco_ids=np.asarray(charuco_ids, dtype=np.int32),
    )


def _draw_preview(cv2: Any, bgr: np.ndarray, pose: CharucoPose | None, samples: int, target_samples: int) -> np.ndarray:
    out = bgr.copy()
    if pose is not None:
        cv2.aruco.drawDetectedCornersCharuco(out, pose.charuco_corners, pose.charuco_ids)
        status = (
            f"samples={samples}/{target_samples} markers={pose.marker_count} "
            f"corners={pose.corner_count} pnp={pose.rms_reprojection_error_px:.2f}px"
        )
        color = (0, 220, 0)
    else:
        status = f"samples={samples}/{target_samples} board=not detected"
        color = (0, 0, 255)
    cv2.putText(out, status, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(
        out,
        "SPACE=capture  d=drop  q=solve  ESC=abort",
        (16, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


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
    print(f"[hand-eye] opened camera {args.camera!r}: {actual_w}x{actual_h}")
    return cap


def _solve_hand_eye(cv2: Any, samples: list[Sample], method: int) -> np.ndarray:
    R_gripper2base = [s.T_base_gripper[:3, :3] for s in samples]
    t_gripper2base = [s.T_base_gripper[:3, 3] for s in samples]
    R_target2cam = [s.T_camera_board[:3, :3] for s in samples]
    t_target2cam = [s.T_camera_board[:3, 3] for s in samples]
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=method,
    )
    T_gripper_camera = np.eye(4, dtype=np.float64)
    T_gripper_camera[:3, :3] = R_cam2gripper
    T_gripper_camera[:3, 3] = np.asarray(t_cam2gripper, dtype=np.float64).reshape(3)
    return T_gripper_camera


def _board_positions_in_base(samples: list[Sample], T_gripper_camera: np.ndarray) -> np.ndarray:
    return np.stack(
        [(s.T_base_gripper @ T_gripper_camera @ s.T_camera_board)[:3, 3] for s in samples],
        axis=0,
    )


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("ERROR: PyYAML is required.") from e
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _write_samples(path: Path, samples: list[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, s in enumerate(samples):
        rows.append(
            {
                "index": i,
                "q_sim_rad": s.q_sim_rad.astype(float).tolist(),
                "T_base_gripper": s.T_base_gripper.tolist(),
                "T_camera_board": s.T_camera_board.tolist(),
                "image_path": s.image_path,
                "marker_count": s.marker_count,
                "corner_count": s.corner_count,
                "reprojection_error_px": s.reprojection_error_px,
            }
        )
    with open(path, "w") as f:
        json.dump({"samples": rows}, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate wrist-camera hand-eye transform with a fixed ChArUco board.")
    p.add_argument("--camera", default="/dev/video1", help="OpenCV camera source, e.g. /dev/video1 or 1.")
    p.add_argument("--backend", choices=("any", "v4l2"), default="v4l2")
    p.add_argument("--fourcc", default="MJPG")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--port", default="/dev/ttyACM0", help="SO101 follower serial port.")
    p.add_argument("--robot-id", default="eva-follower", help="LeRobot calibration id.")
    p.add_argument("--calibrate", action="store_true", help="Allow LeRobot calibration if needed.")
    p.add_argument(
        "--disable-torque",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable servo torque after connecting so the arm can be moved by hand.",
    )
    p.add_argument("--intrinsics", type=Path, default=INTRINSICS_YAML)
    p.add_argument("--out", type=Path, default=DEFAULT_HAND_EYE_YAML)
    p.add_argument("--session-dir", type=Path, default=DEFAULT_SESSION_DIR)
    p.add_argument("--squares-x", type=int, default=5)
    p.add_argument("--squares-y", type=int, default=7)
    p.add_argument("--square-m", type=float, default=0.0285)
    p.add_argument("--marker-m", type=float, default=0.0170)
    p.add_argument("--dictionary", default="DICT_5X5_50")
    p.add_argument("--min-corners", type=int, default=16)
    p.add_argument("--samples", type=int, default=20, help="Recommended number of samples to collect.")
    p.add_argument(
        "--method",
        choices=("tsai", "park", "horaud", "andreff", "daniilidis"),
        default="tsai",
        help="OpenCV hand-eye solver.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cv2 = _require_cv2()
    K, dist, intrinsics_data = _load_intrinsics(args.intrinsics)
    if int(intrinsics_data.get("image_width", args.width)) != args.width or int(intrinsics_data.get("image_height", args.height)) != args.height:
        print(
            f"[hand-eye] WARNING: intrinsics are for "
            f"{intrinsics_data.get('image_width')}x{intrinsics_data.get('image_height')}, "
            f"but capture requested {args.width}x{args.height}.",
            file=sys.stderr,
        )

    dictionary = cv2.aruco.getPredefinedDictionary(_dictionary_id(args.dictionary))
    board = _make_board(cv2, args.squares_x, args.squares_y, args.square_m, args.marker_m, dictionary)
    detector_params = _make_detector_params(cv2)
    charuco_detector = _make_charuco_detector(cv2, board, detector_params)
    fk = FK(URDF_PATH)

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    print(f"[hand-eye] connecting robot {args.robot_id} on {args.port}")
    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
    robot.connect(calibrate=args.calibrate)
    if args.disable_torque:
        robot.bus.disable_torque()
        print("[hand-eye] torque disabled; move the arm by hand between samples")

    cap = _open_camera(cv2, args)
    images_dir = args.session_dir / "images"
    samples: list[Sample] = []

    method_map = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    print()
    print("[hand-eye] Fix the ChArUco board in the workspace. Move the wrist camera to varied poses.")
    print("[hand-eye] Controls: SPACE=capture, d=drop last, q=solve, ESC=abort.")
    print("[hand-eye] Use varied rotations; pure translations make hand-eye calibration degenerate.")

    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                raise SystemExit("ERROR: camera returned no frame")
            pose = _detect_charuco_pose(
                cv2,
                bgr,
                board,
                dictionary,
                detector_params,
                charuco_detector,
                K,
                dist,
                args.min_corners,
            )
            preview = _draw_preview(cv2, bgr, pose, len(samples), args.samples)
            cv2.imshow("charuco hand-eye calibration", preview)
            key = cv2.waitKey(20) & 0xFF

            if key == ord(" "):
                if pose is None:
                    print("[hand-eye] board not detected well enough; not capturing")
                    continue
                q = _read_q_sim_rad(robot)
                T_bg = fk_T_base_gripper(fk, q)
                images_dir.mkdir(parents=True, exist_ok=True)
                image_path = images_dir / f"sample_{len(samples):03d}_{int(time.time() * 1000)}.png"
                cv2.imwrite(str(image_path), bgr)
                samples.append(
                    Sample(
                        q_sim_rad=q,
                        T_base_gripper=T_bg,
                        T_camera_board=pose.T_camera_board,
                        image_path=str(image_path),
                        marker_count=pose.marker_count,
                        corner_count=pose.corner_count,
                        reprojection_error_px=pose.rms_reprojection_error_px,
                    )
                )
                print(
                    f"[hand-eye] captured {len(samples)}: corners={pose.corner_count} "
                    f"pnp={pose.rms_reprojection_error_px:.3f}px "
                    f"board_dist={np.linalg.norm(pose.T_camera_board[:3, 3]):.3f}m"
                )
            elif key == ord("d"):
                if samples:
                    dropped = samples.pop()
                    print(f"[hand-eye] dropped sample {Path(dropped.image_path).name}; now {len(samples)} samples")
            elif key == ord("q"):
                if len(samples) < 8:
                    print(f"[hand-eye] need at least 8 samples; have {len(samples)}")
                    continue
                break
            elif key == 27:
                print("[hand-eye] aborted; nothing written")
                return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot.disconnect()

    print(f"[hand-eye] solving with {len(samples)} samples, method={args.method}")
    T_gripper_camera = _solve_hand_eye(cv2, samples, method_map[args.method])
    board_positions = _board_positions_in_base(samples, T_gripper_camera)
    mean = board_positions.mean(axis=0)
    std = board_positions.std(axis=0)
    max_std_mm = float(np.max(std) * 1000.0)

    print("[hand-eye] T_gripper_camera:")
    print(np.array2string(T_gripper_camera, precision=6, suppress_small=True))
    print(
        "[hand-eye] fixed-board consistency std xyz mm: "
        f"[{std[0] * 1000:.2f}, {std[1] * 1000:.2f}, {std[2] * 1000:.2f}]"
    )
    if max_std_mm > 8.0:
        print("[hand-eye] WARNING: consistency std is high; collect more varied, sharper samples.")

    payload = {
        "T_gripper_camera": T_gripper_camera.tolist(),
        "T_ee_cam": T_gripper_camera.tolist(),
        "parent_frame": "gripper_link",
        "child_frame": "camera",
        "calibration_metadata": {
            "type": "charuco_hand_eye",
            "method": args.method,
            "n_samples": len(samples),
            "camera_intrinsics": str(args.intrinsics),
            "image_width": args.width,
            "image_height": args.height,
            "board": {
                "squares_x": args.squares_x,
                "squares_y": args.squares_y,
                "square_length_m": args.square_m,
                "marker_length_m": args.marker_m,
                "dictionary": args.dictionary,
            },
            "board_position_mean_base_m": mean.tolist(),
            "board_position_std_base_m": std.tolist(),
            "mean_pnp_reprojection_error_px": float(np.mean([s.reprojection_error_px for s in samples])),
        },
    }
    _write_yaml(args.out, payload)
    _write_samples(args.session_dir / "samples.json", samples)
    print(f"[hand-eye] wrote {args.out}")
    print(f"[hand-eye] wrote {args.session_dir / 'samples.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
