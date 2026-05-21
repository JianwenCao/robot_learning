"""Calibrate a camera from a ChArUco board.

Default board:
    7 columns x 5 rows, square side 2.85 cm, marker side 1.70 cm.

This is the OpenCV-contrib ChArUco calibration path:
detect ArUco markers -> interpolate ChArUco corners ->
cv2.aruco.calibrateCameraCharuco[Extended].

Typical live capture:
    python -m deploy.calibrate_camera_charuco --cam-index 0 --width 1280 --height 720

Calibrate from an existing image folder:
    python -m deploy.calibrate_camera_charuco --images /tmp/charuco_frames/*.png
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "camera_intrinsics.yaml"
DEFAULT_SESSION_DIR = PROJECT_ROOT / "deploy" / "charuco_calib_session"

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
AUTO_DICTS = (
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_6X6_1000",
)


@dataclass
class Detection:
    path: str
    image_size: tuple[int, int]
    dictionary_name: str
    marker_count: int
    corner_count: int
    charuco_corners: np.ndarray
    charuco_ids: np.ndarray


def _require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "ERROR: Python cannot import cv2. Install OpenCV contrib, e.g.\n"
            "  pip install opencv-contrib-python\n"
            "or use the project environment that provides cv2.aruco."
        ) from e
    if not hasattr(cv2, "aruco"):
        raise SystemExit(
            "ERROR: cv2.aruco is missing. You need opencv-contrib-python, "
            "not the plain opencv-python package."
        )
    return cv2


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


def _detect_markers(cv2: Any, gray: np.ndarray, dictionary: Any, params: Any):
    aruco = cv2.aruco
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
        return detector.detectMarkers(gray)
    return aruco.detectMarkers(gray, dictionary, parameters=params)


def _make_dictionary(cv2: Any, name_or_id: str):
    return cv2.aruco.getPredefinedDictionary(_dictionary_id(name_or_id))


def _detect_best_dictionary(cv2: Any, gray: np.ndarray, candidates: list[tuple[str, Any, Any]], params: Any):
    best_name = candidates[0][0]
    best_board = candidates[0][2]
    best_result = ([], None, [])
    best_count = -1
    for name, dictionary, board in candidates:
        marker_corners, marker_ids, rejected = _detect_markers(cv2, gray, dictionary, params)
        count = 0 if marker_ids is None else int(len(marker_ids))
        if count > best_count:
            best_name = name
            best_board = board
            best_result = (marker_corners, marker_ids, rejected)
            best_count = count
    return best_name, best_board, best_result


def _interpolate_charuco(cv2: Any, gray: np.ndarray, board: Any, marker_corners: Any, marker_ids: Any):
    aruco = cv2.aruco
    if marker_ids is None or len(marker_ids) == 0:
        return 0, None, None
    if hasattr(aruco, "interpolateCornersCharuco"):
        return aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)
    detector = aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _marker_corners, _marker_ids = detector.detectBoard(gray)
    count = 0 if charuco_ids is None else int(len(charuco_ids))
    return count, charuco_corners, charuco_ids


def _draw_detection(cv2: Any, image: np.ndarray, marker_corners: Any, marker_ids: Any, charuco_corners: Any, charuco_ids: Any):
    out = image.copy()
    aruco = cv2.aruco
    if marker_ids is not None and len(marker_ids) > 0:
        aruco.drawDetectedMarkers(out, marker_corners, marker_ids)
    if charuco_ids is not None and len(charuco_ids) > 0:
        aruco.drawDetectedCornersCharuco(out, charuco_corners, charuco_ids)
    return out


def detect_image(
    cv2: Any,
    image: np.ndarray,
    image_path: str,
    candidates: list[tuple[str, Any, Any]],
    params: Any,
    min_corners: int,
    preview_dir: Path | None,
) -> Detection | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    dictionary_name, board, (marker_corners, marker_ids, _rejected) = _detect_best_dictionary(
        cv2, gray, candidates, params
    )
    _count, charuco_corners, charuco_ids = _interpolate_charuco(cv2, gray, board, marker_corners, marker_ids)
    marker_count = 0 if marker_ids is None else int(len(marker_ids))
    corner_count = 0 if charuco_ids is None else int(len(charuco_ids))

    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        drawn = _draw_detection(cv2, image, marker_corners, marker_ids, charuco_corners, charuco_ids)
        suffix = "ok" if corner_count >= min_corners else "reject"
        cv2.imwrite(str(preview_dir / f"{Path(image_path).stem}_{suffix}.png"), drawn)

    if charuco_corners is None or charuco_ids is None or corner_count < min_corners:
        return None

    h, w = gray.shape[:2]
    return Detection(
        path=image_path,
        image_size=(w, h),
        dictionary_name=dictionary_name,
        marker_count=marker_count,
        corner_count=corner_count,
        charuco_corners=np.asarray(charuco_corners, dtype=np.float32),
        charuco_ids=np.asarray(charuco_ids, dtype=np.int32),
    )


def _collect_from_images(
    cv2: Any,
    patterns: list[str],
    candidates: list[tuple[str, Any, Any]],
    params: Any,
    min_corners: int,
    preview_dir: Path | None,
) -> list[Detection]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    paths = [p for p in paths if Path(p).is_file()]
    if not paths:
        raise SystemExit("ERROR: no input images matched --images.")

    detections: list[Detection] = []
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"[skip] failed to read {path}", file=sys.stderr)
            continue
        det = detect_image(cv2, image, path, candidates, params, min_corners, preview_dir)
        if det is None:
            print(f"[reject] {path}: too few ChArUco corners", file=sys.stderr)
            continue
        print(f"[accept] {path}: dict={det.dictionary_name} markers={det.marker_count} corners={det.corner_count}")
        detections.append(det)
    return detections


def _collect_live(
    cv2: Any,
    args: argparse.Namespace,
    candidates: list[tuple[str, Any, Any]],
    params: Any,
    session_dir: Path,
) -> list[Detection]:
    camera_source = int(args.camera) if str(args.camera).isdigit() else str(args.camera)
    backend = cv2.CAP_V4L2 if args.backend == "v4l2" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_source, backend)
    if args.fourcc:
        fourcc = cv2.VideoWriter_fourcc(*args.fourcc.upper())
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps:
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: failed to open camera source {args.camera!r}.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_text = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))
    print(
        f"Opened camera {args.camera!r}: {actual_w}x{actual_h} "
        f"fps={actual_fps:.1f} fourcc={fourcc_text!r} backend={args.backend}"
    )

    accepted_dir = session_dir / "accepted"
    preview_dir = session_dir / "preview"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    detections: list[Detection] = []
    last_capture_t = 0.0
    print("Live capture controls: SPACE=save, a=toggle auto, c=calibrate, q=quit")
    print("Move the board through image corners, center, near/far, and tilted poses.")
    auto = args.auto
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("ERROR: camera returned no frame.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dictionary_name, board, (marker_corners, marker_ids, _) = _detect_best_dictionary(
                cv2, gray, candidates, params
            )
            _, charuco_corners, charuco_ids = _interpolate_charuco(cv2, gray, board, marker_corners, marker_ids)
            marker_count = 0 if marker_ids is None else int(len(marker_ids))
            corner_count = 0 if charuco_ids is None else int(len(charuco_ids))
            vis = _draw_detection(cv2, frame, marker_corners, marker_ids, charuco_corners, charuco_ids)
            text = (
                f"samples={len(detections)}/{args.samples} markers={marker_count} "
                f"corners={corner_count} dict={dictionary_name} auto={'on' if auto else 'off'}"
            )
            color = (0, 220, 0) if corner_count >= args.min_corners else (0, 0, 255)
            cv2.putText(vis, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
            cv2.imshow("charuco camera calibration", vis)

            now = time.monotonic()
            want_capture = auto and corner_count >= args.min_corners and now - last_capture_t >= args.auto_interval
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                auto = not auto
            if key == ord("c"):
                break
            if key == ord(" "):
                want_capture = True

            if want_capture:
                name = f"charuco_{len(detections):03d}_{int(time.time() * 1000)}.png"
                img_path = accepted_dir / name
                cv2.imwrite(str(img_path), frame)
                det = detect_image(cv2, frame, str(img_path), candidates, params, args.min_corners, preview_dir)
                last_capture_t = now
                if det is None:
                    print(f"[reject] live frame: corners={corner_count}")
                else:
                    detections.append(det)
                    print(
                        f"[accept] {img_path}: dict={det.dictionary_name} "
                        f"markers={det.marker_count} corners={det.corner_count}"
                    )
                if len(detections) >= args.samples:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return detections


def _calibrate(cv2: Any, board: Any, detections: list[Detection], flags: int):
    image_sizes = {d.image_size for d in detections}
    if len(image_sizes) != 1:
        raise SystemExit(f"ERROR: all calibration images must have the same size; got {sorted(image_sizes)}")
    image_size = detections[0].image_size
    all_corners = [d.charuco_corners for d in detections]
    all_ids = [d.charuco_ids for d in detections]

    aruco = cv2.aruco
    if hasattr(aruco, "calibrateCameraCharucoExtended"):
        ret, K, dist, rvecs, tvecs, std_int, std_ext, per_view = aruco.calibrateCameraCharucoExtended(
            all_corners,
            all_ids,
            board,
            image_size,
            None,
            None,
            flags=flags,
        )
        per_view_errors = [float(x) for x in np.asarray(per_view).reshape(-1)]
    elif hasattr(aruco, "calibrateCameraCharuco"):
        ret, K, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
            all_corners,
            all_ids,
            board,
            image_size,
            None,
            None,
            flags=flags,
        )
        std_int = None
        std_ext = None
        per_view_errors = _compute_per_view_errors(cv2, board, detections, K, dist, rvecs, tvecs)
    else:
        object_points = []
        image_points = []
        for det in detections:
            obj, img = board.matchImagePoints(det.charuco_corners, det.charuco_ids)
            object_points.append(obj)
            image_points.append(img)
        if hasattr(cv2, "calibrateCameraExtended"):
            ret, K, dist, rvecs, tvecs, std_int, std_ext, per_view = cv2.calibrateCameraExtended(
                object_points,
                image_points,
                image_size,
                None,
                None,
                flags=flags,
            )
            per_view_errors = [float(x) for x in np.asarray(per_view).reshape(-1)]
        else:
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                object_points,
                image_points,
                image_size,
                None,
                None,
                flags=flags,
            )
            std_int = None
            std_ext = None
            per_view_errors = _compute_per_view_errors(cv2, board, detections, K, dist, rvecs, tvecs)

    return float(ret), np.asarray(K), np.asarray(dist).reshape(-1), rvecs, tvecs, std_int, std_ext, per_view_errors, image_size


def _filter_detections_by_error(
    detections: list[Detection],
    per_view_errors: list[float],
    max_sample_error: float | None,
    min_samples: int,
) -> list[Detection]:
    if max_sample_error is None:
        return detections
    keep = [
        det for det, err in zip(detections, per_view_errors)
        if float(err) <= float(max_sample_error)
    ]
    if len(keep) < min_samples:
        print(
            f"[filter] requested --max-sample-error {max_sample_error:.3f}px "
            f"would keep only {len(keep)} samples; keeping all {len(detections)} samples instead.",
            file=sys.stderr,
        )
        return detections
    print(
        f"[filter] kept {len(keep)}/{len(detections)} samples with "
        f"per-view error <= {max_sample_error:.3f}px"
    )
    return keep


def _board_chessboard_corners(board: Any) -> np.ndarray:
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype=np.float32)
    return np.asarray(board.chessboardCorners, dtype=np.float32)


def _compute_per_view_errors(cv2: Any, board: Any, detections: list[Detection], K: np.ndarray, dist: np.ndarray, rvecs: Any, tvecs: Any) -> list[float]:
    board_corners = _board_chessboard_corners(board)
    errors: list[float] = []
    for det, rvec, tvec in zip(detections, rvecs, tvecs):
        ids = det.charuco_ids.reshape(-1)
        obj = board_corners[ids]
        img = det.charuco_corners.reshape(-1, 2)
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        projected = projected.reshape(-1, 2)
        err = np.linalg.norm(projected - img, axis=1)
        errors.append(float(np.sqrt(np.mean(err * err))))
    return errors


def _mean_reprojection_error(cv2: Any, board: Any, detections: list[Detection], K: np.ndarray, dist: np.ndarray, rvecs: Any, tvecs: Any) -> float:
    board_corners = _board_chessboard_corners(board)
    total_err_sq = 0.0
    total_points = 0
    for det, rvec, tvec in zip(detections, rvecs, tvecs):
        ids = det.charuco_ids.reshape(-1)
        obj = board_corners[ids]
        img = det.charuco_corners.reshape(-1, 2)
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = projected.reshape(-1, 2) - img
        total_err_sq += float(np.sum(err * err))
        total_points += int(len(img))
    return math.sqrt(total_err_sq / max(total_points, 1))


def _yaml_matrix(rows: int, cols: int, data: np.ndarray) -> dict[str, Any]:
    return {"rows": rows, "cols": cols, "data": [float(x) for x in np.asarray(data).reshape(-1)]}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("ERROR: PyYAML is required to write calibration YAML. Install pyyaml.") from e
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _write_report(path: Path, detections: list[Detection], per_view_errors: list[float], args: argparse.Namespace) -> None:
    rows = []
    for i, det in enumerate(detections):
        rows.append(
            {
                "index": i,
                "path": det.path,
                "dictionary": det.dictionary_name,
                "markers": det.marker_count,
                "charuco_corners": det.corner_count,
                "rms_reprojection_error_px": per_view_errors[i] if i < len(per_view_errors) else None,
            }
        )
    report = {
        "created_wall_time": time.time(),
        "board": {
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_length_m": args.square_m,
            "marker_length_m": args.marker_m,
            "dictionary": args.dictionary,
        },
        "samples": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def _calibration_flags(cv2: Any, args: argparse.Namespace) -> int:
    flags = 0
    if args.fix_aspect_ratio:
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if args.zero_tangent_dist:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if args.fix_k3:
        flags |= cv2.CALIB_FIX_K3
    if args.rational_model:
        flags |= cv2.CALIB_RATIONAL_MODEL
    return flags


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate camera intrinsics with an OpenCV ChArUco board.")
    p.add_argument("--images", nargs="*", default=None, help="Image paths/globs. If omitted, live camera capture is used.")
    p.add_argument("--camera", default="0", help="OpenCV camera source, e.g. 1 or /dev/video1.")
    p.add_argument("--cam-index", dest="camera", help=argparse.SUPPRESS)
    p.add_argument("--backend", choices=("any", "v4l2"), default="v4l2", help="OpenCV capture backend for live capture.")
    p.add_argument("--fourcc", default="MJPG", help="Requested capture FourCC, e.g. MJPG, YUYV, or empty string.")
    p.add_argument("--fps", type=float, default=30.0, help="Requested capture FPS.")
    p.add_argument("--width", type=int, default=1280, help="Requested capture width.")
    p.add_argument("--height", type=int, default=720, help="Requested capture height.")
    p.add_argument("--samples", type=int, default=35, help="Number of accepted live samples to collect.")
    p.add_argument("--auto", action="store_true", help="Automatically save valid live frames.")
    p.add_argument("--auto-interval", type=float, default=0.8, help="Seconds between auto-saved frames.")
    p.add_argument("--session-dir", type=Path, default=DEFAULT_SESSION_DIR, help="Directory for captured frames and diagnostics.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output camera_intrinsics.yaml path.")
    p.add_argument("--camera-name", default="charuco_camera", help="camera_name field for the output YAML.")
    p.add_argument("--squares-x", type=int, default=7, help="Number of board squares in X/columns.")
    p.add_argument("--squares-y", type=int, default=5, help="Number of board squares in Y/rows.")
    p.add_argument("--square-m", type=float, default=0.0285, help="Chessboard square side length in meters.")
    p.add_argument("--marker-m", type=float, default=0.0170, help="ArUco marker side length in meters.")
    p.add_argument("--dictionary", default="DICT_4X4_1000", help="OpenCV ArUco dictionary name, numeric id, or auto.")
    p.add_argument("--min-corners", type=int, default=8, help="Minimum interpolated ChArUco corners required to accept an image.")
    p.add_argument("--min-samples", type=int, default=15, help="Minimum accepted images required before calibration.")
    p.add_argument(
        "--max-sample-error",
        type=float,
        default=None,
        help="Optional two-pass filter: discard samples whose first-pass per-view RMS error exceeds this many pixels.",
    )
    p.add_argument("--preview", action="store_true", help="Write detection overlay images for --images mode.")
    p.add_argument("--clean-session", action="store_true", help="Delete the existing session dir before live capture.")
    p.add_argument("--fix-aspect-ratio", action="store_true", help="OpenCV CALIB_FIX_ASPECT_RATIO.")
    p.add_argument("--zero-tangent-dist", action="store_true", help="OpenCV CALIB_ZERO_TANGENT_DIST.")
    p.add_argument("--fix-k3", action="store_true", help="OpenCV CALIB_FIX_K3.")
    p.add_argument("--rational-model", action="store_true", help="OpenCV CALIB_RATIONAL_MODEL for wide-angle lenses.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cv2 = _require_cv2()
    aruco = cv2.aruco

    if str(args.dictionary).lower() == "auto":
        candidates = []
        for name in AUTO_DICTS:
            dictionary = _make_dictionary(cv2, name)
            candidate_board = _make_board(
                cv2, args.squares_x, args.squares_y, args.square_m, args.marker_m, dictionary
            )
            candidates.append((name, dictionary, candidate_board))
        board = candidates[0][2]
    else:
        dictionary = _make_dictionary(cv2, str(args.dictionary))
        board = _make_board(cv2, args.squares_x, args.squares_y, args.square_m, args.marker_m, dictionary)
        candidates = [(str(args.dictionary), dictionary, board)]
    params = _make_detector_params(cv2)

    if args.images is None:
        if args.clean_session and args.session_dir.exists():
            shutil.rmtree(args.session_dir)
        detections = _collect_live(cv2, args, candidates, params, args.session_dir)
    else:
        preview_dir = args.session_dir / "preview" if args.preview else None
        detections = _collect_from_images(cv2, args.images, candidates, params, args.min_corners, preview_dir)

    if len(detections) < args.min_samples:
        raise SystemExit(
            f"ERROR: only {len(detections)} usable samples; need at least {args.min_samples}. "
            "Collect more views with board coverage near image edges and several tilted poses."
        )
    candidate_boards = {name: candidate_board for name, _dictionary, candidate_board in candidates}
    most_common_dictionary = max(
        {d.dictionary_name for d in detections},
        key=lambda name: sum(d.dictionary_name == name for d in detections),
    )
    board = candidate_boards[most_common_dictionary]
    if str(args.dictionary).lower() == "auto":
        print(f"Using detected dictionary for calibration: {most_common_dictionary}")

    flags = _calibration_flags(cv2, args)
    rms, K, dist, rvecs, tvecs, std_int, _std_ext, per_view_errors, image_size = _calibrate(cv2, board, detections, flags)
    filtered = _filter_detections_by_error(detections, per_view_errors, args.max_sample_error, args.min_samples)
    if len(filtered) != len(detections):
        detections = filtered
        rms, K, dist, rvecs, tvecs, std_int, _std_ext, per_view_errors, image_size = _calibrate(cv2, board, detections, flags)
    mean_err = _mean_reprojection_error(cv2, board, detections, K, dist, rvecs, tvecs)
    w, h = image_size
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 0.0, (w, h))

    payload = {
        "image_width": int(w),
        "image_height": int(h),
        "camera_name": args.camera_name,
        "camera_matrix": _yaml_matrix(3, 3, K),
        "distortion_model": "plumb_bob",
        "distortion_coefficients": _yaml_matrix(1, int(len(dist)), dist),
        "rectification_matrix": _yaml_matrix(3, 3, np.eye(3)),
        "projection_matrix": _yaml_matrix(3, 4, np.column_stack([new_K, np.zeros(3)])),
        "calibration_error": {
            "rms_reprojection_error_px": float(rms),
            "mean_reprojection_error_px": float(mean_err),
        },
        "calibration_board": {
            "type": "charuco",
            "squares_x": int(args.squares_x),
            "squares_y": int(args.squares_y),
            "square_length_m": float(args.square_m),
            "marker_length_m": float(args.marker_m),
            "dictionary": args.dictionary,
        },
    }
    if std_int is not None:
        payload["std_deviation_intrinsics"] = [float(x) for x in np.asarray(std_int).reshape(-1)]

    _write_yaml(args.output, payload)
    report_path = args.session_dir / "calibration_report.json"
    _write_report(report_path, detections, per_view_errors, args)

    print("\nCalibration complete")
    print(f"  output: {args.output}")
    print(f"  report: {report_path}")
    print(f"  image_size: {w}x{h}")
    print(f"  samples: {len(detections)}")
    print(f"  RMS reprojection error: {rms:.4f} px")
    print(f"  mean reprojection error: {mean_err:.4f} px")
    print("  camera_matrix:")
    print(np.array2string(K, precision=6, suppress_small=True))
    print(f"  distortion: {np.array2string(dist, precision=6, suppress_small=True)}")

    worst = sorted(enumerate(per_view_errors), key=lambda x: x[1], reverse=True)[:5]
    if worst:
        print("  worst samples:")
        for idx, err in worst:
            print(f"    {err:.4f} px  {detections[idx].path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
