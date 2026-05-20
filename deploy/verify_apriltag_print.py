"""Print-quality smoke test for vendored AprilTag prints.

Captures one frame from the wrist USB cam, runs the ``tagStandard41h12``
detector, and prints per-tag ``decision_margin`` + on-image side length.
Healthy thresholds (see ``deploy/README.md`` Step 3):

* ``decision_margin >= 30`` — print + lighting are good.
* ``side_px >= 25`` — tag is large enough on the sensor for stable PnP.

Run after printing each batch of tags, **before** mounting them on the
cubes. No arm / FK / hand-eye needed — pure detector sanity check.

Usage:
    python -m deploy.verify_apriltag_print               # /dev/video0 (default)
    python -m deploy.verify_apriltag_print --cam-index 2 # other USB cam

Upstream detector: https://github.com/pupil-labs/apriltags
AprilTag algorithm + tag families: https://github.com/AprilRobotics/apriltag
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cam-index", type=int, default=0,
                   help="OpenCV camera index (default 0).")
    p.add_argument("--family", default="tagStandard41h12",
                   help="AprilTag family (default matches the vendored prints).")
    args = p.parse_args()

    try:
        from pupil_apriltags import Detector
    except ImportError:
        print("ERROR: pupil-apriltags not installed. "
              "Run deploy/setup_inference_pc.sh.", file=sys.stderr)
        return 1

    det = Detector(families=args.family)
    cap = cv2.VideoCapture(args.cam_index)
    if not cap.isOpened():
        print(f"ERROR: failed to open camera index {args.cam_index}. "
              "Check the USB cam is plugged in / try a different --cam-index.",
              file=sys.stderr)
        return 1

    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("ERROR: no frame captured from the camera.", file=sys.stderr)
        return 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = det.detect(gray)

    if not detections:
        print("No tags detected. Common causes: tag not in view, motion blur, "
              "glossy paper. Re-print on matte paper / hold closer / hold still.")
        return 0

    print(f"Detected {len(detections)} tag(s):")
    for t in detections:
        side_px = float(np.linalg.norm(t.corners[0] - t.corners[1]))
        margin_ok = "OK" if t.decision_margin >= 30 else "LOW"
        side_ok = "OK" if side_px >= 25 else "SMALL"
        print(f"  id={t.tag_id:3d}  margin={t.decision_margin:5.1f} ({margin_ok})"
              f"  side={side_px:5.1f} px ({side_ok})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
