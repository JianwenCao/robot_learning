"""End-to-end AprilTag → base-frame xy sanity check.

Uses the same components the closed-loop deploy uses (driver wrist cam,
URDF FK, ``AprilTagDetector`` loading ``camera_intrinsics.yaml`` +
``deploy/hand_eye.yaml``) to localise one cube tag and print its xy in the
**robot base frame**. Run this after hand-eye calibration but before
loading a policy — it catches a stale ``hand_eye.yaml``, swapped
intrinsics, or a tag-size mismatch in seconds.

Usage:
    python -m deploy.verify_apriltag_chain                          # red cube (id 0), 15 mm
    python -m deploy.verify_apriltag_chain --tag-id 1               # blue cube
    python -m deploy.verify_apriltag_chain --tag-id 99 --tag-size 0.030   # calibration tag

See ``deploy/README.md`` Step 5 for expected results + the troubleshooting
table (off by 5–10 cm → redo hand-eye; off by ≈ tag size → wrong
--tag-size; off by ~5 % → undistort skipped).

Upstream detector: https://github.com/pupil-labs/apriltags
AprilTag algorithm + tag families: https://github.com/AprilRobotics/apriltag
"""
from __future__ import annotations

import argparse
import sys

from deploy.calibrate_hand_eye import fk_T_base_ee
from deploy.cube_detector import AprilTagDetector
from deploy.driver import FK, URDF_PATH, LerobotSO101Driver


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tag-id", type=int, default=0,
                   help="Target tag ID. 0=red, 1=blue, 2=yellow, 3=green, "
                        "4=purple, 5=orange, 99=calibration tag.")
    p.add_argument("--tag-size", type=float, default=0.015,
                   help="Tag edge length in metres (default 15 mm for "
                        "cube tags; pass 0.030 for the calibration tag).")
    args = p.parse_args()

    drv = LerobotSO101Driver()
    drv.connect()
    try:
        fk = FK(URDF_PATH)
        det = AprilTagDetector(target_id=args.tag_id, tag_size_m=args.tag_size)

        q = drv.read_proprio_sim_rad()
        T_be = fk_T_base_ee(fk, q)
        rgb = drv.capture_wrist_rgb_hwc()
        (x, y), ok = det.pose(rgb, T_be)
    finally:
        drv.disconnect()

    print(f"target_id = {args.tag_id}, tag_size = {args.tag_size*1000:.0f} mm")
    print(f"cube xy (base frame, m): ({x:+.3f}, {y:+.3f})   valid={ok}")
    if not ok:
        print("Tag not in view of the current wrist pose. Move the arm or "
              "the cube into view and re-run.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
