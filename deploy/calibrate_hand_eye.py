"""Interactive hand-eye calibration for the SO-ARM101 wrist camera.

Computes ``T_ee_cam`` (rigid transform: end-effector frame → wrist-cam
frame) via Tsai-Lenz hand-eye calibration and writes
``deploy/hand_eye.yaml``. See ``docs/STATE_APRILTAG_PLAN.md`` §3 for the
canonical procedure and ``deploy/README.md`` Step 4 for context.

Workflow
--------

1. Place the 30 mm ID-99 AprilTag flat on the table, visible to the wrist
   cam.
2. Connect arm + camera (this script does that automatically).
3. By any means available — leader-arm teleop, manual back-driving with
   torque off, or scripted joint commands — move the wrist to **≥ 12
   distinct poses** where the tag remains in view. Vary EE position
   **and** orientation; pure translation gives a degenerate solve.
4. At each pose, focus the OpenCV preview window and press **SPACE** to
   capture. The script reads servo joints (→ FK → ``T_base_ee``) and the
   tag detection (→ PnP → ``T_cam_tag``) simultaneously.
5. When you have enough samples, press **Q** to solve. The script runs
   ``cv2.calibrateHandEye`` with ``CALIB_HAND_EYE_TSAI`` and writes the
   result.

Hardware torque
---------------

If your bus driver doesn't expose a torque-off API, the simplest way to
move the arm by hand is to physically unplug the servo bus before
running the script — but you also lose joint read-back. Easier: use the
LeRobot leader-arm teleop in another terminal, and just snapshot in this
script when the arm reaches a pose you like.

Quality check
-------------

After solving, the script computes the tag position in base frame for
every captured sample (``T_base_tag = T_base_ee · T_ee_cam · T_cam_tag``)
and prints the std across samples. The tag is physically fixed, so a
correct hand-eye gives near-zero std. Plan §3 requires **≤ 5 mm** by
ruler, and the script warns if the analytical std exceeds that bound.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from deploy.driver import (
    EE_LOCAL_OFFSET,
    FK,
    INTRINSICS_YAML,
    JOINT_NAMES,
    LerobotSO101Driver,
    URDF_PATH,
    _load_intrinsics,
)

DEPLOY_DIR = Path(__file__).resolve().parent
HAND_EYE_YAML = DEPLOY_DIR / "hand_eye.yaml"


def fk_T_base_ee(fk: FK, joint_pos_rad: np.ndarray) -> np.ndarray:
    """Return 4×4 ``T_base_ee`` for the current joint angles.

    Position part matches :meth:`FK.ee_xyz` exactly (same EE_LOCAL_OFFSET
    applied to the gripper_link pose). Rotation part is the gripper_link
    orientation in base frame (kinpy's chain returns this without
    additional offset, which is what we want — the offset only affects
    translation of the ee reference point, not its rotation).
    """
    arm_vals = {n: float(v) for n, v in zip(JOINT_NAMES[:5], joint_pos_rad[:5])}
    th = [arm_vals[n] for n in fk.chain.get_joint_parameter_names()]
    T = fk.chain.forward_kinematics(th)
    out = np.eye(4)
    out[:3, :3] = T.rot_mat
    out[:3, 3] = np.asarray(T.pos, dtype=np.float64) + T.rot_mat @ EE_LOCAL_OFFSET
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", default="tagStandard41h12",
                        help="AprilTag family used for the calibration tag.")
    parser.add_argument("--tag-id", type=int, default=99,
                        help="Tag ID printed on the calibration tag.")
    parser.add_argument("--tag-size", type=float, default=0.030,
                        help="Calibration tag edge length in metres (default 30 mm).")
    parser.add_argument("--n-poses", type=int, default=12,
                        help="Target number of samples (script lets you collect more).")
    parser.add_argument("--out", type=str, default=str(HAND_EYE_YAML),
                        help="Output YAML path.")
    args = parser.parse_args()

    # --- Intrinsics + FK + detector + hardware ----------------------------
    K, dist = _load_intrinsics()
    if K is None:
        print(f"ERROR: camera_intrinsics.yaml not found at {INTRINSICS_YAML}", file=sys.stderr)
        return 1
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    fk = FK(URDF_PATH)

    try:
        from pupil_apriltags import Detector as AprilDetector
    except ImportError:
        print("ERROR: pupil-apriltags not installed. Run deploy/setup_inference_pc.sh.",
              file=sys.stderr)
        return 1
    det = AprilDetector(families=args.family)

    print("[calib] connecting to arm + camera...")
    driver = LerobotSO101Driver()
    driver.connect()
    print("[calib] connected.")

    # --- Capture loop ------------------------------------------------------
    R_g2b: list[np.ndarray] = []  # gripper-in-base (R_ee_in_base)
    t_g2b: list[np.ndarray] = []
    R_t2c: list[np.ndarray] = []  # tag-in-cam (R_tag_in_cam)
    t_t2c: list[np.ndarray] = []
    samples: list[dict] = []      # store full transforms for verify pass

    print()
    print("=== Hand-eye calibration ===")
    print(f"Tag: family={args.family}, id={args.tag_id}, size={args.tag_size*1000:.0f} mm.")
    print(f"Target: collect at least {args.n_poses} samples (more is better).")
    print("Move the arm to varied EE positions AND orientations between samples.")
    print()
    print("Controls (focus the OpenCV preview window):")
    print("  SPACE — capture the current pose")
    print("  D     — drop the most recent sample")
    print("  Q     — finish + solve (needs ≥ 4 samples for a usable solve)")
    print("  ESC   — abort without writing the YAML")
    print()

    try:
        while True:
            rgb = driver.capture_wrist_rgb_hwc()
            rgb_undist = cv2.undistort(rgb, K, dist)
            gray = cv2.cvtColor(rgb_undist, cv2.COLOR_RGB2GRAY)
            detections = det.detect(
                gray, estimate_tag_pose=True,
                camera_params=(fx, fy, cx, cy),
                tag_size=args.tag_size,
            )
            target = next((d for d in detections if d.tag_id == args.tag_id), None)

            # Build preview (BGR for cv2.imshow).
            disp = cv2.cvtColor(rgb_undist, cv2.COLOR_RGB2BGR).copy()
            cv2.putText(
                disp, f"Samples: {len(R_g2b)}/{args.n_poses}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            if target is not None:
                z_cm = float(np.linalg.norm(target.pose_t)) * 100.0
                cv2.putText(
                    disp, f"Tag {target.tag_id} OK ({z_cm:.1f} cm)",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                pts = np.asarray(target.corners, dtype=np.int32)
                cv2.polylines(disp, [pts], True, (0, 255, 0), 2)
            else:
                cv2.putText(
                    disp, f"Tag {args.tag_id} NOT VISIBLE",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )
            cv2.imshow("hand-eye calibration", disp)

            key = cv2.waitKey(50) & 0xFF
            if key == ord(" "):
                if target is None:
                    print("[calib] no target tag in view — move arm to keep it visible.")
                    continue
                q_sim = driver.read_proprio_sim_rad()
                T_be = fk_T_base_ee(fk, q_sim)
                R_g2b.append(T_be[:3, :3].copy())
                t_g2b.append(T_be[:3, 3].copy())
                T_ct = np.eye(4)
                T_ct[:3, :3] = np.asarray(target.pose_R, dtype=np.float64)
                T_ct[:3, 3] = np.asarray(target.pose_t, dtype=np.float64).flatten()
                R_t2c.append(T_ct[:3, :3].copy())
                t_t2c.append(T_ct[:3, 3].copy())
                samples.append({"q": q_sim.copy(), "T_be": T_be, "T_ct": T_ct})
                print(
                    f"[calib] sample {len(R_g2b)} captured "
                    f"(tag distance: {np.linalg.norm(T_ct[:3, 3]) * 100:.1f} cm)"
                )
            elif key == ord("d"):
                if R_g2b:
                    R_g2b.pop(); t_g2b.pop(); R_t2c.pop(); t_t2c.pop(); samples.pop()
                    print(f"[calib] dropped last sample, now {len(R_g2b)} samples.")
            elif key == ord("q"):
                if len(R_g2b) < 4:
                    print(f"[calib] need ≥ 4 samples to solve (have {len(R_g2b)}).")
                    continue
                break
            elif key == 27:  # ESC
                print("[calib] aborted by user (ESC), nothing written.")
                return 1
    finally:
        cv2.destroyAllWindows()
        driver.disconnect()

    # --- Solve hand-eye ----------------------------------------------------
    print(f"\n[calib] solving hand-eye with {len(R_g2b)} samples (Tsai-Lenz)...")
    R_c2g, t_c2g = cv2.calibrateHandEye(
        R_gripper2base=R_g2b,
        t_gripper2base=t_g2b,
        R_target2cam=R_t2c,
        t_target2cam=t_t2c,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )
    T_ee_cam = np.eye(4)
    T_ee_cam[:3, :3] = R_c2g
    T_ee_cam[:3, 3] = t_c2g.flatten()
    print("[calib] T_ee_cam =")
    print(T_ee_cam)

    # --- Self-consistency check -------------------------------------------
    # Tag is physically fixed, so T_base_tag = T_be · T_ee_cam · T_ct
    # should be constant across samples. The std across samples is a
    # direct readout of the calibration residual.
    tag_positions = np.stack(
        [(s["T_be"] @ T_ee_cam @ s["T_ct"])[:3, 3] for s in samples], axis=0,
    )
    mean = tag_positions.mean(axis=0)
    std = tag_positions.std(axis=0)
    print()
    print(f"[calib] tag-position consistency across {len(samples)} samples:")
    print(f"  mean (xyz, mm): [{mean[0]*1000:+.1f}, {mean[1]*1000:+.1f}, {mean[2]*1000:+.1f}]")
    print(f"  std  (xyz, mm): [{std[0]*1000: .2f}, {std[1]*1000: .2f}, {std[2]*1000: .2f}]")
    if float(np.max(std)) > 0.005:
        print(
            "[calib] WARNING: max std > 5 mm. Possible causes: too few samples, "
            "samples too co-planar (pose variety too small), tag not flat, "
            "camera mount loose. Consider collecting more / better samples and re-running."
        )
    else:
        print("[calib] OK: max std within 5 mm tolerance.")

    # --- Write hand_eye.yaml ----------------------------------------------
    out_path = Path(args.out)
    with open(out_path, "w") as f:
        yaml.safe_dump(
            {
                "T_ee_cam": T_ee_cam.tolist(),
                "calibration_metadata": {
                    "n_samples": len(samples),
                    "tag_family": args.family,
                    "tag_id": int(args.tag_id),
                    "tag_size_m": float(args.tag_size),
                    "residual_mean_mm": (mean * 1000).tolist(),
                    "residual_std_mm": (std * 1000).tolist(),
                },
            },
            f,
            default_flow_style=False,
        )
    print(f"\n[calib] wrote {out_path}")
    print(
        "Next: verify physically by commanding the EE to the projected tag "
        "(x, y) and confirming ≤ 5 mm offset with a ruler. See "
        "docs/STATE_APRILTAG_PLAN.md §3 step 6."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
