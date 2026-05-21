"""Deploy the state-only AprilTag PPO policy on the real SO-ARM101.

This mirrors the sim command:

    uv run play --task Isaac-SO-ARM101-PickPlace-Bowl-StateAprilTag-Play-v0 --actor-only ...

The actor receives the same 27-D observation:

    joint_pos_rel(6), joint_vel_rel(6), gripper_state(1),
    bowl_xy(2), ee_proj_xy(2), ee_to_bowl_xy(2), last_action(6),
    cube_pos_xy_from_apriltag(2)

AprilTag inference uses:

    T_base_tag = T_base_gripper(q) @ T_gripper_camera @ T_camera_tag

Before policy rollout, the arm is homed to the sim reset pose:

    deploy.driver.HOME_POSE_RAD
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deploy.driver import (
    FK,
    FPS,
    HOME_POSE_RAD,
    JOINT_DEFAULTS_RAD,
    JOINT_NAMES,
    MAX_RAD_PER_STEP,
    QDOT_EWMA_ALPHA,
    URDF_PATH,
    _decode_action,
    _gripper_pct_from_sim_rad,
    _gripper_sim_rad_from_pct,
    _slew_limit,
)
from deploy.ppo_actor import PPOActorState, STATE_APRILTAG_STATE_DIM
from deploy.verify_apriltag_base_transform import (
    DEFAULT_HAND_EYE_YAML,
    _choose_detection,
    _detect_apriltags,
    _load_hand_eye,
    _load_intrinsics,
    _open_camera,
    _read_q_sim_rad,
    _require_cv2,
    fk_T_base_gripper,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = (
    PROJECT_ROOT
    / "isaac_so_arm101/logs/rsl_rl/pickplace_bowl_state_apriltag"
    / "2026-05-20_13-49-40/model_1450.pt"
)
DEFAULT_DEBUG_DIR = PROJECT_ROOT / "deploy/runs/real_policy_apriltag"
DEVICE = "cpu"
DEFAULT_BOWL_XY = "0.24,0.00"


def _wrap_angle_near(value: float, reference: float) -> float:
    return float(reference + ((value - reference + np.pi) % (2.0 * np.pi) - np.pi))


class RealSO101:
    def __init__(self, robot: Any):
        self.robot = robot

    def read_proprio_sim_rad(self) -> np.ndarray:
        q = _read_q_sim_rad(self.robot)
        q[4] = _wrap_angle_near(float(q[4]), float(JOINT_DEFAULTS_RAD[4]))
        return q

    def send_joint_targets_sim_rad(self, target_sim_rad: np.ndarray) -> None:
        cmd: dict[str, float] = {}
        for i, name in enumerate(JOINT_NAMES[:5]):
            cmd[f"{name}.pos"] = float(target_sim_rad[i]) * (180.0 / np.pi)
        cmd["gripper.pos"] = float(np.clip(_gripper_pct_from_sim_rad(float(target_sim_rad[5])), 0.0, 100.0))
        self.robot.send_action(cmd)


def _build_state_27(
    q_rad: np.ndarray,
    qdot_rad: np.ndarray,
    bowl_xy: np.ndarray,
    ee_xy: np.ndarray,
    last_action: np.ndarray,
    cube_xy: np.ndarray,
) -> np.ndarray:
    joint_pos_rel = (q_rad - JOINT_DEFAULTS_RAD)[:6]
    joint_vel_rel = qdot_rad[:6]
    gripper_state = q_rad[5:6]
    ee_to_bowl = bowl_xy - ee_xy
    state = np.concatenate(
        [
            joint_pos_rel,
            joint_vel_rel,
            gripper_state,
            bowl_xy,
            ee_xy,
            ee_to_bowl,
            last_action,
            cube_xy,
        ]
    ).astype(np.float32)
    if state.shape != (STATE_APRILTAG_STATE_DIM,):
        raise RuntimeError(f"state shape mismatch: {state.shape}, expected {(STATE_APRILTAG_STATE_DIM,)}")
    return state


def _home_robot(
    io: RealSO101,
    target_rad: np.ndarray = HOME_POSE_RAD,
    timeout_s: float = 8.0,
    tol: float = 0.035,
) -> np.ndarray:
    dt = 1.0 / FPS
    deadline = time.time() + timeout_s
    next_tick = time.time()
    q = io.read_proprio_sim_rad()
    while time.time() < deadline:
        residual = q - target_rad
        if float(np.max(np.abs(residual))) < tol:
            print(f"[deploy] homed: residual={dict(zip(JOINT_NAMES, residual.round(3).tolist()))}")
            return q
        io.send_joint_targets_sim_rad(target_rad)
        next_tick += dt
        sleep_for = next_tick - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.time()
        q = io.read_proprio_sim_rad()
    residual = q - target_rad
    print(f"[deploy] WARNING: homing timed out; residual={dict(zip(JOINT_NAMES, residual.round(3).tolist()))}")
    return q


def _connect_robot(args: argparse.Namespace):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
    robot.connect(calibrate=args.calibrate)
    return robot


def _detect_cube_xyz(
    *,
    cv2: Any,
    cap: Any,
    detector: Any,
    K: np.ndarray,
    dist: np.ndarray,
    tag_size: float,
    tag_id: int | None,
    T_gripper_camera: np.ndarray,
    T_base_gripper: np.ndarray,
) -> tuple[np.ndarray | None, dict | None, np.ndarray | None, np.ndarray, list[dict]]:
    ok, bgr = cap.read()
    if not ok:
        raise RuntimeError("camera returned no frame")
    undist, detections = _detect_apriltags(cv2, detector, bgr, K, dist, tag_size)
    target = _choose_detection(detections, tag_id)
    if target is None:
        return None, None, None, undist, detections
    T_base_tag = T_base_gripper @ T_gripper_camera @ target["T_camera_tag"]
    return T_base_tag[:3, 3].astype(np.float32), target, T_base_tag, undist, detections


def _draw_preview(
    cv2: Any,
    image: np.ndarray,
    detections: list[dict],
    target: dict | None,
    *,
    step: int,
    cube_xyz: np.ndarray | None,
    bowl_xy: np.ndarray,
    ee_xy: np.ndarray,
    cube_valid: bool,
    held_last: bool,
    grasped: bool,
    action: np.ndarray | None,
) -> np.ndarray:
    vis = image.copy()
    for det in detections:
        pts = det["corners"].astype(np.int32)
        is_target = target is not None and det["id"] == target["id"]
        color = (0, 255, 0) if is_target else (0, 180, 255)
        cv2.polylines(vis, [pts], True, color, 2)
        center = tuple(det["center"].astype(int))
        cv2.putText(
            vis,
            f"id={det['id']} m={det['margin']:.0f}",
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    if cube_xyz is None:
        line1 = f"step={step} cube: not detected"
    else:
        line1 = (
            f"step={step} cube_xyz=({cube_xyz[0]:+.3f},{cube_xyz[1]:+.3f},{cube_xyz[2]:+.3f}) "
            f"ee_xy=({ee_xy[0]:+.3f},{ee_xy[1]:+.3f}) "
            f"bowl=({bowl_xy[0]:+.3f},{bowl_xy[1]:+.3f})"
        )
    line2 = f"valid={cube_valid} held_last={held_last} grasped={grasped}"
    if action is not None:
        line2 += f" action=[{', '.join(f'{x:+.2f}' for x in action)}]"
    cv2.putText(vis, line1, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis, line2, (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "q/ESC=stop rollout", (16, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def _write_debug_row(debug_path: Path | None, row: dict[str, Any]) -> None:
    if debug_path is None:
        return
    with open(debug_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _parse_xy(text: str, name: str) -> np.ndarray:
    try:
        x_str, y_str = text.split(",", 1)
        return np.array([float(x_str), float(y_str)], dtype=np.float32)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"{name} must be 'x,y' in metres, got {text!r}") from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real SO101 PPO deploy with AprilTag xy observation.")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument(
        "--bowl-xy",
        default=DEFAULT_BOWL_XY,
        help=(
            "Bowl xy in robot base frame, metres. Defaults to 0.24,0.00, "
            "the midpoint of the sim command range x=[0.18,0.30], y=[-0.15,0.15]."
        ),
    )
    p.add_argument("--tag-id", type=int, default=5, help="AprilTag id on the target cube.")
    p.add_argument("--tag-size", type=float, default=0.014, help="AprilTag effective edge length in metres.")
    p.add_argument("--family", default="tagStandard41h12")
    p.add_argument("--camera", default="/dev/video1")
    p.add_argument("--backend", choices=("any", "v4l2"), default="v4l2")
    p.add_argument("--fourcc", default="MJPG")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0, help="Camera FPS request; policy loop still runs at sim FPS unless --policy-fps overrides.")
    p.add_argument("--policy-fps", type=float, default=float(FPS))
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--robot-id", default="eva-follower")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--intrinsics", type=Path, default=PROJECT_ROOT / "camera_intrinsics.yaml")
    p.add_argument("--hand-eye", type=Path, default=DEFAULT_HAND_EYE_YAML)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--dropout-hold", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grasp-xy-tol", type=float, default=0.04)
    p.add_argument("--home-timeout", type=float, default=8.0)
    p.add_argument("--no-confirm", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Load policy and run one synthetic 27-D forward only.")
    p.add_argument("--debug-dump", action="store_true")
    p.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True, help="Show live camera preview with AprilTag overlay.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bowl_xy = _parse_xy(args.bowl_xy, "--bowl-xy")
    if not args.ckpt.exists():
        raise SystemExit(f"ERROR: checkpoint not found: {args.ckpt}")

    policy = PPOActorState.from_checkpoint(args.ckpt, map_location=DEVICE).to(DEVICE)
    print(f"[deploy] loaded actor checkpoint: {args.ckpt}")
    print(f"[deploy] obs_dim={STATE_APRILTAG_STATE_DIM}, action_dim=6, tag_size={args.tag_size:.4f}m, tag_id={args.tag_id}")

    fk = FK(URDF_PATH)
    if args.dry_run:
        q = JOINT_DEFAULTS_RAD.copy()
        state = _build_state_27(
            q,
            np.zeros(6, dtype=np.float32),
            bowl_xy,
            fk.ee_xyz(q)[:2],
            np.zeros(6, dtype=np.float32),
            np.array([0.20, 0.00], dtype=np.float32),
        )
        with torch.no_grad():
            action = policy(torch.from_numpy(state).unsqueeze(0))[0].cpu().numpy()
        print(f"[deploy] dry-run state.shape={state.shape}, action={np.round(action, 4).tolist()}")
        return 0

    cv2 = _require_cv2()
    K, dist, intr_w, intr_h = _load_intrinsics(args.intrinsics)
    if intr_w != args.width or intr_h != args.height:
        print(f"[deploy] WARNING: intrinsics are {intr_w}x{intr_h}, capture requested {args.width}x{args.height}", file=sys.stderr)
    T_gripper_camera = _load_hand_eye(args.hand_eye)

    try:
        from pupil_apriltags import Detector
    except ImportError as e:
        raise SystemExit("ERROR: pupil-apriltags is required.") from e
    detector = Detector(families=args.family)

    debug_path: Path | None = None
    if args.debug_dump:
        run_dir = DEFAULT_DEBUG_DIR / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        debug_path = run_dir / "rollout.jsonl"
        with open(run_dir / "meta.json", "w") as f:
            json.dump(
                {
                    "ckpt": str(args.ckpt),
                    "bowl_xy": bowl_xy.tolist(),
                    "tag_id": args.tag_id,
                    "tag_size": args.tag_size,
                    "hand_eye": str(args.hand_eye),
                    "intrinsics": str(args.intrinsics),
                },
                f,
                indent=2,
            )
        print(f"[deploy] debug dump: {run_dir}")

    robot = _connect_robot(args)
    cap = _open_camera(cv2, args)
    io = RealSO101(robot)

    try:
        q0 = io.read_proprio_sim_rad()
        print(f"[deploy] current q_sim_rad={np.round(q0, 3).tolist()}")
        if not args.no_confirm:
            input("[deploy] Clear workspace. Press Enter to move arm to sim home pose, Ctrl-C to abort...")
        q_prev = _home_robot(io, timeout_s=args.home_timeout)
        ee_home = fk.ee_xyz(q_prev)
        print(f"[deploy] home ee_xyz={np.round(ee_home, 3).tolist()}")

        if not args.no_confirm:
            input("[deploy] Place cube/tag and bowl at the configured positions. Press Enter to start policy...")

        dt = 1.0 / float(args.policy_fps)
        qdot_filt = np.zeros(6, dtype=np.float32)
        last_action = np.zeros(6, dtype=np.float32)
        last_cube_xyz = np.zeros(3, dtype=np.float32)
        have_cube = False
        grasped = False
        n_valid = 0
        n_drop = 0
        next_tick = time.time()

        for step in range(args.max_steps):
            q = io.read_proprio_sim_rad()
            qdot_raw = (q - q_prev) / dt
            qdot_filt = (QDOT_EWMA_ALPHA * qdot_raw + (1.0 - QDOT_EWMA_ALPHA) * qdot_filt).astype(np.float32)
            q_prev = q
            T_base_gripper = fk_T_base_gripper(fk, q)
            ee_xy = T_base_gripper[:2, 3].astype(np.float32)

            cube_valid = False
            held_last_cube_xyz = False
            target_margin = None
            T_base_tag = None
            preview_image = None
            detections: list[dict] = []
            target = None
            cube_xyz, target, T_base_tag, preview_image, detections = _detect_cube_xyz(
                cv2=cv2,
                cap=cap,
                detector=detector,
                K=K,
                dist=dist,
                tag_size=args.tag_size,
                tag_id=args.tag_id,
                T_gripper_camera=T_gripper_camera,
                T_base_gripper=T_base_gripper,
            )
            if cube_xyz is not None:
                last_cube_xyz = cube_xyz
                have_cube = True
                cube_valid = True
                n_valid += 1
                target_margin = float(target["margin"]) if target is not None else None
            else:
                n_drop += 1
                held_last_cube_xyz = have_cube and args.dropout_hold

            if not have_cube:
                if args.preview and preview_image is not None:
                    vis = _draw_preview(
                        cv2,
                        preview_image,
                        detections,
                        target,
                        step=step,
                        cube_xyz=None,
                        bowl_xy=bowl_xy,
                        ee_xy=ee_xy,
                        cube_valid=False,
                        held_last=False,
                        grasped=grasped,
                        action=None,
                    )
                    cv2.imshow("real policy AprilTag deploy", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("[deploy] rollout stopped from preview window")
                        break
                io.send_joint_targets_sim_rad(HOME_POSE_RAD)
                print(f"[deploy] step={step}: target tag not detected yet; holding home")
                time.sleep(dt)
                continue

            if not cube_valid and not args.dropout_hold:
                if args.preview and preview_image is not None:
                    vis = _draw_preview(
                        cv2,
                        preview_image,
                        detections,
                        target,
                        step=step,
                        cube_xyz=last_cube_xyz,
                        bowl_xy=bowl_xy,
                        ee_xy=ee_xy,
                        cube_valid=False,
                        held_last=False,
                        grasped=grasped,
                        action=None,
                    )
                    cv2.imshow("real policy AprilTag deploy", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("[deploy] rollout stopped from preview window")
                        break
                io.send_joint_targets_sim_rad(target_rad if "target_rad" in locals() else HOME_POSE_RAD)
                print(f"[deploy] step={step}: target tag dropout; holding previous joint command")
                time.sleep(dt)
                continue

            cube_xyz_obs = last_cube_xyz
            state = _build_state_27(q, qdot_filt, bowl_xy, ee_xy, last_action, cube_xyz_obs[:2])
            with torch.no_grad():
                action = policy(torch.from_numpy(state).unsqueeze(0).to(DEVICE))[0].cpu().numpy()
            last_action = action.astype(np.float32)

            target_rad = _decode_action(action)
            target_rad = _slew_limit(target_rad, q, max_step=MAX_RAD_PER_STEP)
            io.send_joint_targets_sim_rad(target_rad)

            if not grasped and action[5] < 0.0:
                xy_dist = float(np.linalg.norm(last_cube_xyz[:2] - ee_xy))
                if xy_dist <= args.grasp_xy_tol:
                    grasped = True
                    print(f"[deploy] grasp latched at step={step}, xy_dist={xy_dist*1000:.1f}mm")

            if args.preview and preview_image is not None:
                vis = _draw_preview(
                    cv2,
                    preview_image,
                    detections,
                    target,
                    step=step,
                    cube_xyz=cube_xyz_obs,
                    bowl_xy=bowl_xy,
                    ee_xy=ee_xy,
                    cube_valid=cube_valid,
                    held_last=held_last_cube_xyz,
                    grasped=grasped,
                    action=action,
                )
                cv2.imshow("real policy AprilTag deploy", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[deploy] rollout stopped from preview window")
                    break

            if step % max(int(args.policy_fps), 1) == 0:
                print(
                    f"[deploy] step={step:04d} ee_xy={np.round(ee_xy,3).tolist()} "
                    f"cube_xyz={np.round(cube_xyz_obs,3).tolist()} bowl_xy={np.round(bowl_xy,3).tolist()} "
                    f"action={np.round(action,2).tolist()} valid={cube_valid} held_last={held_last_cube_xyz} "
                    f"margin={target_margin} grasped={grasped}"
                )

            _write_debug_row(
                debug_path,
                {
                    "step": step,
                    "q_sim_rad": q.tolist(),
                    "qdot_rad": qdot_filt.tolist(),
                    "state": state.tolist(),
                    "action": action.tolist(),
                    "target_rad": target_rad.tolist(),
                    "ee_xy": ee_xy.tolist(),
                    "cube_xyz": cube_xyz_obs.tolist(),
                    "cube_valid": cube_valid,
                    "held_last_cube_xyz": held_last_cube_xyz,
                    "tag_margin": target_margin,
                    "T_base_tag": None if T_base_tag is None else T_base_tag.tolist(),
                    "grasped": grasped,
                },
            )

            next_tick += dt
            sleep_for = next_tick - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.time()

        print(
            "[deploy] rollout done. "
            f"policy_valid={n_valid}, policy_dropouts={n_drop}, "
            f"grasped={grasped}"
        )
    finally:
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()
        robot.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
