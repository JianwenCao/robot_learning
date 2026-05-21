"""Deploy the RGB vision student policy on the real SO-ARM101.

The actor receives the same deployable inputs as the sim student:

    policy state(25) + wrist RGB image(3, 240, 320)

No AprilTag, depth, or mask is used at inference.
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
    _slew_limit,
)
from deploy.ppo_actor import PPOActorVision, STATE_DIM
from deploy.verify_apriltag_base_transform import (
    _load_intrinsics,
    _open_camera,
    _read_q_sim_rad,
    _require_cv2,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = PROJECT_ROOT / "deploy/runs/vision_student.pt"
DEFAULT_DEBUG_DIR = PROJECT_ROOT / "deploy/runs/real_policy_vision"
DEFAULT_INTRINSICS = PROJECT_ROOT.parent / "eva_follower" / "intrinsics.yaml"
DEVICE = "cpu"
DEFAULT_BOWL_XY = "0.24,0.00"
POLICY_IMAGE_WIDTH = 320
POLICY_IMAGE_HEIGHT = 240


def _wrap_angle_near(value: float, reference: float) -> float:
    return float(reference + ((value - reference + np.pi) % (2.0 * np.pi) - np.pi))


class RealSO101:
    def __init__(self, robot: Any):
        self.robot = robot

    def read_proprio_sim_rad(self) -> np.ndarray:
        q = _read_q_sim_rad(self.robot)
        # q[4] = _wrap_angle_near(float(q[4]), float(JOINT_DEFAULTS_RAD[4]))
        return q

    def send_joint_targets_sim_rad(self, target_sim_rad: np.ndarray) -> None:
        cmd: dict[str, float] = {}
        for i, name in enumerate(JOINT_NAMES[:5]):
            cmd[f"{name}.pos"] = float(target_sim_rad[i]) * (180.0 / np.pi)
        cmd["gripper.pos"] = float(np.clip(_gripper_pct_from_sim_rad(float(target_sim_rad[5])), 0.0, 100.0))
        self.robot.send_action(cmd)


def _connect_robot(args: argparse.Namespace):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
    robot.connect(calibrate=args.calibrate)
    return robot


def _home_robot(
    io: RealSO101,
    target_rad: np.ndarray = HOME_POSE_RAD,
    timeout_s: float = 8.0,
    tol: float = 0.035,
    duration_s: float = 5.0,
) -> np.ndarray:
    """Move to HOME with a fixed smooth interpolation from the current pose."""
    dt = 1.0 / FPS
    q_start = io.read_proprio_sim_rad()
    next_tick = time.time()
    n_steps = max(int(round(float(duration_s) * FPS)), 1)
    for i in range(n_steps + 1):
        alpha = i / n_steps
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        cmd_rad = ((1.0 - alpha) * q_start + alpha * target_rad).astype(np.float32)
        io.send_joint_targets_sim_rad(cmd_rad)
        next_tick += dt
        sleep_for = next_tick - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.time()

    deadline = time.time() + max(float(timeout_s) - float(duration_s), 0.0)
    q = io.read_proprio_sim_rad()
    while time.time() < deadline:
        residual = q - target_rad
        if float(np.max(np.abs(residual))) < tol:
            print(f"[deploy-vision] homed: residual={dict(zip(JOINT_NAMES, residual.round(3).tolist()))}")
            return q
        io.send_joint_targets_sim_rad(target_rad)
        time.sleep(dt)
        q = io.read_proprio_sim_rad()
    residual = q - target_rad
    print(f"[deploy-vision] WARNING: homing timed out; residual={dict(zip(JOINT_NAMES, residual.round(3).tolist()))}")
    return q


def _parse_xy(text: str, name: str) -> np.ndarray:
    try:
        x_str, y_str = text.split(",", 1)
        return np.array([float(x_str), float(y_str)], dtype=np.float32)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"{name} must be 'x,y' in metres, got {text!r}") from e


def _build_state_25(
    q_rad: np.ndarray,
    qdot_rad: np.ndarray,
    bowl_xy: np.ndarray,
    ee_xy: np.ndarray,
    last_action: np.ndarray,
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
        ]
    ).astype(np.float32)
    if state.shape != (STATE_DIM,):
        raise RuntimeError(f"state shape mismatch: {state.shape}, expected {(STATE_DIM,)}")
    return state


def _read_policy_image(cv2: Any, cap: Any, K: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    ok, bgr = cap.read()
    if not ok:
        raise RuntimeError("camera returned no frame")
    undist = cv2.undistort(bgr, K, dist)
    resized = cv2.resize(undist, (POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return undist, rgb, tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real SO101 deploy for RGB vision student policy.")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--bowl-xy", default=DEFAULT_BOWL_XY)
    p.add_argument("--camera", default="/dev/video1")
    p.add_argument("--backend", choices=("any", "v4l2"), default="v4l2")
    p.add_argument("--fourcc", default="MJPG")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--policy-fps", type=float, default=float(FPS))
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--robot-id", default="eva-follower")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument(
        "--max-rad-per-step",
        type=float,
        default=float(MAX_RAD_PER_STEP),
        help="Per-policy-step joint target slew limit in rad. Set <=0 to disable.",
    )
    p.add_argument("--home-timeout", type=float, default=8.0)
    p.add_argument(
        "--home-duration",
        type=float,
        default=5.0,
        help="Seconds for smooth interpolation from current pose to HOME before rollout.",
    )
    p.add_argument("--no-confirm", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--debug-dump", action="store_true")
    p.add_argument("--preview", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bowl_xy = _parse_xy(args.bowl_xy, "--bowl-xy")
    if not args.ckpt.exists():
        raise SystemExit(f"ERROR: checkpoint not found: {args.ckpt}")
    if not args.intrinsics.exists():
        raise SystemExit(f"ERROR: intrinsics not found: {args.intrinsics}")

    policy = PPOActorVision.from_checkpoint(args.ckpt, map_location=DEVICE).to(DEVICE)
    print(f"[deploy-vision] loaded actor checkpoint: {args.ckpt}")
    print(f"[deploy-vision] obs=state25+rgb3x240x320, action_dim=6")
    if args.max_rad_per_step > 0.0:
        print(f"[deploy-vision] joint slew limit: {args.max_rad_per_step:.4f} rad/step")
    else:
        print("[deploy-vision] joint slew limit: disabled")
    print(f"[deploy-vision] homing interpolation: duration={args.home_duration:.1f}s, timeout={args.home_timeout:.1f}s")

    if args.dry_run:
        state = _build_state_25(
            JOINT_DEFAULTS_RAD.copy(),
            np.zeros(6, dtype=np.float32),
            bowl_xy,
            np.array([0.2421, -0.0007], dtype=np.float32),
            np.zeros(6, dtype=np.float32),
        )
        image = torch.zeros(1, 3, POLICY_IMAGE_HEIGHT, POLICY_IMAGE_WIDTH)
        with torch.no_grad():
            action = policy(torch.from_numpy(state).unsqueeze(0), image)[0].cpu().numpy()
        print(f"[deploy-vision] dry-run state.shape={state.shape}, image.shape={tuple(image.shape)}, action={np.round(action, 4).tolist()}")
        return 0

    fk = FK(URDF_PATH)

    cv2 = _require_cv2()
    K, dist, intr_w, intr_h = _load_intrinsics(args.intrinsics)
    if intr_w != args.width or intr_h != args.height:
        print(f"[deploy-vision] WARNING: intrinsics are {intr_w}x{intr_h}, capture requested {args.width}x{args.height}", file=sys.stderr)

    debug_path: Path | None = None
    debug_dir: Path | None = None
    if args.debug_dump:
        debug_dir = DEFAULT_DEBUG_DIR / time.strftime("%Y%m%d-%H%M%S")
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / "rollout.jsonl"
        with open(debug_dir / "meta.json", "w") as f:
            json.dump({"ckpt": str(args.ckpt), "bowl_xy": bowl_xy.tolist(), "intrinsics": str(args.intrinsics)}, f, indent=2)
        print(f"[deploy-vision] debug dump: {debug_dir}")

    robot = _connect_robot(args)
    cap = _open_camera(cv2, args)
    io = RealSO101(robot)

    try:
        q0 = io.read_proprio_sim_rad()
        print(f"[deploy-vision] current q_sim_rad={np.round(q0, 3).tolist()}")
        if not args.no_confirm:
            input("[deploy-vision] Clear workspace. Press Enter to move arm to sim home pose, Ctrl-C to abort...")
        q_prev = _home_robot(io, timeout_s=args.home_timeout, duration_s=float(args.home_duration))
        print(f"[deploy-vision] home ee_xyz={np.round(fk.ee_xyz(q_prev), 3).tolist()}")
        if not args.no_confirm:
            input("[deploy-vision] Place cube and bowl at configured positions. Press Enter to start policy...")

        dt = 1.0 / float(args.policy_fps)
        qdot_filt = np.zeros(6, dtype=np.float32)
        last_action = np.zeros(6, dtype=np.float32)
        next_tick = time.time()

        for step in range(args.max_steps):
            q = io.read_proprio_sim_rad()
            qdot_raw = (q - q_prev) / dt
            qdot_filt = (QDOT_EWMA_ALPHA * qdot_raw + (1.0 - QDOT_EWMA_ALPHA) * qdot_filt).astype(np.float32)
            q_prev = q
            ee_xy = fk.ee_xyz(q)[:2].astype(np.float32)
            preview_bgr, policy_rgb, image_t = _read_policy_image(cv2, cap, K, dist)
            state = _build_state_25(q, qdot_filt, bowl_xy, ee_xy, last_action)
            with torch.no_grad():
                action = policy(torch.from_numpy(state).unsqueeze(0).to(DEVICE), image_t.to(DEVICE))[0].cpu().numpy()
            last_action = action.astype(np.float32)

            target_rad = _decode_action(action)
            if args.max_rad_per_step > 0.0:
                target_rad = _slew_limit(target_rad, q, max_step=float(args.max_rad_per_step))
            io.send_joint_targets_sim_rad(target_rad)

            if args.preview:
                vis = preview_bgr.copy()
                cv2.putText(vis, f"step={step} bowl=({bowl_xy[0]:+.2f},{bowl_xy[1]:+.2f}) ee=({ee_xy[0]:+.2f},{ee_xy[1]:+.2f})", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(vis, f"action=[{', '.join(f'{x:+.2f}' for x in action)}] q/ESC=stop", (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("real policy vision deploy", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[deploy-vision] rollout stopped from preview window")
                    break

            if debug_path is not None:
                with open(debug_path, "a") as f:
                    f.write(json.dumps({
                        "step": step,
                        "q_sim_rad": q.tolist(),
                        "state": state.tolist(),
                        "action": action.tolist(),
                        "target_rad": target_rad.tolist(),
                        "ee_xy": ee_xy.tolist(),
                        "policy_image_shape": list(image_t.shape),
                        "policy_image_mean": float(image_t.mean().item()),
                        "policy_image_std": float(image_t.std().item()),
                    }) + "\n")
                if debug_dir is not None:
                    cv2.imwrite(
                        str(debug_dir / f"policy_input_{step:04d}.png"),
                        cv2.cvtColor(policy_rgb, cv2.COLOR_RGB2BGR),
                    )
                if debug_dir is not None and step % max(int(args.policy_fps), 1) == 0:
                    cv2.imwrite(str(debug_dir / f"camera_{step:04d}.png"), preview_bgr)

            if step % max(int(args.policy_fps), 1) == 0:
                print(
                    f"[deploy-vision] step={step:04d} ee_xy={np.round(ee_xy,3).tolist()} "
                    f"img_mean={float(image_t.mean()):.3f} img_std={float(image_t.std()):.3f} "
                    f"action={np.round(action,2).tolist()}"
                )

            next_tick += dt
            sleep_for = next_tick - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.time()
    finally:
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()
        robot.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
