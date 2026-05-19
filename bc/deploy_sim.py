"""Closed-loop deployment in Isaac sim.

Two modes:
  --zero-action       : feed zero actions through env.step. Sanity-checks
                        the wiring (env launch, reset, step, camera read,
                        joint-state read). Used for step 7 of BC_EVAL1_PLAN.
  (default rollout)   : load a trained BC ckpt, run k'=EXECUTE_K-step
                        action chunking, log success / metrics. Used for
                        steps 8 and 9.

Action conversion (deg-absolute → env's rad-delta-around-home):
  arm_action_i = (deg2rad(bc_target_deg_i) - default_pos_i) / arm_scale
  gripper:  bc_continuous > THRESH → open command (+1), else close (-1)

Notes on bowl override:
  The task's ``bowl_pose`` command samples a uniform xy each reset. To
  evaluate at a specific bowl, we (1) build the env, (2) reset, (3)
  manually overwrite the command buffer at slot 0–1 with our (x, y). The
  bowl pose marker in the viewport will move to the override on the next
  visualisation tick.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

# -------------------------- App launch boilerplate ---------------------------
from isaaclab.app import AppLauncher

_parser = argparse.ArgumentParser(description="BC closed-loop deploy in Isaac sim")
_parser.add_argument("--task", type=str,
                     default="Isaac-SO-ARM101-PickPlace-Bowl-Play-v0")
_parser.add_argument("--zero-action", action="store_true",
                     help="No policy; just step env with zero action for the sanity check.")
_parser.add_argument("--run", type=str, default="bc_eval1_v1",
                     help="Run name under bc/runs/")
_parser.add_argument("--ckpt", type=str, default="best.pt")
_parser.add_argument("--max-steps", type=int, default=300)
_parser.add_argument("--rollouts", type=int, default=1)
_parser.add_argument("--execute-k", type=int, default=4)
_parser.add_argument("--gripper-thresh", type=float, default=25.0,
                     help="BC continuous gripper > this → open command")
_parser.add_argument("--control-stride", type=int, default=2,
                     help="Hold each BC target for this many sim ticks. "
                          "Demos are 30 Hz; sim runs at 50 Hz, so stride=2 "
                          "→ 25 Hz effective ≈ demo rate.")
_parser.add_argument("--episode-length-s", type=float, default=15.0,
                     help="Override env's episode_length_s so we get more "
                          "than the default 5 s to complete pick-and-place.")
_parser.add_argument("--init-gripper-closed", action="store_true",
                     help="After reset, force gripper joint to 0 to match the "
                          "demos' initial state (sim home is gripper=0.5 rad).")
_parser.add_argument("--bowl-xy", type=str, default=None,
                     help="Comma-separated 'x,y' (m) to force the bowl. "
                          "If omitted, the task samples it randomly per reset.")
_parser.add_argument("--seed", type=int, default=0)
_parser.add_argument("--video", action="store_true",
                     help="If set, periodically dump the wrist-cam frame to disk.")
AppLauncher.add_app_launcher_args(_parser)
args = _parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --------------------------- Now safe to import ------------------------------
import gymnasium as gym                                                  # noqa: E402
import isaac_so_arm101.tasks                                              # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg                            # noqa: E402

from bc.config import ACTION_DIM, CHUNK_K, IMG_H, IMG_W, RUNS_DIR        # noqa: E402
from bc.model import GoalCondBCPolicy                                     # noqa: E402
from bc.normalize import Stats                                            # noqa: E402

ARM_SCALE = 0.5  # JointPositionActionCfg.scale in joint_pos_env_cfg.py
ARM_DEFAULT_RAD = np.array(
    [0.0, 0.0, 0.0, 1.57, 0.0], dtype=np.float64,
)  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll


def _force_bowl_xy(env, xy: tuple[float, float]) -> None:
    """Overwrite the active bowl-pose command for env 0 with (x, y, 0)."""
    cmd_mgr = env.unwrapped.command_manager
    term_name = "bowl_pose"
    if term_name not in cmd_mgr._terms:
        raise RuntimeError(f"command term '{term_name}' not found")
    term = cmd_mgr.get_term(term_name)
    # Command buffer shape: (N, 7) = (x, y, z, qw, qx, qy, qz)
    buf = term.command
    buf[0, 0] = float(xy[0])
    buf[0, 1] = float(xy[1])
    buf[0, 2] = 0.0


def _read_wrist_rgb_chw_u8(env) -> np.ndarray:
    """Wrist-cam RGB → (3, H, W) uint8 for env 0.

    The TiledCamera returns RGB at the env's configured size (128 × 72).
    Internally the layout is (N, H, W, C) float in [0,1] or uint8 0..255.
    """
    cam = env.unwrapped.scene["wrist_cam"]
    rgb = cam.data.output["rgb"]                          # (N, H, W, C)
    img = rgb[0]                                          # (H, W, C)
    if img.dtype != torch.uint8:
        img = (img.clamp(0, 1) * 255.0).to(torch.uint8)
    img = img.permute(2, 0, 1).contiguous()               # (C, H, W)
    return img.cpu().numpy()


def _read_proprio_deg(env) -> np.ndarray:
    """Robot joint pos in degrees (6,): arm + gripper.

    The articulation joint order is: shoulder_pan, shoulder_lift,
    elbow_flex, wrist_flex, wrist_roll, gripper — matching the demo schema.
    """
    robot = env.unwrapped.scene["robot"]
    q_rad = robot.data.joint_pos[0].detach().cpu().numpy()  # (6,)
    q_deg = np.rad2deg(q_rad)
    return q_deg.astype(np.float32)


def _bc_target_to_env_action(bc_target_deg: np.ndarray, gripper_thresh: float,
                              device: str) -> torch.Tensor:
    """Convert BC's 6-D absolute joint target (deg) → env's 6-D action.

    Arm: (rad - default) / scale.
    Gripper: continuous BC value > thresh → +1 (open command), else -1 (close).
    """
    q_rad = np.deg2rad(bc_target_deg[:5].astype(np.float64))
    arm_action = (q_rad - ARM_DEFAULT_RAD) / ARM_SCALE
    grip_action = 1.0 if bc_target_deg[5] > gripper_thresh else -1.0
    a = np.concatenate([arm_action, [grip_action]]).astype(np.float32)
    return torch.from_numpy(a).to(device).unsqueeze(0)            # (1, 6)


def _check_success(env) -> bool:
    """Use the env's real ``task_success`` predicate (mdp/terminations.py)."""
    from isaac_so_arm101.tasks.pickplace.mdp.terminations import task_success
    return bool(task_success(env.unwrapped)[0].item())


# ====================================================================== main
def main() -> int:
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    env_cfg = parse_env_cfg(args.task, num_envs=1)
    if args.episode_length_s and args.episode_length_s > 0:
        env_cfg.episode_length_s = args.episode_length_s
    env = gym.make(args.task, cfg=env_cfg)
    device = env.unwrapped.device

    # Load policy if needed.
    policy = stats = None
    if not args.zero_action:
        run_dir = RUNS_DIR / args.run
        stats = Stats.load(run_dir / "stats.json")
        ck = torch.load(run_dir / args.ckpt, map_location=device, weights_only=False)
        policy = GoalCondBCPolicy(k=CHUNK_K).to(device)
        policy.load_state_dict(ck["model"])
        policy.eval()
        print(f"Loaded BC ckpt: epoch={ck['epoch']} val_l1={ck['val_l1']:.4f}")

    print(f"obs space: {env.observation_space}")
    print(f"act space: {env.action_space}")

    bowl_xy = None
    if args.bowl_xy is not None:
        bowl_xy = tuple(float(s) for s in args.bowl_xy.split(","))
        assert len(bowl_xy) == 2

    successes = 0
    for r in range(args.rollouts):
        print(f"\n=== rollout {r + 1}/{args.rollouts} ===")
        env.reset()
        if bowl_xy is not None:
            _force_bowl_xy(env, bowl_xy)
        if args.init_gripper_closed:
            robot = env.unwrapped.scene["robot"]
            grip_idx = robot.find_joints("gripper")[0][0]
            zero = torch.zeros((1,), device=robot.device)
            robot.write_joint_position_to_sim(zero, joint_ids=[grip_idx])
            robot.write_joint_velocity_to_sim(zero, joint_ids=[grip_idx])
            print("  forced gripper=0 (closed) to match demos")
        # Read the actual bowl that the task will use this episode
        bowl_cmd = env.unwrapped.command_manager.get_term("bowl_pose").command[0, :3]
        bowl_xyz = bowl_cmd.detach().cpu().numpy().astype(np.float32)
        print(f"bowl_xyz (robot base): {bowl_xyz.round(3)}")

        # Verify wrist + joints work right after reset.
        img = _read_wrist_rgb_chw_u8(env)
        proprio_deg = _read_proprio_deg(env)
        print(f"  wrist img: shape={img.shape}, dtype={img.dtype}, "
              f"range=[{img.min()}, {img.max()}]")
        print(f"  proprio @ reset (deg): {proprio_deg.round(2)}")

        chunk = None; step_in_chunk = 0; stride_count = 0
        cur_action = None
        success = False
        t0 = time.time()
        terminated_internally = False
        for t in range(args.max_steps):
            if args.zero_action:
                cur_action = torch.zeros((1, ACTION_DIM), device=device)
            else:
                # Re-query BC every `control_stride` sim ticks.
                if cur_action is None or stride_count >= args.control_stride:
                    if chunk is None or step_in_chunk >= args.execute_k:
                        img = _read_wrist_rgb_chw_u8(env)
                        proprio_deg = _read_proprio_deg(env)
                        img_t = torch.from_numpy(img).to(device).unsqueeze(0)
                        prop_n = stats.normalize("proprio", proprio_deg)
                        bowl_n = stats.normalize("bowl", bowl_xyz)
                        prop_t = torch.from_numpy(prop_n).to(device).unsqueeze(0)
                        bowl_t = torch.from_numpy(bowl_n).to(device).unsqueeze(0)
                        with torch.no_grad():
                            out_n = policy(img_t, prop_t, bowl_t)                # (1, k, 6)
                        chunk = stats.denormalize("action", out_n[0].cpu().numpy())
                        step_in_chunk = 0
                    bc_target = chunk[step_in_chunk]
                    step_in_chunk += 1
                    stride_count = 0
                    cur_action = _bc_target_to_env_action(
                        bc_target, args.gripper_thresh, device)
                stride_count += 1

            _, _, terminated, truncated, _ = env.step(cur_action)

            # Check success BEFORE the env potentially resets internally.
            if _check_success(env):
                success = True
                print(f"  SUCCESS detected at t={t+1}")
                break

            if (t + 1) % 25 == 0:
                proprio_deg = _read_proprio_deg(env)
                print(f"  t={t+1:4d}  q_deg={proprio_deg.round(2)}")

            # Stop on internal termination (time_out or task termination).
            if bool(terminated[0].item()) or bool(truncated[0].item()):
                terminated_internally = True
                print(f"  episode terminated internally at t={t+1} "
                      f"(terminated={bool(terminated[0])}, "
                      f"truncated={bool(truncated[0])})")
                break

        dt = time.time() - t0
        print(f"  rollout {r+1}: {'SUCCESS' if success else 'fail'} "
              f"in {t+1} steps ({dt:.1f}s)")
        if success:
            successes += 1

    print(f"\n{'='*40}")
    print(f"Total: {successes}/{args.rollouts} successful rollouts")
    print(f"{'='*40}")
    env.close()
    return 0


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
