# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import isaac_so_arm101.scripts.rsl_rl.cli_args as cli_args # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# Sim-side debug dump — mirrors deploy/deploy_real.py --debug-dump output layout so
# a sim rollout and a real rollout can be diffed step-by-step.
parser.add_argument("--debug-dump", action="store_true",
                    help="Save per-step wrist image (RGB+mask composite) + a state/action JSONL "
                         "for one fixed-length episode. Forces num_envs=1.")
parser.add_argument("--dump-steps", type=int, default=250,
                    help="Number of steps to dump (default 250 = 5 s @ 50 Hz, matches deploy).")
parser.add_argument("--dump-bowl-xy", type=str, default=None,
                    help="Comma-separated 'x,y' metres in robot base frame. Forces the bowl_pose "
                         "command for env 0 so the sim rollout uses the same target as the real run.")
parser.add_argument("--dump-out", type=str, default=None,
                    help="Output directory for the dump. Default: <ckpt_dir>/debug_dump/<timestamp>/")
# Behaviour-diagnosis mode: run N full episodes, log per-step trajectory + per-episode
# summary stats, then aggregate. Designed to answer "is the policy lifting-and-holding,
# or gaming the success latches?". Forces num_envs=1 for clean per-episode bookkeeping.
parser.add_argument("--diag-rollouts", type=int, default=0,
                    help="Run N episodes with detailed behaviour logging. 0 = disabled (default).")
parser.add_argument("--diag-out", type=str, default=None,
                    help="Output directory for diagnosis. Default: <ckpt_dir>/diag/<timestamp>/")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
# debug dump needs wrist cam → enable cameras + single env
if args_cli.debug_dump:
    args_cli.enable_cameras = True
    if args_cli.num_envs is None:
        args_cli.num_envs = 1
# diag mode also needs camera (vision policy) and single-env bookkeeping.
# Run user's choice of headed/headless — they pass --headless if they want.
if args_cli.diag_rollouts > 0:
    args_cli.enable_cameras = True
    if args_cli.num_envs is None:
        args_cli.num_envs = 1

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


# Sim-side mirror of deploy.driver constants. Must match exactly — the action
# decoding here only exists to log ``target_sim_rad`` for diffing; the env's
# action manager still does the real decode internally.
_JOINT_DEFAULTS_RAD = [0.0, 0.0, 0.0, 1.57, 0.0, 0.5]
_ARM_ACTION_SCALE = 0.5
_GRIPPER_OPEN_RAD = 0.5
_GRIPPER_CLOSE_RAD = 0.0


def _force_bowl_xy(env, xy):
    """Overwrite the active bowl_pose command for env 0 to pin a fixed eval xy."""
    cmd_mgr = env.unwrapped.command_manager
    if "bowl_pose" not in cmd_mgr._terms:
        raise RuntimeError("command term 'bowl_pose' not found")
    buf = cmd_mgr.get_term("bowl_pose").command
    buf[0, 0] = float(xy[0])
    buf[0, 1] = float(xy[1])
    buf[0, 2] = 0.0


def _decode_target(action6):
    """Same arm/gripper decode as deploy.driver._decode_action — for log parity."""
    import numpy as np
    arm = [_JOINT_DEFAULTS_RAD[i] + _ARM_ACTION_SCALE * float(action6[i]) for i in range(5)]
    grip = _GRIPPER_OPEN_RAD if float(action6[5]) > 0.0 else _GRIPPER_CLOSE_RAD
    return np.array(arm + [grip], dtype=np.float32)


def _run_debug_dump(env, policy, args_cli, ckpt_path, log_dir, dt):
    """Run a fixed-length episode, dumping wrist image + state/action per step.

    Layout matches ``deploy/deploy_real.py`` ``--debug-dump``:
        <out>/step_XXXX.png   RGB|mask composite, 72×256
        <out>/log.jsonl       per-step: t, state, action, q_sim_rad, ee_xy, target_sim_rad,
                              plus critic-only ground truth (block_xy, is_grasped)
        <out>/meta.json       ckpt, bowl_xy, fps, dt, num_envs, mask_source
    """
    import json
    import time as _time
    import numpy as np
    import cv2

    fps = int(round(1.0 / dt)) if dt > 0 else 50

    # output dir
    if args_cli.dump_out:
        out = os.path.abspath(args_cli.dump_out)
    else:
        stamp = _time.strftime("%Y%m%d-%H%M%S")
        out = os.path.join(log_dir, "debug_dump", stamp)
    os.makedirs(out, exist_ok=True)

    # bowl override (optional). Apply *after* the env's auto-reset so the
    # command buffer is populated; if not provided, log whatever was sampled.
    bowl_xy = None
    if args_cli.dump_bowl_xy:
        x, y = (float(s) for s in args_cli.dump_bowl_xy.split(","))
        bowl_xy = (x, y)
        # The env auto-resets on make; rebroadcast via explicit reset so the
        # bowl command is freshly sampled, then immediately override env 0.
        env.reset()
        _force_bowl_xy(env, bowl_xy)
    bowl_cmd_buf = env.unwrapped.command_manager.get_term("bowl_pose").command
    bowl_xy_logged = bowl_cmd_buf[0, :2].detach().cpu().numpy().tolist()

    meta = {
        "ckpt": str(ckpt_path),
        "task": args_cli.task,
        "bowl_xy": bowl_xy_logged,
        "fps": fps,
        "dt": float(dt),
        "num_envs": int(env.unwrapped.num_envs),
        "dump_steps": int(args_cli.dump_steps),
        "joint_defaults_rad": _JOINT_DEFAULTS_RAD,
        "mask_source": "semantic_segmentation",
        "image_shape": [4, 72, 128],
        "state_layout": {
            "joint_pos_rel": [0, 6],
            "joint_vel_rel": [6, 12],
            "gripper_state": [12, 13],
            "bowl_xy": [13, 15],
            "ee_proj_xy": [15, 17],
            "ee_to_bowl_xy": [17, 19],
            "last_action": [19, 25],
        },
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[play] --debug-dump active: writing to {out}")

    log_path = os.path.join(out, "log.jsonl")
    obs = env.get_observations()
    defaults = np.array(_JOINT_DEFAULTS_RAD, dtype=np.float32)

    with open(log_path, "w") as logf:
        for t in range(int(args_cli.dump_steps)):
            with torch.inference_mode():
                action = policy(obs)
                next_obs, _, _, _ = env.step(action)

            # env 0 only
            state = obs["policy"][0].detach().cpu().numpy().astype(np.float32)
            img = obs["wrist_image"][0].detach().cpu().numpy()           # (4, H, W) in [0,1]
            critic = obs["critic"][0].detach().cpu().numpy().astype(np.float32) \
                if "critic" in obs.keys() else None
            act_np = action[0].detach().cpu().numpy().astype(np.float32)

            # composite PNG: RGB | mask×3, side-by-side
            rgb_u8 = (img[:3].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
            mask_u8 = (img[3:4].repeat(3, axis=0).transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
            composite = np.concatenate([rgb_u8, mask_u8], axis=1)
            cv2.imwrite(os.path.join(out, f"step_{t:04d}.png"),
                        cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

            # derived diagnostics (matching deploy_real log fields)
            q_sim_rad = (state[0:6] + defaults).tolist()
            ee_xy = state[15:17].tolist()
            target_sim_rad = _decode_target(act_np).tolist()

            row = {
                "t": int(t),
                "state": state.tolist(),
                "action": act_np.tolist(),
                "q_sim_rad": q_sim_rad,
                "ee_xy": ee_xy,
                "target_sim_rad": target_sim_rad,
            }
            if critic is not None:
                # critic layout: 6+6+1+2+2+2+3+2+3+1+6 = 34
                # block_position is at [19:22], block_to_bowl_xy [22:24], is_grasped [27:28]
                row["block_xyz_gt"] = critic[19:22].tolist()
                row["block_to_bowl_xy_gt"] = critic[22:24].tolist()
                row["is_grasped_gt"] = float(critic[27])
            logf.write(json.dumps(row) + "\n")

            if (t + 1) % 30 == 0:
                print(f"  t={t+1:4d}/{args_cli.dump_steps}  "
                      f"action={np.round(act_np, 2).tolist()}  "
                      f"ee_xy={[round(v, 3) for v in ee_xy]}")

            obs = next_obs

    print(f"[play] debug dump written to {out}")


def _run_diag(env, policy, args_cli, ckpt_path, log_dir, dt):
    """Run N episodes with detailed per-step + per-episode behaviour logging.

    Diagnostic for "is the policy actually doing lift-and-hold, or gaming the
    success latches?". Single-env so per-episode bookkeeping is unambiguous.

    Output layout:
        <out>/meta.json          ckpt, task, n_episodes, fps, dt, env_constants
        <out>/steps.jsonl        per-step row across all episodes (ep, t, … fields)
        <out>/episodes.jsonl     per-episode summary row
        <out>/summary.txt        aggregate stats across all episodes
    """
    import json
    import time as _time
    import numpy as np

    n_eps = int(args_cli.diag_rollouts)
    fps = int(round(1.0 / dt)) if dt > 0 else 50

    if args_cli.diag_out:
        out = os.path.abspath(args_cli.diag_out)
    else:
        stamp = _time.strftime("%Y%m%d-%H%M%S")
        out = os.path.join(log_dir, "diag", stamp)
    os.makedirs(out, exist_ok=True)

    # Optional bowl override — share --dump-bowl-xy with --debug-dump.
    bowl_xy = None
    if args_cli.dump_bowl_xy:
        x, y = (float(s) for s in args_cli.dump_bowl_xy.split(","))
        bowl_xy = (x, y)
        env.reset()
        _force_bowl_xy(env, bowl_xy)

    meta = {
        "ckpt": str(ckpt_path),
        "task": args_cli.task,
        "n_episodes_requested": n_eps,
        "num_envs": int(env.unwrapped.num_envs),
        "fps": fps,
        "dt": float(dt),
        "joint_defaults_rad": _JOINT_DEFAULTS_RAD,
        "lift_latch_threshold_m": 0.07,
        "over_bowl_high_threshold_m": 0.08,
        "over_bowl_xy_threshold_m": 0.06,
        "bowl_xy_override": bowl_xy,
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[diag] writing to {out}")

    defaults = np.array(_JOINT_DEFAULTS_RAD, dtype=np.float32)
    steps_f = open(os.path.join(out, "steps.jsonl"), "w")
    eps_f = open(os.path.join(out, "episodes.jsonl"), "w")

    obs = env.get_observations()
    ep_idx = 0
    t_in_ep = 0
    ep_state = _new_episode_state()
    max_steps_safety = 600  # absolute cap per episode (sim is 250 steps @ 50 Hz)
    completed = []

    while ep_idx < n_eps:
        with torch.inference_mode():
            action = policy(obs)
            next_obs, rewards_step, dones, _extras = env.step(action)

        # --- pull env-0 data for this step ---
        state = obs["policy"][0].detach().cpu().numpy().astype(np.float32)
        critic = obs["critic"][0].detach().cpu().numpy().astype(np.float32) \
            if "critic" in obs.keys() else None
        act_np = action[0].detach().cpu().numpy().astype(np.float32)
        rew_np = float(rewards_step[0].detach().cpu().item()) \
            if hasattr(rewards_step, "detach") else float(rewards_step[0])
        done_np = bool(dones[0].detach().cpu().item()) if hasattr(dones, "detach") else bool(dones[0])

        # critic layout: 6+6+1+2+2+2+(block_xyz=3 at 19:22)+(block_to_bowl_xy=2 at 22:24)
        #               +(gripper_to_block=3 at 24:27)+(is_grasped=1 at 27:28)+...
        cube_xyz = critic[19:22].tolist() if critic is not None else [None, None, None]
        cube_to_bowl = critic[22:24].tolist() if critic is not None else [None, None]
        is_grasped = float(critic[27]) if critic is not None else None

        q_sim_rad = (state[0:6] + defaults).tolist()
        ee_xy = state[15:17].tolist()
        gripper_state_obs = float(state[12])           # observed gripper opening
        gripper_cmd = 1.0 if act_np[5] > 0.0 else 0.0  # binary command

        # latches off the env instance
        was_grasped_latch = bool(getattr(env.unwrapped, "_was_grasped", None)[0].item()) \
            if getattr(env.unwrapped, "_was_grasped", None) is not None else None
        was_over_bowl_latch = bool(getattr(env.unwrapped, "_was_over_bowl_above_rim", None)[0].item()) \
            if getattr(env.unwrapped, "_was_over_bowl_above_rim", None) is not None else None

        # --- step JSONL row ---
        row = {
            "ep": ep_idx,
            "t": t_in_ep,
            "q_sim_rad": [round(v, 4) for v in q_sim_rad],
            "joint_pos_rel": [round(float(v), 4) for v in state[0:6]],
            "joint_vel_rel": [round(float(v), 4) for v in state[6:12]],
            "ee_xy": [round(v, 4) for v in ee_xy],
            "ee_to_bowl_xy": [round(float(v), 4) for v in state[17:19]],
            "cube_xyz": [round(float(v), 4) for v in cube_xyz],
            "cube_to_bowl_xy": [round(float(v), 4) for v in cube_to_bowl],
            "is_grasped": is_grasped,
            "gripper_state_obs": round(gripper_state_obs, 4),
            "gripper_cmd": gripper_cmd,
            "action": [round(float(v), 4) for v in act_np],
            "was_grasped_latch": was_grasped_latch,
            "was_over_bowl_latch": was_over_bowl_latch,
            "reward": round(rew_np, 4),
            "done": done_np,
        }
        steps_f.write(json.dumps(row) + "\n")

        # --- update per-episode trackers ---
        _update_episode_state(ep_state, row, t_in_ep)

        t_in_ep += 1

        # --- episode boundary? ---
        if done_np or t_in_ep >= max_steps_safety:
            summary = _finalize_episode(ep_state, ep_idx, t_in_ep, row, bowl_xy_obs=state[13:15].tolist())
            eps_f.write(json.dumps(summary) + "\n")
            completed.append(summary)
            print(
                f"[diag] ep {ep_idx+1:3d}/{n_eps}  "
                f"n_steps={summary['n_steps']:3d}  "
                f"max_z={summary['max_cube_z']:.3f}  "
                f"steps>0.07={summary['steps_cube_above_07']:3d}  "
                f"steps>0.10={summary['steps_cube_above_10']:3d}  "
                f"continuous_held={summary['max_run_above_07']:3d}  "
                f"lift_latch={summary['lift_latch_first_step']!s:>4}  "
                f"bowl_latch={summary['over_bowl_latch_first_step']!s:>4}  "
                f"terminal_dist={summary['terminal_cube_to_bowl_dist']:.3f}  "
                f"by={summary['terminated_by']}  "
                f"succ={summary['succeeded']}"
            )
            ep_idx += 1
            t_in_ep = 0
            ep_state = _new_episode_state()

        obs = next_obs

    steps_f.close()
    eps_f.close()

    # --- aggregate summary ---
    n = len(completed)
    if n == 0:
        print("[diag] no episodes completed")
        return
    def col(k): return [e[k] for e in completed if e[k] is not None]
    def fmt(v): return f"{v:.3f}" if isinstance(v, float) else str(v)
    succ = sum(1 for e in completed if e["succeeded"])
    by_counts = {}
    for e in completed:
        by_counts[e["terminated_by"]] = by_counts.get(e["terminated_by"], 0) + 1
    max_z_arr = np.array(col("max_cube_z"), dtype=np.float32)
    steps07_arr = np.array(col("steps_cube_above_07"), dtype=np.float32)
    steps10_arr = np.array(col("steps_cube_above_10"), dtype=np.float32)
    held_arr = np.array(col("max_run_above_07"), dtype=np.float32)
    term_dist_arr = np.array(col("terminal_cube_to_bowl_dist"), dtype=np.float32)
    ep_len_arr = np.array(col("n_steps"), dtype=np.float32)
    grip_trans_arr = np.array(col("n_gripper_transitions"), dtype=np.float32)
    lift_latch_rate = sum(1 for e in completed if e["lift_latch_first_step"] is not None) / n
    bowl_latch_rate = sum(1 for e in completed if e["over_bowl_latch_first_step"] is not None) / n
    both_latches_rate = sum(1 for e in completed
                            if e["lift_latch_first_step"] is not None
                            and e["over_bowl_latch_first_step"] is not None) / n
    # "real lift" = held cube > 0.07 m for at least 10 consecutive steps
    real_lift_rate = sum(1 for e in completed if e["max_run_above_07"] >= 10) / n

    lines = [
        f"=== Aggregate diagnosis ({n} episodes) ===",
        f"  ckpt:                          {ckpt_path}",
        f"  task:                          {args_cli.task}",
        f"  success rate:                  {succ}/{n} = {succ/n:.3f}",
        f"  lift latch fire rate:          {lift_latch_rate:.3f}   (cube ever > 0.07 m)",
        f"  over-bowl-above-rim latch:     {bowl_latch_rate:.3f}",
        f"  both latches fired:            {both_latches_rate:.3f}",
        f"  REAL lift rate (held≥10 step): {real_lift_rate:.3f}   ← key diagnostic for 'actually lifting and holding'",
        f"  terminated_by distribution:    {by_counts}",
        f"",
        f"  Per-episode statistics (mean ± std, min..max):",
        f"    n_steps:                     {ep_len_arr.mean():.1f} ± {ep_len_arr.std():.1f}   ({ep_len_arr.min():.0f}..{ep_len_arr.max():.0f})",
        f"    max_cube_z (m):              {max_z_arr.mean():.4f} ± {max_z_arr.std():.4f}   ({max_z_arr.min():.4f}..{max_z_arr.max():.4f})",
        f"    steps cube > 0.07 m:         {steps07_arr.mean():.1f} ± {steps07_arr.std():.1f}",
        f"    steps cube > 0.10 m:         {steps10_arr.mean():.1f} ± {steps10_arr.std():.1f}",
        f"    max continuous run > 0.07:   {held_arr.mean():.1f} ± {held_arr.std():.1f}   ← if this is < 5, policy is 'spiking' not 'holding'",
        f"    terminal cube-to-bowl dist:  {term_dist_arr.mean():.4f} ± {term_dist_arr.std():.4f}",
        f"    gripper transitions/ep:      {grip_trans_arr.mean():.1f} ± {grip_trans_arr.std():.1f}",
        f"",
        f"  Interpretation hints:",
        f"    real_lift_rate <<  success_rate  →  policy is gaming the latches (success without sustained lift)",
        f"    real_lift_rate ≈   success_rate  →  policy is genuinely lifting; failures are about precision",
        f"    max continuous run > 0.07 < 10 across most eps → spike-lifts, not holds",
    ]
    summary_txt = "\n".join(lines)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(summary_txt + "\n")
    print()
    print(summary_txt)
    print()
    print(f"[diag] full per-step trace: {os.path.join(out, 'steps.jsonl')}")
    print(f"[diag] per-episode rows:   {os.path.join(out, 'episodes.jsonl')}")


def _new_episode_state():
    return {
        "max_cube_z": -1e9,
        "steps_cube_above_07": 0,
        "steps_cube_above_10": 0,
        "cur_run_above_07": 0,
        "max_run_above_07": 0,
        "lift_latch_first_step": None,
        "over_bowl_latch_first_step": None,
        "gripper_open_count": 0,
        "gripper_close_count": 0,
        "prev_gripper_cmd": None,
        "n_gripper_transitions": 0,
        "prev_action": None,
        "action_rate_sum": 0.0,
        "action_rate_max": 0.0,
        "ee_xmin": 1e9, "ee_xmax": -1e9, "ee_ymin": 1e9, "ee_ymax": -1e9,
    }


def _update_episode_state(s, row, t):
    cz = row["cube_xyz"][2]
    if cz is not None:
        if cz > s["max_cube_z"]:
            s["max_cube_z"] = cz
        if cz > 0.07:
            s["steps_cube_above_07"] += 1
            s["cur_run_above_07"] += 1
            if s["cur_run_above_07"] > s["max_run_above_07"]:
                s["max_run_above_07"] = s["cur_run_above_07"]
        else:
            s["cur_run_above_07"] = 0
        if cz > 0.10:
            s["steps_cube_above_10"] += 1
    if row["was_grasped_latch"] and s["lift_latch_first_step"] is None:
        s["lift_latch_first_step"] = t
    if row["was_over_bowl_latch"] and s["over_bowl_latch_first_step"] is None:
        s["over_bowl_latch_first_step"] = t
    if row["gripper_cmd"] > 0.5:
        s["gripper_open_count"] += 1
    else:
        s["gripper_close_count"] += 1
    if s["prev_gripper_cmd"] is not None and row["gripper_cmd"] != s["prev_gripper_cmd"]:
        s["n_gripper_transitions"] += 1
    s["prev_gripper_cmd"] = row["gripper_cmd"]
    if s["prev_action"] is not None:
        import numpy as np
        d = np.array(row["action"]) - np.array(s["prev_action"])
        rate = float(np.linalg.norm(d))
        s["action_rate_sum"] += rate
        if rate > s["action_rate_max"]:
            s["action_rate_max"] = rate
    s["prev_action"] = row["action"]
    ex, ey = row["ee_xy"][0], row["ee_xy"][1]
    if ex < s["ee_xmin"]: s["ee_xmin"] = ex
    if ex > s["ee_xmax"]: s["ee_xmax"] = ex
    if ey < s["ee_ymin"]: s["ee_ymin"] = ey
    if ey > s["ee_ymax"]: s["ee_ymax"] = ey


def _finalize_episode(s, ep_idx, n_steps, last_row, bowl_xy_obs):
    import numpy as np
    cx, cy = last_row["cube_xyz"][0], last_row["cube_xyz"][1]
    cz = last_row["cube_xyz"][2]
    bx, by = float(bowl_xy_obs[0]), float(bowl_xy_obs[1])
    if cx is not None:
        terminal_dist = float(np.hypot(cx - bx, cy - by))
    else:
        terminal_dist = None

    succeeded = bool(s["lift_latch_first_step"] is not None
                     and s["over_bowl_latch_first_step"] is not None
                     and terminal_dist is not None and terminal_dist < 0.06
                     and cz is not None and cz < 0.06
                     and last_row["gripper_cmd"] > 0.5)

    # Reasoning for terminated_by:
    # - If cube z is far below table top (< -0.05) → fell off table
    # - If n_steps reached the env episode_length (250) → time_out
    # - Otherwise success-like (early term) — env doesn't actually emit success-term, so rare
    if cz is not None and cz < -0.05:
        terminated_by = "block_off_table"
    elif n_steps >= 250:
        terminated_by = "time_out"
    else:
        terminated_by = "other"

    return {
        "ep": ep_idx,
        "n_steps": n_steps,
        "terminated_by": terminated_by,
        "succeeded": succeeded,
        "max_cube_z": round(s["max_cube_z"], 4),
        "steps_cube_above_07": s["steps_cube_above_07"],
        "steps_cube_above_10": s["steps_cube_above_10"],
        "max_run_above_07": s["max_run_above_07"],
        "lift_latch_first_step": s["lift_latch_first_step"],
        "over_bowl_latch_first_step": s["over_bowl_latch_first_step"],
        "gripper_open_count": s["gripper_open_count"],
        "gripper_close_count": s["gripper_close_count"],
        "n_gripper_transitions": s["n_gripper_transitions"],
        "mean_action_rate": round(s["action_rate_sum"] / max(1, n_steps), 4),
        "max_action_rate": round(s["action_rate_max"], 4),
        "ee_x_range": [round(s["ee_xmin"], 4), round(s["ee_xmax"], 4)],
        "ee_y_range": [round(s["ee_ymin"], 4), round(s["ee_ymax"], 4)],
        "terminal_cube_xyz": [round(v, 4) if v is not None else None for v in last_row["cube_xyz"]],
        "terminal_cube_to_bowl_dist": round(terminal_dist, 4) if terminal_dist is not None else None,
        "terminal_gripper_cmd": last_row["gripper_cmd"],
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return 1
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0

    # ------------------------------------------------------------------ debug dump
    # Sim-side counterpart to deploy/deploy_real.py --debug-dump. Same file layout
    # (step_XXXX.png composite + log.jsonl + meta.json) so the two folders can
    # be diffed image-by-image / row-by-row to characterise the sim↔real gap.
    if args_cli.debug_dump:
        _run_debug_dump(env, policy, args_cli, resume_path, log_dir, dt)
        env.close()
        return 0

    # ------------------------------------------------------------------ diag
    # Per-episode behaviour audit. Runs N episodes, dumps per-step + per-episode
    # rows and an aggregate summary that distinguishes "real lift-and-hold" from
    # "spike latches then drop". See _run_diag.
    if args_cli.diag_rollouts > 0:
        _run_diag(env, policy, args_cli, resume_path, log_dir, dt)
        env.close()
        return 0

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    return 0


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()