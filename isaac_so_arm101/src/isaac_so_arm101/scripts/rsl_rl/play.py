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
# Sim-side debug dump — mirrors bc/deploy_real.py --debug-dump output layout so
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


# Sim-side mirror of bc.deploy_real constants. Must match exactly — the action
# decoding here only exists to log ``target_sim_rad`` for diffing; the env's
# action manager still does the real decode internally.
_JOINT_DEFAULTS_RAD = [0.0, 0.0, 0.0, 1.57, 0.0, 0.5]
_ARM_ACTION_SCALE = 0.5
_GRIPPER_OPEN_RAD = 0.5
_GRIPPER_CLOSE_RAD = 0.0


def _force_bowl_xy(env, xy):
    """Overwrite the active bowl_pose command for env 0 (matches bc/deploy_sim.py)."""
    cmd_mgr = env.unwrapped.command_manager
    if "bowl_pose" not in cmd_mgr._terms:
        raise RuntimeError("command term 'bowl_pose' not found")
    buf = cmd_mgr.get_term("bowl_pose").command
    buf[0, 0] = float(xy[0])
    buf[0, 1] = float(xy[1])
    buf[0, 2] = 0.0


def _decode_target(action6):
    """Same arm/gripper decode as bc.deploy_real._decode_action — for log parity."""
    import numpy as np
    arm = [_JOINT_DEFAULTS_RAD[i] + _ARM_ACTION_SCALE * float(action6[i]) for i in range(5)]
    grip = _GRIPPER_OPEN_RAD if float(action6[5]) > 0.0 else _GRIPPER_CLOSE_RAD
    return np.array(arm + [grip], dtype=np.float32)


def _run_debug_dump(env, policy, args_cli, ckpt_path, log_dir, dt):
    """Run a fixed-length episode, dumping wrist image + state/action per step.

    Layout matches ``bc/deploy_real.py`` ``--debug-dump``:
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
    # Sim-side counterpart to bc/deploy_real.py --debug-dump. Same file layout
    # (step_XXXX.png composite + log.jsonl + meta.json) so the two folders can
    # be diffed image-by-image / row-by-row to characterise the sim↔real gap.
    if args_cli.debug_dump:
        _run_debug_dump(env, policy, args_cli, resume_path, log_dir, dt)
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