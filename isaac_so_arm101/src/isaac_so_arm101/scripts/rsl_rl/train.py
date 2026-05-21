# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

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
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--actor-only",
    action="store_true",
    default=False,
    help="Warm-start only actor.* and std from --checkpoint, leaving critic/optimizer fresh. "
         "Use for cross-task transfer when policy obs matches but critic obs differs.",
)
# Stage-3 warm-start critic carry-over (EVAL1_PLAN §7.2 intervention #5).
# When set, AFTER the distill checkpoint warm-starts the actor via
# runner.load(--load_run), we overlay the Stage-1 teacher's ``critic.*`` keys
# onto the policy. Teacher critic shape (policy+critic → [256,128,64] → 1)
# matches Stage-3 critic shape layer-for-layer, so this is a clean
# nn.Module.load_state_dict(filtered, strict=False) call. Without it, the
# fresh-random critic destroys the warm-started actor within ~50 iters via
# noisy advantage estimates.
parser.add_argument(
    "--teacher_ckpt",
    type=str,
    default=None,
    help="Optional path to Stage-1 teacher PPO checkpoint (model_*.pt). When set, "
         "overlays the teacher's critic.* keys onto the policy after the distill "
         "warm-start has loaded the actor. Stage-3 vision PPO only.",
)
parser.add_argument(
    "--distill_rollout_policy",
    type=str,
    default="student",
    choices=("student", "teacher"),
    help="Distillation only: use student rollouts for DAgger-style training "
         "or teacher rollouts for simple behavior cloning.",
)
parser.add_argument(
    "--dump_cnn_inputs",
    action="store_true",
    default=False,
    help="Distillation only: periodically save the RGB tensors fed into the vision CNN.",
)
parser.add_argument(
    "--cnn_dump_interval",
    type=int,
    default=200,
    help="Save CNN input images every N env steps when --dump_cnn_inputs is set.",
)
parser.add_argument(
    "--cnn_dump_max_envs",
    type=int,
    default=4,
    help="Number of env images to save at each --dump_cnn_inputs step.",
)
parser.add_argument(
    "--distill_then_ppo",
    action="store_true",
    default=False,
    help="Distillation only: after teacher/student distillation, warm-start a pure PPO runner "
         "from the distilled student and continue PPO training in the same environment.",
)
parser.add_argument(
    "--distill_iterations",
    type=int,
    default=500,
    help="Number of distillation iterations to run when --distill_then_ppo is set. "
         "--max_iterations overrides this value for the distillation phase.",
)
parser.add_argument(
    "--ppo_iterations",
    type=int,
    default=1000,
    help="Number of pure PPO iterations to run after distillation when --distill_then_ppo is set.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

import omni
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaac_so_arm101.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _make_pickplace_ppo_cfg(agent_cfg: RslRlBaseRunnerCfg) -> RslRlBaseRunnerCfg:
    """Build the vision PPO cfg used after student distillation."""
    from isaac_so_arm101.tasks.pickplace.agents.rsl_rl_ppo_cfg import PickPlaceBowlPPORunnerCfg

    ppo_cfg = PickPlaceBowlPPORunnerCfg()
    ppo_cfg.device = agent_cfg.device
    ppo_cfg.seed = agent_cfg.seed
    ppo_cfg.logger = agent_cfg.logger
    ppo_cfg.run_name = agent_cfg.run_name
    ppo_cfg.resume = False
    ppo_cfg.load_run = ""
    ppo_cfg.load_checkpoint = "model_.*.pt"
    ppo_cfg.max_iterations = int(args_cli.ppo_iterations)
    return ppo_cfg


def _overlay_teacher_critic_if_requested(runner: OnPolicyRunner, agent_cfg: RslRlBaseRunnerCfg) -> None:
    """Optionally overlay a teacher PPO critic onto a PPO runner."""
    if args_cli.teacher_ckpt is None:
        return
    if not os.path.isfile(args_cli.teacher_ckpt):
        raise FileNotFoundError(f"--teacher_ckpt path does not exist: {args_cli.teacher_ckpt}")
    print(f"[INFO]: Loading teacher critic from: {args_cli.teacher_ckpt}")
    teacher_data = torch.load(args_cli.teacher_ckpt, map_location=agent_cfg.device, weights_only=False)
    # RSL-RL saves under "model_state_dict"; fall back gracefully if the
    # checkpoint was written by a different convention.
    if isinstance(teacher_data, dict) and "model_state_dict" in teacher_data:
        teacher_sd = teacher_data["model_state_dict"]
    else:
        teacher_sd = teacher_data
    critic_sd = {k: v for k, v in teacher_sd.items() if k.startswith("critic.")}
    if not critic_sd:
        raise RuntimeError(
            f"--teacher_ckpt {args_cli.teacher_ckpt} contained no 'critic.*' keys; "
            f"got top-level keys {list(teacher_sd.keys())[:8]}..."
        )
    # strict=False because we only ship critic.* keys; everything else
    # (actor.*, actor_cnn.*, std, normalizers) is expected to be "missing"
    # from this filtered overlay and stay at its post-distill-warm-start value.
    #
    # Bypass the custom ``PickPlaceVisionActorCritic.load_state_dict``
    # (and the ``rsl_rl.modules.ActorCritic.load_state_dict`` it wraps) by
    # calling ``nn.Module.load_state_dict`` directly. Both wrappers return
    # ``bool`` (distill-vs-resume signal for the runner) rather than the
    # standard ``(missing, unexpected)`` tuple, which would break the
    # unpack below. ``nn.Module.load_state_dict`` gives the proper tuple
    # and just copies tensors into matching parameter slots — exactly what
    # the critic overlay needs.
    policy = runner.alg.policy
    missing, unexpected = torch.nn.Module.load_state_dict(policy, critic_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"Teacher critic load: unexpected keys {unexpected}")
    print(
        f"[INFO]: Teacher critic loaded — overlaid {len(critic_sd)} critic.* keys "
        f"(Pinto asymmetric A-C handoff). Actor / CNN / std unchanged."
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if (
        args_cli.distill_rollout_policy is not None
        and hasattr(agent_cfg, "algorithm")
        and hasattr(agent_cfg.algorithm, "rollout_policy")
    ):
        agent_cfg.algorithm.rollout_policy = args_cli.distill_rollout_policy
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.distill_then_ppo and agent_cfg.class_name == "DistillationRunner":
        agent_cfg.max_iterations = (
            args_cli.max_iterations if args_cli.max_iterations is not None else int(args_cli.distill_iterations)
        )
    else:
        agent_cfg.max_iterations = (
            args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
        )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if args_cli.actor_only:
        if args_cli.checkpoint is None:
            raise ValueError("--actor-only requires --checkpoint")
        if os.path.isfile(args_cli.checkpoint):
            resume_path = os.path.abspath(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    elif agent_cfg.resume or agent_cfg.class_name == "DistillationRunner":
        if agent_cfg.class_name == "DistillationRunner" and args_cli.checkpoint and os.path.isfile(args_cli.checkpoint):
            resume_path = os.path.abspath(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    raw_env = env
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    if args_cli.dump_cnn_inputs:
        if not hasattr(runner.alg, "cnn_input_dump_dir"):
            raise RuntimeError("--dump_cnn_inputs is only supported by PickPlaceBCDistillation")
        runner.alg.cnn_input_dump_dir = os.path.join(log_dir, "cnn_inputs")
        runner.alg.cnn_input_dump_interval = int(args_cli.cnn_dump_interval)
        runner.alg.cnn_input_dump_max_envs = int(args_cli.cnn_dump_max_envs)
        runner.alg.cnn_input_dump_env = raw_env
        print(f"[INFO]: dumping CNN input images to {runner.alg.cnn_input_dump_dir}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if args_cli.actor_only:
        print(f"[INFO]: Actor-only warm-start from: {resume_path}")
        data = torch.load(resume_path, map_location=agent_cfg.device, weights_only=False)
        state_dict = data["model_state_dict"] if isinstance(data, dict) and "model_state_dict" in data else data
        actor_state_dict = {
            key: value for key, value in state_dict.items()
            if key.startswith("actor") or key == "std"
        }
        if not actor_state_dict:
            raise RuntimeError(f"--actor-only found no actor.* or std keys in checkpoint: {resume_path}")
        result = torch.nn.Module.load_state_dict(runner.alg.policy, actor_state_dict, strict=False)
        missing = getattr(result, "missing_keys", result[0])
        unexpected = getattr(result, "unexpected_keys", result[1])
        if unexpected:
            raise RuntimeError(f"--actor-only load had unexpected keys: {unexpected}")
        n_actor = sum(1 for key in actor_state_dict if key.startswith("actor"))
        n_critic_missing = sum(1 for key in missing if key.startswith("critic"))
        print(
            f"[INFO]: Actor-only warm-start loaded {len(actor_state_dict)} keys "
            f"({n_actor} actor.*, std={'std' in actor_state_dict}); "
            f"left {n_critic_missing} critic.* keys fresh."
        )
    elif agent_cfg.resume or agent_cfg.class_name == "DistillationRunner":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # Stage-3 critic warm-start (EVAL1_PLAN §7.2 intervention #5).
    # Apply AFTER runner.load so the distill warm-start of the actor has
    # already been applied — we overlay only critic.* keys without touching
    # the actor. Skips silently if --teacher_ckpt is not provided.
    if isinstance(runner, OnPolicyRunner):
        _overlay_teacher_critic_if_requested(runner, agent_cfg)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    if args_cli.distill_then_ppo:
        if agent_cfg.class_name != "DistillationRunner":
            raise RuntimeError("--distill_then_ppo requires a DistillationRunner task such as *-Student-v0")
        if args_cli.ppo_iterations <= 0:
            raise ValueError("--ppo_iterations must be positive when --distill_then_ppo is set")

        distill_iterations = int(agent_cfg.max_iterations)
        print(
            f"[INFO]: Hybrid training enabled: distill {distill_iterations} iterations, "
            f"then pure PPO {int(args_cli.ppo_iterations)} iterations."
        )
        runner.learn(num_learning_iterations=distill_iterations, init_at_random_ep_len=True)
        distill_ckpt_path = os.path.join(log_dir, f"model_distill_{distill_iterations}.pt")
        runner.save(distill_ckpt_path)
        print(f"[INFO]: Saved distillation checkpoint for PPO warm-start: {distill_ckpt_path}")

        ppo_agent_cfg = _make_pickplace_ppo_cfg(agent_cfg)
        ppo_agent_cfg.max_iterations = int(args_cli.ppo_iterations)
        ppo_runner = OnPolicyRunner(env, ppo_agent_cfg.to_dict(), log_dir=log_dir, device=ppo_agent_cfg.device)
        ppo_runner.add_git_repo_to_log(__file__)
        print(f"[INFO]: Warm-starting pure PPO actor from distilled student: {distill_ckpt_path}")
        ppo_runner.load(distill_ckpt_path, load_optimizer=False)
        ppo_runner.current_learning_iteration = distill_iterations
        _overlay_teacher_critic_if_requested(ppo_runner, ppo_agent_cfg)
        dump_yaml(os.path.join(log_dir, "params", "ppo_agent.yaml"), ppo_agent_cfg)
        ppo_runner.learn(num_learning_iterations=ppo_agent_cfg.max_iterations, init_at_random_ep_len=False)
    else:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
