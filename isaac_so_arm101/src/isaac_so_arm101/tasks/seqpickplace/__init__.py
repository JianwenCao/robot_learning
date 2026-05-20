# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for SO-ARM101 Eval-3 (sequential pick-and-place)."""

import gymnasium as gym

from . import agents

# Vision env (Stage-3 target). Stage-3 vision PPO cfg isn't written yet —
# the placeholder rsl_rl entry point lets the rsl_rl train script load
# this env without erroring. Swap to the Stage-3 cfg when it lands.
gym.register(
    id="Isaac-SO-ARM101-SeqPickPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101SeqPickPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.teacher_ppo_cfg:SeqPickPlaceTeacherPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-SeqPickPlace-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101SeqPickPlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.teacher_ppo_cfg:SeqPickPlaceTeacherPPORunnerCfg",
    },
    disable_env_checker=True,
)

# State-only teacher sharing the vision env cfg — wrist camera renders
# each tick but its output is unused (wrist_image not in obs_groups).
# Use only for diagnostic comparison; requires --enable_cameras.
gym.register(
    id="Isaac-SO-ARM101-SeqPickPlace-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101SeqPickPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.teacher_ppo_cfg:SeqPickPlaceTeacherPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-SeqPickPlace-Teacher-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101SeqPickPlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.teacher_ppo_cfg:SeqPickPlaceTeacherPPORunnerCfg",
    },
    disable_env_checker=True,
)

# Camera-free state-only teacher variant — wrist_cam and wrist_image both
# nulled. Skips RTX render entirely (~2-3× faster than Teacher-v0). No
# --enable_cameras flag required.
gym.register(
    id="Isaac-SO-ARM101-SeqPickPlace-Teacher-Fast-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101SeqPickPlaceTeacherFastEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.teacher_ppo_cfg:SeqPickPlaceTeacherPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-SeqPickPlace-Teacher-Fast-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101SeqPickPlaceTeacherFastEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.teacher_ppo_cfg:SeqPickPlaceTeacherPPORunnerCfg",
    },
    disable_env_checker=True,
)

# State-only + AprilTag deploy path. Camera-free env, PolicyCfg extended
# with per-cube noisy xy + visibility flags. The seq_goal vector encodes
# the current sub-goal target color + bowl xy + step idx — the deploy
# loop advances the step counter between sub-goals; the post-grasp
# freeze re-keys to the new target on each transition.
gym.register(
    id="Isaac-SO-ARM101-SeqPickPlace-StateAprilTag-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101SeqPickPlaceStateAprilTagEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.state_apriltag_ppo_cfg:SeqPickPlaceStateAprilTagPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-SeqPickPlace-StateAprilTag-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101SeqPickPlaceStateAprilTagEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.state_apriltag_ppo_cfg:SeqPickPlaceStateAprilTagPPORunnerCfg",
    },
    disable_env_checker=True,
)
