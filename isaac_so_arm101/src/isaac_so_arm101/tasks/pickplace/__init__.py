# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the SO-ARM101 single-block pick-and-place task."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-SO-ARM101-PickPlace-Bowl-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceBowlEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceBowlPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-PickPlace-Bowl-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceBowlEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceBowlPPORunnerCfg",
    },
    disable_env_checker=True,
)

# Teacher variant — same env cfg (scene/rewards/DR identical to the vision
# task), different RSL-RL cfg (state-only obs_groups, no CNN). Trained
# first; its rollouts label the vision student via DAgger / BC distillation
# in the §7 fallback recipe.
gym.register(
    id="Isaac-SO-ARM101-PickPlace-Bowl-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceBowlEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.teacher_ppo_cfg:PickPlaceBowlTeacherPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-PickPlace-Bowl-Teacher-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceBowlEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.teacher_ppo_cfg:PickPlaceBowlTeacherPPORunnerCfg",
    },
    disable_env_checker=True,
)

# Vision student — DAgger distillation from the state teacher (model_*.pt
# in pickplace_bowl_teacher). Same env cfg as the teacher (we need both
# state and image obs in the obs dict — the env already produces all of
# policy / critic / wrist_image groups). The runner cfg points at
# DistillationRunner which auto-loads the teacher checkpoint via
# StudentTeacher.load_state_dict (PPO ``actor.*`` → teacher MLP).
gym.register(
    id="Isaac-SO-ARM101-PickPlace-Bowl-Student-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceBowlEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.distill_cfg:PickPlaceBowlDistillRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-PickPlace-Bowl-Student-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceBowlEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.distill_cfg:PickPlaceBowlDistillRunnerCfg",
    },
    disable_env_checker=True,
)

# EVAL1_PLAN §9 alternative path — pretrained ResNet-18 backbone, cold-start
# PPO (no teacher, no distillation). Separate env cfg (only the curriculum
# schedule differs from the §7 path) and separate agent cfg (new actor-critic
# class). Distinct task ID so log dirs don't collide with the §7 production
# pipeline.
gym.register(
    id="Isaac-SO-ARM101-PickPlace-Bowl-Pretrained-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pretrained_env_cfg:SoArm101PickPlaceBowlPretrainedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.pretrained_ppo_cfg:PickPlaceBowlPretrainedPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-PickPlace-Bowl-Pretrained-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pretrained_env_cfg:SoArm101PickPlaceBowlPretrainedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.pretrained_ppo_cfg:PickPlaceBowlPretrainedPPORunnerCfg",
    },
    disable_env_checker=True,
)
