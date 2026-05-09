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
