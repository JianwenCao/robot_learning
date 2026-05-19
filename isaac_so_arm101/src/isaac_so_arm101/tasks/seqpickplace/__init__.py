# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for SO-ARM101 Eval-3 (sequential pick-and-place)."""

import gymnasium as gym

gym.register(
    id="Isaac-SO-ARM101-SeqPickPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101SeqPickPlaceEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-SeqPickPlace-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101SeqPickPlaceEnvCfg_PLAY",
    },
    disable_env_checker=True,
)

# Camera-free state-only teacher variant.
gym.register(
    id="Isaac-SO-ARM101-SeqPickPlace-Teacher-Fast-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101SeqPickPlaceTeacherFastEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-SeqPickPlace-Teacher-Fast-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101SeqPickPlaceTeacherFastEnvCfg_PLAY",
    },
    disable_env_checker=True,
)
