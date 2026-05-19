# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for SO-ARM101 Bonus-B (singulation)."""

import gymnasium as gym

gym.register(
    id="Isaac-SO-ARM101-Singulation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101SingulationEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Singulation-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101SingulationEnvCfg_PLAY",
    },
    disable_env_checker=True,
)

# Camera-free state-only teacher variant.
gym.register(
    id="Isaac-SO-ARM101-Singulation-Teacher-Fast-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101SingulationTeacherFastEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-Singulation-Teacher-Fast-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101SingulationTeacherFastEnvCfg_PLAY",
    },
    disable_env_checker=True,
)
