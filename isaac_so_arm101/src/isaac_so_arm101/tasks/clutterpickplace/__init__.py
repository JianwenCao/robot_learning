# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the SO-ARM101 Eval-2 (clutter pick-and-place) task."""

import gymnasium as gym

gym.register(
    id="Isaac-SO-ARM101-ClutterPickPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101ClutterPickPlaceEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-ClutterPickPlace-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101ClutterPickPlaceEnvCfg_PLAY",
    },
    disable_env_checker=True,
)

# Camera-free state-only teacher variant — much faster wall-clock since
# the wrist TiledCamera spawn and the wrist_image obs group are both
# nulled. Pair with a teacher_ppo_cfg.py whose obs_groups concatenates
# policy + critic so the actor sees the privileged cube state directly.
gym.register(
    id="Isaac-SO-ARM101-ClutterPickPlace-Teacher-Fast-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101ClutterPickPlaceTeacherFastEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-ClutterPickPlace-Teacher-Fast-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101ClutterPickPlaceTeacherFastEnvCfg_PLAY",
    },
    disable_env_checker=True,
)
