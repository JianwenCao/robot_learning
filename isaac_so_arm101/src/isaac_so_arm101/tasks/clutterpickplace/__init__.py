# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the SO-ARM101 Eval-2 (clutter pick-and-place) task."""

import gymnasium as gym

from . import agents

# ---------------------------------------------------------------------------
# Vision task (Stage 3 — vision PPO with FiLM-conditioned actor + teacher critic).
# ---------------------------------------------------------------------------

gym.register(
    id="Isaac-SO-ARM101-ClutterPickPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101ClutterPickPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ClutterPickPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-ClutterPickPlace-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101ClutterPickPlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ClutterPickPlacePPORunnerCfg",
    },
    disable_env_checker=True,
)

# Stage-2 distillation student. Same env cfg as the vision task — the
# env produces all of policy / critic / goal / wrist_image groups; the
# distill runner cfg selects which go to the student vs teacher.
gym.register(
    id="Isaac-SO-ARM101-ClutterPickPlace-Student-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101ClutterPickPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.distill_cfg:ClutterPickPlaceDistillRunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-ClutterPickPlace-Student-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101ClutterPickPlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.distill_cfg:ClutterPickPlaceDistillRunnerCfg",
    },
    disable_env_checker=True,
)

# ---------------------------------------------------------------------------
# Stage-1 teacher tasks — state-only PPO. Same scene + rewards + DR as the
# vision task; only obs_groups differ (wired in
# :class:`agents.teacher_ppo_cfg.ClutterPickPlaceTeacherPPORunnerCfg`).
# ---------------------------------------------------------------------------

# Camera-rendered teacher — wrist_cam is spawned and rendered each step,
# output discarded. Useful when the teacher's training distribution must
# match the student's visually. Requires ``--enable_cameras``.
gym.register(
    id="Isaac-SO-ARM101-ClutterPickPlace-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101ClutterPickPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.teacher_ppo_cfg:ClutterPickPlaceTeacherPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-ClutterPickPlace-Teacher-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101ClutterPickPlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.teacher_ppo_cfg:ClutterPickPlaceTeacherPPORunnerCfg",
    },
    disable_env_checker=True,
)

# Camera-free teacher (recommended default for Stage 1) — wrist_cam +
# wrist_image obs group both nulled. ~2-3× faster wall-clock per iter.
# No ``--enable_cameras`` needed.
gym.register(
    id="Isaac-SO-ARM101-ClutterPickPlace-Teacher-Fast-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101ClutterPickPlaceTeacherFastEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.teacher_ppo_cfg:ClutterPickPlaceTeacherPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-ClutterPickPlace-Teacher-Fast-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101ClutterPickPlaceTeacherFastEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.teacher_ppo_cfg:ClutterPickPlaceTeacherPPORunnerCfg",
    },
    disable_env_checker=True,
)

# State-only + AprilTag deploy path — camera-free env, PolicyCfg extended
# with cube_positions_xy_noisy + cube_visible_flags (sim-side mirror of
# pupil-apriltags pose injection on the real arm). See
# ``docs/STATE_APRILTAG_PLAN.md``. Single-stage PPO (no distillation, no
# vision warm-start). No ``--enable_cameras`` needed.
gym.register(
    id="Isaac-SO-ARM101-ClutterPickPlace-StateAprilTag-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101ClutterPickPlaceStateAprilTagEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.state_apriltag_ppo_cfg:ClutterPickPlaceStateAprilTagPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-ClutterPickPlace-StateAprilTag-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101ClutterPickPlaceStateAprilTagEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.state_apriltag_ppo_cfg:ClutterPickPlaceStateAprilTagPPORunnerCfg",
    },
    disable_env_checker=True,
)
