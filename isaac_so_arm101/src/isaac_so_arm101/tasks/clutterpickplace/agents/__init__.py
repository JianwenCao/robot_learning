# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configs for the Eval-2 clutter pick-and-place task.

Three runner cfgs, one per pipeline stage (see ``docs/EVAL2_PLAN.md`` §7):

* :mod:`teacher_ppo_cfg`   — Stage 1 state teacher (no CNN, no FiLM).
* :mod:`distill_cfg`       — Stage 2 vision distillation (DAgger).
* :mod:`rsl_rl_ppo_cfg`    — Stage 3 vision PPO with FiLM + teacher critic.
"""
