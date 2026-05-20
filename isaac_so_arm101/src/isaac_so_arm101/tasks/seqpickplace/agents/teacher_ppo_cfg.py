# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO config for the SO-ARM101 Eval-3 sequential pick-and-place **teacher**.

State-only teacher for the §9 three-stage pipeline (see EVAL3_PLAN.md).
The actor reads ``policy + critic`` directly so binding C (visual color →
one-hot bit) is bypassed — the teacher gets ``current_target_block_position``,
all 4 active cube positions, and the 11-D ``seq_goal_vector`` as state input.

We reuse :class:`PickPlaceVisionActorCritic` from Eval-1; that class auto-
disables both ``actor_cnn`` and ``critic_cnn`` when ``wrist_image`` is
absent from ``obs_groups``, so it behaves as a plain symmetric MLP A-C
in the teacher regime. No new actor-critic subclass needed for Stage 1.

Hyperparameters track EVAL3_PLAN.md §8: γ=0.98 (15-s episode × 50 Hz
needs a long horizon for the step-2 release to back-propagate through
TD(λ)), 3000 max_iterations (vs Eval-1's 1500 — 3× longer task), entropy
0.006 (Franka Lift stock; entropy ramp deferred to a curriculum hook
once we see Stage-1 metrics).
"""

import rsl_rl.runners.on_policy_runner as _on_policy_runner

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from isaac_so_arm101.tasks.pickplace.agents.vision_actor_critic import (
    PickPlaceVisionActorCritic,
)


def _register_class() -> None:
    setattr(
        _on_policy_runner,
        PickPlaceVisionActorCritic.__name__,
        PickPlaceVisionActorCritic,
    )


_register_class()


@configclass
class SeqPickPlaceTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """State-only PPO for the Eval-3 sequential teacher."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "seqpickplace_teacher"
    empirical_normalization = False

    obs_groups = {
        "policy": ["policy", "critic"],
        "critic": ["policy", "critic"],
    }

    policy = RslRlPpoActorCriticCfg(
        class_name="PickPlaceVisionActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
