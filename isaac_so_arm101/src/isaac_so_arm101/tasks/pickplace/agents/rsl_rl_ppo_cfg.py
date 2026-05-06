# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO config for the pick-and-place task with asymmetric Actor-Critic.

The actor reads the deployable ``policy`` observation group only; the
critic additionally reads the privileged ``critic`` group. This is the
standard sim-to-real recipe — the critic accelerates training but is
discarded at deploy. See EVAL1_PLAN §3.8 / §3.9 for hyperparameter
rationale.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PickPlaceBowlPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Lift-task style: 24 steps × 4096 envs × 1500 iters ≈ 150 M steps.
    # State-only Day-3 target (EVAL1_PLAN §4.2 Step I) is similar; vision
    # training in Day 4 bumps max_iterations to ~5000.
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 100
    experiment_name = "pickplace_bowl"
    empirical_normalization = False

    # Asymmetric Actor-Critic: actor sees only the deployable group, critic
    # sees deployable + privileged. The keys here ("policy" / "critic") are
    # the ObsGroup names defined in :class:`ObservationsCfg`.
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "critic"],
    }

    # Hyperparameters mirror the lift task's PPO cfg exactly — that one
    # converges reliably for SO-ARM100/101 manipulation, and EVAL1_PLAN
    # §3.8 prescribes the same starting point for state-only training.
    policy = RslRlPpoActorCriticCfg(
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
