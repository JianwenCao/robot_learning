# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO config for the SO-ARM101 Eval-3 state-only + AprilTag path.

Single-stage from-scratch PPO on
:class:`SoArm101SeqPickPlaceStateAprilTagEnvCfg`. The actor sees only the
deployable ``policy`` group (proprio + seq_goal + per-cube noisy xy +
visibility flags). The critic additionally reads the privileged ``critic``
group (GT positions of all 4 active cubes + current-target xyz +
ee→target distance) for low-variance value estimation.

The asymmetric A-C is the same recipe used by Eval-1/Eval-2 state_apriltag
paths. Hyperparameters mirror :class:`SeqPickPlaceTeacherPPORunnerCfg`
(γ=0.98, 3000 iters, entropy 0.006) — same MDP, just a wider obs vector
(+12 cube positions + 6 visibility flags).
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
class SeqPickPlaceStateAprilTagPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for the Eval-3 state-only + AprilTag-noise path."""

    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 50
    experiment_name = "seqpickplace_state_apriltag"
    empirical_normalization = False

    # Asymmetric A-C: actor sees deployable policy obs only; critic gets
    # all privileged signals.
    obs_groups = {
        "policy": ["policy"],
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
        # 0.005 → 0.01 (2026-05-20): Eval-2 cfg documented 0.005 as the cause
        # of v3's stay-still stall — over-tight trust region throttles per-iter
        # learning before reach can lock in. Eval-3 inherits the same arch +
        # an even harder long-horizon MDP, so use the working 0.01 instead.
        # std_max=0.2 gripper cap below still prevents the binary-gripper σ
        # blow-up that originally motivated the tightening.
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        # Per-dim init noise + cap, same as Eval-2 state_apriltag: arm at
        # σ=1.0 for reach exploration, gripper σ=0.1 (capped at 0.2) so
        # binary closure stays decisive across sub-goals.
        self.policy.init_noise_std = [1.0, 1.0, 1.0, 1.0, 1.0, 0.1]
        self.policy.std_max = [1e3, 1e3, 1e3, 1e3, 1e3, 0.2]
