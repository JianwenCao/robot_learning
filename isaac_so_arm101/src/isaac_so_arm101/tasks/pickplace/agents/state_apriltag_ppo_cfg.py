# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO config for the SO-ARM101 state-only + AprilTag deploy path.

Single-stage from-scratch PPO on the camera-free env
:class:`SoArm101PickPlaceBowlStateAprilTagEnvCfg`. The actor sees the
deployable ``policy`` group **plus** the new ``cube_pos_xy_noisy`` obs
term — i.e. all proprio + a noisy 2-D cube position that mirrors what
the AprilTag pipeline produces at deploy. The critic still sees the full
privileged state. See ``docs/STATE_APRILTAG_PLAN.md`` for the rationale.

Key differences vs :mod:`teacher_ppo_cfg`:

* Asymmetric A-C: ``obs_groups = {"policy": ["policy"], "critic": ["policy", "critic"]}``.
  The teacher path is *symmetric* on privileged state because the teacher
  acts as a distillation oracle. Here we want the actor distribution to
  match what the real-arm deploy will feed it, so we exclude the privileged
  critic obs from the actor.
* ``experiment_name = "pickplace_bowl_state_apriltag"`` so logs land in a
  separate folder from the teacher's.

Reuses :class:`PickPlaceVisionActorCritic` — that class auto-disables the
CNN when ``wrist_image`` isn't in ``obs_groups``, behaving as a stock
symmetric MLP A-C otherwise. The teacher cfg uses the same trick.
"""

import rsl_rl.runners.on_policy_runner as _on_policy_runner

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from .vision_actor_critic import PickPlaceVisionActorCritic


def _register_class() -> None:
    setattr(
        _on_policy_runner,
        PickPlaceVisionActorCritic.__name__,
        PickPlaceVisionActorCritic,
    )


_register_class()


@configclass
class PickPlaceBowlStateAprilTagPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for state-only + AprilTag-noise PPO.

    Hyperparameters mirror :class:`PickPlaceBowlTeacherPPORunnerCfg` —
    same network width, same algorithm settings — because the MDP is
    identical except for the extra 2-D obs and the asymmetric A-C split.
    """

    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 50
    experiment_name = "pickplace_bowl_state_apriltag"
    empirical_normalization = False

    # Asymmetric A-C: actor sees only the deployable obs (proprio + noisy
    # cube_pos_xy), critic sees policy + privileged ground-truth.
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
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
