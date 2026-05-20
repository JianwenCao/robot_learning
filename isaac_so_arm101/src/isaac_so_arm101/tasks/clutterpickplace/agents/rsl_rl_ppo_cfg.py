# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO config for the SO-ARM101 Eval-2 vision task (Stage 3).

Asymmetric A-C:

* Actor reads ``policy + goal + wrist_image`` — FiLM-conditioned
  ResNet-18 backbone with the target-color one-hot from ``goal``.
* Critic reads ``policy + critic + goal`` (state-only; no image).

The critic is initialized from the Stage 1 teacher's critic via the
``--teacher_ckpt`` flag (``scripts/rsl_rl/train.py``) — see EVAL2_PLAN
§7.1 intervention #5. The actor is warm-started from the Stage 2
distill student via ``--load_run / --checkpoint`` and the
:meth:`PickPlaceVisionActorCritic.load_state_dict` distill-branch.
"""

import rsl_rl.runners.on_policy_runner as _on_policy_runner

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from .vision_actor_critic import ClutterPickPlaceVisionActorCritic


def _register_class() -> None:
    """Inject :class:`ClutterPickPlaceVisionActorCritic` into RSL-RL's runner scope.

    RSL-RL resolves ``policy.class_name`` strings via :func:`eval` against
    :mod:`rsl_rl.runners.on_policy_runner`'s globals.
    """
    setattr(
        _on_policy_runner,
        ClutterPickPlaceVisionActorCritic.__name__,
        ClutterPickPlaceVisionActorCritic,
    )


_register_class()


@configclass
class ClutterPickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Stage 3 vision PPO with FiLM-conditioned ResNet actor + teacher critic."""

    # 1024 envs × 16 steps = 16,384 transitions/iter. 2500 iters → ~41 M
    # env-steps — color discrimination needs more samples than Eval-1's
    # 2000-iter budget.
    num_steps_per_env = 16
    max_iterations = 2500
    save_interval = 100
    experiment_name = "clutterpickplace"
    empirical_normalization = False

    # Asymmetric A-C. Actor gets image + goal (FiLM cond); critic gets
    # privileged state + goal. ``ClutterPickPlaceVisionActorCritic._encode_*``
    # routes ``goal`` to the CNN's FiLM head automatically when present.
    obs_groups = {
        "policy": ["policy", "goal", "wrist_image"],
        "critic": ["policy", "critic", "goal"],
    }

    policy = RslRlPpoActorCriticCfg(
        class_name="ClutterPickPlaceVisionActorCritic",
        # Stage-3 ``init_noise_std`` is FORCED in load_state_dict's distill
        # branch — the saved distill ``std=0.1`` is dropped and this value
        # used instead (EVAL2_PLAN §7.1 intervention #2). 0.5 keeps binary-
        # gripper exploration alive in the first ~500 iters.
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Higher entropy than late-Eval-1 (which dropped to 0.003 once
        # stage-3 was beyond the imitation basin). For Eval-2 we keep
        # 0.006 longer because target-color exploration matters longer
        # than Eval-1's geometric grasp exploration.
        entropy_coef=0.006,
        num_learning_epochs=8,
        num_mini_batches=16,
        learning_rate=1.0e-4,
        schedule="adaptive",
        # γ=0.98 matches the Stage 1 teacher — needed because the
        # release-in-bowl=30 reward pays every post-release step and
        # γ=0.9 would chop ~80 % of that tail (EVAL2_PLAN §7.1 #3).
        gamma=0.98,
        lam=0.95,
        # Tight KL band — without it adaptive LR + entropy bonus can let
        # σ blow up (Eval-1 observed σ=2.12 by iter 2100 at desired_kl=0.01).
        desired_kl=0.005,
        max_grad_norm=1.0,
    )

    # CNN config passed through to ClutterPickPlaceVisionActorCritic →
    # PickPlaceVisionActorCritic → _ResNetSpatialSoftmaxCNN.
    # Stored in policy kwargs by appending below.

    def __post_init__(self):
        # RslRlPpoActorCriticCfg is a frozen dataclass-like — we pass
        # extra kwargs through ``to_dict`` indirectly. The cleanest way
        # for runner consumption: monkeypatch into ``policy.__dict__``
        # after construction so RSL-RL's ``**policy_cfg_kwargs`` unpacks
        # them into ``ClutterPickPlaceVisionActorCritic.__init__``.
        # Configclass behaves like a regular dataclass — direct
        # attribute set works.
        self.policy.cnn_class = "resnet"
        self.policy.cnn_kwargs = {
            "truncate_at": "layer2",   # 9×16 spatial map at 72×128 (cube ~1 cell)
            "film_cond_dim": 6,        # NUM_COLORS one-hot
        }
