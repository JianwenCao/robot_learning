# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO config for the §9 pretrained-backbone cold-start path.

EVAL1_PLAN §9 alternative path. Parallel to :mod:`rsl_rl_ppo_cfg` (the §7
production path) — kept in a separate file so the verified Stage 1–3
pipeline's hyperparameters are not touched.

Hyperparameter rationale (matches EVAL1_PLAN §9.4):

* ``init_noise_std=1.0``  — cold-start needs wide exploration; nothing to
  preserve from an imitation basin (no Stage-2 warm-start).
* ``gamma=0.98``          — match the §4 reward shape (long-horizon
  ``release_in_bowl`` latch). Changing γ at the same time as the encoder
  would conflate the experiment with a recipe-transfer test.
* ``desired_kl=0.01``     — looser than Stage 3's 0.005; we have no
  imitation basin to drift out of, and the encoder's gradient is naturally
  smaller than the head's so KL moves slower.
* ``max_iterations=4000`` — ~2× the warm-started Stage 3 budget; cold-start
  needs longer.

Class registration mirrors :mod:`rsl_rl_ppo_cfg`: importing this module
injects :class:`PickPlaceResNetActorCritic` into the RSL-RL runner's
namespace so the ``class_name`` string in the policy cfg resolves.
"""

import rsl_rl.runners.on_policy_runner as _on_policy_runner

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

from .pretrained_resnet_actor_critic import PickPlaceResNetActorCritic


def _register_class() -> None:
    """Inject :class:`PickPlaceResNetActorCritic` into RSL-RL's runner scope.

    Same mechanism as :func:`agents.rsl_rl_ppo_cfg._register_class`: RSL-RL
    resolves ``policy.class_name`` via :func:`eval` against the runner
    module's globals.
    """
    setattr(
        _on_policy_runner,
        PickPlaceResNetActorCritic.__name__,
        PickPlaceResNetActorCritic,
    )


_register_class()


@configclass
class PickPlaceBowlPretrainedPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config — pretrained ResNet-18 actor (frozen trunk), cold-start, no teacher.

    Diverges from :class:`PickPlaceBowlPPORunnerCfg` in three ways
    (everything else mirrored):

    1. ``policy.class_name`` → ``PickPlaceResNetActorCritic`` (with
       ``freeze_backbone=True`` as the actor-critic default).
    2. Cold-start exploration / iteration budget (``init_noise_std=1.0``,
       ``max_iterations=4000``, ``desired_kl=0.01``).
    3. ``entropy_coef`` left at 0.006 throughout — no warm-start basin to
       escape, so we don't need the 0.006 → 0.003 decay the §7 path used.

    **Frozen backbone (supervisor recommendation, diverges from §9.4
    fine-tune spec).** The ResNet-18 trunk has ``requires_grad=False`` and
    runs under ``torch.no_grad()`` in the forward pass — see
    :class:`pretrained_resnet_actor_critic._PretrainedActorEncoder` for the
    full rationale. Only the depth/mask CNN, fused 1×1 + spatial-softmax
    head, actor MLP, value MLP, and σ parameter train. This obviates the
    decoupled-encoder-LR / 2-group-optimizer scheme §9.4 originally
    specified (no encoder grads to manage) — single LR ``1e-4`` for all
    trainable parameters is fine.

    Fallback if the frozen ImageNet trunk plateaus (likely if frozen
    features don't separate the 2 cm cube against the wood table well
    enough): swap the backbone to R3M or MVP, both manipulation-pretrained.
    Don't unfreeze ImageNet — the literature is consistent that
    manipulation-pretrained > fine-tuned ImageNet for sim-to-real RL
    encoders.
    """

    # ~130 M env-steps at 1024 envs × 16 steps × 4000 iters — roughly 2× the
    # warm-started Stage 3 budget. Cold-start needs more time before
    # reward signal stabilizes the policy.
    num_steps_per_env = 16
    # v8: cut iteration budget. v7 peaked at iter 1500 (SR≈0.62) before
    # collapsing. With the σ-clamp in PickPlaceResNetActorCritic we expect
    # the peak to hold rather than collapse, so the working answer should
    # be readable well before 2000 iters.
    max_iterations = 2000
    save_interval = 50
    # Separate experiment dir so logs don't collide with the §7 ``pickplace_bowl``
    # run history.
    experiment_name = "pickplace_bowl_pretrained"
    empirical_normalization = False

    # Asymmetric A-C — same wiring as the §7 Stage 3 path. Image goes to the
    # actor only; critic reads the privileged state group.
    obs_groups = {
        "policy": ["policy", "wrist_image"],
        "critic": ["policy", "critic"],
    }

    policy = RslRlPpoActorCriticCfg(
        class_name="PickPlaceResNetActorCritic",
        # v7.1 (2026-05-15): refined exploration tuning.
        # v7 hit success_rate=0.61 at iter 593 but σ then grew unboundedly
        # (3.97 by iter 593) until numerical instability crashed the run
        # at iter ~899 (mean reward went to -1.5e+20, value loss inf,
        # std went negative → torch.normal crashed).
        # v7.1 fixes by:
        #   - noise_std_type "scalar" → "log" (σ = exp(log_std), can't go negative)
        #   - init_noise_std 1.5 → 1.0 (matches v5.4's stable cold-start init)
        #   - entropy_coef 0.012 → 0.008 (was 0.006 in v5.4; modest bump only)
        #   - desired_kl 0.025 → 0.015 (was 0.01 in v5.4; modest relaxation only)
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # v8.1: revert v8's tight KL + low entropy. The first v8 attempt
        # (desired_kl=0.005, entropy_coef=0.004) saw the adaptive KL halve
        # LR to the 1e-5 floor within 5 iters of training start: with
        # init_noise_std=1.0 the consecutive-policy KL at random init
        # already exceeds 2 × 0.005, so the controller halved LR every
        # iter until it bottomed. With nothing learning, mean_reward went
        # to ~0 by iter 100.
        # The σ-clamp at exp(0.0)=1.0 in the actor (see
        # PickPlaceResNetActorCritic) is now what prevents the v7
        # σ-runaway, so KL no longer needs to do double duty. Keep
        # v7's working desired_kl=0.015 / entropy_coef=0.008.
        entropy_coef=0.008,
        num_learning_epochs=8,
        num_mini_batches=16,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.015,
        max_grad_norm=1.0,
    )
