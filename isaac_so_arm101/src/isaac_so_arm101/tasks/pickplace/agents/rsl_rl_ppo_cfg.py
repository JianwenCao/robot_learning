# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO config for the SO-ARM101 pick-and-place vision task.

The actor reads the deployable ``policy`` state group **plus** the
``wrist_rgb`` image group (encoded by a CNN); the critic additionally reads
the privileged ``critic`` group. This is the asymmetric Actor-Critic setup
prescribed in EVAL1_PLAN §3.4 / §3.9.

Class-injection mechanism
-------------------------

``rsl_rl.runners.on_policy_runner.OnPolicyRunner`` resolves the policy
class via ``eval(self.policy_cfg.pop("class_name"))``, which looks up the
name in that module's globals. To make our custom
:class:`PickPlaceVisionActorCritic` discoverable, we register it into
:mod:`rsl_rl.runners.on_policy_runner`'s namespace at *import time* of this
module. Importing this config (which happens at gym registration in
:mod:`tasks.pickplace.__init__`) is enough to make the class available
before any training starts.
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
    """Inject :class:`PickPlaceVisionActorCritic` into RSL-RL's runner scope.

    RSL-RL resolves ``policy.class_name`` strings via :func:`eval` against
    the runner module's globals, so the class needs to be visible there.
    """
    setattr(
        _on_policy_runner,
        PickPlaceVisionActorCritic.__name__,
        PickPlaceVisionActorCritic,
    )


_register_class()


@configclass
class PickPlaceBowlPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config — vision policy with asymmetric critic.

    Hyperparameters mirror the upstream lift task's PPO cfg, with two changes
    motivated by EVAL1_PLAN §3.10:

    * ``max_iterations`` bumped (vision needs more samples than state-only).
    * ``policy.class_name`` swapped to the CNN-based actor-critic.
    """

    # 24 steps × 2048 envs × 4000 iters ≈ 200 M env steps. Plan §3.10
    # budget is 100–300 M steps. 4000 iters covers the bootstrap-decay
    # curriculum end (~iter 3900 at 2048 envs × 24 steps = 49 152
    # env-steps/iter, vs 192 k env-steps for warmup+decay) plus a bit
    # of from-scratch tail. Kill earlier if per-stage success curves
    # plateau on TB (``Curriculum/log_metrics/release_from_scratch``).
    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 100
    experiment_name = "pickplace_bowl"
    empirical_normalization = False

    # Asymmetric A-C wiring. ``wrist_rgb`` goes to the actor only — the
    # critic already has ground-truth ``block_position`` etc. via the
    # privileged ``critic`` group, so feeding it the image is redundant
    # compute. Removing it cuts ~1 CNN forward+backward per iter and
    # halves the encoder param count of the optimizer. The actor still
    # gets the image (it's the only one that needs to localize the block
    # from vision). Setting :attr:`PickPlaceVisionActorCritic.critic_cnn`
    # to ``None`` happens automatically when ``wrist_rgb`` isn't in
    # ``obs_groups["critic"]``.
    obs_groups = {
        "policy": ["policy", "wrist_rgb"],
        "critic": ["policy", "critic"],
    }

    policy = RslRlPpoActorCriticCfg(
        class_name="PickPlaceVisionActorCritic",
        # Lowered 1.0 → 0.5 after run 9 (2026-05-09, p_grasped=0.10 floor +
        # reach=1.0). At σ=1.0 even the bootstrapped 10% of envs lost their
        # pre-set grasp within ~30 simulated steps because random gripper
        # noise (action ~ N(0,1) per dim) repeatedly crossed the
        # closed/open threshold. The bootstrap therefore never produced a
        # sustained "grasp held → continued reward" trajectory PPO could
        # credit-assign back to "close gripper at cube" — Episode_Reward/grasp
        # decayed 0.044 → 0.011 over 250 iters, gfs stayed at 0.0000, and
        # mean reward eventually collapsed to 0.19. With σ=0.5 the
        # stochastic action range halves, bootstrapped grasps should
        # persist long enough (≥100 steps) for PPO to learn that
        # closure → reward. Still wide enough to maintain exploration on
        # the reach + grasp-discovery sub-task in non-bootstrapped envs.
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # ``entropy_coef`` halved 0.006 → 0.003 after the second resume
        # of run 2026-05-07_00-23-09 (resumed dir 2026-05-07_09-17-59).
        # ``desired_kl=0.005`` alone slowed σ growth but didn't reverse
        # it: σ drifted 2.06 → 2.31 over 2200 iters, then collapsed to
        # 1.90 only after reward had crashed from +38 to -1.5 (policy
        # locked into a "do-nothing" pattern paying steady action-rate
        # penalties). Halving entropy_coef weakens the entropy bonus
        # term that was actively pushing σ up; combined with the tight
        # KL clamp, σ should now decay monotonically. We trade a bit of
        # exploration for stability — acceptable since the policy was
        # already roughly converged at iter 2400 (σ=2.15, reward ~38).
        entropy_coef=0.003,
        num_learning_epochs=5,
        # 8 mini-batches keeps the per-batch sample count at 2048×24/8 =
        # 6144 — same as the 1024-env / 4-mini-batch baseline that runs 1-3
        # were tuned against. With 2048 envs, sticking to 4 mini-batches
        # would double the minibatch (12288 samples), which smooths
        # gradients but halves the number of SGD steps per update.
        # Holding minibatch size constant gives more SGD steps per iter
        # (40 vs 20) → better convergence rate at the cost of slightly
        # more GPU compute.
        num_mini_batches=8,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        # ``desired_kl`` halved from the lift-task default 0.01 → 0.005 after
        # observing action-noise σ blow up to 2.12 by iter 2100 of the
        # initial vision run (logs/rsl_rl/pickplace_bowl/2026-05-07_00-23-09).
        # The adaptive LR schedule was raising the LR whenever KL undershot
        # the target, which compounded with the entropy coef to widen σ
        # exponentially. A tighter KL band keeps updates conservative —
        # smaller σ growth, less variance in the per-iter reward, slower
        # but more stable convergence. Restored to 0.01 only if convergence
        # gets too slow (more than ~200 iters with no improvement).
        desired_kl=0.005,
        max_grad_norm=1.0,
    )
