# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO config for the SO-ARM101 pick-and-place **teacher** task.

The teacher policy is the first half of the §7 teacher–student distillation
fallback in ``EVAL1_PLAN.md``: it solves the same MDP as the vision env, but
with **privileged state** as the actor input (block pose, distances, grasp
flag). Because the actor sees ground-truth state directly, the credit-
assignment problem that blocks end-to-end vision PPO disappears — state
PPO solved this MDP at the Day-3 milestone, and ManiSkill3-style cube-
grasp converges in 1k–1.5k iters in that regime.

Once the teacher is mature (target: ``release_from_scratch ≥ 0.6`` in
sim), it serves as the *action label oracle* for a vision student trained
via DAgger / BC distillation. The student sees only ``wrist_image`` +
proprio (deployable) and learns to mimic the teacher's actions.

Architecture choice
-------------------

We **reuse** :class:`PickPlaceVisionActorCritic` rather than write a new
class. That class already auto-disables the CNN when the image group
is not in ``obs_groups`` (see ``actor_uses_image`` / ``critic_uses_image``
branches in its ``__init__``). With ``obs_groups`` set to state-only
groups, both ``actor_cnn`` and ``critic_cnn`` are ``None`` and the class
behaves as a stock symmetric MLP actor-critic.

The wrist camera is still rendered every step in the underlying env
(it's wired into the scene by ``joint_pos_env_cfg.py``), but the obs
group ``wrist_image`` is *not* in ``obs_groups`` here, so the rendered
tensor is computed-and-discarded each step. ~2× wall-clock overhead vs
a no-camera teacher env, but no new env cfg needed — keeps the change
minimal and ensures the teacher's training distribution (DR, scene,
rewards) matches the env the student will eventually run in.
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

    Same trick as ``rsl_rl_ppo_cfg.py`` — RSL-RL resolves
    ``policy.class_name`` strings via :func:`eval` against the runner
    module's globals, so the class needs to be visible there. Idempotent
    if the vision cfg already registered it.
    """
    setattr(
        _on_policy_runner,
        PickPlaceVisionActorCritic.__name__,
        PickPlaceVisionActorCritic,
    )


_register_class()


@configclass
class PickPlaceBowlTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for the state-only teacher.

    Inherits the same hyperparameter philosophy as the vision PPO config
    (ManiSkill3-style: γ=0.9, num_steps_per_env=16, mini_batches=16,
    learning_epochs=8) but with two intentional changes:

    * ``obs_groups`` symmetric on the combined state — actor + critic both
      read ``policy + critic`` (deployable proprio + privileged
      block_pose / distances / grasp flag).
    * No CNN — :class:`PickPlaceVisionActorCritic` auto-skips the conv
      stack when the image group isn't in ``obs_groups``.
    """

    # Match stock Isaac Lab Franka Lift PPO config (rsl_rl_ppo_cfg.py at
    # ``isaaclab_tasks/manager_based/manipulation/lift/config/franka/agents``).
    # That config converges this same MDP class (state-based PPO on
    # binary-gripper pick-and-place) reliably in ~1500 iters at 4096
    # envs. We had drifted away from it across runs 11-15 with
    # task-specific tweaks (γ=0.9 from ManiSkill, more mini-batches for
    # GPU saturation, smaller init noise) — every drift compounded the
    # divergence. This is a verbatim revert.
    num_steps_per_env = 24
    # Single-stage from-scratch training (no more two-stage). Stage 1's
    # "lift to z=0.10" objective baked the wrong wrist posture; stage 2
    # couldn't unlearn it cleanly. The new task design (latch-based
    # transport, goal_z=0, release reward from start) lets the teacher
    # learn pick + transport + place + release in one shot, no need
    # for a staged warm-up.
    max_iterations = 1500
    save_interval = 50
    experiment_name = "pickplace_bowl_teacher"
    empirical_normalization = False

    # Symmetric A-C — both actor and critic see deployable + privileged.
    # ``wrist_image`` is intentionally absent; the vision env still renders
    # it but the policy never reads it. (Saves writing a no-camera env cfg
    # variant; doubles per-iter render cost vs a stripped scene, accepted
    # for code-simplicity reasons documented at the top of this file.)
    obs_groups = {
        "policy": ["policy", "critic"],
        "critic": ["policy", "critic"],
    }

    policy = RslRlPpoActorCriticCfg(
        class_name="PickPlaceVisionActorCritic",
        # Stock σ=1.0 across all dims, INCLUDING the binary gripper.
        # Stock Franka Lift uses BinaryJointPositionActionCfg too, so
        # this is a known-good config for our action space. Our prior
        # gripper-σ override (0.1, then 0.2) is also disabled in the
        # actor-critic class to honor stock semantics — see the
        # gripper_init_std variable in vision_actor_critic.py.
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Reverted 0.02 → 0.006 (stock Franka Lift) after run-17 TB
        # diagnostic (2026-05-10, 350 iters, bootstrap=0): entropy_coef=0.02
        # over-corrected — σ inflated 1.0 → 1.60 → 1.49 (high plateau),
        # value_function loss converged to 0.0000, surrogate loss bouncing
        # around 0, LR clamped to 1e-4 floor. PPO stopped updating because
        # entropy bonus dominated the policy gradient. The σ=1.5 regime
        # makes the binary gripper action essentially random (P(open)=
        # P(close)=50% even with biased μ), so no sustained close-and-
        # hold trajectory ever appeared in 614k frames except one
        # transient grasp at iter 66. Stock entropy (0.006) lets σ decay
        # naturally to a useful exploration band; bootstrap p=0.10 (set
        # in pickplace_env_cfg.EventCfg) supplies the rollouts that
        # exploration alone couldn't generate.
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
