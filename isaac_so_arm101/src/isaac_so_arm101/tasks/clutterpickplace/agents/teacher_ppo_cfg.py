# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO config for the SO-ARM101 Eval-2 **state teacher**.

Stage 1 of the §7 three-stage pipeline (``docs/EVAL2_PLAN.md``): the
teacher solves the targeted-pick-and-place MDP using **privileged
state** as the actor input (target & distractor cube poses, ee→cube
distances, target_color one-hot, lift/grasp flags). Because the actor
sees ground-truth target pose directly, the perception bottleneck that
blocks end-to-end vision PPO disappears.

Once mature (target: ``release_from_scratch ≥ 0.6`` in sim) it serves
as the action oracle for Stage 2's vision student via DAgger, and its
critic state-dict overlays Stage 3's vision PPO critic via
``--teacher_ckpt``.

Architecture choice — reuse :class:`PickPlaceVisionActorCritic`
-------------------------------------------------------------

We reuse the Eval-1 actor-critic class rather than write a new one.
Its ``__init__`` already auto-disables ``actor_cnn`` / ``critic_cnn``
when the ``wrist_image`` group is absent from ``obs_groups``, so with
state-only obs groups it behaves as a stock symmetric MLP A-C — no
code changes needed.

Two env cfgs are registered against this PPO cfg (see
:mod:`tasks.clutterpickplace.__init__`):

* ``Isaac-SO-ARM101-ClutterPickPlace-Teacher-v0`` uses the shared
  vision-task env cfg — wrist camera is spawned and rendered every
  step even though the obs group ``wrist_image`` isn't in
  ``obs_groups``. Requires ``--enable_cameras``. Matches the
  student's training distribution exactly.
* ``Isaac-SO-ARM101-ClutterPickPlace-Teacher-Fast-v0`` uses a
  stripped env cfg (no ``wrist_cam``, no ``wrist_image`` obs).
  Skips RTX render path; ~2-3× faster wall-clock per iter. No
  ``--enable_cameras`` needed. State-side obs / rewards / DR are
  identical to the vision env — only the unused image pipeline is
  removed.
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
    """Inject :class:`PickPlaceVisionActorCritic` into RSL-RL's runner scope.

    RSL-RL resolves ``policy.class_name`` strings via :func:`eval`
    against the runner module's globals; the class needs to be visible
    there. Idempotent if the Eval-1 cfg already registered it during
    its own import.
    """
    setattr(
        _on_policy_runner,
        PickPlaceVisionActorCritic.__name__,
        PickPlaceVisionActorCritic,
    )


_register_class()


@configclass
class ClutterPickPlaceTeacherPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for the Eval-2 state teacher.

    Hyperparameters mirror Eval-1's ``PickPlaceBowlTeacherPPORunnerCfg``
    (the recipe that worked first try) — only ``experiment_name``
    changes. The Eval-2 state schema is wider than Eval-1 (one-hot +6,
    distractor pose +3) but the MDP shape is identical, so the same
    hyperparams should converge.

    Env-step budget: ``2048 envs × 32 steps/iter × 1500 iters ≈ 98 M``,
    same as Eval-1.
    """

    # Bumped from RSL-RL default 24 → 32 (matches Eval-1 teacher) to
    # grow the per-iter batch (more on-policy samples per learning
    # update) at no wall-clock cost — collection time is PhysX-bound.
    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 50
    experiment_name = "clutterpickplace_teacher"
    empirical_normalization = False

    # Symmetric A-C on privileged + deployable state, **plus** the target-
    # color one-hot in the ``goal`` group. ``wrist_image`` is intentionally
    # absent — PickPlaceVisionActorCritic sees its absence and skips
    # constructing the CNNs, so this is a pure MLP A-C.
    obs_groups = {
        "policy": ["policy", "critic", "goal"],
        "critic": ["policy", "critic", "goal"],
    }

    policy = RslRlPpoActorCriticCfg(
        class_name="PickPlaceVisionActorCritic",
        # Scalar init kept here for the RslRl dataclass schema (it
        # expects a float). The actual per-dim noise vector is patched
        # into ``policy.init_noise_std`` in ``__post_init__`` below
        # (PickPlaceVisionActorCritic accepts list/tuple as well as
        # scalar). See ``__post_init__`` for the rationale.
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Bumped 0.003 → 0.006 back (v7) after v5/v6 showed entropy=0.003
        # + desired_kl=0.005 + per-dim noise was over-tight: arm σ flat
        # at init, entropy collapsed to 6.79 by iter 200 (vs v3's 10.10),
        # policy under-explored, reach barely moved. Stock Eval-1 value.
        # With per-dim init [arm 1.0, gripper 0.1] the entropy bonus
        # pushes arm σ slightly up (good for exploration) AND pushes
        # gripper σ slightly up — but gripper σ starts so low (0.1)
        # that even 2x growth (0.2) keeps it decisive. Best of both worlds.
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        # Tightened 0.01 → 0.005 after the first Stage-1 run (logs/...
        # 2026-05-19_21-39-16) blew σ up: σ went 1.79 → 2.32 → 2.85 →
        # 2.13 by iter 400, Mean reward degraded 0.06 → -3.58 as the
        # policy thrashed and then locked into a "do-nothing" attractor
        # paying steady action-rate / joint-vel penalties. This is the
        # same failure mode Eval-1 documented (vision-PPO σ blowup);
        # the Eval-1 fix was desired_kl=0.005, which we adopt here.
        # The Eval-2 teacher MDP has wider state (68d vs Eval-1's 59d)
        # and a wrong_block_in_bowl=-20 penalty, both of which make
        # the value-function gradient noisier than stock Franka Lift —
        # hence stock desired_kl=0.01 was too loose.
        desired_kl=0.005,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        # Per-dim init_noise_std: [arm0, arm1, arm2, arm3, arm4, gripper].
        # Arm dims at σ=1.0 (matches Eval-1 teacher exploration that
        # produced reach gradient by iter 100 in v3). Gripper dim at
        # σ=0.1 — the binary gripper threshold (action[5] > 0) flips
        # ~50 % per step at σ=1.0 even with biased μ, so the cube
        # cannot survive sustained closure. v3 confirmed this: 1500
        # iters with scalar σ=1.0 → zero lift events. v4 with scalar
        # σ=0.5 reached worse (arm under-explored). Per-dim split lets
        # arm and gripper exploration scales decouple.
        #
        # PickPlaceVisionActorCritic accepts list/tuple via its
        # ``init_noise_std`` constructor arg; we monkey-patch the
        # RslRl dataclass which type-hints float only.
        self.policy.init_noise_std = [1.0, 1.0, 1.0, 1.0, 1.0, 0.1]

        # Per-dim σ upper cap. v7 confirmed the per-dim init alone is
        # not enough — entropy_coef=0.006 pushed gripper σ from 0.1 to
        # ~0.4 by iter 700 (mean σ peaked at 2.27), at which point
        # binary closure breaks again and the lifts that appeared at
        # iter 200-600 disappeared. Cap = [arm 5×∞, gripper 0.2] lets
        # PPO push gripper σ DOWN further but blocks entropy bonus
        # from inflating it. Arm dims uncapped (1e3 ≈ ∞).
        self.policy.std_max = [1e3, 1e3, 1e3, 1e3, 1e3, 0.2]
