# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL distillation config for the SO-ARM101 pick-and-place vision student.

Stage 3 of the §7 teacher–student pipeline (after the state teacher
converged at ``model_700.pt`` of ``pickplace_bowl_teacher``):

* Student reads ``policy + wrist_image`` (deployable obs — what the real
  robot sees) → CNN+MLP → 6-D action.
* Teacher reads ``policy + critic`` (privileged state) → MLP → 6-D action.
* Loss: MSE between student action and teacher action.
* Algorithm: RSL-RL's stock :class:`rsl_rl.algorithms.Distillation` —
  on-policy DAgger (student rolls out, teacher labels every state).

RSL-RL auto-loads the teacher: when the runner's checkpoint path points
at a PPO-trained ``model_*.pt`` whose keys start with ``actor.``,
:meth:`StudentTeacher.load_state_dict` recognises the format and copies
the actor weights into ``self.teacher``. The student starts from random
init.

Class-injection mechanism
-------------------------

Same trick as ``rsl_rl_ppo_cfg.py`` — RSL-RL's
``DistillationRunner._construct_algorithm`` resolves the policy class
via ``eval(self.policy_cfg.pop("class_name"))`` against
``rsl_rl.runners.distillation_runner``'s globals. We register our
:class:`PickPlaceVisionStudentTeacher` into that namespace at import
time, which happens when the gym task is loaded.
"""

import rsl_rl.runners.distillation_runner as _distillation_runner

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)

from .vision_student_teacher import PickPlaceVisionStudentTeacher


def _register_class() -> None:
    """Inject :class:`PickPlaceVisionStudentTeacher` into RSL-RL's runner scope.

    RSL-RL resolves ``policy.class_name`` strings via :func:`eval` against
    the runner module's globals, so the class needs to be visible there.
    """
    setattr(
        _distillation_runner,
        PickPlaceVisionStudentTeacher.__name__,
        PickPlaceVisionStudentTeacher,
    )


_register_class()


@configclass
class PickPlaceBowlDistillRunnerCfg(RslRlDistillationRunnerCfg):
    """DAgger distillation: vision student ← state teacher (model_700.pt).

    Hyperparameters follow Isaac Lab's AnymalD distillation example
    (locomotion velocity → flat distillation), adapted for our setup:

    * ``num_steps_per_env`` 120 (AnymalD) → **24** — matches our teacher's
      PPO rollout length. Cube grasping episodes are 250 steps (5 s × 50 Hz);
      24 steps/env is ~10% of episode length, enough for a meaningful
      partial trajectory in each rollout chunk.
    * ``num_envs`` 1024 (vs AnymalD's 4096) — image rollouts are 4× heavier
      than state-only locomotion; we run half the teacher's 2048 to stay
      well under the 32 GB VRAM ceiling.
    * ``learning_rate`` 1e-3 → **1e-4** — student is regressing a known
      target, not exploring; conservative LR avoids overshoot.
    * ``num_learning_epochs`` 2 → **5** — image regression benefits from
      more SGD steps per rollout batch.
    * ``loss_type="mse"`` — standard action regression. Tried-and-true
      for continuous control; alternative ``huber`` only matters if the
      teacher's action distribution has heavy tails (it doesn't —
      σ ≈ 0.94 Gaussian at convergence).
    """

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    # 1024 envs × 24 steps = 24,576 transitions per rollout-iter; with
    # 5 learning epochs that's 122k gradient updates per iter, plenty for
    # the small student network (≈ 217k params).
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "pickplace_bowl_student"
    empirical_normalization = False

    # Student reads deployable obs (proprio + image); teacher reads
    # privileged state. ``policy + wrist_image`` matches the live env's
    # actor obs group (see ObservationsCfg in pickplace_env_cfg.py).
    obs_groups = {
        "policy": ["policy", "wrist_image"],
        "teacher": ["policy", "critic"],
    }

    # ------------------------------------------------------------------
    # Policy (student-teacher)
    # ------------------------------------------------------------------
    policy = RslRlDistillationStudentTeacherCfg(
        class_name="PickPlaceVisionStudentTeacher",
        # Low init noise: student is matching a known target action,
        # not exploring. Stock RSL-RL distillation uses 0.1.
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        # Hidden dims match the teacher's PPO config (256,128,64) so the
        # teacher state-dict loads cleanly into the teacher slot, and the
        # student MLP is sized comparably to the teacher.
        student_hidden_dims=[256, 128, 64],
        teacher_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    # ------------------------------------------------------------------
    # Algorithm (Distillation)
    # ------------------------------------------------------------------
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-4,
        # gradient_length: number of env steps the gradient flows back
        # through (BPTT-style for recurrent; for non-recurrent this is
        # effectively the truncation length for advantage / loss
        # accumulation). RSL-RL default 15 — leave as-is.
        gradient_length=15,
        max_grad_norm=1.0,
        loss_type="mse",
        optimizer="adam",
    )
