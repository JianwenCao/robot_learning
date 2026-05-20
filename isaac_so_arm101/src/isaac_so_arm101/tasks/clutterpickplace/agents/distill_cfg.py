# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL distillation config for the SO-ARM101 Eval-2 vision student (Stage 2).

* Student reads ``policy + goal + wrist_image`` → ResNet+FiLM CNN + MLP → 6-D action.
* Teacher reads ``policy + critic + goal`` (privileged state) → MLP → 6-D action.
* Loss: MSE between student action and teacher action.

RSL-RL auto-loads the teacher: when the runner's checkpoint path points
at a PPO-trained ``model_*.pt`` whose keys start with ``actor.``,
:meth:`StudentTeacher.load_state_dict` recognises the format and copies
the actor weights into ``self.teacher``. Our Stage-1 teacher
(``model_*.pt`` from ``clutterpickplace_teacher``) has the right shape
because the teacher and student MLPs share identical hidden dims and
the teacher's privileged-state vector matches the ``policy + critic + goal``
group set.
"""

import rsl_rl.runners.distillation_runner as _distillation_runner

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)

from .vision_student_teacher import ClutterPickPlaceVisionStudentTeacher


def _register_class() -> None:
    setattr(
        _distillation_runner,
        ClutterPickPlaceVisionStudentTeacher.__name__,
        ClutterPickPlaceVisionStudentTeacher,
    )


_register_class()


@configclass
class ClutterPickPlaceDistillRunnerCfg(RslRlDistillationRunnerCfg):
    """DAgger distillation: vision student ← state teacher.

    Hyperparameters mirror Eval-1's ``PickPlaceBowlDistillRunnerCfg``
    with one knob (``num_envs`` reduced to 1024 in the train script if
    needed; not enforced here so the env cfg's default 2048 carries
    through and we let the cmd-line cut it if VRAM forces it).
    """

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "clutterpickplace_student"
    empirical_normalization = False

    # Student: deployable obs (proprio + image + goal). Teacher: privileged
    # state + goal. Both include ``goal`` so the student's FiLM head and
    # the teacher's MLP both see the target-color one-hot.
    obs_groups = {
        "policy": ["policy", "goal", "wrist_image"],
        "teacher": ["policy", "critic", "goal"],
    }

    policy = RslRlDistillationStudentTeacherCfg(
        class_name="ClutterPickPlaceVisionStudentTeacher",
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[256, 128, 64],
        teacher_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-4,
        gradient_length=15,
        max_grad_norm=1.0,
        loss_type="mse",
        optimizer="adam",
    )
