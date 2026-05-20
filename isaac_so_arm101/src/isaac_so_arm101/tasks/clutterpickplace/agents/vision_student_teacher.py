# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Eval-2 vision student + state teacher for distillation (Stage 2).

Subclasses :class:`pickplace.agents.vision_student_teacher.PickPlaceVisionStudentTeacher`
to:

1. Use the **frozen ResNet-18 + FiLM** encoder for the student
   (matching the Stage 3 actor architecture so the distill checkpoint
   loads layer-for-layer into Stage 3).
2. Route the ``goal`` obs group (target-color one-hot) to the
   encoder's FiLM head.

DAgger ordering is unchanged from Eval-1 — RSL-RL's stock distillation
algorithm rolls the student, queries the teacher for an action label,
and minimizes MSE between the two.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.networks import MLP

from isaac_so_arm101.tasks.pickplace.agents.vision_actor_critic import (
    DRQ_PAD_PIXELS,
    _ResNetSpatialSoftmaxCNN,
    _random_shift_pad,
)
from isaac_so_arm101.tasks.pickplace.agents.vision_student_teacher import (
    PickPlaceVisionStudentTeacher,
)

DEFAULT_GOAL_GROUP = "goal"


class ClutterPickPlaceVisionStudentTeacher(PickPlaceVisionStudentTeacher):
    """Vision student with FiLM-conditioned ResNet encoder + state teacher MLP."""

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        student_hidden_dims=(256, 128, 64),
        teacher_hidden_dims=(256, 128, 64),
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        image_group_name: str = "wrist_image",
        image_feat_dim: int = 128,
        goal_group_name: str = DEFAULT_GOAL_GROUP,
        film_cond_dim: int = 6,
        truncate_at: str = "layer2",
        **kwargs,
    ):
        # Don't call super().__init__ — it builds a _SpatialSoftmaxCNN
        # student_cnn we'd just have to throw away. Replicate the parent
        # init body, then swap in a ResNet+FiLM encoder.
        nn.Module.__init__(self)
        if kwargs:
            print(
                "ClutterPickPlaceVisionStudentTeacher.__init__ got unexpected kwargs (ignored): "
                + str(list(kwargs.keys()))
            )
        from rsl_rl.networks import EmpiricalNormalization
        from torch.distributions import Normal

        self.loaded_teacher = False
        self.obs_groups = obs_groups
        self.image_group_name = image_group_name
        self.goal_group_name = goal_group_name

        # ---- Student input: image → ResNet+FiLM, state stays 1-D ----------
        student_state_dim = self._sum_state_dims(obs, obs_groups["policy"], image_group_name)
        student_uses_image = image_group_name in obs_groups["policy"]

        if student_uses_image:
            img_sample = obs[image_group_name]
            assert (
                img_sample.dim() == 4
            ), f"Expected image obs of shape (N, C, H, W); got {tuple(img_sample.shape)}"
            img_in_shape = (
                int(img_sample.shape[1]),
                int(img_sample.shape[2]),
                int(img_sample.shape[3]),
            )
            self.student_cnn: nn.Module = _ResNetSpatialSoftmaxCNN(
                in_shape=img_in_shape,
                out_dim=image_feat_dim,
                freeze=True,
                truncate_at=truncate_at,
                film_cond_dim=film_cond_dim,
            )
            student_in = student_state_dim + image_feat_dim
        else:
            self.student_cnn = None
            student_in = student_state_dim

        self.student = MLP(student_in, num_actions, list(student_hidden_dims), activation)
        print(f"Student CNN: {self.student_cnn}")
        print(f"Student MLP: {self.student}")

        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(student_state_dim)
        else:
            self.student_obs_normalizer = nn.Identity()

        # ---- Teacher: pure MLP over privileged state ----------------------
        teacher_dim = self._sum_state_dims(obs, obs_groups["teacher"], image_group_name)
        self.teacher = MLP(teacher_dim, num_actions, list(teacher_hidden_dims), activation)
        self.teacher.eval()
        print(f"Teacher MLP: {self.teacher}")

        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(teacher_dim)
        else:
            self.teacher_obs_normalizer = nn.Identity()

        # ---- Action noise --------------------------------------------------
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type {noise_std_type!r}")

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    # ---- encode override — pass goal one-hot to FiLM ---------------------

    def _gather_film_cond(self, obs) -> torch.Tensor | None:
        if self.goal_group_name in obs:
            return self._safe(obs[self.goal_group_name])
        return None

    def _encode_student(self, obs) -> torch.Tensor:
        state = self._gather_student_state(obs)
        state = self.student_obs_normalizer(state)
        if self.student_cnn is None:
            return state
        img = self._safe(obs[self.image_group_name])
        # DrQ ±4 px at train time (same as parent — closes the Stage 2→3
        # distribution-shift gap; see EVAL1_PLAN §7.2 intervention #1).
        if self.training:
            img = _random_shift_pad(img, DRQ_PAD_PIXELS)
        film_cond = self._gather_film_cond(obs)
        feat = self.student_cnn(img, film_cond=film_cond)
        return torch.cat([state, feat], dim=-1)
