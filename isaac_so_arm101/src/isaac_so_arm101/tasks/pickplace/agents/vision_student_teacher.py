# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision-based student + state-based teacher for distillation.

Subclasses :class:`rsl_rl.modules.StudentTeacher` to support 4-D image
observations in the student group. The parent class asserts every student
obs group is 1-D — we bypass that and route the ``wrist_image`` group
(N, 5, 72, 128) through a Spatial-Softmax CNN encoder before concatenating
with the 1-D state and feeding the student MLP.

The teacher is unchanged from the parent — pure MLP over the privileged
state group (``policy + critic``). At distillation time, the parent class's
``load_state_dict`` auto-detects PPO-trained teacher checkpoints (keys
starting with ``actor.``) and loads them into the teacher MLP. Our state
teacher (``model_700.pt`` from ``pickplace_bowl_teacher``) was trained as
:class:`PickPlaceVisionActorCritic` with the image group absent from
``obs_groups``, so its ``actor_cnn`` is ``None`` and the saved weights
are pure MLP — matches the teacher slot here cleanly.

Architecture:

    Student (deployable):
        wrist_image (N, 5, 72, 128)  ──[_SpatialSoftmaxCNN]── 128-D feat ─┐
        policy state (~25 D proprio + bowl + last_action) ─────── concat ─┴── MLP [256,128,64] → 6 actions

    Teacher (sim-only, frozen):
        policy + critic (~59 D privileged state) ── MLP [256,128,64] → 6 actions

Registered into ``rsl_rl.runners.distillation_runner``'s namespace at
import time of :mod:`agents.distill_cfg` so the runner's
``eval(class_name)`` can resolve the class name.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules import StudentTeacher
from rsl_rl.networks import MLP, EmpiricalNormalization
from torch.distributions import Normal

# Reuse the spatial-softmax CNN already proven on this env. Same first-layer
# shape (5-channel input), same output dim (128) — drop-in compatible with
# the rest of our pipeline.
from .vision_actor_critic import _SpatialSoftmaxCNN

DEFAULT_IMAGE_GROUP = "wrist_image"


class PickPlaceVisionStudentTeacher(StudentTeacher):
    """Vision student (CNN+MLP) + state teacher (MLP), bypassing parent's 1-D assertion.

    Constructor signature mirrors :class:`rsl_rl.modules.StudentTeacher` —
    RSL-RL instantiates it via
    ``cls(obs, obs_groups, num_actions, **policy_cfg_kwargs)``. We keep that
    contract intact so the runner doesn't need any patching beyond the
    ``class_name`` lookup (see :func:`agents.distill_cfg._register_class`).

    Args:
        image_group_name: name of the obs group carrying ``(N, C, H, W)``
            image tensors. Default ``"wrist_image"``.
        image_feat_dim: output dim of the CNN encoder. Default 128.
        Other args are forwarded to :class:`StudentTeacher` semantics (we
        replicate the bits we need rather than calling ``super().__init__``).
    """

    is_recurrent = False

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
        image_group_name: str = DEFAULT_IMAGE_GROUP,
        image_feat_dim: int = 128,
        **kwargs,
    ):
        # Deliberately DO NOT call super().__init__ — parent asserts every
        # student obs group is 1-D, which crashes on (N, 5, 72, 128). We
        # replicate the parts of the parent we still need (teacher MLP,
        # action distribution, normalizers, load_state_dict semantics).
        nn.Module.__init__(self)
        if kwargs:
            print(
                "PickPlaceVisionStudentTeacher.__init__ got unexpected kwargs (ignored): "
                + str(list(kwargs.keys()))
            )

        self.loaded_teacher = False
        self.obs_groups = obs_groups
        self.image_group_name = image_group_name

        # ------------------------------------------------------------------
        # Student input: image goes through CNN, state stays 1-D.
        # ------------------------------------------------------------------
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
            self.student_cnn: nn.Module = _SpatialSoftmaxCNN(img_in_shape, image_feat_dim)
            student_in = student_state_dim + image_feat_dim
        else:
            self.student_cnn = None
            student_in = student_state_dim

        # Standard MLP — same as parent class.
        self.student = MLP(student_in, num_actions, list(student_hidden_dims), activation)
        print(f"Student CNN: {self.student_cnn}")
        print(f"Student MLP: {self.student}")

        # Student normalization is only over the *state* portion (CNN has
        # its own LayerNorm at the head; image features are well-conditioned).
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(student_state_dim)
        else:
            self.student_obs_normalizer = nn.Identity()

        # ------------------------------------------------------------------
        # Teacher: pure MLP over privileged state (unchanged from parent).
        # ------------------------------------------------------------------
        teacher_dim = self._sum_state_dims(obs, obs_groups["teacher"], image_group_name)
        self.teacher = MLP(teacher_dim, num_actions, list(teacher_hidden_dims), activation)
        self.teacher.eval()
        print(f"Teacher MLP: {self.teacher}")

        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(teacher_dim)
        else:
            self.teacher_obs_normalizer = nn.Identity()

        # ------------------------------------------------------------------
        # Action noise — same as parent class.
        # ------------------------------------------------------------------
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type {noise_std_type!r}")

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    @staticmethod
    def _sum_state_dims(obs, group_names, image_group_name) -> int:
        """Sum last-dim sizes of all *non-image* groups in ``group_names``."""
        total = 0
        for name in group_names:
            if name == image_group_name:
                continue
            t = obs[name]
            assert (
                t.dim() == 2
            ), f"State obs group {name!r} must be 1-D per env; got shape {tuple(t.shape)}"
            total += t.shape[-1]
        return total

    @staticmethod
    def _safe(t: torch.Tensor) -> torch.Tensor:
        """Detach from any inference-mode storage so backprop works."""
        out = torch.empty_like(t)
        out.copy_(t)
        return out

    def _gather_student_state(self, obs) -> torch.Tensor:
        parts = []
        for g in self.obs_groups["policy"]:
            if g == self.image_group_name:
                continue
            parts.append(self._safe(obs[g]))
        if parts:
            return torch.cat(parts, dim=-1)
        # No state, only image — return empty tensor with correct batch
        n = obs[self.image_group_name].shape[0]
        return torch.empty(n, 0, device=obs[self.image_group_name].device)

    def _gather_teacher_state(self, obs) -> torch.Tensor:
        parts = [self._safe(obs[g]) for g in self.obs_groups["teacher"]]
        return torch.cat(parts, dim=-1)

    def _encode_student(self, obs) -> torch.Tensor:
        state = self._gather_student_state(obs)
        state = self.student_obs_normalizer(state)
        if self.student_cnn is None:
            return state
        img = self._safe(obs[self.image_group_name])
        feat = self.student_cnn(img)
        return torch.cat([state, feat], dim=-1)

    # ----------------------------------------------------------------------
    # StudentTeacher interface
    # ----------------------------------------------------------------------

    def update_distribution(self, x):
        mean = self.student(x)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs):
        """Student rollout action — sample from distribution (DAgger style)."""
        x = self._encode_student(obs)
        self.update_distribution(x)
        return self.distribution.sample()

    def act_inference(self, obs):
        """Deterministic student action — used at deploy / play."""
        x = self._encode_student(obs)
        return self.student(x)

    def evaluate(self, obs):
        """Teacher action — the distillation label. Frozen, no grad."""
        teacher_obs = self._gather_teacher_state(obs)
        teacher_obs = self.teacher_obs_normalizer(teacher_obs)
        with torch.no_grad():
            return self.teacher(teacher_obs)

    # The parent's ``get_student_obs`` / ``get_teacher_obs`` are concat-only
    # 1-D paths; override so downstream consumers (storage bookkeeping etc.)
    # get the encoded student features and the teacher state respectively.
    def get_student_obs(self, obs):
        return self._encode_student(obs)

    def get_teacher_obs(self, obs):
        return self._gather_teacher_state(obs)

    def update_normalization(self, obs):
        if self.student_obs_normalization:
            self.student_obs_normalizer.update(self._gather_student_state(obs))

    # ``load_state_dict`` inherited from parent works as-is:
    #   • Detects PPO checkpoint via ``"actor."`` keys → loads into ``self.teacher``
    #   • Detects distillation checkpoint via ``"student."`` keys → loads both
    # Our PPO teacher (``model_700.pt`` from pickplace_bowl_teacher) has
    # ``actor.{0..6}.{weight,bias}`` keys which map to ``teacher.{0..6}.*`` —
    # architectures match (input dim 59 = policy(25)+critic(34), 3 hidden
    # layers [256,128,64] ELU, output 6).
