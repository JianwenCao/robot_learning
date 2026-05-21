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
from rsl_rl.algorithms import Distillation

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)

from .vision_student_teacher import PickPlaceVisionStudentTeacher


class PickPlaceBCDistillation(Distillation):
    """Distillation with selectable rollout policy.

    ``rollout_policy="student"`` keeps RSL-RL's default DAgger-style
    behavior: student actions step the env, teacher actions are labels.
    ``rollout_policy="teacher"`` runs simple behavior cloning rollouts:
    teacher actions step the env and the student regresses to those same
    teacher actions.
    """

    def __init__(self, *args, rollout_policy: str = "student", **kwargs):
        super().__init__(*args, **kwargs)
        if rollout_policy not in {"student", "teacher"}:
            raise ValueError(
                f"rollout_policy={rollout_policy!r}; expected 'student' or 'teacher'"
            )
        self.rollout_policy = rollout_policy
        self.cnn_input_dump_dir: str | None = None
        self.cnn_input_dump_interval = 200
        self.cnn_input_dump_max_envs = 4
        self.cnn_input_dump_env = None
        self._cnn_input_dump_step = 0

    def _maybe_dump_cnn_inputs(self, obs) -> None:
        if not self.cnn_input_dump_dir:
            return
        if self._cnn_input_dump_step % max(int(self.cnn_input_dump_interval), 1) != 0:
            self._cnn_input_dump_step += 1
            return
        if "wrist_image" not in obs:
            self._cnn_input_dump_step += 1
            return

        import os
        import json
        import cv2
        import numpy as np

        os.makedirs(self.cnn_input_dump_dir, exist_ok=True)
        img = obs["wrist_image"].detach()
        n = min(int(self.cnn_input_dump_max_envs), int(img.shape[0]))
        for env_i in range(n):
            frame = img[env_i]
            if frame.shape[0] < 3:
                continue
            rgb = frame[:3].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
            rgb_u8 = (rgb * 255.0).round().astype(np.uint8)
            path = os.path.join(
                self.cnn_input_dump_dir,
                f"step_{self._cnn_input_dump_step:06d}_env_{env_i:03d}.png",
            )
            cv2.imwrite(path, cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
        env = self.cnn_input_dump_env
        if env is not None:
            meta_path = os.path.join(self.cnn_input_dump_dir, "metadata.jsonl")
            record = {"step": int(self._cnn_input_dump_step), "envs": []}
            for env_i in range(n):
                item = {"env": int(env_i)}
                for attr, key in (
                    ("_vision_block_rgb", "cube_rgb"),
                    ("_vision_table_rgb", "table_rgb"),
                    ("_vision_background_rgb", "background_rgb"),
                    ("_vision_robot_rgb", "robot_mean_rgb"),
                ):
                    value = getattr(env.unwrapped if hasattr(env, "unwrapped") else env, attr, None)
                    if value is not None:
                        item[key] = [float(x) for x in value[env_i].detach().cpu().tolist()]
                record["envs"].append(item)
            with open(meta_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        self._cnn_input_dump_step += 1

    def act(self, obs):
        self._maybe_dump_cnn_inputs(obs)
        student_actions = self.policy.act(obs).detach()
        teacher_actions = self.policy.evaluate(obs).detach()
        self.transition.actions = (
            teacher_actions if self.rollout_policy == "teacher" else student_actions
        )
        self.transition.privileged_actions = teacher_actions
        self.transition.observations = obs
        return self.transition.actions


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
    setattr(
        _distillation_runner,
        PickPlaceBCDistillation.__name__,
        PickPlaceBCDistillation,
    )


_register_class()


@configclass
class PickPlaceDistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    class_name: str = "PickPlaceBCDistillation"
    rollout_policy: str = "student"


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

    # Student reads deployable obs (proprio + RGB image); teacher reads the
    # same 27-D StateAprilTag policy vector used by the robust teacher
    # checkpoint.
    obs_groups = {
        "policy": ["policy", "wrist_image"],
        "teacher": ["state_apriltag_policy"],
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
    algorithm = PickPlaceDistillationAlgorithmCfg(
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
