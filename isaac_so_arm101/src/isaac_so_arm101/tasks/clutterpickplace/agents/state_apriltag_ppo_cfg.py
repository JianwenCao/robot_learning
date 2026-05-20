# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO config for the SO-ARM101 Eval-2 state-only + AprilTag deploy path.

Single-stage from-scratch PPO on the camera-free env
:class:`SoArm101ClutterPickPlaceStateAprilTagEnvCfg`. The actor sees the
deployable ``policy`` group (proprio + bowl_xy + per-cube noisy xy +
visibility flags) AND the ``goal`` group (target color one-hot). The
critic additionally sees the full privileged ``critic`` group
(GT target/distractor positions, ee→target, target_is_grasped). See
``docs/STATE_APRILTAG_PLAN.md`` §6/§7 for the deploy-side mirror.

Reuses :class:`PickPlaceVisionActorCritic` — that class auto-disables the
CNN when ``wrist_image`` isn't in ``obs_groups``, so this cfg yields a
plain MLP actor-critic with FiLM-free goal conditioning (goal one-hot
just concatenates into the state vector like every other proprio term).
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
    setattr(
        _on_policy_runner,
        PickPlaceVisionActorCritic.__name__,
        PickPlaceVisionActorCritic,
    )


_register_class()


@configclass
class ClutterPickPlaceStateAprilTagPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for the Eval-2 state-only + AprilTag-noise path.

    Hyperparameters mirror :class:`ClutterPickPlaceTeacherPPORunnerCfg`
    (same MDP shape, just a wider obs vector: +12 for cube positions,
    +6 for visibility flags). Same per-dim init noise + std cap to keep
    the binary gripper decisive while the arm explores.
    """

    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 50
    experiment_name = "clutterpickplace_state_apriltag"
    empirical_normalization = False

    # Asymmetric A-C: actor sees deployable (policy) + goal one-hot;
    # critic additionally sees privileged GT cube positions / distances /
    # grasp flag for stable value estimation.
    obs_groups = {
        "policy": ["policy", "goal"],
        "critic": ["policy", "goal", "critic"],
    }

    policy = RslRlPpoActorCriticCfg(
        class_name="PickPlaceVisionActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        # Tightened to 0.005 like the Eval-2 teacher cfg — wider obs +
        # wrong_block_in_bowl penalty make value gradient noisier than
        # stock; loose desired_kl let σ blow up in the first Eval-2 run.
        desired_kl=0.005,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        # Per-dim init: arm dims at σ=1.0 for reach exploration; gripper
        # at σ=0.1 so binary closure stays decisive. Cap gripper σ at
        # 0.2 so the entropy bonus can't inflate it past the binary
        # threshold. Same recipe Eval-2 teacher uses.
        self.policy.init_noise_std = [1.0, 1.0, 1.0, 1.0, 1.0, 0.1]
        self.policy.std_max = [1e3, 1e3, 1e3, 1e3, 1e3, 0.2]
