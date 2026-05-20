# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO config for the SO-ARM101 Eval-2 state-only + AprilTag deploy path.

Single-stage from-scratch PPO on the camera-free env
:class:`SoArm101ClutterPickPlaceStateAprilTagEnvCfg`. The actor sees the
deployable target-only ``policy`` group (proprio + bowl_xy + target-cube
noisy xy), matching Eval-1's 27-D actor input. The critic additionally sees
the target color goal and full privileged ``critic`` group (GT
target/distractor positions, ee→target, target_is_grasped).

Reuses :class:`PickPlaceVisionActorCritic` — that class auto-disables the
CNN when ``wrist_image`` isn't in ``obs_groups``, so this cfg yields a
plain MLP actor-critic. Goal one-hot is critic-only in the state-AprilTag
path; the deploy actor is color-blind and keyed externally by AprilTag ID.
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

    Hyperparameters mirror Eval-1's state-AprilTag runner. The actor obs
    shape is intentionally identical (27-D target-only stream); the critic
    is wider because it receives privileged target/distractor state.
    """

    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 50
    experiment_name = "clutterpickplace_state_apriltag"
    empirical_normalization = False

    # Asymmetric A-C: actor sees only the deployable policy stream
    # (27-D = base + target_cube_pos_xy_noisy; color-blind per
    # EVAL2_PLAN.md §2). The target is keyed externally by re-keying
    # the AprilTag detector ID at deploy, so the policy never needs the
    # ``target_color_onehot`` goal vector. Critic keeps "goal" for the
    # extra inductive bias; it's privileged and only used at training.
    obs_groups = {
        "policy": ["policy"],
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
        # Back to 0.01 (matches Eval-1's working baseline) after Eval-2 v3
        # stalled at iter 770 with reach=0.07, lift=0.05, release=0.0.
        # The 0.005 over-tightening (chosen in v2 to prevent σ inflation)
        # throttled PPO's per-iter learning to the point that the policy
        # couldn't escape the "stay still" local optimum. The per-dim
        # ``std_max=0.2`` gripper cap below already prevents the binary-
        # gripper σ blowup that originally motivated tightening desired_kl,
        # so we can safely loosen the trust region back to stock.
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    def __post_init__(self):
        # Per-dim init: arm dims at σ=1.0 for reach exploration; gripper
        # at σ=0.1 so binary closure stays decisive. Cap gripper σ at
        # 0.2 so the entropy bonus can't inflate it past the binary
        # threshold. Same recipe Eval-2 teacher uses.
        self.policy.init_noise_std = [1.0, 1.0, 1.0, 1.0, 1.0, 0.1]
        self.policy.std_max = [1e3, 1e3, 1e3, 1e3, 1e3, 0.2]
