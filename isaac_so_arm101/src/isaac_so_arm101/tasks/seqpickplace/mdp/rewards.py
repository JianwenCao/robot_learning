# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the Eval-3 sequential pick-and-place task.

Each step has the same reward shape as Eval-1/Eval-2 (reach → lift →
transport → place → release) but indexed at the *current* target cube
(``env._target_cube_idx_per_step[:, env._seq_step_idx]``) and the
*current* target bowl (``cmd.current_target_bowl_xy()``). When the
release predicate fires for an env, :class:`SequentialGoalCommand` reads
``env._seq_step_release_indicator`` next ``_update_command`` call and
bumps ``env._seq_step_idx``, which automatically retargets every reward
term to the next sub-goal.

The 3-step success rate is tracked as a separate latch
``env._seq_success_per_step_latch`` (``(N, 3)`` bool) so TB metrics can
report per-step completion alongside the all-steps-done rate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from .events import COLOR_NAMES, N_GOAL_STEPS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_cube_pos_w(env: "ManagerBasedRLEnv", cube_prefix: str = "cube_") -> torch.Tensor:
    parts = []
    for name in COLOR_NAMES:
        cube: RigidObject = env.scene[f"{cube_prefix}{name}"]
        parts.append(cube.data.root_pos_w[:, :3])
    return torch.stack(parts, dim=1)


def _all_cube_lin_vel_w(env: "ManagerBasedRLEnv", cube_prefix: str = "cube_") -> torch.Tensor:
    parts = []
    for name in COLOR_NAMES:
        cube: RigidObject = env.scene[f"{cube_prefix}{name}"]
        parts.append(cube.data.root_lin_vel_w)
    return torch.stack(parts, dim=1)


def _current_target_palette_idx(env: "ManagerBasedRLEnv") -> torch.Tensor:
    step = env._seq_step_idx.clamp(max=N_GOAL_STEPS - 1)
    return env._target_cube_idx_per_step.gather(1, step.view(-1, 1)).squeeze(1)


def _current_target_pos_w(env: "ManagerBasedRLEnv", cube_prefix: str = "cube_") -> torch.Tensor:
    idx = _current_target_palette_idx(env)
    return _all_cube_pos_w(env, cube_prefix).gather(
        1, idx.view(-1, 1, 1).expand(-1, 1, 3)
    ).squeeze(1)


def _current_target_vel_w(env: "ManagerBasedRLEnv", cube_prefix: str = "cube_") -> torch.Tensor:
    idx = _current_target_palette_idx(env)
    return _all_cube_lin_vel_w(env, cube_prefix).gather(
        1, idx.view(-1, 1, 1).expand(-1, 1, 3)
    ).squeeze(1)


def _current_bowl_w(env: "ManagerBasedRLEnv", command_name: str = "seq_goal") -> torch.Tensor:
    """``(N, 2)`` current bowl xy in world frame."""
    robot: Articulation = env.scene["robot"]
    cmd = env.command_manager.get_term(command_name)
    bowl_b = cmd.current_target_bowl_xy()
    return robot.data.root_pos_w[:, :2] + bowl_b


def _lifted_mask(
    env: "ManagerBasedRLEnv", minimal_height: float = 0.025, cube_prefix: str = "cube_"
) -> torch.Tensor:
    z = _current_target_pos_w(env, cube_prefix)[:, 2]
    lifted_now = z > minimal_height
    if not hasattr(env, "_seq_was_grasped"):
        env._seq_was_grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._seq_was_grasped |= lifted_now
    return env._seq_was_grasped


def _over_bowl_high_mask(
    env: "ManagerBasedRLEnv",
    r_safe: float = 0.06,
    rim_clearance: float = 0.08,  # 2026-05-20: 0.12 → 0.08 (see Eval-1)
    command_name: str = "seq_goal",
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    target_w = _current_target_pos_w(env, cube_prefix)
    bowl_w = _current_bowl_w(env, command_name)
    over = (target_w[:, 2] > rim_clearance) & (
        torch.norm(target_w[:, :2] - bowl_w, dim=1) < r_safe
    )
    if not hasattr(env, "_seq_was_over_bowl_above_rim"):
        env._seq_was_over_bowl_above_rim = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._seq_was_over_bowl_above_rim |= over
    return env._seq_was_over_bowl_above_rim


# ---------------------------------------------------------------------------
# Step-active mask — kills rewards for envs that have already finished all 3.
# ---------------------------------------------------------------------------


def _step_active(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return (env._seq_step_idx < N_GOAL_STEPS).float()


# ---------------------------------------------------------------------------
# Reach / lift / transport / place / release — all current-target-aware
# ---------------------------------------------------------------------------


def reach_current_target(
    env: "ManagerBasedRLEnv",
    std: float = 0.05,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    target_w = _current_target_pos_w(env, cube_prefix)
    dist = torch.norm(target_w - ee_w, dim=1)
    return (1.0 - torch.tanh(dist / std)) * _step_active(env)


def lift_current_target(
    env: "ManagerBasedRLEnv",
    minimal_height: float = 0.07,
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    z = _current_target_pos_w(env, cube_prefix)[:, 2]
    return (z > minimal_height).float() * _step_active(env)


def transport_current_target_to_bowl(
    env: "ManagerBasedRLEnv",
    std: float = 0.30,
    minimal_height: float = 0.025,
    command_name: str = "seq_goal",
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    target_w = _current_target_pos_w(env, cube_prefix)
    bowl_w = _current_bowl_w(env, command_name)
    # 3-D distance: assume bowl z = 0 in robot frame → z in world frame
    # equals robot root z (≈ 0). Distance to a "lifted goal" was used in
    # Eval-1 (z=0.10); here we use the simpler xy distance which still
    # closes the post-grasp transport gradient.
    dist = torch.norm(target_w[:, :2] - bowl_w, dim=1)
    was_lifted = _lifted_mask(env, minimal_height, cube_prefix).float()
    return was_lifted * (1.0 - torch.tanh(dist / std)) * _step_active(env)


def release_current_target_in_bowl(
    env: "ManagerBasedRLEnv",
    r_safe: float = 0.06,
    bowl_height: float = 0.06,
    gripper_open_threshold: float = 0.2,
    block_speed_threshold: float = 0.05,
    minimal_height: float = 0.07,
    rim_clearance: float = 0.08,  # 2026-05-20: 0.12 → 0.08
    command_name: str = "seq_goal",
    gripper_joint_name: str = "gripper",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """Fires +1 the step the current target is released in the current bowl.

    Side effect: writes ``env._seq_step_release_indicator`` so the
    :class:`SequentialGoalCommand` can advance the step counter. Also
    writes into ``env._seq_success_per_step_latch[:, current_step]`` so
    per-step success rates can be read at episode end.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    target_w = _current_target_pos_w(env, cube_prefix)
    bowl_w = _current_bowl_w(env, command_name)
    in_xy = torch.norm(target_w[:, :2] - bowl_w, dim=1) < r_safe
    low = target_w[:, 2] < bowl_height

    gripper_idx = robot.find_joints(gripper_joint_name)[0][0]
    gripper_q = robot.data.joint_pos[:, gripper_idx]
    opened = gripper_q > gripper_open_threshold

    settled = torch.norm(_current_target_vel_w(env, cube_prefix), dim=1) < block_speed_threshold

    was_lifted = _lifted_mask(env, minimal_height, cube_prefix)
    was_over_high = _over_bowl_high_mask(
        env, r_safe=r_safe, rim_clearance=rim_clearance,
        command_name=command_name, cube_prefix=cube_prefix,
    )

    indicator = in_xy & low & opened & settled & was_lifted & was_over_high
    step_active = env._seq_step_idx < N_GOAL_STEPS
    indicator = indicator & step_active

    # Write per-step success latch BEFORE writing the advance indicator
    # (read by the command in the next _update_command call).
    if not hasattr(env, "_seq_success_per_step_latch"):
        env._seq_success_per_step_latch = torch.zeros(
            (env.num_envs, N_GOAL_STEPS), dtype=torch.bool, device=env.device
        )
    if indicator.any():
        rows = torch.where(indicator)[0]
        cols = env._seq_step_idx[rows]
        env._seq_success_per_step_latch[rows, cols] = True

    # PDF-strict per-step latch: "correct block placed in the
    # corresponding bowl AND released" — minimal gate (in_xy ∧ low ∧
    # opened) without the safety latches / ``settled``. Logged in
    # parallel as ``success_rate_strict`` so we can tell whether the
    # conservative gate is biasing the headline ``all_steps_success``
    # low. Step-advance (``_seq_step_release_indicator``) still uses
    # the strict-AND-safety gate so the command term doesn't advance
    # on a still-spinning unstable release.
    strict_step_indicator = in_xy & low & opened & step_active
    if not hasattr(env, "_seq_success_per_step_latch_strict"):
        env._seq_success_per_step_latch_strict = torch.zeros(
            (env.num_envs, N_GOAL_STEPS), dtype=torch.bool, device=env.device
        )
    if strict_step_indicator.any():
        rows = torch.where(strict_step_indicator)[0]
        cols = env._seq_step_idx[rows]
        env._seq_success_per_step_latch_strict[rows, cols] = True

    if not hasattr(env, "_seq_step_release_indicator"):
        env._seq_step_release_indicator = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    env._seq_step_release_indicator |= indicator
    return indicator.float()


# ---------------------------------------------------------------------------
# Step bonus — fixed bonus per completed step (independent of dense terms).
# ---------------------------------------------------------------------------


def step_completion_bonus(
    env: "ManagerBasedRLEnv",
    weight_per_step: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Fires once when each sub-goal completes, with a per-step weight.

    The Eval-3 scoring is 4/4/2 for the three steps; if we want the
    RL signal to track grading, set ``weight_per_step=(4.0, 4.0, 2.0)``
    and weight=1.0 in cfg. Defaults to 1/1/1 so cfg weight controls the
    total scale.
    """
    ind = getattr(env, "_seq_step_release_indicator", None)
    if ind is None:
        return torch.zeros(env.num_envs, device=env.device)
    rewards = torch.tensor(weight_per_step, device=env.device, dtype=torch.float32)
    step_clamped = env._seq_step_idx.clamp(max=N_GOAL_STEPS - 1)
    per_env_w = rewards[step_clamped]
    return ind.float() * per_env_w


# ---------------------------------------------------------------------------
# Wrong-cube-in-bowl penalty
# ---------------------------------------------------------------------------


def wrong_cube_in_current_bowl(
    env: "ManagerBasedRLEnv",
    r_safe: float = 0.06,
    bowl_height: float = 0.06,
    command_name: str = "seq_goal",
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """Pays 1.0 (per cube, summed) for any *non-current-target* cube that
    sits inside the current bowl.

    Sized to discourage dumping the wrong block into the active bowl —
    that would block the current step's target from going in.
    """
    all_pos = _all_cube_pos_w(env, cube_prefix)  # (N, 6, 3)
    bowl_w = _current_bowl_w(env, command_name).unsqueeze(1)  # (N, 1, 2)
    in_xy = torch.norm(all_pos[:, :, :2] - bowl_w, dim=2) < r_safe  # (N, 6)
    low = all_pos[:, :, 2] < bowl_height
    # mask out the current target
    target_idx = _current_target_palette_idx(env)  # (N,)
    target_mask = torch.zeros_like(in_xy)
    target_mask.scatter_(1, target_idx.view(-1, 1), True)
    wrong = in_xy & low & ~target_mask
    return wrong.float().sum(dim=1) * _step_active(env)


# ---------------------------------------------------------------------------
# Curriculum metrics — per-step + all-steps success rate.
# ---------------------------------------------------------------------------


def log_seq_success_metrics(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
) -> dict[str, float]:
    latch = getattr(env, "_seq_success_per_step_latch", None)
    if latch is None or env_ids is None:
        return {}
    if isinstance(env_ids, torch.Tensor):
        if env_ids.numel() == 0:
            return {}
    elif len(env_ids) == 0:
        return {}

    outcomes = latch[env_ids].float()  # (n_ended, 3)
    metrics: dict[str, float] = {
        "step0_success": outcomes[:, 0].mean().item(),
        "step1_success": outcomes[:, 1].mean().item(),
        "step2_success": outcomes[:, 2].mean().item(),
        "all_steps_success": (outcomes.sum(dim=1) == N_GOAL_STEPS).float().mean().item(),
        "n_episodes_ended": float(outcomes.shape[0]),
    }
    latch[env_ids] = False

    # PDF-strict counterparts (in_xy ∧ low ∧ opened, no safety latches /
    # no settled). See :func:`release_current_target_in_bowl` for why we
    # log this in parallel.
    strict_latch = getattr(env, "_seq_success_per_step_latch_strict", None)
    if strict_latch is not None:
        strict_out = strict_latch[env_ids].float()
        metrics["step0_success_strict"] = strict_out[:, 0].mean().item()
        metrics["step1_success_strict"] = strict_out[:, 1].mean().item()
        metrics["step2_success_strict"] = strict_out[:, 2].mean().item()
        metrics["success_rate_strict"] = (strict_out.sum(dim=1) == N_GOAL_STEPS).float().mean().item()
        strict_latch[env_ids] = False

    return metrics
