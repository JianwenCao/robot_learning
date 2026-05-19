# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Target-aware reward terms for the clutter pick-and-place task.

Mirrors the structure of :mod:`isaac_so_arm101.tasks.pickplace.mdp.rewards`
but indexes everything by the per-env target cube index
(``env._target_cube_idx``) instead of a fixed "object" scene entity. Adds
two distractor-aware penalty terms that the single-cube task didn't need:

* :func:`distractor_disturb_penalty` — penalizes pushing the wrong cube
  outside the original cluster footprint (proxy for "don't disturb").
* :func:`wrong_block_in_bowl` — penalizes ending up with the distractor
  in the bowl (a sometimes-found shortcut where the policy grasps any
  cube and dumps it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from .events import COLOR_NAMES

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


def _target_pos_w(env: "ManagerBasedRLEnv", cube_prefix: str = "cube_") -> torch.Tensor:
    all_pos = _all_cube_pos_w(env, cube_prefix)
    idx = env._target_cube_idx
    return all_pos.gather(1, idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)


def _target_lin_vel_w(env: "ManagerBasedRLEnv", cube_prefix: str = "cube_") -> torch.Tensor:
    all_vel = _all_cube_lin_vel_w(env, cube_prefix)
    idx = env._target_cube_idx
    return all_vel.gather(1, idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)


def _bowl_xy_w(env: "ManagerBasedRLEnv", command_name: str) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    bowl_b = env.command_manager.get_command(command_name)[:, :2]
    return robot.data.root_pos_w[:, :2] + bowl_b


# Per-episode latches against the TARGET cube — separate from the
# Eval-1 ``env._was_grasped`` etc. so the two tasks don't stomp on each
# other if loaded together.

def _target_lifted_mask(
    env: "ManagerBasedRLEnv",
    minimal_height: float = 0.025,
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """OR-latched "target was lifted above ``minimal_height`` at any prior step"."""
    target_z = _target_pos_w(env, cube_prefix)[:, 2]
    lifted_now = target_z > minimal_height
    if not hasattr(env, "_target_was_grasped"):
        env._target_was_grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._target_was_grasped |= lifted_now
    return env._target_was_grasped


def _target_over_bowl_high_mask(
    env: "ManagerBasedRLEnv",
    r_safe: float = 0.06,
    rim_clearance: float = 0.08,
    command_name: str = "bowl_pose",
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """OR-latched "target was above ``rim_clearance`` AND over bowl xy"."""
    target_w = _target_pos_w(env, cube_prefix)
    bowl_w = _bowl_xy_w(env, command_name)
    over_bowl_high = (target_w[:, 2] > rim_clearance) & (
        torch.norm(target_w[:, :2] - bowl_w, dim=1) < r_safe
    )
    if not hasattr(env, "_target_was_over_bowl_above_rim"):
        env._target_was_over_bowl_above_rim = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._target_was_over_bowl_above_rim |= over_bowl_high
    return env._target_was_over_bowl_above_rim


# ---------------------------------------------------------------------------
# Reach (dense, ungated) — pull EE toward the *target* cube.
# ---------------------------------------------------------------------------


def reach_target_block(
    env: "ManagerBasedRLEnv",
    std: float = 0.05,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """Dense reach reward against the target cube: ``1 - tanh(d/std)``."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    target_w = _target_pos_w(env, cube_prefix)
    dist = torch.norm(target_w - ee_w, dim=1)
    return 1.0 - torch.tanh(dist / std)


# ---------------------------------------------------------------------------
# Lift (sparse indicator on the target cube being above a threshold).
# ---------------------------------------------------------------------------


def target_grasp_event(
    env: "ManagerBasedRLEnv",
    minimal_height: float = 0.07,
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """Indicator the *target* cube is currently above ``minimal_height``."""
    target_z = _target_pos_w(env, cube_prefix)[:, 2]
    return (target_z > minimal_height).float()


# ---------------------------------------------------------------------------
# Transport (dense target→bowl, gated on the per-episode lift latch).
# ---------------------------------------------------------------------------


def target_transport_to_bowl(
    env: "ManagerBasedRLEnv",
    std: float = 0.30,
    minimal_height: float = 0.025,
    command_name: str = "bowl_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """Dense ``target → goal_xyz`` distance, gated on episode lift latch."""
    robot: Articulation = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_b = command[:, :3]
    goal_pos_w = robot.data.root_pos_w + goal_pos_b
    target_w = _target_pos_w(env, cube_prefix)
    distance = torch.norm(goal_pos_w - target_w, dim=1)
    was_lifted = _target_lifted_mask(env, minimal_height, cube_prefix)
    return was_lifted.float() * (1.0 - torch.tanh(distance / std))


# ---------------------------------------------------------------------------
# Place (binary: target inside bowl footprint AND below rim, gated on both
# per-episode latches).
# ---------------------------------------------------------------------------


def target_in_bowl(
    env: "ManagerBasedRLEnv",
    r_safe: float = 0.06,
    bowl_height: float = 0.08,
    minimal_height: float = 0.025,
    rim_clearance: float = 0.08,
    command_name: str = "bowl_pose",
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """1.0 when the *target* cube is inside the bowl AND the two safety
    latches have fired (was lifted + was over bowl from above).
    """
    target_w = _target_pos_w(env, cube_prefix)
    bowl_w = _bowl_xy_w(env, command_name)
    in_xy = torch.norm(target_w[:, :2] - bowl_w, dim=1) < r_safe
    low = target_w[:, 2] < bowl_height
    was_lifted = _target_lifted_mask(env, minimal_height, cube_prefix)
    was_over_high = _target_over_bowl_high_mask(
        env, r_safe=r_safe, rim_clearance=rim_clearance,
        command_name=command_name, cube_prefix=cube_prefix,
    )
    return (in_xy & low & was_lifted & was_over_high).float()


# ---------------------------------------------------------------------------
# Release (place + gripper open + target settled — fires once committed).
# ---------------------------------------------------------------------------


def release_target_in_bowl(
    env: "ManagerBasedRLEnv",
    r_safe: float = 0.06,
    bowl_height: float = 0.06,
    gripper_open_threshold: float = 0.2,
    block_speed_threshold: float = 0.05,
    minimal_height: float = 0.07,
    rim_clearance: float = 0.08,
    command_name: str = "bowl_pose",
    gripper_joint_name: str = "gripper",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """Mirrors :func:`pickplace.mdp.rewards.release_in_bowl` against the target.

    Also OR-latches ``env._target_task_success_latch`` so the per-episode
    success rate can be read by :func:`log_target_success_metrics` from
    the curriculum manager. Distinct latch from Eval-1's so the two
    tasks coexist cleanly when loaded in the same Python process.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    target_w = _target_pos_w(env, cube_prefix)
    bowl_w = _bowl_xy_w(env, command_name)
    in_xy = torch.norm(target_w[:, :2] - bowl_w, dim=1) < r_safe
    low = target_w[:, 2] < bowl_height

    gripper_idx = robot.find_joints(gripper_joint_name)[0][0]
    gripper_q = robot.data.joint_pos[:, gripper_idx]
    opened = gripper_q > gripper_open_threshold

    settled = torch.norm(_target_lin_vel_w(env, cube_prefix), dim=1) < block_speed_threshold

    was_lifted = _target_lifted_mask(env, minimal_height, cube_prefix)
    was_over_high = _target_over_bowl_high_mask(
        env, r_safe=r_safe, rim_clearance=rim_clearance,
        command_name=command_name, cube_prefix=cube_prefix,
    )

    indicator = in_xy & low & opened & settled & was_lifted & was_over_high
    if not hasattr(env, "_target_task_success_latch"):
        env._target_task_success_latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._target_task_success_latch |= indicator
    return indicator.float()


# ---------------------------------------------------------------------------
# Distractor-aware penalties (Eval-2 specific)
# ---------------------------------------------------------------------------


def distractor_disturb_penalty(
    env: "ManagerBasedRLEnv",
    threshold_speed: float = 0.05,
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """Penalty when the *distractor* cube (the non-target member of the
    active pair) is moving — proxy for "knocking over" or "pushing".

    The policy is allowed to brush the distractor at low speed but
    shoving it hard incurs a penalty (set ``weight < 0`` in cfg). Compute
    by gathering the distractor's linear speed per env.
    """
    cmd = env.command_manager.get_term("target_color")
    distractor_idx = cmd.active_indices.gather(
        1, (1 - cmd.target_idx_in_pair).view(-1, 1)
    ).squeeze(1)
    all_vel = _all_cube_lin_vel_w(env, cube_prefix)
    vel = all_vel.gather(1, distractor_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
    speed = torch.norm(vel, dim=1)
    # 0 below threshold, linear in (speed - threshold) above. Bounded at
    # 1 so a violent push doesn't dominate the reward.
    return ((speed - threshold_speed).clamp(min=0.0) / threshold_speed).clamp(max=1.0)


def wrong_block_in_bowl(
    env: "ManagerBasedRLEnv",
    r_safe: float = 0.06,
    bowl_height: float = 0.06,
    minimal_height: float = 0.025,
    command_name: str = "bowl_pose",
    cube_prefix: str = "cube_",
) -> torch.Tensor:
    """Penalty when the *distractor* cube ends up in the bowl footprint.

    Pays 1.0 (multiplied by ``weight < 0`` in cfg) when the distractor
    is inside the bowl xy and below ``bowl_height``. Mirrors
    :func:`target_in_bowl` for the wrong asset — sharply discourages
    the "grasp any cube, hope it's right" shortcut.
    """
    cmd = env.command_manager.get_term("target_color")
    distractor_idx = cmd.active_indices.gather(
        1, (1 - cmd.target_idx_in_pair).view(-1, 1)
    ).squeeze(1)
    all_pos = _all_cube_pos_w(env, cube_prefix)
    dist_w = all_pos.gather(1, distractor_idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
    bowl_w = _bowl_xy_w(env, command_name)
    in_xy = torch.norm(dist_w[:, :2] - bowl_w, dim=1) < r_safe
    low = dist_w[:, 2] < bowl_height
    return (in_xy & low).float()


# ---------------------------------------------------------------------------
# Curriculum metric — TB-logged binary success rate against the target cube.
# ---------------------------------------------------------------------------


def log_target_success_metrics(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
) -> dict[str, float]:
    """Per-episode target-success rate for TB. Same pattern as Eval-1's
    :func:`log_success_metrics`, reading ``env._target_task_success_latch``.
    """
    latch = getattr(env, "_target_task_success_latch", None)
    if latch is None or env_ids is None:
        return {}
    if isinstance(env_ids, torch.Tensor):
        if env_ids.numel() == 0:
            return {}
    elif len(env_ids) == 0:
        return {}

    outcomes = latch[env_ids].float()
    success_rate = outcomes.mean().item()
    n_ended = int(outcomes.numel())
    latch[env_ids] = False
    return {"success_rate": success_rate, "n_episodes_ended": float(n_ended)}
