# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset event terms for the pick-and-place task.

This module currently hosts one specialty event:

* :func:`init_block_in_gripper` — a "grasped-init" bootstrap that, with
  configurable probability, places the block inside the gripper jaws at
  episode start (and pre-closes the gripper around it). It exists as a
  fallback for when state-only PPO from scratch can't discover grasping
  through random exploration alone — by mixing a fraction of episodes
  that *start* grasped, the transport / place / release reward stages
  receive gradient signal early. A curriculum can then ramp the
  bootstrap probability down to 0 once those stages are competent.

Design note: this event must run **after** the default
``reset_root_state_uniform`` (block xy randomization) so it can override
that pose. The ordering is determined by the order in
``EventCfg`` — list this term last in the cfg.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# End-effector position in the robot root frame at home pose with the
# overridden ``gripper=0.5`` open joint. Read once via
# ``scripts/probe_home_pose.py`` (which steps physics one frame so body
# poses are populated) and hardcoded here. The FrameTransformer's
# ``target_pos_w`` is *stale* during reset events because the manager
# applies events before the next physics step, so we cannot read it
# dynamically.
EE_HOME_B: tuple[float, float, float] = (0.2421, -0.0007, 0.0829)


def init_block_in_gripper(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    p_grasped: float = 0.5,
    block_offset_xyz: tuple[float, float, float] = (0.0, 0.0, -0.005),
    gripper_closed_q: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_joint_name: str = "gripper",
):
    """Bootstrap a Bernoulli(``p_grasped``) subset of ``env_ids`` with the
    block already inside the gripper and the jaws pre-closed.

    The block is teleported to the end-effector home position (read from
    :data:`EE_HOME_B` in the robot root frame), offset by ``block_offset_xyz``
    so it sits between the jaws rather than at the ee origin. The gripper
    joint is set to ``gripper_closed_q`` (slightly above 0 so the cube
    isn't penetrated by the moving jaw).

    Curriculum is handled outside this function via a curriculum term that
    decays ``p_grasped`` over training (e.g. 1.0 → 0.0 over 5000 steps).

    Side effect: maintains a per-env ``env._is_bootstrapped`` bool tensor
    flagging which envs received the bootstrap on their *current* episode.
    Reward / metric functions use this to track success rates split by
    bootstrap status — see :func:`mdp.rewards.log_bootstrap_metrics`.
    """
    # Lazy-init the per-env bootstrap flag. Lives on the env instance
    # (not in scene.extras, which gets clobbered by physics state). Reset
    # for every env in env_ids — we'll re-set True for those that win
    # the Bernoulli toss below.
    if not hasattr(env, "_is_bootstrapped"):
        env._is_bootstrapped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if isinstance(env_ids, torch.Tensor):
        n = env_ids.numel()
    else:
        n = len(env_ids)
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if n == 0:
        return
    env._is_bootstrapped[env_ids] = False

    # Decide which envs get the bootstrap this reset.
    if p_grasped <= 0.0:
        return
    coin = torch.rand(n, device=env.device) < p_grasped
    if not torch.any(coin):
        return
    bootstrap_ids = env_ids[coin]
    env._is_bootstrapped[bootstrap_ids] = True
    n_boot = bootstrap_ids.numel()

    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # Block goes at EE_HOME (in robot frame) + offset, transformed to world.
    ee_home_b = torch.tensor(EE_HOME_B, device=env.device, dtype=torch.float32)
    offset = torch.tensor(block_offset_xyz, device=env.device, dtype=torch.float32)
    target_b = ee_home_b + offset  # (3,) in robot frame
    root_pos_w = robot.data.root_pos_w[bootstrap_ids]  # (n_boot, 3)
    block_pos_w = root_pos_w + target_b.unsqueeze(0)

    # Identity quaternion (w, x, y, z) for the block — orientation isn't
    # critical because the cube is symmetric.
    quat = torch.zeros((n_boot, 4), device=env.device)
    quat[:, 0] = 1.0
    pose = torch.cat([block_pos_w, quat], dim=1)
    obj.write_root_pose_to_sim(pose, env_ids=bootstrap_ids)

    zero_vel = torch.zeros((n_boot, 6), device=env.device)
    obj.write_root_velocity_to_sim(zero_vel, env_ids=bootstrap_ids)

    # Pre-close the gripper around the block.
    gripper_idx = robot.find_joints(gripper_joint_name)[0][0]
    new_q = robot.data.joint_pos[bootstrap_ids].clone()
    new_q[:, gripper_idx] = gripper_closed_q
    new_qvel = robot.data.joint_vel[bootstrap_ids].clone()
    new_qvel[:, gripper_idx] = 0.0
    robot.write_joint_state_to_sim(new_q, new_qvel, env_ids=bootstrap_ids)


def decay_p_grasped(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,  # unused — curriculum terms ignore env_ids
    initial: float = 0.8,
    final: float = 0.0,
    warmup_steps: int = 12_000,
    decay_steps: int = 180_000,
    event_term_name: str = "bootstrap_grasped",
) -> dict[str, float]:
    """Linearly decay :func:`init_block_in_gripper`'s ``p_grasped`` parameter.

    Plugged in as a :class:`CurriculumTermCfg`. Reads the global step counter
    (``env.common_step_counter``, which increments by 1 per ``env.step()``
    call regardless of ``num_envs``) and overwrites the
    ``bootstrap_grasped`` event term's ``p_grasped`` on the live cfg.

    Schedule:

        step <= warmup_steps         : p = initial
        warmup < step <= warmup+decay: p = lerp(initial, final, t)
        step > warmup_steps + decay  : p = final

    Defaults: warmup 12 000 step() calls (~ 500 PPO iters at
    num_steps_per_env=24), decay over the next 180 000 (~ 7500 PPO iters).
    The wide decay range (vs the original 60k) was needed because shorter
    decays caused the policy to ride the bootstrap and never learn
    grasp-from-scratch — once p hit 0, reward collapsed.
    Tune via the curriculum cfg's ``params``.

    Returns the new ``p_grasped`` so it shows up in
    ``Curriculum/p_grasped`` on TB for at-a-glance monitoring.
    """
    step = int(env.common_step_counter)
    if step <= warmup_steps:
        p = initial
    elif step >= warmup_steps + decay_steps:
        p = final
    else:
        frac = (step - warmup_steps) / max(decay_steps, 1)
        p = initial + (final - initial) * frac

    # Tolerate missing event term — happens in PLAY mode where bootstrap_grasped
    # is intentionally not registered (we want the pure task at eval). Silently
    # no-op the cfg write; the returned scalar still appears in TB.
    try:
        term_cfg = env.event_manager.get_term_cfg(event_term_name)
        term_cfg.params["p_grasped"] = float(p)
        env.event_manager.set_term_cfg(event_term_name, term_cfg)
    except (ValueError, KeyError):
        pass
    return {"p_grasped": float(p)}


def expand_block_xy_range(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,  # unused — curriculum terms ignore env_ids
    initial_xy: tuple[float, float] = (0.02, 0.02),
    final_xy: tuple[float, float] = (0.10, 0.15),
    warmup_steps: int = 12_000,
    expand_steps: int = 180_000,
    event_term_name: str = "reset_block_position",
) -> dict[str, float]:
    """Linearly expand the block reset's xy randomization radius.

    Plugged in as a :class:`CurriculumTermCfg`. Modifies the live
    ``pose_range`` parameter of an event term that uses
    :func:`isaaclab.envs.mdp.reset_root_state_uniform`.

    The default schedule (warmup_steps=12 000, expand_steps=180 000) holds
    the block in a tight ±2 cm × ±2 cm patch directly under the EE home
    pose for the first 500 PPO iters (at 1024 envs × 24 steps; halve for
    2048 envs). The block is always inside the wrist camera's FoV at home
    pose during this phase, so the CNN sees the cube in every from-scratch
    reset and the RL signal for visual reach is dense. The range then
    expands linearly toward ±10 cm × ±15 cm over the next 7500 iters,
    matching the old (full-task) randomization at curriculum end.

    This replaces the bootstrap-grasp curriculum (decay_p_grasped). That
    curriculum subsidized the *outcome* (block in hand) rather than the
    *pose for grasping* — runs 1-5 showed the policy never learned visual
    reach because 80% of episodes started bootstrapped. The pre-grasp
    geometry curriculum here gives dense gradient on the reach+grasp step
    from iter 0.

    Returns a dict so the value shows up as ``Curriculum/<term>/<key>``
    on TB for live monitoring.
    """
    step = int(env.common_step_counter)
    if step <= warmup_steps:
        frac = 0.0
    elif step >= warmup_steps + expand_steps:
        frac = 1.0
    else:
        frac = (step - warmup_steps) / max(expand_steps, 1)
    rx = initial_xy[0] + (final_xy[0] - initial_xy[0]) * frac
    ry = initial_xy[1] + (final_xy[1] - initial_xy[1]) * frac

    term_cfg = env.event_manager.get_term_cfg(event_term_name)
    term_cfg.params["pose_range"]["x"] = (-rx, rx)
    term_cfg.params["pose_range"]["y"] = (-ry, ry)
    env.event_manager.set_term_cfg(event_term_name, term_cfg)
    return {"x_radius": float(rx), "y_radius": float(ry)}


def reset_was_grasped(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
) -> None:
    """Clear the per-episode lift latch maintained by
    :func:`mdp.rewards._episode_lifted_mask`.

    Wired as an :class:`EventTerm` with ``mode="reset"`` so the latch is
    zeroed for the resetting envs at the start of each new episode. Without
    this term the latch would stay True for the rest of the run after the
    first lift, which would defeat the gating on
    :func:`mdp.rewards.place_in_bowl` / :func:`mdp.rewards.release_in_bowl`
    (the second-and-later episodes of an env would farm those rewards
    without lifting again).

    Idempotent: a no-op if the latch buffer hasn't been allocated yet
    (first reset, before any reward term has touched it).
    """
    flag = getattr(env, "_was_grasped", None)
    if flag is None:
        return
    if isinstance(env_ids, torch.Tensor):
        if env_ids.numel() == 0:
            return
    elif len(env_ids) == 0:
        return
    flag[env_ids] = False


def reset_was_over_bowl_above_rim(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
) -> None:
    """Clear the per-episode over-bowl-above-rim latch maintained by
    :func:`mdp.rewards._episode_over_bowl_high_mask`.

    Same pattern as :func:`reset_was_grasped` — wired as a reset event so
    the latch is fresh per episode. Idempotent: no-op if the latch buffer
    hasn't been allocated yet.
    """
    flag = getattr(env, "_was_over_bowl_above_rim", None)
    if flag is None:
        return
    if isinstance(env_ids, torch.Tensor):
        if env_ids.numel() == 0:
            return
    elif len(env_ids) == 0:
        return
    flag[env_ids] = False


def randomize_wrist_hsv_dr(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    hue_shift_deg_range: tuple[float, float] = (-20.0, 20.0),
    sat_scale_range: tuple[float, float] = (0.65, 1.35),
    val_scale_range: tuple[float, float] = (0.55, 1.45),
) -> None:
    """Per-episode (hue, saturation, value) sample for color-aware DR.

    Stores ``env._wrist_hsv_dr`` of shape ``(num_envs, 3)``: each row is
    ``(hue_shift_rad, sat_scale, val_scale)`` constant across the episode.
    :func:`apply_color_jitter` in obs functions reads this buffer.

    Defaults cover the realistic envelope for indoor lighting + USB
    camera WB drift:

    * **±20° hue** — typical webcam WB error under warm-vs-cool ambient
      light. Wider than ±15° to extend the training distribution past the
      "average" lab condition.
    * **0.65–1.35 saturation** — handles "washed-out" exposures + the
      tendency of cheap USB cams to oversaturate reds/blues.
    * **0.55–1.45 value** — wide enough to cover dim → bright lab
      conditions without an actual light-intensity DR (the dome light is
      a global prim and not straightforward to per-env randomize).

    Constant within an episode, re-sampled at reset — matching how true
    lighting/WB conditions persist across a real-robot rollout.
    """
    if not hasattr(env, "_wrist_hsv_dr"):
        # Default = identity transform (hue=0, sat=1, val=1).
        env._wrist_hsv_dr = torch.tensor([0.0, 1.0, 1.0], device=env.device).repeat(env.num_envs, 1)

    if isinstance(env_ids, torch.Tensor):
        if env_ids.numel() == 0:
            return
    elif len(env_ids) == 0:
        return

    n = len(env_ids) if not isinstance(env_ids, torch.Tensor) else env_ids.numel()
    lo_h, hi_h = hue_shift_deg_range
    lo_s, hi_s = sat_scale_range
    lo_v, hi_v = val_scale_range
    deg_to_rad = math.pi / 180.0
    new = torch.empty((n, 3), device=env.device)
    new[:, 0] = (lo_h + (hi_h - lo_h) * torch.rand(n, device=env.device)) * deg_to_rad
    new[:, 1] = lo_s + (hi_s - lo_s) * torch.rand(n, device=env.device)
    new[:, 2] = lo_v + (hi_v - lo_v) * torch.rand(n, device=env.device)
    env._wrist_hsv_dr[env_ids] = new


def randomize_wrist_image_tint(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    rgb_scale_range: tuple[float, float] = (0.7, 1.3),
    brightness_range: tuple[float, float] = (-0.15, 0.15),
) -> None:
    """Per-episode color tint applied to the wrist RGB obs.

    Samples a per-env ``(r_scale, g_scale, b_scale, brightness_shift)`` at
    reset and stores it on ``env._wrist_image_dr`` (shape ``(num_envs, 4)``).
    :func:`mdp.observations.wrist_image` reads this buffer and applies the
    transform inside the obs function.

    This is a deliberate substitute for material-level cube / table color
    randomization. Isaac Lab's ``randomize_visual_color`` requires
    ``replicate_physics=False``, which would force a scene-level rewrite.
    Tinting the *rendered camera output* instead achieves the same end —
    the encoder sees the cube and table at varied apparent colors across
    episodes — at zero scene-cfg cost. The two differ on specular shading,
    which is negligible for our matte cube + matte table.

    The tint is **constant within an episode** (re-sampled only on reset),
    matching how true material DR would behave. Per-step jitter (small
    additive Gaussian noise, brightness wiggle) lives separately in the obs
    function as the DrQ-style frame-to-frame regularizer.
    """
    if not hasattr(env, "_wrist_image_dr"):
        # 4 cols: r_scale, g_scale, b_scale, brightness_shift. Default is
        # identity so any env that hasn't been touched by this event yet
        # sees an unmodified image.
        env._wrist_image_dr = torch.tensor([1.0, 1.0, 1.0, 0.0], device=env.device).repeat(env.num_envs, 1)

    if isinstance(env_ids, torch.Tensor):
        if env_ids.numel() == 0:
            return
    elif len(env_ids) == 0:
        return

    n = len(env_ids) if not isinstance(env_ids, torch.Tensor) else env_ids.numel()
    lo_s, hi_s = rgb_scale_range
    lo_b, hi_b = brightness_range
    new = torch.empty((n, 4), device=env.device)
    new[:, :3] = lo_s + (hi_s - lo_s) * torch.rand((n, 3), device=env.device)
    new[:, 3] = lo_b + (hi_b - lo_b) * torch.rand((n,), device=env.device)
    env._wrist_image_dr[env_ids] = new


def randomize_camera_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, tuple[float, float]],
    convention: str = "ros",
) -> None:
    """Per-reset uniform DR of a (Tiled)Camera's pose around its default mount.

    Ported from LeIsaac's ``leisaac.enhance.envs.mdp.events.randomize_camera_uniform``
    (used in ``pick_orange_env_cfg.py`` and ``lift_cube_env_cfg.py`` with
    ±25 mm xyz / ±2.5° rpy on the wrist + front cams). Their range is a
    *known-good* envelope for SO101 sim-to-real transfer of teleop demos —
    we want at least that wide for a from-scratch RL policy.

    Layer-2 of the camera setup: the camera's *default* pose is set by
    :class:`TiledCameraCfg.OffsetCfg` (parented to ``gripper_link`` in
    :mod:`joint_pos_env_cfg`). On every episode reset this event samples
    a uniform ``(Δx, Δy, Δz, Δroll, Δpitch, Δyaw)`` and adds it to that
    default world pose, then writes back via
    :meth:`Camera.set_world_poses`. Different envs see different mounts
    each rollout, simulating WOWROBO bracket manufacturing variance and
    forcing the policy to be robust to the day-6 caliper measurement
    being a few mm / a degree off.

    ``pose_range`` keys: ``x``, ``y``, ``z``, ``roll``, ``pitch``,
    ``yaw``. Missing keys default to ``(0.0, 0.0)``.
    ``convention`` selects the view convention for both reading the
    current pose and writing the perturbed one — ``"ros"`` matches
    LeIsaac's call sites and is the standard CV camera convention.
    """
    asset: Camera = env.scene[asset_cfg.name]

    ori_pos_w = asset.data.pos_w[env_ids]
    if convention == "ros":
        ori_quat_w = asset.data.quat_w_ros[env_ids]
    elif convention == "opengl":
        ori_quat_w = asset.data.quat_w_opengl[env_ids]
    elif convention == "world":
        ori_quat_w = asset.data.quat_w_world[env_ids]
    else:
        raise ValueError(f"Unknown camera convention: {convention!r}")

    range_keys = ("x", "y", "z", "roll", "pitch", "yaw")
    range_list = [pose_range.get(k, (0.0, 0.0)) for k in range_keys]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device
    )

    positions = ori_pos_w + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(
        rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    orientations = math_utils.quat_mul(ori_quat_w, orientations_delta)

    asset.set_world_poses(positions, orientations, env_ids, convention)
