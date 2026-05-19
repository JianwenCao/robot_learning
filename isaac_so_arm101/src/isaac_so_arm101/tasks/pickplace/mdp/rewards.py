# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the SO-ARM101 pick-and-place task.

Stage gating is critical: without it the dense reach term dominates and the
policy never moves on to grasp. Each reward returns ``(num_envs,)``; the
weights are set in ``pickplace_env_cfg.RewardsCfg``.

Stages:
    1. ``reach_block``   — dense, ``(1 - is_grasped) * tanh(...)`` until grasp.
    2. ``grasp_event``   — sparse one-shot bonus the first time the block is
       picked up (latched per episode).
    3. ``transport``     — dense, gated on ``is_grasped``: gripper-xy → bowl-xy.
    4. ``place``         — sparse, block xy near bowl AND block low.
    5. ``release``       — terminal, place-condition AND gripper opened AND
       block roughly stationary.

Penalties:
    * action L2 (cheap regularizer)
    * action-rate L2 (smoothness — important for sim-to-real)
    * drop after grasp
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Helpers (kept private so they don't get auto-exported into mdp.*)
# ---------------------------------------------------------------------------


def _grasped_mask(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    grasp_distance: float,
    minimal_height: float,
) -> torch.Tensor:
    """Bool tensor (num_envs,) — "block has been lifted off the table".

    Earlier this also required ``||ee - block|| < grasp_distance`` but that
    was a chicken-and-egg trap: the policy couldn't earn grasp until block
    was lifted, but couldn't lift the block until something gripped it. The
    lift task uses just ``z > minimal_height`` and that's enough — a block
    can't levitate on its own, so any non-trivial lift implies a grasp.
    The ``grasp_distance`` parameter is kept on the call signature for
    backward-compat with reward terms that still pass it.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    block_pos_w = obj.data.root_pos_w
    return block_pos_w[:, 2] > minimal_height


def _bowl_xy_w(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Bowl xy in the *world* frame (matches the block's world-frame xy).

    The command lives in the robot root frame; we transform it to world by
    adding the robot's root xy. (For a fixed-base arm this is just an offset.)
    """
    robot: Articulation = env.scene["robot"]
    bowl_b = env.command_manager.get_command(command_name)[:, :2]
    return robot.data.root_pos_w[:, :2] + bowl_b


def _episode_over_bowl_high_mask(
    env: ManagerBasedRLEnv,
    r_safe: float = 0.06,
    rim_clearance: float = 0.08,
    command_name: str = "bowl_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Per-episode latch: cube was above the rim *while* over the bowl xy at
    some prior step of the current episode.

    Encodes the "must approach from above" constraint on a 5 cm physical
    bowl rim that the **sim doesn't model** (bowl is a 2-D CommandTerm; no
    collision prim). Without this latch, the existing
    :func:`_episode_lifted_mask` only requires the cube to have been lifted
    once *anywhere* — so a policy can lift far from the bowl, drop low,
    then slide laterally at low z into the bowl footprint. That trajectory
    would slam the gripper into the rim on real hardware.

    The latch sets True iff at some step in the episode:

        ``block_z > rim_clearance`` **AND**
        ``||block_xy - bowl_xy|| < r_safe``

    i.e. the cube was simultaneously above the rim height AND within the
    bowl xy footprint. After the latch fires, the policy is free to
    descend (which it must, to release into the bowl) — but it had to
    have approached from above first. Reset by
    :func:`mdp.events.reset_was_over_bowl_above_rim` (event mode=``reset``).

    Stored on ``env._was_over_bowl_above_rim``.

    Returns ``(num_envs,)`` bool.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    block_xy = obj.data.root_pos_w[:, :2]
    block_z = obj.data.root_pos_w[:, 2]
    bowl_w = _bowl_xy_w(env, command_name)
    over_bowl_high = (block_z > rim_clearance) & (
        torch.norm(block_xy - bowl_w, dim=1) < r_safe
    )
    if not hasattr(env, "_was_over_bowl_above_rim"):
        env._was_over_bowl_above_rim = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    env._was_over_bowl_above_rim |= over_bowl_high
    return env._was_over_bowl_above_rim


def _episode_lifted_mask(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    minimal_height: float = 0.025,
) -> torch.Tensor:
    """Per-episode latched flag: "block has been lifted ≥ ``minimal_height`` at any
    step of the current episode".

    Updated every call (idempotent OR-latch); reset to all-False on episode
    reset by :func:`mdp.events.reset_was_grasped` (event-term, mode=``reset``).
    Lives on the env instance under ``env._was_grasped``.

    Used by :func:`place_in_bowl` and :func:`release_in_bowl` to gate the
    bowl-region rewards on the policy having actually grasped + lifted the
    cube at some point in the episode. Without this gate the policy can
    earn place / release by *dragging* the cube laterally with the gripper
    closed at z≈0.01 (block_z stays below ``minimal_height`` so
    :func:`_grasped_mask` returns False, but block xy reaches the bowl
    footprint via lateral pushing) — a strategy that doesn't transfer to
    real hardware. Gating both rewards on this latch closes that shortcut.

    Returns shape ``(num_envs,)`` bool.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    lifted_now = obj.data.root_pos_w[:, 2] > minimal_height
    if not hasattr(env, "_was_grasped"):
        env._was_grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Idempotent OR-latch — safe to call multiple times per step.
    env._was_grasped |= lifted_now
    return env._was_grasped


# ---------------------------------------------------------------------------
# Stage 0 — pre-grasp pose: ee close to block AND jaws open
# ---------------------------------------------------------------------------


def pre_grasp_pose(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    grasp_distance: float = 0.04,
    minimal_height: float = 0.025,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_joint_name: str = "gripper",
    gripper_open_value: float = 1.5,
) -> torch.Tensor:
    """Dense reward for *pre-grasp* posture, gated off once the block has
    been lifted at any prior step of the episode (per-episode latch).

    Returns ``proximity * jaw_openness * (1 - was_lifted_this_episode)``:

    * ``proximity = exp(-||ee - block||² / std²)``.
    * ``jaw_openness = clamp(gripper_q / gripper_open_value, 0, 1)``.
    * ``(1 - was_lifted)`` — gated off the **per-episode lift latch**
      (``env._was_grasped``, also used by ``place``/``release``).

    Why per-episode and not per-step:
        Earlier this term was gated on ``(1 - is_grasped)`` where
        ``is_grasped = block_z > 0.025`` is per-step. Run 11 (2026-05-09,
        new full method, iter 100) showed the failure mode: bootstrap
        envs would lose the grasp within ~30 steps from random gripper
        action noise → block falls → ``is_grasped=False`` → pre_grasp
        flips back on → policy trained to keep jaws open → never
        recovers. ``grasp_bootstrap`` decayed 0.41 → 0.006 across 100
        iters and ``grasp_from_scratch`` stayed flat at 0.000.
        Gating on the **episode lift latch** makes the disable
        permanent for the rest of the episode once the block has been
        lifted ≥ ``minimal_height`` even once: a fumbled grasp can no
        longer pay pre_grasp on the recovery, so the gradient pushes
        toward "regrasp + lift" rather than "open jaws and farm".
        Bootstrap envs spawn with the block already at gripper height
        (≈ 0.09 m > 0.025) so the latch sets at step 0 → bootstrap
        envs see pre_grasp = 0 throughout, eliminating the open-jaws
        attractor for them entirely.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    block_pos_w = obj.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    d2 = torch.sum((block_pos_w - ee_w) ** 2, dim=1)
    proximity = torch.exp(-d2 / (std * std))

    gripper_idx = robot.find_joints(gripper_joint_name)[0][0]
    gripper_q = robot.data.joint_pos[:, gripper_idx]
    jaw_openness = (gripper_q / gripper_open_value).clamp(0.0, 1.0)

    # Per-episode latch (idempotent OR-update; reset by ``mdp.events.reset_was_grasped``).
    was_lifted = _episode_lifted_mask(env, object_cfg, minimal_height)
    return proximity * jaw_openness * (1.0 - was_lifted.float())


# ---------------------------------------------------------------------------
# Stage 1 — reach
# ---------------------------------------------------------------------------


def reach_block(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Dense reach reward, ungated.

    Earlier we gated this term off when the block was grasped (``× (1 -
    is_grasped)``) on the theory that the policy would hover above the
    block to farm reach reward. In practice this created a reward cliff at
    the moment of grasping (reach: 1 → 0 stepwise) which is a gradient
    discontinuity PPO doesn't handle well — and the upstream lift task
    leaves its analogous reach reward ungated and converges fine. So we
    drop the gate.

    Keeping reach ungated only works while its weight stays small relative
    to the lift signal (``grasp`` term, w=15). Run 7 (5.0 → 1.0 reach revert,
    see :class:`pickplace_env_cfg.RewardsCfg`) restored that ratio after a
    w=5 run created a hover-at-cube basin the policy never escaped.

    The ``std`` was also tightened from 0.1 → 0.05 to match the lift task's
    sharper attractor at the cube position.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    dist = torch.norm(obj.data.root_pos_w - ee_frame.data.target_pos_w[..., 0, :], dim=1)
    return 1.0 - torch.tanh(dist / std)


# ---------------------------------------------------------------------------
# Stage 1.4 — EE descent to cube (pre-grasp geometric prerequisite)
# ---------------------------------------------------------------------------


def ee_descent_to_cube(
    env: ManagerBasedRLEnv,
    xy_std: float = 0.04,
    z_band: float = 0.02,
    cube_half_size: float = 0.01,
    minimal_height: float = 0.025,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Dense pre-grasp reward — pays for ``EE close in xy AND below cube top``,
    gated off once the cube is lifted.

    Returns ``near_xy * descended * (1 - lifted_now)`` where:

    * ``near_xy = exp(-||ee_xy - cube_xy||² / xy_std²)`` — Gaussian on 2-D
      proximity, fires within ~4 cm.
    * ``descended = clamp((cube_top_z - ee_z) / z_band, 0, 1)`` — 0 when EE is
      above cube top, 1 when EE is ``z_band`` (2 cm) below cube top. ``cube_top
      = cube_z + cube_half_size`` for our 2 cm cube.
    * ``(1 - lifted_now) = (cube_z <= minimal_height)`` — pays only while the
      cube is on the table. Turns off once the policy succeeds in lifting,
      so `lifting_object` and `transport_to_bowl` take over downstream.

    **Why this term exists (v5.2 fix, 2026-05-13).** Across v3 → v5.1 (1968 +
    798 + 1073 + 418 + 1273 = ~5500 cumulative PPO iters), ``grasp_from_scratch``
    stayed pinned at exactly 0 — from-scratch envs never grasped. The `reach`
    reward (`1 - tanh(d_ee_obj / 0.05)`) is dense on 3-D distance, but its
    gradient is dominated by xy and doesn't reliably pull EE *down* to
    cube-grasping height. The `closed_grasp_signal` rewards "jaws closed near
    cube" but the policy never reaches that geometric prerequisite. This
    term provides the missing **z-descent gradient**: once EE is close in xy,
    it must lower to cube level to earn the bonus. Lowering is the geometric
    prerequisite for the jaws to actually be around the cube when they close.

    Inspired by ManiSkill3 PickCube's per-link reaching reward and Robosuite
    PickPlace's "vertical reach" shaping component — both decompose the
    reach gradient into xy-approach and z-descent rather than a single 3-D
    distance, which empirically improves cold-start grasp discovery.

    **Weight 2.0** in PretrainedRewardsCfg. Combined with reach(1.0) and
    closed_grasp(1.5), the pre-grasp gradient budget is ~4.5/step (vs the
    v5.1 pre-grasp budget of ~1.65/step, a ~3× boost). Post-grasp the term
    turns off, so it can't create a hover attractor — once cube is lifted,
    `lifting_object(15) + transport(~14)` dominate the immediate reward
    landscape.

    Gate is per-step ``block_z <= minimal_height`` (not the episode lift
    latch). If the policy fumbles the grasp and the cube drops back below
    0.025, the descent reward turns back on — encouraging recovery rather
    than punishing it.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    block_pos_w = obj.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    # XY proximity — Gaussian, sharp at 4 cm scale
    d_xy2 = torch.sum((block_pos_w[:, :2] - ee_w[:, :2]) ** 2, dim=1)
    near_xy = torch.exp(-d_xy2 / (xy_std * xy_std))

    # Z descent — 0 when EE above cube top, 1 when EE ≥ z_band below cube top
    cube_top_z = block_pos_w[:, 2] + cube_half_size
    ee_z = ee_w[:, 2]
    descended = ((cube_top_z - ee_z) / z_band).clamp(0.0, 1.0)

    # Pre-grasp gate (per-step, not episode latch): turn off once cube lifted
    pre_grasp = (block_pos_w[:, 2] <= minimal_height).float()

    return near_xy * descended * pre_grasp


# ---------------------------------------------------------------------------
# Stage 2 — is_grasping (physical contact between gripper jaws and cube)
# ---------------------------------------------------------------------------


def is_grasping_contact(
    env: ManagerBasedRLEnv,
    force_threshold: float = 0.05,
    require_both_jaws: bool = True,
    fixed_sensor_name: str = "gripper_contact_fixed",
    moving_sensor_name: str = "gripper_contact_moving",
) -> torch.Tensor:
    """Binary indicator (0/1) — gripper jaws are physically in contact with the cube.

    Reads filtered contact forces from the two ContactSensors on
    ``gripper_link`` and ``moving_jaw_so101_v1_link`` (configured in
    :class:`PickPlaceBowlSceneCfg`). Each sensor's ``data.force_matrix_w``
    contains forces between its prim and the filter set (here, just the
    cube). The L2 norm of this force vector being above ``force_threshold``
    on **both** jaws means the cube is squeezed between them — a true
    physical grasp, not a kinematic proxy.

    **Why this replaces v3–v6's ``closed_grasp_signal`` (v7 fix, 2026-05-15).**
    The kinematic proxy paid for "jaws closed near cube" — but PPO could
    earn it by closing jaws *in air* near the cube without actually
    grasping. v5.3/v5.4 saw the policy converge on "lower EE to cube level
    with jaws nearly open" instead of committing to a grasp action,
    because the kinematic proxy didn't sharply distinguish "grasping" from
    "hovering with closed jaws nearby". A contact sensor is the
    unambiguous physical signal.

    **Requiring both jaws** (``require_both_jaws=True``) is the
    discriminating choice: a one-sided push (one jaw touching cube while
    the other is in air) is not a grasp. Both jaws in contact means the
    cube is between them, which mechanically implies the cube can be
    lifted by closing further. ManiSkill3's ``agent.is_grasping(cube)``
    uses the same two-jaw-contact convention.

    Sim-to-real: this signal goes into the **reward function only**, never
    into the policy observation. The deployed real arm has no tactile
    sensors; the trained policy never depends on contact info as input.
    This is the standard asymmetric actor-critic pattern (privileged info
    in critic/reward but not in actor observations).
    """
    fixed_sensor = env.scene[fixed_sensor_name]
    moving_sensor = env.scene[moving_sensor_name]

    # force_matrix_w shape: (num_envs, num_bodies_on_sensor=1, num_filter_prims=1, 3)
    # take L2 norm over the xyz force vector, then check threshold
    fixed_force = torch.norm(fixed_sensor.data.force_matrix_w.reshape(env.num_envs, -1, 3), dim=-1)
    moving_force = torch.norm(moving_sensor.data.force_matrix_w.reshape(env.num_envs, -1, 3), dim=-1)

    # max over any contact pair on that link (in our case there's only one filter prim)
    fixed_in_contact = (fixed_force.max(dim=-1).values > force_threshold)
    moving_in_contact = (moving_force.max(dim=-1).values > force_threshold)

    if require_both_jaws:
        contact = fixed_in_contact & moving_in_contact
    else:
        contact = fixed_in_contact | moving_in_contact

    return contact.float()


# ---------------------------------------------------------------------------
# Stage 1.5 — closed-grasp signal (dense, ManiSkill3-style contact proxy)
# ---------------------------------------------------------------------------


def closed_grasp_signal(
    env: ManagerBasedRLEnv,
    std: float = 0.03,
    gripper_closed_threshold: float = 0.3,
    pre_lift_height: float = 0.025,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_joint_name: str = "gripper",
) -> torch.Tensor:
    """Dense ``proximity × closedness`` reward — generates policy gradient on the
    grasp **action** itself.

    Returns ``proximity * closedness`` where:

    * ``proximity = exp(-||ee - block||² / std²)`` — sharp Gaussian, fires only
      within ~3 cm of the cube.
    * ``closedness = clamp((closed_thr - gripper_q) / closed_thr, 0, 1)`` — 1
      when the gripper joint is fully closed (q ≈ 0), 0 at ``closed_thr``.
      Home pose has ``gripper=0.5`` (open), so a threshold of 0.3 means the
      policy must *actively* close the jaws past the open default to get any
      payout.

    **Why this term exists (§9 v4 fix, 2026-05-13).** The §9.5 v3 design
    relied on init-state bootstrap (``init_block_in_gripper`` with
    ``p_grasped=0.5``) to seed grasp competence. The 2026-05-12 v3 run
    showed this *fails* — across 1968 PPO iters, ``grasp_from_scratch`` stayed
    pinned at exactly 0 while bootstrap envs maintained ~48 % grasp rate.
    Diagnosis: init-state subsidy trains the *post-grasp* action sequence
    (lift → transport → release) and feeds the critic a value-function
    learning signal, but on from-scratch rollouts the policy never randomly
    stumbles into a successful grasp, so there is **no positive advantage
    on grasp-leading actions** and PPO has no gradient toward
    "close jaws when near cube". As ``p_grasped`` decayed (0.50 → 0.24), the
    overall ``success_rate`` collapsed monotonically (0.148 → 0.040).

    This term provides the missing gradient directly. PPO's policy gradient
    on ``closed_grasp_signal`` is positive iff the policy closes jaws *while*
    the gripper is near the cube — exactly the grasp-leading action sequence
    bootstrap alone can't teach. Once the cube lifts off, this term keeps
    paying (gripper stays near the lifted cube, jaws stay closed), so it
    composes additively with ``lifting_object`` and ``transport_to_bowl``
    rather than competing.

    **Why this is the ManiSkill3 pattern.** ``agent.is_grasping(cube)`` in
    ManiSkill3 pays +1/step whenever the gripper is in contact with the cube,
    no height gate. That's a contact-state reward whose policy gradient
    rewards the close-jaws-on-cube action. We can't query contacts on this
    asset (``activate_contact_sensors=False``), so we approximate with the
    kinematic proxy above — gripper close to cube + gripper joint closed.
    Same gradient signal, no ContactSensor plumbing.

    **Weight tuning.** ``RewardsCfg.weight=3.0``: with max signal 1.0/step
    over 250 steps that's a 750 budget per episode, large enough to bias the
    policy toward closing on-cube but smaller than the ``lifting_object``
    (15) + ``transport_to_bowl`` (16) + ``release_in_bowl`` (30) budget so
    the post-grasp incentives still dominate. At release time the term goes
    to 0 (jaws open), removing any disincentive to release — verified
    arithmetically: release-and-stay-in-bowl pays ~46/step, hover-with-grasp
    pays ~35/step, so the release decision is still strictly preferred.

    **Gated on ``block_z < pre_lift_height`` (v4.1, 2026-05-13).** The v4
    initial design was ungated, which the 2026-05-13 v4 run
    (``pickplace_bowl_pretrained/2026-05-13_01-46-…``, killed iter 798)
    showed creates a *new* hover-with-grasp attractor: bootstrap envs hit
    92 % grasp rate, but ``release_in_bowl`` decayed 0.017 → 0.004 and
    ``success_rate`` fell to 0 because "hover-with-cube" paid
    ``closed_grasp(3) + lift(15) + transport(16) = 34/step`` indefinitely,
    competing with ``release(46/step)`` which required an exploration cost
    the policy never paid. Gating on ``block_z < pre_lift_height`` turns
    this term into a *pre-grasp-only* signal: it pays while the cube is
    still on the table (z < 0.025), then turns off the moment the cube
    lifts. After lift, ``lifting_object`` + ``transport_to_bowl`` carry the
    policy, and ``release_in_bowl`` is the unambiguous next-stage payoff
    (no closed_grasp competition). This also concentrates the gradient on
    *from-scratch* envs (bootstrap envs spawn with the cube ≈ 9 cm above
    the table, so they earn ~0 from this term — bootstrap learning still
    happens through lift/transport/release, exactly as the §9.5 design
    intended).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    block_pos_w = obj.data.root_pos_w
    block_z = block_pos_w[:, 2]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    d2 = torch.sum((block_pos_w - ee_w) ** 2, dim=1)
    proximity = torch.exp(-d2 / (std * std))

    gripper_idx = robot.find_joints(gripper_joint_name)[0][0]
    gripper_q = robot.data.joint_pos[:, gripper_idx]
    closedness = ((gripper_closed_threshold - gripper_q) / gripper_closed_threshold).clamp(0.0, 1.0)

    pre_lift = (block_z < pre_lift_height).float()
    return proximity * closedness * pre_lift


# ---------------------------------------------------------------------------
# Stage 2 — grasp event (one-shot per episode)
# ---------------------------------------------------------------------------


def grasp_event(
    env: ManagerBasedRLEnv,
    grasp_distance: float = 0.04,
    minimal_height: float = 0.025,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Indicator (0/1) that the block is currently grasped.

    Note: this is *not* one-shot — it pays out every step the block is held.
    Combined with ``transport`` (also gated on grasp) this works well in
    practice and avoids the bookkeeping of an episode-state buffer that
    Isaac Lab reward terms don't carry by default.
    """
    grasped = _grasped_mask(env, object_cfg, ee_frame_cfg, grasp_distance, minimal_height)
    return grasped.float()


# ---------------------------------------------------------------------------
# Stage 3 — transport (gripper xy → bowl xy, gated on grasp)
# ---------------------------------------------------------------------------


def transport_to_bowl(
    env: ManagerBasedRLEnv,
    std: float = 0.30,
    minimal_height: float = 0.025,
    command_name: str = "bowl_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # Kept for backward compat with old callers that pass these:
    grasp_distance: float = 0.04,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Dense **cube → goal_xyz** distance, gated on the per-episode lift latch.

    Returns

        was_lifted_this_episode * (1 - tanh(||cube - goal_xyz|| / std))

    where the gate is the per-episode ``env._was_grasped`` latch (True if
    the cube has been ≥ ``minimal_height`` at *any* prior step of the
    current episode), maintained by :func:`_episode_lifted_mask` and
    cleared each reset by :func:`mdp.events.reset_was_grasped`.

    Why latch (not per-step lift gate, which is what stock Franka Lift
    uses): stock task's goal is a 3-D pose ABOVE the table (z=0.25-0.50)
    and the cube ends up *held* at that goal. The per-step lift gate
    enforces "cube must currently be lifted" which works because the
    goal IS lifted. Our task is place-into-bowl where the bowl sits on
    the table at z≈0 — after release, cube ends up at cube_center ≈
    0.01 (bowl on table, cube on bowl floor). With per-step lift gate,
    transport reward goes to 0 the moment the cube enters the bowl
    (cube_z drops below 0.025) even though the policy just *succeeded*
    at placing. Latch instead pays continuously after lift-once-per-
    episode regardless of cube z, so policy is rewarded for placing
    AND for staying placed.

    Drag-on-table exploit prevention: same as the latch on
    place_in_bowl / release_in_bowl — drag without lifting → latch
    stays False → transport pays 0.

    ``grasp_distance`` and ``ee_frame_cfg`` kwargs kept on signature
    for backward-compat with cfg call sites; unused.
    """
    del grasp_distance, ee_frame_cfg  # unused; kept for cfg backward-compat
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    goal_pos_b = command[:, :3]  # goal in robot root frame
    goal_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, goal_pos_b
    )
    cube_pos_w = obj.data.root_pos_w
    distance = torch.norm(goal_pos_w - cube_pos_w, dim=1)
    was_lifted = _episode_lifted_mask(env, object_cfg, minimal_height)
    return was_lifted.float() * (1.0 - torch.tanh(distance / std))


# ---------------------------------------------------------------------------
# Stage 4 — place (block over bowl AND low)
# ---------------------------------------------------------------------------


def place_in_bowl(
    env: ManagerBasedRLEnv,
    r_safe: float = 0.06,
    bowl_height: float = 0.08,
    minimal_height: float = 0.025,
    rim_clearance: float = 0.08,
    command_name: str = "bowl_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for the block being inside the bowl footprint **after a grasp
    AND an over-the-rim approach**.

    Gated on TWO per-episode latches:

    * :func:`_episode_lifted_mask` (block was lifted ≥ ``minimal_height``
      at some prior step) — closes the drag-on-table exploit.
    * :func:`_episode_over_bowl_high_mask` (block was simultaneously above
      ``rim_clearance`` AND within ``r_safe`` of bowl xy at some prior step)
      — closes the lateral-slide-into-bowl exploit. Sim has no physical
      bowl prim, so without this latch the policy can descend to z≈0.01
      far from the bowl, then slide in at low z — a trajectory that
      would crash the gripper into the real 5 cm rim. Requiring the cube
      to have been over the bowl at safe height at some prior step forces
      an over-the-top descent trajectory.

    Geometric "in bowl" condition: block xy within ``r_safe`` of bowl xy
    AND block z below ``bowl_height``. ``rim_clearance=0.08`` puts the
    cube *bottom* 1 cm above the 5 cm rim plus a 2 cm cube — i.e. the
    cube body fully clears the rim before any descent begins.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    block_xy = obj.data.root_pos_w[:, :2]
    block_z = obj.data.root_pos_w[:, 2]
    bowl_w = _bowl_xy_w(env, command_name)
    in_xy = torch.norm(block_xy - bowl_w, dim=1) < r_safe
    low = block_z < bowl_height
    was_lifted = _episode_lifted_mask(env, object_cfg, minimal_height)
    was_over_high = _episode_over_bowl_high_mask(
        env, r_safe=r_safe, rim_clearance=rim_clearance,
        command_name=command_name, object_cfg=object_cfg,
    )
    return (in_xy & low & was_lifted & was_over_high).float()


# ---------------------------------------------------------------------------
# Stage 4.5 — release proximity (dense partial credit for "approaching release")
# ---------------------------------------------------------------------------


def release_proximity(
    env: ManagerBasedRLEnv,
    r_safe: float = 0.06,
    bowl_height: float = 0.06,
    gripper_open_value: float = 1.5,
    minimal_height: float = 0.025,
    rim_clearance: float = 0.04,
    command_name: str = "bowl_pose",
    gripper_joint_name: str = "gripper",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense partial-credit reward for "approaching release configuration".

    Returns ``was_lifted × was_over_high × xy_factor × z_factor × open_factor``
    where each factor is a smooth [0, 1] decay:

    * ``xy_factor = clamp(1 - ||cube_xy - bowl||/r_safe, 0, 1)`` — 1 inside the
      bowl footprint, decays linearly to 0 at the bowl boundary. Wider region
      than the binary in_xy gate of ``release_in_bowl`` so partial credit
      starts as soon as the cube is near the bowl.
    * ``z_factor = clamp(1 - cube_z/bowl_height, 0, 1)`` — 1 when cube on bowl
      floor (z=0), decays linearly to 0 at the bowl_height (default 6cm).
      Rewards lowering the cube past the rim into the bowl.
    * ``open_factor = clamp(gripper_q / gripper_open_value, 0, 1)`` — 0 when
      jaws fully closed (q=0), 1 when fully open (q=1.5). Continuous gradient
      toward opening, in contrast to the binary opened-gate of release_in_bowl.
    * ``was_lifted`` (latch at z>0.025) and ``was_over_high`` (latch at
      z>0.04 over bowl) — same gates as release_in_bowl, ensure the term
      only pays in the right neighborhood (cube must have been lifted and
      approached the bowl from above first).

    **Why this term exists (v6 fix, 2026-05-15).** Across 8 prior reward
    variants, `release_in_bowl` was the only release-stage reward — a
    6-AND-gate conjunction (in_xy ∧ low_z ∧ jaws_open ∧ settled ∧
    was_lifted ∧ was_over_high). The policy got 0 signal until all 6 gates
    held simultaneously, which is an exploration wall. PPO never learned
    the coordinated "lower + open + wait" 3-step action sequence.

    This term provides the missing **dense gradient** along the release
    approach: as the cube lowers (z_factor↑), enters the bowl xy
    (xy_factor↑), and jaws open (open_factor↑), reward increases
    monotonically. The exploration wall becomes a gradient hill.

    Mirrors the staged-gating pattern from ManiSkill3 PickCube and
    Robosuite PickPlace where each stage has a dense partial-credit term
    alongside the binary stage-success indicator. The pattern that
    succeeded for grasp (`ee_descent_to_cube` + `closed_grasp_signal`)
    applied to release.

    **Weight 8.0** in PretrainedRewardsCfg. Max signal 1.0/step. Below
    release_in_bowl(30) so full release still strictly dominates partial
    release. Above closed_grasp(1.5) so the policy is incentivized to open
    jaws over the bowl rather than keep them closed.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    block_xy = obj.data.root_pos_w[:, :2]
    block_z = obj.data.root_pos_w[:, 2]
    bowl_w = _bowl_xy_w(env, command_name)

    # Continuous proximity factors
    xy_dist = torch.norm(block_xy - bowl_w, dim=1)
    xy_factor = (1.0 - xy_dist / r_safe).clamp(0.0, 1.0)
    z_factor = (1.0 - block_z / bowl_height).clamp(0.0, 1.0)

    gripper_idx = robot.find_joints(gripper_joint_name)[0][0]
    gripper_q = robot.data.joint_pos[:, gripper_idx]
    open_factor = (gripper_q / gripper_open_value).clamp(0.0, 1.0)

    # Same neighborhood gates as release_in_bowl
    was_lifted = _episode_lifted_mask(env, object_cfg, minimal_height)
    was_over_high = _episode_over_bowl_high_mask(
        env, r_safe=r_safe, rim_clearance=rim_clearance,
        command_name=command_name, object_cfg=object_cfg,
    )

    gate = was_lifted.float() * was_over_high.float()
    return gate * xy_factor * z_factor * open_factor


# ---------------------------------------------------------------------------
# Stage 5 — release (terminal: place + gripper open + block stationary)
# ---------------------------------------------------------------------------


def release_in_bowl(
    env: ManagerBasedRLEnv,
    r_safe: float = 0.06,
    bowl_height: float = 0.06,
    gripper_open_threshold: float = 0.2,
    block_speed_threshold: float = 0.05,
    minimal_height: float = 0.07,
    rim_clearance: float = 0.08,
    command_name: str = "bowl_pose",
    gripper_joint_name: str = "gripper",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminal release reward — fires when the policy actually lets go.

    Gated on the same per-episode lift latch as :func:`place_in_bowl` so the
    drag-to-bowl shortcut can't farm release either. Without this gate, the
    policy can keep the gripper open the whole episode (it spawns open at
    home pose with ``gripper=0.5``), nudge the block laterally to the bowl
    via random arm motion, and immediately satisfy
    ``opened AND settled AND in_xy AND low`` without any lift — exactly the
    failure mode flagged at iter ~650 of run 2026-05-08_22-36-11.

    All conditions must hold simultaneously:

    * episode lift latch (block was raised ≥ ``minimal_height`` at some step),
    * place condition (block xy near bowl, block low),
    * gripper joint position above ``gripper_open_threshold`` (i.e. open),
    * block linear speed below ``block_speed_threshold`` m/s (settled).

    ``minimal_height`` defaults to **0.07 m** (the lift-once latch — see
    :func:`_episode_lifted_mask`). ``rim_clearance`` defaults to **0.08 m**
    and gates the *over-the-bowl* approach latch — see
    :func:`_episode_over_bowl_high_mask`. Sim has no physical bowl prim
    (bowl = 2-D goal command), so the two latches together encode the
    real bowl's 5 cm rim:

    * lift-once latch (≥ 0.07): closes the drag-on-table exploit.
    * over-bowl-above-rim latch (≥ 0.08 *while* over bowl xy): closes the
      lateral-slide-in exploit. Forces the descent to come from above
      the rim, not from the side.

    Both must hold for release to pay out, on top of the geometric
    in-bowl + open + settled gate. Without the second latch, a policy
    could lift far from the bowl, lower to z≈0.02, then slide in at low
    z — which would crash the real gripper into the rim.

    See :func:`mdp.terminations.task_success` for why the gripper joint is
    resolved by name inside the function rather than via ``SceneEntityCfg``.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # place
    block_xy = obj.data.root_pos_w[:, :2]
    block_z = obj.data.root_pos_w[:, 2]
    bowl_w = _bowl_xy_w(env, command_name)
    in_xy = torch.norm(block_xy - bowl_w, dim=1) < r_safe
    low = block_z < bowl_height

    # gripper open (gripper joint is in [0, 1.7] roughly; >0.2 means opened)
    gripper_idx = robot.find_joints(gripper_joint_name)[0][0]
    gripper_q = robot.data.joint_pos[:, gripper_idx]
    opened = gripper_q > gripper_open_threshold

    # block roughly stationary
    settled = torch.norm(obj.data.root_lin_vel_w, dim=1) < block_speed_threshold

    # episode lift latch — must have grasped + lifted at some prior step
    was_lifted = _episode_lifted_mask(env, object_cfg, minimal_height)

    # episode over-bowl-above-rim latch — must have approached from above
    was_over_high = _episode_over_bowl_high_mask(
        env, r_safe=r_safe, rim_clearance=rim_clearance,
        command_name=command_name, object_cfg=object_cfg,
    )

    indicator = (in_xy & low & opened & settled & was_lifted & was_over_high)

    # Per-episode task-success latch (side effect, read by
    # :func:`log_success_metrics` CurriculumTerm). OR-update: latch flips
    # to True the first frame the success gate holds and stays True for
    # the rest of the episode. Cleared inside the CurriculumTerm at
    # reset_idx (see comment in log_success_metrics). Same gate as the
    # ``task_success`` predicate in :mod:`mdp.terminations`, so the TB
    # success_rate matches what a binary task_success termination would
    # measure if we used one.
    if not hasattr(env, "_task_success_latch"):
        env._task_success_latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._task_success_latch |= indicator

    return indicator.float()


# ---------------------------------------------------------------------------
# Penalties
# ---------------------------------------------------------------------------


def block_dropped(
    env: ManagerBasedRLEnv,
    drop_height: float = 0.005,
    r_safe: float = 0.06,
    command_name: str = "bowl_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalty for dropping the block on the table away from the bowl.

    Returns ``1.0`` when the block is on the table (z < drop_height) AND
    not within the bowl radius. Multiplied by a negative weight in cfg.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    block_z = obj.data.root_pos_w[:, 2]
    bowl_w = _bowl_xy_w(env, command_name)
    far_from_bowl = torch.norm(obj.data.root_pos_w[:, :2] - bowl_w, dim=1) >= r_safe
    on_table = block_z < drop_height
    return (on_table & far_from_bowl).float()


# ---------------------------------------------------------------------------
# Metrics — wired as a CurriculumTerm in cfg, not a RewardTerm. CurriculumTerms
# whose function returns a ``dict[str, float]`` get logged automatically by
# Isaac Lab's CurriculumManager as ``Curriculum/<term_name>/<key>``. The earlier
# attempt to side-effect-write into ``env.extras["log"]`` from a reward
# function was lost because Isaac Lab replaces ``extras["log"]`` between the
# reward step and the runner's read, so reward-side-effect writes never reach
# TensorBoard. Curriculum-return-dict writes do.
# ---------------------------------------------------------------------------


def log_success_metrics(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
) -> dict[str, float]:
    """Per-episode binary task-success rate.

    Wired as a :class:`CurriculumTermCfg`. Returns
    ``{"success_rate": float, "n_episodes_ended": int}`` which Isaac Lab's
    :class:`CurriculumManager` logs as
    ``Curriculum/log_success/success_rate`` (and ``.../n_episodes_ended``).

    **What this measures.** For each env that just ended an episode this
    step (``env_ids`` is the resetting-envs index list — the
    CurriculumManager only invokes terms at ``_reset_idx`` time), reads
    the per-env ``env._task_success_latch`` flag. The latch is
    True iff the :func:`release_in_bowl` indicator (in_xy ∧ low ∧
    gripper_open ∧ settled ∧ was_lifted) fired at any step of that
    episode — i.e. iff the policy actually placed the block in the bowl
    and released. This is exactly the gate used by
    :func:`mdp.terminations.task_success`, so the TB scalar matches the
    binary-success rate a paper / report would quote.

    Mean across the resetting envs = success rate for episodes that
    ended this step. Across an iteration RSL-RL averages these
    per-reset-step values for the TB log; with ~1024 envs and 250-step
    episodes, ~4 envs reset per step on average so the moving estimate
    smooths over ~100 episode outcomes per PPO iter.

    **Latch maintenance.** Updated each step (idempotent OR-update) inside
    :func:`release_in_bowl` as a side effect — the only reward term
    that runs every step *and* knows the success indicator. Cleared
    in-place by THIS function after reading, so successive episodes on
    the same env start with a fresh latch. No separate ``mode="reset"``
    event term is needed (curriculum.compute fires before reset events
    in :meth:`ManagerBasedRLEnv._reset_idx`, so an event-time clear
    would be redundant).

    Returns an empty dict on the first call (before any reward step has
    populated the latch) — Isaac Lab handles that gracefully.
    """
    latch = getattr(env, "_task_success_latch", None)
    if latch is None:
        return {}
    if env_ids is None:
        return {}
    if isinstance(env_ids, torch.Tensor):
        if env_ids.numel() == 0:
            return {}
    elif len(env_ids) == 0:
        return {}

    outcomes = latch[env_ids].float()
    success_rate = outcomes.mean().item()
    n_ended = int(outcomes.numel())

    # Clear latch for the ended envs so next episode starts fresh.
    # Done here rather than in a reset event because curriculum.compute()
    # runs BEFORE event_manager.apply(mode="reset") in _reset_idx — clearing
    # at event time would be a no-op (we'd have already read+cleared).
    latch[env_ids] = False

    metrics: dict[str, float] = {
        "success_rate": success_rate,
        "n_episodes_ended": float(n_ended),
    }

    # Diagnostic: fraction of ended episodes that ever satisfied the
    # over-bowl-above-rim precondition. If success_rate stays low while
    # this stays low, the policy isn't even attempting the safe-approach
    # trajectory. If this is high but success_rate is low, the approach
    # is fine but some other gate (settled / opened / low) is failing.
    over_high = getattr(env, "_was_over_bowl_above_rim", None)
    if over_high is not None:
        over_high_outcomes = over_high[env_ids].float()
        metrics["over_bowl_high_rate"] = over_high_outcomes.mean().item()

    return metrics


def log_bootstrap_metrics(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,  # CurriculumTerm requires this be mandatory (no default); unused here
    r_safe: float = 0.06,
    bowl_height: float = 0.06,
    gripper_open_threshold: float = 0.2,
    block_speed_threshold: float = 0.05,
    minimal_height: float = 0.025,
    command_name: str = "bowl_pose",
    gripper_joint_name: str = "gripper",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, float]:
    """Per-bootstrap-status TB metrics. Returns a dict, logged as
    ``Curriculum/log_metrics/<key>`` by Isaac Lab's CurriculumManager.

    Reads ``env._is_bootstrapped`` (set by :func:`init_block_in_gripper`) and
    returns:

    * ``p_bootstrapped`` — fraction of envs currently in the bootstrap
      regime. Equals the curriculum's ``p_grasped`` parameter in
      expectation; useful as a sanity check that the curriculum is
      actually decaying.
    * ``release_bootstrap`` — mean release-success indicator across
      bootstrapped envs only. Should be near 1 for a competent policy on
      the easy regime.
    * ``release_from_scratch`` — mean release-success indicator across
      non-bootstrapped envs only. **The key signal**: if this stays near
      0 while ``release_bootstrap`` is near 1, the policy is riding the
      subsidy and will collapse when ``p_grasped`` decays.
    * ``grasp_bootstrap`` / ``grasp_from_scratch`` — same split for the
      lift signal (block above ``minimal_height``). Ramps up earlier than
      release; useful for catching from-scratch failure before it's
      hopeless.

    Wired as a :class:`CurriculumTermCfg`. Has no side effects on the env;
    the dict it returns is the only output.
    """
    flag = getattr(env, "_is_bootstrapped", None)
    if flag is None:
        return {}

    is_boot = flag.float()
    is_scratch = 1.0 - is_boot
    n_boot = is_boot.sum()
    n_scratch = is_scratch.sum()

    # Lift / grasp signal: block above minimal_height. Cheap to compute
    # and gives a much earlier signal than release does.
    obj: RigidObject = env.scene[object_cfg.name]
    block_z = obj.data.root_pos_w[:, 2]
    grasp_signal = (block_z > minimal_height).float()

    # Release signal: replicate release_in_bowl so we don't recurse.
    # NOTE: keep the gating identical to :func:`release_in_bowl` (now also
    # requires the per-episode lift latch ``env._was_grasped``) so the
    # ``release_from_scratch`` TB metric reflects the same event the
    # policy is actually rewarded for. Without the latch, this metric
    # would over-count drag-to-bowl episodes that no longer earn reward.
    robot: Articulation = env.scene[robot_cfg.name]
    block_xy = obj.data.root_pos_w[:, :2]
    bowl_w = _bowl_xy_w(env, command_name)
    in_xy = torch.norm(block_xy - bowl_w, dim=1) < r_safe
    low = block_z < bowl_height
    gripper_idx = robot.find_joints(gripper_joint_name)[0][0]
    opened = robot.data.joint_pos[:, gripper_idx] > gripper_open_threshold
    settled = torch.norm(obj.data.root_lin_vel_w, dim=1) < block_speed_threshold
    was_lifted = getattr(env, "_was_grasped", None)
    if was_lifted is None:
        # Latch hasn't been initialized yet (first call before any reward
        # term touched it); treat as all-False.
        was_lifted = torch.zeros_like(in_xy, dtype=torch.bool)
    was_over_high = getattr(env, "_was_over_bowl_above_rim", None)
    if was_over_high is None:
        was_over_high = torch.zeros_like(in_xy, dtype=torch.bool)
    release_signal = (
        in_xy & low & opened & settled & was_lifted & was_over_high
    ).float()

    metrics: dict[str, float] = {"p_bootstrapped": is_boot.mean().item()}
    if n_boot > 0:
        metrics["release_bootstrap"] = (release_signal * is_boot).sum().item() / n_boot.item()
        metrics["grasp_bootstrap"] = (grasp_signal * is_boot).sum().item() / n_boot.item()
    if n_scratch > 0:
        metrics["release_from_scratch"] = (release_signal * is_scratch).sum().item() / n_scratch.item()
        metrics["grasp_from_scratch"] = (grasp_signal * is_scratch).sum().item() / n_scratch.item()
    return metrics
