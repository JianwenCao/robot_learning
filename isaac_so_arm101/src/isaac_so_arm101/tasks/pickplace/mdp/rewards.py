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
    std: float = 0.15,
    grasp_distance: float = 0.04,
    minimal_height: float = 0.025,
    command_name: str = "bowl_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Dense gripper-xy → bowl-xy reward, only active while holding the block.

    Gating on ``is_grasped`` is what stops the policy from learning to fly
    the empty gripper over the bowl.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_xy_w = ee_frame.data.target_pos_w[..., 0, :2]
    bowl_w = _bowl_xy_w(env, command_name)
    dist = torch.norm(ee_xy_w - bowl_w, dim=1)
    grasped = _grasped_mask(env, object_cfg, ee_frame_cfg, grasp_distance, minimal_height)
    return grasped.float() * (1.0 - torch.tanh(dist / std))


# ---------------------------------------------------------------------------
# Stage 4 — place (block over bowl AND low)
# ---------------------------------------------------------------------------


def place_in_bowl(
    env: ManagerBasedRLEnv,
    r_safe: float = 0.06,
    bowl_height: float = 0.08,
    minimal_height: float = 0.025,
    command_name: str = "bowl_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for the block being inside the bowl footprint **after a grasp**.

    Gated on the per-episode :func:`_episode_lifted_mask` latch (block was
    lifted ≥ ``minimal_height`` at some prior step of this episode). Without
    this gate the policy can drag the block laterally with the gripper
    closed at z≈0.01 to reach the bowl footprint without ever lifting,
    which doesn't transfer to real hardware. With it, place reward only
    starts paying out after the policy has *grasped + lifted* the block
    at least once in the episode — making lift the necessary precondition
    for any place / release reward.

    The geometric "in bowl" condition itself is unchanged — block xy
    within ``r_safe`` of bowl xy AND block z below ``bowl_height`` (so
    a held block hovering above the bowl rim still pays out, covering
    the gripper-jaw closure depth). The earlier rationale — that the
    BowlPoseCommand rejection sampler keeps ``‖block − bowl‖ ≥ 0.10``
    at reset, preventing init-overlap farming — is preserved; the new
    gate covers the *dragging* exploit specifically.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    block_xy = obj.data.root_pos_w[:, :2]
    block_z = obj.data.root_pos_w[:, 2]
    bowl_w = _bowl_xy_w(env, command_name)
    in_xy = torch.norm(block_xy - bowl_w, dim=1) < r_safe
    low = block_z < bowl_height
    was_lifted = _episode_lifted_mask(env, object_cfg, minimal_height)
    return (in_xy & low & was_lifted).float()


# ---------------------------------------------------------------------------
# Stage 5 — release (terminal: place + gripper open + block stationary)
# ---------------------------------------------------------------------------


def release_in_bowl(
    env: ManagerBasedRLEnv,
    r_safe: float = 0.06,
    bowl_height: float = 0.06,
    gripper_open_threshold: float = 0.2,
    block_speed_threshold: float = 0.05,
    minimal_height: float = 0.025,
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

    return (in_xy & low & opened & settled & was_lifted).float()


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
    release_signal = (in_xy & low & opened & settled & was_lifted).float()

    metrics: dict[str, float] = {"p_bootstrapped": is_boot.mean().item()}
    if n_boot > 0:
        metrics["release_bootstrap"] = (release_signal * is_boot).sum().item() / n_boot.item()
        metrics["grasp_bootstrap"] = (grasp_signal * is_boot).sum().item() / n_boot.item()
    if n_scratch > 0:
        metrics["release_from_scratch"] = (release_signal * is_scratch).sum().item() / n_scratch.item()
        metrics["grasp_from_scratch"] = (grasp_signal * is_scratch).sum().item() / n_scratch.item()
    return metrics
