# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based RL env configuration for SO-ARM101 single-block pick-and-place.

Source-of-truth scaffold for Eval 1 of Project 3. The bowl is **not** a
scene prim — it is a 2-D goal sampled per episode by the command manager
(``bowl_pose``). The block is a 2 cm cube whose initial xy is randomized
within the workspace. Success is judged geometrically.

See ``EVAL1_PLAN.md`` (project root) for the design rationale; this file
is the concrete realization of §3 of that document. The state-only Day-3
milestone is done; this file is now the Day-4 vision configuration —
``wrist_cam`` is parented to ``gripper_link`` and ``wrist_rgb`` is the
deployable image observation. Block ground-truth lives only on the
asymmetric *critic* group from now on.
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaac_so_arm101.tasks.pickplace.mdp as mdp
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Wrist-camera resolution
# ---------------------------------------------------------------------------
#
# 16:9 to match the real wrist USB cam (1280×720 native — see
# ``camera_intrinsics.yaml``). 128×72 is a 1/10 scale of native, which
# preserves the FOV exactly and keeps the per-env render cost cheap. Real
# horizontal FOV ≈ 102° — matters for "block out of frame" search behavior
# (EVAL1_PLAN §4); square-cropping would lose ~32° and hurt search.
#
# The CNN encoder in :mod:`agents.vision_actor_critic` is sized for these
# dimensions; bump them in lockstep if the cube becomes too small to
# resolve at the typical 0.20 m gripper-down standoff.
WRIST_RGB_WIDTH = 128
WRIST_RGB_HEIGHT = 72


@configclass
class PickPlaceBowlSceneCfg(InteractiveSceneCfg):
    """Scene with: ground, gray table, SO-ARM101 robot, block, ee frame, wrist cam.

    No bowl prim — the bowl lives only as a ``CommandTerm`` (see
    :class:`CommandsCfg`). The robot, ee_frame, object, and wrist_cam slots
    are filled in by ``joint_pos_env_cfg`` (matches the lift-task pattern
    upstream).
    """

    # filled in by joint_pos_env_cfg
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg | DeformableObjectCfg = MISSING
    # Wrist camera: TiledCamera (batched render) parented to gripper_link.
    # Intrinsics are baked from camera_intrinsics.yaml in joint_pos_env_cfg.
    wrist_cam: TiledCameraCfg = MISSING

    # Flat gray cuboid table matching eval color (#B8ADA9 → linear sRGB
    # ≈ (0.722, 0.678, 0.663)). 1 m × 1 m × 2 cm thick; top of the table
    # sits at z=0 so block init z=0.01 is "block on table".
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.2, 0.0, -0.01]),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.722, 0.678, 0.663),  # #B8ADA9
                roughness=0.7,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


# ---------------------------------------------------------------------------
# Commands — bowl goal
# ---------------------------------------------------------------------------


@configclass
class CommandsCfg:
    """Bowl goal sampled per episode within the SO-ARM101 reachable workspace.

    Default ranges match §3.1 of EVAL1_PLAN. Tighten these once we measure
    the *actual* reachable workspace on the real robot. We reuse Isaac Lab's
    :class:`UniformPoseCommandCfg` rather than a custom command — only the
    (x, y) component is consumed downstream by obs / reward / termination.
    """

    bowl_pose = mdp.BowlPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # set in joint_pos_env_cfg (visualization marker only)
        # Bowl resampled only on episode reset (resampling time ≥ episode
        # length so it never resamples mid-episode).
        resampling_time_range=(6.0, 6.0),
        debug_vis=True,
        # Rejection sampling keeps the bowl ≥ 10 cm from the block at xy.
        # Without this, ~10–20 % of resets put the block inside r_safe of
        # the bowl, which lets a stationary block farm the place reward
        # without ever being grasped (observed at iter 99 of the smoke run).
        target_asset_name="object",
        min_distance=0.10,
        max_attempts=8,
        ranges=mdp.BowlPoseCommandCfg.Ranges(
            pos_x=(0.10, 0.30),
            pos_y=(-0.15, 0.15),
            pos_z=(0.0, 0.0),  # bowl on the table
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@configclass
class ActionsCfg:
    """Joint-position around home; gripper is a binary open/close.

    Filled in by :mod:`joint_pos_env_cfg`. We choose joint-position control
    (rather than IK) because it maps 1:1 to Feetech ``goal_position`` writes
    on the real arm — the only conversion is rad ↔ servo counts.
    """

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


# ---------------------------------------------------------------------------
# Observations — policy + asymmetric critic groups
# ---------------------------------------------------------------------------


@configclass
class ObservationsCfg:
    """Three obs groups: deployable state, privileged state, wrist image.

    The state-only Day-3 milestone is done; we no longer feed
    ``block_position`` to the policy. The actor must now infer the block's
    location from ``wrist_rgb`` alone, while the asymmetric critic still
    receives privileged ground-truth (block pose, distances, ``is_grasped``).

    Three separate groups (rather than two) so that the image keeps its 4-D
    layout ``(N, C, H, W)`` and isn't flattened into the 1-D state vector.
    The custom :class:`PickPlaceVisionActorCritic` reads ``wrist_rgb``
    through its CNN encoder before concatenating with the state features.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Deployable *state* observations — 1-D, concatenated.

        Reproducible on the real arm from servo telemetry + the bowl arg +
        URDF FK. **No** block position here — the policy must localize the
        block via ``wrist_rgb`` (see :class:`WristRgbCfg`).
        """

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        gripper_state = ObsTerm(func=mdp.gripper_state)
        bowl_xy = ObsTerm(func=mdp.bowl_xy, params={"command_name": "bowl_pose"})
        ee_proj_xy = ObsTerm(func=mdp.ee_proj_xy)
        ee_to_bowl_xy = ObsTerm(func=mdp.ee_to_bowl_xy, params={"command_name": "bowl_pose"})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations — discarded at deploy.

        Includes everything the policy sees plus block pose, distances, and
        the heuristic ``is_grasped`` flag. RSL-RL with asymmetric A-C reads
        this group via ``obs_groups``.
        """

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        gripper_state = ObsTerm(func=mdp.gripper_state)
        bowl_xy = ObsTerm(func=mdp.bowl_xy, params={"command_name": "bowl_pose"})
        ee_proj_xy = ObsTerm(func=mdp.ee_proj_xy)
        ee_to_bowl_xy = ObsTerm(func=mdp.ee_to_bowl_xy, params={"command_name": "bowl_pose"})
        block_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        block_to_bowl_xy = ObsTerm(func=mdp.block_to_bowl_xy, params={"command_name": "bowl_pose"})
        gripper_to_block = ObsTerm(func=mdp.gripper_to_block)
        is_grasped = ObsTerm(func=mdp.is_grasped)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class WristImageCfg(ObsGroup):
        """5-channel wrist-camera image — ``(N, 5, H, W)``, **not** concatenated.

        Channels: RGB (0–2) + clipped/normalized depth (3) + binary cube
        mask (4). All in ``[0, 1]``. Sim and real-side deploy feed the
        same shape into the same CNN; on real, channels 3 and 4 come from
        Depth Anything 3 and HSV thresholding respectively. See
        :func:`mdp.wrist_image` for per-step DR (brightness, noise, depth
        scale jitter mimicking DA3 artifacts) and
        :func:`mdp.randomize_wrist_image_tint` for per-episode color tint
        (sampled at reset).

        Single-term group so the image keeps its spatial layout
        downstream. The custom CNN-based actor-critic recognizes this
        group by name (``wrist_image``) and routes it through the conv
        encoder. ``corrupt=False`` on the play variant via
        ``params={"corrupt": False}`` (set in
        ``SoArm101PickPlaceBowlEnvCfg_PLAY.__post_init__``).
        """

        wrist_image = ObsTerm(func=mdp.wrist_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True  # single term — concat is a no-op

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    wrist_image: WristImageCfg = WristImageCfg()


# ---------------------------------------------------------------------------
# Events — resets + (later) domain randomization
# ---------------------------------------------------------------------------


@configclass
class EventCfg:
    """Reset events. Domain randomization knobs (visual / dynamics) are
    layered in once state-only training is solved (EVAL1_PLAN §3.7)."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Clear the per-episode lift latch (``env._was_grasped``) maintained
    # by :func:`mdp.rewards._episode_lifted_mask`. Must run on every reset
    # so the latch can't persist into the next episode and let the policy
    # farm place / release without lifting again. See
    # :func:`mdp.events.reset_was_grasped` for rationale.
    reset_lift_latch = EventTerm(func=mdp.reset_was_grasped, mode="reset")

    reset_block_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # IMPORTANT: ``pose_range`` is a *delta* added to the asset's
            # default position (``InitialStateCfg.pos = [0.2, 0.0, 0.01]``),
            # not an absolute range. The block's default x is 0.2 (mid-
            # workspace), so x∈[-0.1, 0.1] keeps the block inside the bowl
            # workspace x∈[0.10, 0.30]. Same logic for y around 0.0.
            #
            # Initial pose_range tightened ±2 cm × ±2 cm to put the block
            # directly under the EE home pose (EE_HOME_B≈(0.24, 0, 0.08),
            # block default (0.2, 0, 0.01) — only 4 cm of x offset, ~7 cm
            # descent for grasp). This is a "geometric pre-grasp curriculum"
            # — the block is always inside the wrist camera's FoV at home
            # pose, so the CNN gets dense visual signal. The range is
            # expanded by ``CurriculumCfg.block_range_expand`` toward the
            # full ±10 cm × ±15 cm over training. See
            # :func:`mdp.events.expand_block_xy_range` for the schedule.
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    # Bootstrap-grasp event — bumped 0.10 → 0.50 after run-13 (2026-05-09
    # post-gripper-σ-fix, 400 iters) confirmed the diagnosis: with only
    # 10% bootstrap, PPO never sees enough sustained-grasp trajectories
    # to credit-assign back to "close gripper at cube". grasp_from_scratch
    # peaked at 0.0009 around iter 150 then decayed back to 0; pre_grasp
    # crept back to 0.30; grasp Episode_Reward fell 0.083 → 0.016 over
    # iters 100→400. With p=0.50 half the rollouts contain post-grasp
    # value-function tail that propagates back via advantage to the
    # close-and-hold action.
    #
    # Risk: ``bootstrap-curriculum-pitfall`` memory documents that policy
    # can ride the subsidy and never learn from-scratch grasp. Mitigation:
    # ``grasp_from_scratch`` and ``release_from_scratch`` curriculum
    # metrics are computed *only* over the non-bootstrapped 50%, so they
    # remain a clean signal of from-scratch competence. Stop the run if
    # they don't break above 0.005 by iter ~500.
    bootstrap_grasped = EventTerm(
        func=mdp.init_block_in_gripper,
        mode="reset",
        params={"p_grasped": 0.50, "gripper_closed_q": 0.05},
    )

    # Per-episode wrist-image color tint — substitutes for material-level
    # cube/table-color DR (see :func:`mdp.randomize_wrist_image_tint`
    # docstring for why this lives in obs-space rather than scene-space).
    # rgb_scale ±30% covers ManiSkill3 / CS6341 cube color envelopes
    # (the wood block on a near-gray table is well within ±30% of either
    # axis). brightness ±0.15 layers a small global offset on top, which
    # is the obs-space proxy for dome-light intensity DR.
    randomize_wrist_image_tint = EventTerm(
        func=mdp.randomize_wrist_image_tint,
        mode="reset",
        params={
            "rgb_scale_range": (0.7, 1.3),
            "brightness_range": (-0.15, 0.15),
        },
    )

    # Wrist-cam extrinsic DR — ported from LeIsaac (cf. pick_orange_env_cfg.py
    # ±25 mm / ±2.5° on the wrist + front cams; their known-good envelope
    # for SO101 teleop sim-to-real). Per-reset uniform jitter on top of
    # the OffsetCfg-defined default mount, simulating WOWROBO bracket
    # variance and small caliper-measurement error on day 6. Wider than
    # the EVAL1_PLAN §3.8 first cut (±2 mm / ±1°) — at this RL stage
    # there's no reason to be tighter than what teleop demos already
    # transfer through.
    randomize_wrist_cam_pose = EventTerm(
        func=mdp.randomize_camera_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("wrist_cam"),
            "pose_range": {
                "x": (-0.025, 0.025),
                "y": (-0.025, 0.025),
                "z": (-0.025, 0.025),
                "roll": (-2.5 * math.pi / 180, 2.5 * math.pi / 180),
                "pitch": (-2.5 * math.pi / 180, 2.5 * math.pi / 180),
                "yaw": (-2.5 * math.pi / 180, 2.5 * math.pi / 180),
            },
            "convention": "ros",
        },
    )


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


@configclass
class RewardsCfg:
    """Stage-gated rewards (see EVAL1_PLAN §3.5).

    Magnitudes here matter as much as signs: too-large reach or grasp
    weights cause the policy to camp at intermediate stages and never
    explore release. The numbers below are the EVAL1_PLAN starting point.
    """

    # Pre-grasp pose reward re-enabled at w=2.0 after runs 9 & 10
    # (2026-05-09 with bootstrap floor p=0.10 and σ=0.5 respectively)
    # both stalled at gfs=0.0000 despite stable reach saturation. The
    # bootstrap subsidy alone wasn't propagating into from-scratch grasp
    # discovery — bootstrapped envs skip the "approach with open jaws +
    # close around block" decision sequence the actor's CNN is supposed
    # to learn from vision, so credit assignment from grasp reward back
    # to "close gripper at cube" never bridged. ``pre_grasp_pose``
    # provides exactly that intermediate gradient: it pays
    # ``proximity * jaw_openness * (1 - is_grasped)`` per step, gated
    # off the moment a grasp succeeds. Per its docstring rationale,
    # this decomposes the temporally-extended grasp prerequisite into a
    # step-level signal PPO can hill-climb on, and the once-grasped
    # gating ensures attempting a grasp is *net-positive*: lose ≤ 2/step
    # pre_grasp, gain +15/step grasp. The original disable-comment
    # (above the reach term) called bumping reach the "lower-risk
    # version of this idea" — that lower-risk version failed at iter
    # 615 (run 7 reach=5) and again at iter 250 (run 10 reach=1 +
    # bootstrap + low σ), so we're back to the design's intended fix.
    # Weight halved 2.0 → 1.0 alongside the per-episode-latch gating fix in
    # ``mdp.pre_grasp_pose`` (run-11 diagnostic 2026-05-09: at w=2.0 the
    # term saturated at 0.96/episode while ``grasp`` paid only 0.015 →
    # policy converged to "park EE near cube with jaws open" attractor;
    # ``grasp_bootstrap`` decayed 0.41→0.006 over 100 iters). The
    # latch fix removes the open-jaws-after-fumble exploit; halving the
    # weight additionally shallows the open-jaws basin so PPO's
    # exploration noise can find the close-and-lift action sequence.
    pre_grasp = RewTerm(func=mdp.pre_grasp_pose, params={"std": 0.05}, weight=1.0)

    # Reach weight reverted 5.0 → 1.0 after run 7 (2026-05-08_23-57-07,
    # 615 iters, 30 M env-steps) plateaued at reach-only with grasp=0
    # across every env. At w=5 the reach reward is 5× upstream lift's
    # ``reaching_object`` (Franka lift uses w=1.0 std=0.10; our local
    # ``tasks/lift/lift_env_cfg.py`` uses w=1.0 std=0.05). The earlier
    # bump's premise — "reach has to be large enough to compete with
    # bootstrap-derived transport+place+release" — no longer applies
    # since run 6 disabled bootstrap (``p_grasped=0.0``). With nothing
    # else paying out, w=5 created a steep narrow attractor at the
    # cube position and the policy converged to "hover above cube
    # with gripper open," exactly the failure mode upstream avoids by
    # keeping reach an order of magnitude weaker than the lift bonus.
    # The lift channel itself is already in place — ``grasp`` below
    # uses ``_grasped_mask`` which is now a pure ``block_z > 0.025``
    # height check (see :func:`mdp.rewards._grasped_mask` docstring),
    # making it the structural twin of upstream's ungated
    # ``object_is_lifted``. ``std=0.05`` is kept matching the local
    # SO-ARM lift idiom (Franka uses 0.10, but our cube and
    # workspace are smaller).
    reach = RewTerm(func=mdp.reach_block, params={"std": 0.05}, weight=1.0)
    grasp = RewTerm(func=mdp.grasp_event, weight=15.0)
    transport = RewTerm(func=mdp.transport_to_bowl, params={"std": 0.15}, weight=4.0)
    place = RewTerm(func=mdp.place_in_bowl, weight=5.0)
    # Release weight bumped 10 → 20 so the released-into-bowl state pays
    # ~20/step continuously vs the hold-over-bowl state at ~6.4/step. With
    # the success termination removed, this differential makes release
    # the strictly better strategy.
    release = RewTerm(func=mdp.release_in_bowl, weight=20.0)

    # Penalties — small in early training; CurriculumCfg below ramps them
    # up to discourage jittery actions once the policy is competent.
    action_l2 = RewTerm(func=mdp.action_l2, weight=-1e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    drop = RewTerm(func=mdp.block_dropped, weight=-2.0)

    # NOTE: per-bootstrap-status metrics live in CurriculumCfg.log_metrics,
    # not here. CurriculumTerm dict returns are auto-logged to TB; reward
    # side-effect writes to extras["log"] are not.


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@configclass
class TerminationsCfg:
    """Time-out and a workspace-box safety termination.

    Note: we intentionally do **not** use the ``task_success`` predicate as
    a termination. Earlier we did and observed (in the v2 1500-iter run)
    that the policy actively avoided releasing because the success
    termination ended the episode after one step of release reward,
    making "hold and hover" 5× more lucrative than "release". With
    success-termination removed, ``release_in_bowl`` reward accumulates
    per-step until time_out, and the policy learns to release. ``success``
    is still tracked as a TB metric (``Metrics/task_success`` via the
    reward term ``release``).
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    block_off_table = DoneTerm(func=mdp.block_off_table)


# ---------------------------------------------------------------------------
# Curriculum — match the lift task: ramp jitter penalties later in training
# ---------------------------------------------------------------------------


@configclass
class CurriculumCfg:
    """Action/joint-vel penalty ramp + geometric pre-grasp curriculum.

    The action/joint_vel ramps mirror the upstream lift task's curriculum
    (kicks in at step 10 000 ≈ iter 417). The ``block_range_expand`` term
    drives the new (run 6) pre-grasp curriculum: block xy randomization
    starts at ±2 cm × ±2 cm (block always under EE home, always in wrist
    camera FoV) and expands to ±10 cm × ±15 cm over training, replacing
    the run-5 bootstrap-grasp curriculum that failed to teach visual
    reach+grasp.
    """

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 10000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 10000},
    )
    # Per-bootstrap-status TB metrics (zero side effects on training).
    # Returns a dict each step; CurriculumManager logs each key as
    # ``Curriculum/log_metrics/<key>``. The two scalars to actually
    # watch on TB are ``release_from_scratch`` and ``grasp_from_scratch``
    # — these are the deployable-regime success rates that runs 1-4
    # were missing visibility into.
    log_metrics = CurrTerm(func=mdp.log_bootstrap_metrics)

    # Run-6 replacement for the failed bootstrap-grasp curriculum. The
    # block xy randomization radius starts tight (±2 cm) so the cube is
    # always under the EE home pose and inside the wrist camera FoV,
    # making visual reach learnable from iter 0. Range expands linearly
    # to ±10 cm × ±15 cm (the original full-task range) over the
    # 12k+180k = 192k env-step schedule, matching the prior decay_steps.
    # See :func:`mdp.events.expand_block_xy_range` for the math.
    block_range_expand = CurrTerm(
        func=mdp.expand_block_xy_range,
        params={
            "initial_xy": (0.02, 0.02),
            "final_xy": (0.10, 0.15),
            "warmup_steps": 12_000,
            "expand_steps": 180_000,
            "event_term_name": "reset_block_position",
        },
    )


# ---------------------------------------------------------------------------
# Top-level env cfg
# ---------------------------------------------------------------------------


@configclass
class PickPlaceBowlEnvCfg(ManagerBasedRLEnvCfg):
    """SO-ARM101 single-object pick-and-place into a bowl-goal.

    Concrete robot/ee/object/action wiring lives in
    :mod:`joint_pos_env_cfg` so this file stays robot-agnostic.
    """

    scene: PickPlaceBowlSceneCfg = PickPlaceBowlSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 50 Hz control rate (sim 100 Hz physics, decimation=2). Matches
        # EVAL1_PLAN §0 / §3 — the same rate the deploy loop runs on the
        # real Feetech bus, so action-scale dynamics carry over directly.
        self.decimation = 2
        self.episode_length_s = 6.0  # 300 steps @ 50 Hz
        self.viewer.eye = (2.5, 2.5, 1.5)

        self.sim.dt = 0.01  # 100 Hz physics
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
