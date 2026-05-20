# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based RL env configuration for SO-ARM101 single-block pick-and-place.

Source-of-truth scaffold for Eval 1 of Project 3. The bowl is **not** a
scene prim — it is a 2-D goal sampled per episode by the command manager
(``bowl_pose``). The block is a 2 cm cube whose initial xy is randomized
within the workspace. Success is judged geometrically.

See ``docs/EVAL1_PLAN.md`` (project root) for the design rationale; this file
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
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
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

    # v7 (2026-05-15): contact sensors on both gripper jaws, filtered for the
    # cube object. PhysX reports contact forces between the listed prims and
    # the filter set. ``is_grasping_contact`` reward function reads
    # ``env.scene[<name>].data.net_forces_w`` to derive a true physical
    # contact signal (replaces v3–v6's kinematic ``closed_grasp_signal``
    # proxy). Sim-only — the deployed policy never sees this data; it's
    # used only in the reward function during training. Both jaws are
    # monitored separately because a real grasp involves contact on BOTH
    # the fixed jaw (``gripper_link``) and the moving jaw
    # (``moving_jaw_so101_v1_link``); requiring contact on both filters out
    # one-sided pushes that aren't true grasps.
    gripper_contact_fixed = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    gripper_contact_moving = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1_link",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )

    # Flat gray cuboid table matching eval color (#B8ADA9 → linear sRGB
    # ≈ (0.722, 0.678, 0.663)). Top of the table sits at z=0 so block
    # init z=0.01 is "block on table".
    #
    # Sim-to-real geometry match: on the real rig the SO-ARM101 base is
    # clamped at the table edge, so the table extends ~0 cm behind the
    # arm base (origin). Previously the table was 1 m square centered at
    # x=0.2, putting 30 cm of table *behind* the base — which the deployed
    # wrist-cam never sees on the real setup, creating a background
    # distribution mismatch. New x-extent is 0.6 m: back edge at x=-0.05
    # (5 cm clearance for the ~9 cm base footprint), front edge at x=0.55
    # (≥25 cm beyond the max reach of 0.30, plenty of visual context).
    # y unchanged at 1 m — the side bezels aren't a sim-to-real issue.
    # Workspace dependencies (bowl ranges [0.10, 0.30] × ±0.15, block init
    # x=0.20 ± 0.10) are all comfortably inside the new table.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.25, 0.0, -0.01]),
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 1.0, 0.02),
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
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        # Rejection sampling keeps the bowl ≥ 15 cm from the block at xy.
        # Without this, ~10–20 % of resets put the block inside r_safe of
        # the bowl, which lets a stationary block farm the place reward
        # without ever being grasped (observed at iter 99 of the smoke run).
        # Bumped 0.12 → 0.15 (2026-05-20): the prior 12 cm still allowed
        # block/bowl pairs that visually overlap from the wrist-cam standoff
        # (~20 cm), leaving ambiguity about which one the policy should
        # reach for. 15 cm is roughly half the bowl x-range (0.15–0.28),
        # so the two assets reliably land in clearly different parts of the
        # workspace and "transport" is the only path to reward.
        # max_attempts bumped 8 → 16 so the tighter constraint has enough
        # budget to converge (rejection rate is higher with min_distance up).
        target_asset_name="object",
        min_distance=0.15,
        max_attempts=16,
        ranges=mdp.BowlPoseCommandCfg.Ranges(
            # 2026-05-20: re-expanded back to the URDF-reachable limits
            # x∈[0.10, 0.30], y∈±0.15 as an experiment. Background:
            # we'd previously tightened to x∈(0.15, 0.28), y∈±0.12
            # because ~10-20 % of episodes hit unreachable goals at the
            # limit and added gradient noise. Re-trying the wider range
            # now because (a) the upstream MuammerBay/isaac_so_arm101
            # lift task uses even wider (y∈±0.20) without trouble,
            # (b) we have a converged teacher + critic warm-start now
            # (vs the cold-start runs where unreachable goals hurt
            # most), and (c) wider workspace forces the vision policy
            # to generalize over the full reachable set rather than a
            # convenience subset. If success_rate stalls / σ inflates
            # in the first ~500 PPO iters at Stage 3, revert to the
            # comfortable band (x∈[0.15, 0.28], y∈±0.12).
            pos_x=(0.10, 0.30),
            pos_y=(-0.15, 0.15),
            # Goal z = 0 (table level). After release, the 2 cm cube sits
            # on the bowl floor (which sits on the table) → cube center
            # at z = 0.01. With std=0.20 in transport's tanh, the 1 cm
            # offset between cube center and goal is 95% saturated —
            # negligible. Lift gate is now per-EPISODE-latch (cube was
            # lifted at some step) NOT per-step, so transport keeps
            # paying after release when the cube settles in the bowl
            # (whereas per-step gate would zero out the reward at the
            # moment of successful placement — wrong for our task).
            pos_z=(0.0, 0.0),
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
        """4-channel wrist-camera image — ``(N, 4, H, W)``, **not** concatenated.

        Channels: RGB (0–2) + binary cube mask (3). All in ``[0, 1]``. Sim
        and real-side deploy feed the same shape into the same CNN; on
        real, channel 3 comes from HSV thresholding (``cv2.inRange``). See
        :func:`mdp.wrist_image` for per-step DR (RGB brightness + noise;
        mask morphology jitter + dropout) and
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
    """Reset events — VERBATIM stock Isaac Lab Franka Lift recipe.

    Stock Franka Lift converges this exact MDP class (state-PPO + binary
    gripper + pick-and-place) reliably in ~1500 iters. Their EventCfg has
    *only* two terms: ``reset_all`` and ``reset_object_position``. No
    bootstrap, no DR, no latch reset. Run-18 diagnostic (2026-05-10)
    confirmed our additions to this base (bootstrap, visual DR,
    cam-pose DR, lift-latch reset) all created variance in the policy
    gradient that pushed σ to inflate and PPO to oscillate. We strip
    everything Franka doesn't have and trust the proven recipe.

    With release/place/latch all gone from RewardsCfg, the lift latch is
    no longer used anywhere — so reset_was_grasped is dropped too.
    Visual DR is gone for the teacher; if/when distillation produces a
    vision student, DR can be added back at that stage (the camera is
    still rendered each step but no obs term reads it for the teacher).
    """

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Re-enabled for stage-2 (release fine-tune). Clears the per-episode
    # ``env._was_grasped`` lift latch, which ``release_in_bowl`` gates on
    # (only pays after the policy has lifted the cube ≥ 0.025 m at some
    # prior step in the same episode → drag-on-table exploit blocked).
    reset_lift_latch = EventTerm(func=mdp.reset_was_grasped, mode="reset")

    # Clears the per-episode ``env._was_over_bowl_above_rim`` latch — the
    # "approached from above the rim" precondition that
    # ``place_in_bowl`` / ``release_in_bowl`` / ``task_success`` all gate
    # on. Without this reset the latch sticks True after the first
    # over-bowl-high pass, defeating the gate on subsequent episodes.
    reset_rim_latch = EventTerm(func=mdp.reset_was_over_bowl_above_rim, mode="reset")

    reset_block_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # 2026-05-20: re-expanded to URDF-reachable limits to
            # match CommandsCfg.bowl_pose. Block default is
            # (0.2, 0, 0.01); deltas x∈(-0.10, 0.10), y∈(-0.15, 0.15)
            # → absolute x∈[0.10, 0.30], y∈[-0.15, 0.15] — same band
            # as the bowl goal so all (block, goal) pairs remain
            # solvable. See the bowl ranges docstring for the rollback
            # plan if Stage-3 training degrades.
            "pose_range": {"x": (-0.10, 0.10), "y": (-0.15, 0.15), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


@configclass
class RewardsCfg:
    """Reward terms — direct port of Isaac Lab's stock Franka Lift recipe.

    Reference: ``isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg``
    on this same Isaac Lab install. That task is a known-good state-PPO
    pick-and-place benchmark (skrl reports it converging in ~1500 iters at
    4096 envs). Across runs 11–15 of this repo we drifted away from the
    stock recipe — added ``pre_grasp_pose`` (camping attractor), bootstrap
    event (curriculum pitfall), per-episode lift latch, ``release_in_bowl``,
    ``block_dropped`` — each fix accumulating reward-stack complexity that
    fought itself. This commit reverts to the stock 4-term recipe verbatim.

    The 4 terms (matching stock):
      * ``reaching_object``  — dense ee→cube tanh, w=1.0, std=0.10.
      * ``lifting_object``   — sparse indicator block_z > 0.04, w=15.0.
      * ``object_goal_tracking`` — dense cube→goal_xyz tanh, **lift-gated
        per-step**, w=16.0, std=0.30. Goal is bowl_xy at z=0.10 (set in
        ``CommandsCfg.bowl_pose.ranges.pos_z``).
      * ``object_goal_tracking_fine_grained`` — same fn at std=0.05, w=5.0.

    Penalties:
      * ``action_rate``, ``joint_vel`` — tiny (-1e-4) → ramped via curriculum.

    Removed (vs prior state):
      * ``pre_grasp_pose`` — the open-jaws camping attractor.
      * ``place_in_bowl``, ``release_in_bowl`` — over-engineered for a teacher
        that just needs to grasp+transport. Will re-add a single sparse
        release term as a fine-tune stage *after* base teacher converges.
      * ``block_dropped`` — stock has no analog. With per-step lift gate
        on goal_tracking, dropping already costs reward implicitly.
    """

    # SO-ARM-tuned values from upstream isaac_so_arm101/tasks/lift/
    # (proven recipe in this same repo). std=0.05 sharper than Franka's
    # 0.10 — couples reach gradient tighter to the cube position because
    # SO-ARM workspace is smaller. Lift threshold 0.025 (vs Franka's
    # 0.04) matches the smaller cube and shorter lift travel.
    reaching_object = RewTerm(func=mdp.reach_block, params={"std": 0.05}, weight=1.0)

    # ``minimal_height`` bumped 0.025 → 0.07 to match
    # ``release_in_bowl``'s lift latch (which is gated on the real bowl's
    # 5 cm rim — see release_in_bowl docstring). Aligning lift-bonus and
    # release-latch thresholds means the policy gets continuous lift
    # feedback exactly when it's clearing rim height, which is the height
    # band the deployed trajectory needs anyway.
    lifting_object = RewTerm(
        func=mdp.grasp_event,
        params={"minimal_height": 0.07},
        weight=15.0,
    )

    # Dense cube→goal tracking, lift-gated *per-step* (stock semantics —
    # not the per-episode latch we had). Goal pose lives in the
    # ``bowl_pose`` command at z=0.10 (above the bowl, lifted).
    # ``minimal_height`` matches lifting_object (0.025).
    object_goal_tracking = RewTerm(
        func=mdp.transport_to_bowl,
        params={"std": 0.30, "minimal_height": 0.025, "command_name": "bowl_pose"},
        weight=16.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.transport_to_bowl,
        params={"std": 0.05, "minimal_height": 0.025, "command_name": "bowl_pose"},
        weight=5.0,
    )

    # Stage-2 fine-tune (release into bowl). Re-added at moderate weight
    # after stage-1 teacher converged with mean reward ≈ 118
    # (lift saturated 12.4, transport 9.8, fine-grained 0.9). The
    # release_in_bowl predicate gates on: cube xy near bowl AND
    # cube_z < bowl_height AND gripper open AND cube settled AND
    # per-episode lift latch (env._was_grasped). Same predicate used by
    # Isaac Lab's stack/cubes_stacked termination (verified equivalent
    # via upstream survey 2026-05-10), applied as a per-step reward
    # rather than a termination so the policy can keep collecting reward
    # for staying released-in-bowl through episode end.
    #
    # Weight rationale: stage-1 hover saturates at reach(1) + lift(15) +
    # transport(16) + transport_fine(5) ≈ 37/step. Setting release w=30
    # makes "release-and-stay" pay ~31/step (small reach loss after EE
    # leaves cube). Slightly *less* than hover per step BUT stays high
    # for all remaining episode steps after release, while hover requires
    # constant precise control to maintain. With γ=0.98, value-function
    # PV of "release-and-stay" trajectories should beat "hover-forever"
    # via the smaller variance of post-release rewards. Starting from
    # the converged stage-1 checkpoint (vs random init) avoids the
    # high-σ chaos that destabilized prior runs (run-18 used w=50 +
    # bootstrap p=0.10 from random init → σ inflated, training failed).
    release_in_bowl = RewTerm(func=mdp.release_in_bowl, weight=30.0)

    # OLD COMMENT (pre-stage-2): NO release reward — stage 1 is a
    # strict Franka Lift match. Run-18
    # (2026-05-10, w=50, p=0.10 bootstrap) showed release_in_bowl created
    # a high-variance reward landscape that destabilized PPO: 50/step in
    # a narrow region only, toggling on/off as cube enters/leaves the
    # release condition → policy gradient blew up → σ inflated → PPO
    # oscillated. Stock Franka has no release; their teacher hovers at
    # goal indefinitely. Stage 2 (after stage-1 teacher converges) will
    # resume from the stage-1 checkpoint with a small release reward
    # added as a fine-tune.

    # Penalties — initial -1e-4 ramped to -1e-2 by ``CurriculumCfg`` at
    # 10 k env-steps. Stock Franka Lift ramps to -1e-1 (10× heavier);
    # we kept -1e-2 to give the SO-ARM more action-space freedom early.
    # Stock recipe doesn't include action_l2; we keep it off too.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # Soft standoff penalty protecting the real wrist camera from crashing
    # into the table. ``margin=0.03`` (3 cm) chosen to clear plausible grasp
    # postures (gripper_link sits ~9 cm above the EE tip; with the cam
    # offset along gripper +y/-z, world cam_z at typical grasp height
    # stays well above 3 cm). Weight -50 puts max single-step penalty at
    # ~-1.5 (cam right on the table) — same order as ``reaching_object``
    # but well below the post-grasp terms, so it steers without dominating.
    wrist_cam_clearance = RewTerm(
        func=mdp.wrist_cam_table_clearance,
        params={"margin": 0.03, "table_top_z": 0.0},
        weight=-50.0,
    )


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
    """Action/joint-vel penalty ramp + block-xy expand for Stage 3 warm-start.

    Stock Franka has exactly two curriculum terms (action_rate, joint_vel
    ramps at 10 000 env-steps). We add ``block_range_expand`` back with a
    **warm-start-adapted** schedule (NOT the original cold-start schedule):

    * Warm-up: ±3 cm × ±3 cm for the first 5 000 env-steps (~150 PPO iters
      at 2048 envs × 16 steps).
    * Expand: linearly to the live full width (±10 cm × ±15 cm in
      reset_block_position) over the next 30 000 env-steps (~900 iters).
    * After ~35 000 env-steps the schedule is at its final width and the
      curriculum is effectively a no-op.

    Why: with the warm-started actor sitting in the imitation basin and a
    freshly-initialized critic, exposing the policy to full block-xy
    variance from iter 0 multiplies advantage-estimate variance and pushes
    the actor out of the basin before the critic can catch up. A short
    narrow-start period gives the critic a low-variance return distribution
    to fit against; once it has caught up, the curriculum expands to full
    width and PPO trains on the real task.

    Cost on Stage 1 (teacher) is negligible — the schedule completes in
    ~35 PPO iters out of 1500. On Stage 2 (distill) it's slightly
    beneficial: the student first imitates the teacher on easy block
    positions, then on hard ones. EVAL1_PLAN §7.2 intervention #4.
    """

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 10000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 10000},
    )
    # Stage-3-adapted block-xy expand. Ramps the reset_block_position event's
    # pose_range from ±3 cm to the live full width (±10×±15). Live full-width
    # is set in EventCfg.reset_block_position above; if you tighten that,
    # tighten ``final_xy`` here to match.
    block_range_expand = CurrTerm(
        func=mdp.expand_block_xy_range,
        params={
            "initial_xy": (0.03, 0.03),
            "final_xy":   (0.10, 0.15),
            "warmup_steps":  5_000,
            "expand_steps":  30_000,
            "event_term_name": "reset_block_position",
        },
    )
    # Binary task-success rate (% of ended episodes where release_in_bowl
    # fired at any step). Updated as a side effect of ``release_in_bowl``;
    # read here at reset_idx. Emits ``Curriculum/log_success/success_rate``
    # and ``.../n_episodes_ended`` to TB. See
    # :func:`mdp.rewards.log_success_metrics` for the measurement details.
    log_success = CurrTerm(func=mdp.log_success_metrics, params={})


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
        self.episode_length_s = 5.0  # 250 steps @ 50 Hz, matches upstream lift
        self.viewer.eye = (2.5, 2.5, 1.5)

        self.sim.dt = 0.01  # 100 Hz physics
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
