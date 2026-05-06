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
is the concrete realization of §3 of that document. State-only training
(no wrist camera yet) is the Day 3 milestone before vision is layered on.
"""

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
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class PickPlaceBowlSceneCfg(InteractiveSceneCfg):
    """Scene with: ground, gray table, SO-ARM101 robot, block, ee frame.

    No bowl prim — the bowl lives only as a ``CommandTerm`` (see
    :class:`CommandsCfg`). The robot, ee_frame, and object slots are filled
    in by ``joint_pos_env_cfg`` (matches the lift-task pattern upstream).
    """

    # filled in by joint_pos_env_cfg
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

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

    bowl_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # set in joint_pos_env_cfg (visualization marker only)
        # Resampling time matches episode_length_s so the bowl is fixed
        # within an episode.
        resampling_time_range=(6.0, 6.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
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
    """Policy obs is what the real robot can produce; critic obs is privileged.

    Day-1 / Day-3 milestone is *state-only* training: ``block_position``
    is included in the policy group as a stand-in for the wrist camera so
    we can sanity-check reward and DR before turning vision on. When we
    move to vision, that term gets pulled out of ``policy`` and replaced
    with a ``wrist_rgb`` term.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Deployable observations (must be reproducible on the real arm)."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        gripper_state = ObsTerm(func=mdp.gripper_state)
        bowl_xy = ObsTerm(func=mdp.bowl_xy, params={"command_name": "bowl_pose"})
        ee_proj_xy = ObsTerm(func=mdp.ee_proj_xy)
        ee_to_bowl_xy = ObsTerm(func=mdp.ee_to_bowl_xy, params={"command_name": "bowl_pose"})
        # State-only stand-in for the wrist camera — replace with wrist_rgb
        # once Day-3 success criteria are met (see EVAL1_PLAN §4.2 Step I).
        block_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations — discarded at deploy.

        Includes everything the policy sees plus block pose, distances and
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

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ---------------------------------------------------------------------------
# Events — resets + (later) domain randomization
# ---------------------------------------------------------------------------


@configclass
class EventCfg:
    """Reset events. Domain randomization knobs (visual / dynamics) are
    layered in once state-only training is solved (EVAL1_PLAN §3.7)."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_block_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # IMPORTANT: ``pose_range`` is a *delta* added to the asset's
            # default position (``InitialStateCfg.pos = [0.2, 0.0, 0.01]``),
            # not an absolute range. The block's default x is 0.2 (mid-
            # workspace), so x∈[-0.1, 0.1] keeps the block inside the bowl
            # workspace x∈[0.10, 0.30]. Same logic for y around 0.0.
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.15, 0.15), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
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

    reach = RewTerm(func=mdp.reach_block, params={"std": 0.1}, weight=1.0)
    grasp = RewTerm(func=mdp.grasp_event, weight=5.0)
    transport = RewTerm(func=mdp.transport_to_bowl, params={"std": 0.15}, weight=2.0)
    place = RewTerm(func=mdp.place_in_bowl, weight=5.0)
    release = RewTerm(func=mdp.release_in_bowl, weight=10.0)

    # Penalties — small in early training; CurriculumCfg below ramps them
    # up to discourage jittery actions once the policy is competent.
    action_l2 = RewTerm(func=mdp.action_l2, weight=-1e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    drop = RewTerm(func=mdp.block_dropped, weight=-2.0)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@configclass
class TerminationsCfg:
    """Time-out, success, and a workspace-box safety termination."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.task_success)
    block_off_table = DoneTerm(func=mdp.block_off_table)


# ---------------------------------------------------------------------------
# Curriculum — match the lift task: ramp jitter penalties later in training
# ---------------------------------------------------------------------------


@configclass
class CurriculumCfg:
    """Tightens action / joint-vel penalties after the policy has learned
    the task, the same shape as :class:`SoArm100LiftCubeEnvCfg`'s curriculum."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 10000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 10000},
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
