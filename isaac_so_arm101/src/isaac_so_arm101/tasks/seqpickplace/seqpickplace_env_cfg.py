# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based RL env cfg for SO-ARM101 Eval-3 (sequential pick-and-place).

Scene shape mirrors Eval-2 (six color cubes spawned, subset placed in
workspace per episode, gray table, robot, wrist cam). Differences:

* Four cubes per episode are placed in the workspace (spread, not
  clustered), and the other two are parked off-table.
* :class:`SequentialGoalCommand` (custom) handles the full 3-step
  schedule + bowl-position randomization + step-advancement on success.
  No separate ``bowl_pose`` command — the three bowl positions live
  inside this single command term.
* Reward stack is current-step-aware (re-targets every time the policy
  releases a cube into the current bowl).
* Episode length is 15 s (3 × Eval-1's 5 s budget per step).
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaac_so_arm101.tasks.seqpickplace.mdp as mdp
from isaac_so_arm101.tasks import _multicube_sim
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
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


WRIST_RGB_WIDTH = 128
WRIST_RGB_HEIGHT = 72


@configclass
class SeqSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    wrist_cam: TiledCameraCfg = MISSING

    cube_blue:   RigidObjectCfg = MISSING
    cube_yellow: RigidObjectCfg = MISSING
    cube_purple: RigidObjectCfg = MISSING
    cube_orange: RigidObjectCfg = MISSING
    cube_green:  RigidObjectCfg = MISSING
    cube_red:    RigidObjectCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.25, 0.0, -0.01]),
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 1.0, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.722, 0.678, 0.663),
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


@configclass
class CommandsCfg:
    seq_goal = mdp.SequentialGoalCommandCfg(
        asset_name="robot",
        resampling_time_range=(15.0, 15.0),
        # Render the single per-rollout bowl as a red sphere in the
        # viewer (matches Eval-1/2's bowl marker pattern). All 3
        # sequential placements target this one bowl — only its position
        # is randomized between rollouts, not between steps. Marker impl
        # lives in :class:`SequentialGoalCommand._debug_vis_callback`.
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        gripper_state = ObsTerm(func=mdp.gripper_state)
        seq_goal = ObsTerm(func=mdp.seq_goal_vector, params={"command_name": "seq_goal"})
        ee_proj_xy = ObsTerm(func=mdp.ee_proj_xy)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        gripper_state = ObsTerm(func=mdp.gripper_state)
        seq_goal = ObsTerm(func=mdp.seq_goal_vector, params={"command_name": "seq_goal"})
        ee_proj_xy = ObsTerm(func=mdp.ee_proj_xy)
        all_blocks = ObsTerm(func=mdp.all_active_block_positions)
        current_target = ObsTerm(func=mdp.current_target_block_position)
        ee_to_target = ObsTerm(func=mdp.current_target_gripper_to_block)
        target_to_bowl_xy = ObsTerm(
            func=mdp.current_target_block_to_bowl_xy, params={"command_name": "seq_goal"}
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class WristImageCfg(ObsGroup):
        """4-channel ``RGB + current_target_mask`` wrist obs.

        Mirrors clutterpickplace's contract: the mask is the per-color
        instance mask of the *current step's* target cube, drawn from
        the TiledCamera's ``semantic_segmentation`` output (each cube
        carries a ``class:cube_<color>`` tag). Corrupted to mimic
        Florence-2's noise profile at deploy; Play cfgs disable.
        """

        wrist_image = ObsTerm(func=mdp.wrist_rgb_mask_dr, params={"corrupt": True})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    wrist_image: WristImageCfg = WristImageCfg()


@configclass
class SeqStateAprilTagObservationsCfg(ObservationsCfg):
    """Eval-3 obs variant for the state-only + AprilTag deploy path.

    Adds two terms to ``PolicyCfg``:

    * ``cube_positions_xy_noisy`` ``(N, NUM_COLORS*2=12)`` — per-cube noisy
      xy in robot frame, mirroring the per-frame pupil-apriltags output
      on the real arm. Re-keys the post-grasp freeze on every sub-goal
      transition (current target palette idx read via
      :func:`_current_target_palette_idx`).
    * ``cube_visible_flags`` ``(N, NUM_COLORS=6)`` — 1 if the tag was
      detected this step (cube active + on table + not dropped), else 0.

    The seq_goal vector already exposes the current sub-goal target color
    one-hot, current bowl xy, and step idx — those stay in the policy
    obs group via the existing ``seq_goal`` term.
    """

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        cube_positions_xy_noisy = ObsTerm(func=mdp.cube_positions_xy_noisy)
        cube_visible_flags = ObsTerm(func=mdp.cube_visible_flags)

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_latches = EventTerm(func=mdp.reset_seq_latches, mode="reset")
    place_blocks = EventTerm(
        func=mdp.place_seq_blocks,
        mode="reset",
        params={
            # 4 cubes spread independently with pairwise rejection.
            # Workspace 0.15×0.30 m² is enough to fit 4 cubes at
            # ≥ 8 cm pairwise separation (≥ 6 cm edge gap for 2 cm
            # cubes — comfortably wider than the SO-ARM gripper finger
            # span). Bumped from 6 cm per EVAL3_PLAN.md §2 to give
            # more grasp margin. If rejection sampling stops fitting
            # the 4th cube reliably (max_attempts warnings), widen
            # block_x / block_y rather than dropping back.
            "block_x": (0.13, 0.28),
            "block_y": (-0.15, 0.15),
            "min_block_separation": 0.08,
            "table_z": 0.01,
            "max_attempts": 80,
            # Bowl in a tighter range than blocks → cubes can spill into
            # the y-edges where the bowl can't spawn (better separation chance).
            "bowl_x": (0.18, 0.26),
            "bowl_y": (-0.08, 0.08),
            # Bumped 0.08 → 0.15 (2026-05-20) to match Eval-1/Eval-3's
            # spacing so cubes land clearly away from the bowl. NOTE:
            # the bowl excludes ~π·15² ≈ 707 cm² (clipped to the 450 cm²
            # cube workspace) — with 4 cubes also needing ≥ 6 cm
            # pairwise separation, rejection sampling is *tight*. If
            # ``place_seq_blocks`` starts logging "max_attempts
            # exhausted" warnings, either drop this back toward 0.10 or
            # widen the cube workspace (block_x/block_y).
            "min_bowl_block_separation": 0.15,
            "command_name": "seq_goal",
            "cube_prefix": "cube_",
        },
    )
    randomize_wrist_tint = EventTerm(
        func=mdp.randomize_wrist_image_tint,
        mode="reset",
        params={
            "rgb_scale_range": (0.55, 1.45),
            "brightness_range": (-0.20, 0.20),
        },
    )
    randomize_wrist_hsv = EventTerm(
        func=mdp.randomize_wrist_hsv_dr,
        mode="reset",
        params={
            "hue_shift_deg_range": (-20.0, 20.0),
            "sat_scale_range": (0.65, 1.35),
            "val_scale_range": (0.55, 1.45),
        },
    )


@configclass
class RewardsCfg:
    """Reward stack — dense reach/lift/transport on the current target,
    sparse release + per-step bonus on completion.

    The per-step weights ``(4.0, 4.0, 2.0)`` mirror the Eval-3 grading
    (4/4/2 pts) so the value function aligns with the score the human
    grader will write down. ``release_current_target_in_bowl`` weight=30
    matches Eval-1; ``step_completion_bonus`` weight=1.0 then scales the
    per-step (4, 4, 2) values directly.
    """

    reaching_object = RewTerm(func=mdp.reach_current_target, params={"std": 0.05}, weight=1.0)
    lifting_object = RewTerm(
        func=mdp.lift_current_target, params={"minimal_height": 0.07}, weight=15.0
    )
    object_goal_tracking = RewTerm(
        func=mdp.transport_current_target_to_bowl,
        params={"std": 0.30, "minimal_height": 0.025, "command_name": "seq_goal"},
        weight=16.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.transport_current_target_to_bowl,
        params={"std": 0.05, "minimal_height": 0.025, "command_name": "seq_goal"},
        weight=5.0,
    )
    release_in_bowl = RewTerm(func=mdp.release_current_target_in_bowl, weight=30.0)
    step_bonus = RewTerm(
        func=mdp.step_completion_bonus,
        params={"weight_per_step": (4.0, 4.0, 2.0)},
        weight=1.0,
    )
    wrong_in_bowl = RewTerm(func=mdp.wrong_cube_in_current_bowl, weight=-15.0)

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")}
    )

    # Wrist-cam standoff from table — see Eval-1 RewardsCfg for rationale.
    wrist_cam_clearance = RewTerm(
        func=mdp.wrist_cam_table_clearance,
        params={"margin": 0.03, "table_top_z": 0.0},
        weight=-50.0,
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    block_off_table = DoneTerm(func=mdp.active_block_off_table)
    # Optional positive termination when all 3 steps done — saves env
    # time for the bonus speed metric. Comment out if you want the
    # policy to "stay placed" through to time_out instead.
    all_done = DoneTerm(func=mdp.all_steps_done)


@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 10000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 10000},
    )
    log_success = CurrTerm(func=mdp.log_seq_success_metrics, params={})


@configclass
class SeqPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """SO-ARM101 sequential 3-step pick-and-place env."""

    # num_envs default = 2048 — same rationale as clutterpickplace.
    # Episodes are 3× longer here (15 s) but per-step cost is identical.
    scene: SeqSceneCfg = SeqSceneCfg(
        num_envs=_multicube_sim.DEFAULT_TRAIN_NUM_ENVS,
        env_spacing=_multicube_sim.ENV_SPACING,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 15 s = 750 steps @ 50 Hz, enough for 3 sub-goals × 5 s each.
        self.episode_length_s = 15.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        _multicube_sim.apply_multicube_sim_settings(self)
