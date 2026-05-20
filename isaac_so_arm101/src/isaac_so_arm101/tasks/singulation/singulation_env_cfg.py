# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based RL env cfg for SO-ARM101 Bonus-B (singulation).

Scene shape mirrors Eval-2/3 (six color cubes, gray table, robot, wrist
cam). Per-episode event :func:`mdp.sample_active_set` randomizes:

* how many cubes are in the workspace (3 or 4),
* their arrangement (one of 11 families — stacks, flat clusters,
  pyramids, mixed; see ``mdp.events.ARRANGEMENT_SPECS``),
* which palette colors fill the active slots,
* yaw of the whole arrangement.

The bowl is **not a scene prim** — exposed as a 2-D xy via
:class:`mdp.SingulationBowlPoseCommand` so the chained P2 (Eval-3
pick-and-place policy) has the same ``bowl_xy`` state slot it expects
after handoff. P1 keeps cubes out of the bowl xy via the
``bowl_avoidance`` reward.

Episode length: 12 s — Eval-1's per-pick budget is ~3 s; 4-stack /
3-1 pyramid disassembly needs 3 sequential grasps + headroom.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaac_so_arm101.tasks.singulation.mdp as mdp
from isaac_so_arm101.tasks import _multicube_sim
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
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
class SingulationSceneCfg(InteractiveSceneCfg):
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
    """Single bowl-xy goal — same schema P2 reads after the singulation
    handoff. Bowl is not a scene prim (no visual asset), only a 2-D xy
    in robot frame, rejection-sampled ≥ 15 cm from the cluster centre."""

    bowl_pose = mdp.SingulationBowlPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # filled by joint_pos_env_cfg
        resampling_time_range=(12.0, 12.0),
        debug_vis=True,
        min_distance=0.15,
        max_attempts=16,
        ranges=UniformPoseCommandCfg.Ranges(
            pos_x=(0.18, 0.30),
            pos_y=(-0.15, 0.15),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
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
        ee_proj_xy = ObsTerm(func=mdp.ee_proj_xy)
        bowl_xy = ObsTerm(func=mdp.bowl_xy, params={"command_name": "bowl_pose"})
        n_active = ObsTerm(func=mdp.n_active_onehot)
        arrangement = ObsTerm(func=mdp.arrangement_onehot)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        gripper_state = ObsTerm(func=mdp.gripper_state)
        ee_proj_xy = ObsTerm(func=mdp.ee_proj_xy)
        bowl_xy = ObsTerm(func=mdp.bowl_xy, params={"command_name": "bowl_pose"})
        n_active = ObsTerm(func=mdp.n_active_onehot)
        arrangement = ObsTerm(func=mdp.arrangement_onehot)
        active_mask = ObsTerm(func=mdp.active_block_mask)
        all_cube_positions = ObsTerm(func=mdp.all_cube_positions_robot_frame)
        min_pairwise_xy = ObsTerm(func=mdp.min_pairwise_xy_active)
        mean_pairwise_xy = ObsTerm(func=mdp.mean_pairwise_xy_active)
        n_off_table = ObsTerm(func=mdp.n_cubes_off_table)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class WristImageCfg(ObsGroup):
        wrist_image = ObsTerm(func=mdp.wrist_rgb_union_mask_dr, params={"corrupt": True})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    wrist_image: WristImageCfg = WristImageCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_latches = EventTerm(func=mdp.reset_singulation_latches, mode="reset")
    sample_active = EventTerm(
        func=mdp.sample_active_set,
        mode="reset",
        params={
            # arrangement_weights=None → uses DEFAULT_ARRANGEMENT_WEIGHTS
            # from events.py (stacks 0.25 / clusters 0.40 / pyramids 0.20
            # / mixed 0.15).
            "arrangement_weights": None,
            "cube_size": 0.02,
            "table_z": 0.01,
            "stack_lateral_jitter": 0.005,
            "cluster_inter_spacing": 0.021,
            "cluster_position_jitter": 0.003,
            "mixed_gap": 0.07,
            "center_x": (0.16, 0.22),
            "center_y": (-0.08, 0.08),
            "cube_prefix": "cube_",
        },
    )
    randomize_cube_physics = EventTerm(
        func=mdp.randomize_cube_physics,
        mode="reset",
        params={
            "mass_range": (0.016, 0.024),
            "friction_range": (0.7, 1.3),
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
    """Singulation reward stack.

    Dense:
    * ``min_pairwise``     +5  — push the worst-case pair apart.
    * ``mean_pairwise``    +2  — broad spreading signal.
    * ``all_on_table``     +3  — unstacking signal (key for the stack case).
    * ``reach_closest``    +1  — engage with the closest active pair.
    * ``lift_then_place``  +3  — bias toward grasp-and-place (sim2real).

    Sparse:
    * ``success``          +50 — once cleanly separated AND on table.

    Penalties:
    * ``overspeed``        -3  — discourage flinging cubes.
    * ``bowl_avoidance``   -5  — keep singulated cubes out of bowl xy.
    * ``action_rate``    -1e-4 → -1e-2 via curriculum.
    * ``joint_vel``      -1e-4 → -1e-2.
    """

    min_pairwise = RewTerm(
        func=mdp.min_pairwise_xy, params={"cap": 0.10}, weight=5.0
    )
    mean_pairwise = RewTerm(
        func=mdp.mean_pairwise_xy, params={"cap": 0.10}, weight=2.0
    )
    all_on_table = RewTerm(
        func=mdp.all_cubes_on_table, params={"height_threshold": 0.05}, weight=3.0
    )
    reach_closest = RewTerm(
        func=mdp.reach_closest_pair, params={"std": 0.10}, weight=1.0
    )
    lift_then_place = RewTerm(
        func=mdp.lift_then_place,
        params={"z_lo": 0.07, "z_hi": 0.20, "gripper_closed_threshold": 0.25},
        weight=3.0,
    )
    success = RewTerm(
        func=mdp.singulation_success,
        params={"min_separation": 0.05, "on_table_height": 0.05},
        weight=50.0,
    )
    overspeed = RewTerm(
        func=mdp.cube_overspeed_penalty, params={"speed_cap": 0.30}, weight=-3.0
    )
    bowl_avoid = RewTerm(
        func=mdp.bowl_avoidance,
        params={"bowl_command_name": "bowl_pose", "near_threshold": 0.06},
        weight=-5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")}
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cube_off = DoneTerm(func=mdp.active_cube_off_table)
    # Positive termination on success — success is a stable absorbing
    # state, so γ-discounting on early termination incentivises *fast*
    # singulation. Comment out if the policy needs to keep the cubes
    # separated through time_out instead.
    done = DoneTerm(func=mdp.singulation_done)


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
    log_success = CurrTerm(func=mdp.log_singulation_metrics, params={})


@configclass
class SingulationEnvCfg(ManagerBasedRLEnvCfg):
    """SO-ARM101 singulation env (Bonus-B)."""

    scene: SingulationSceneCfg = SingulationSceneCfg(
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
        # 12 s = 600 policy steps @ 50 Hz — fits ~3 grasp-lift-place
        # cycles for a 4-stack at ~3.5 s/cycle (per Eval-1 deploy
        # timing) plus settle headroom.
        self.episode_length_s = 12.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        _multicube_sim.apply_multicube_sim_settings(self)
