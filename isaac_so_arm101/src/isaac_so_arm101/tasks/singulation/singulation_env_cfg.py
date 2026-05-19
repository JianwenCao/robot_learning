# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based RL env cfg for SO-ARM101 Bonus-B (singulation).

Scene shape mirrors Eval-2/3 (six color cubes, gray table, robot, wrist
cam). Per-episode event :func:`mdp.sample_active_set` randomizes:

* how many cubes are in the workspace (3 or 4),
* their arrangement (vertical stack vs flat cluster), and
* which palette colors fill the active slots.

No bowl. No goal-color command. The policy is conditioned only on
``n_active_onehot`` (2-D) + ``arrangement_onehot`` (2-D) so it can adapt
its strategy to the initial config — different motions are needed to
take down a 4-block tower vs to scatter a 3-block flat cluster.

Episode length: 10 s (twice Eval-1's per-pick budget; singulation
typically requires multiple cube manipulations).
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
    """No commands — singulation has no per-step goal vector."""
    pass


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
        n_active = ObsTerm(func=mdp.n_active_onehot)
        arrangement = ObsTerm(func=mdp.arrangement_onehot)
        active_mask = ObsTerm(func=mdp.active_block_mask)
        all_cube_positions = ObsTerm(func=mdp.all_cube_positions_robot_frame)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class WristImageCfg(ObsGroup):
        wrist_image = ObsTerm(func=mdp.wrist_rgb_dr, params={"corrupt": True})

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
            "n_active_choices": (3, 4),
            "stacked_prob": 0.5,
            "cube_size": 0.02,
            "table_z": 0.01,
            "stack_lateral_jitter": 0.003,
            "cluster_inter_spacing": 0.023,
            "cluster_position_jitter": 0.002,
            "center_x": (0.16, 0.22),
            "center_y": (-0.08, 0.08),
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
    """Singulation reward stack.

    * ``min_pairwise``     +5  — push the worst-case pair apart.
    * ``mean_pairwise``    +2  — broad spreading signal.
    * ``all_on_table``     +3  — unstacking signal (key for the stack case).
    * ``reach_closest``    +1  — engage with the closest active pair.
    * ``success``          +50 — once cleanly separated AND on table.
    * ``overspeed``        -3  — discourage flinging cubes.
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
    success = RewTerm(
        func=mdp.singulation_success,
        params={"min_separation": 0.05, "on_table_height": 0.05},
        weight=50.0,
    )
    overspeed = RewTerm(
        func=mdp.cube_overspeed_penalty, params={"speed_cap": 0.30}, weight=-3.0
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")}
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cube_off = DoneTerm(func=mdp.active_cube_off_table)
    # Optional positive termination on success — comment to require the
    # policy to keep cubes separated through to time_out.
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

    # num_envs default = 2048. Singulation is the heaviest of the three
    # multi-cube tasks (stacks → many cube↔cube contact pairs), so the
    # PhysX buffer expansions in _multicube_sim are most important here.
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
        # 10 s = 500 policy steps @ 50 Hz — enough to clear a 4-block
        # stack AND spread the resulting cubes.
        self.episode_length_s = 10.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        _multicube_sim.apply_multicube_sim_settings(self)
