# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based RL env cfg for SO-ARM101 Eval-2 (targeted clutter pick-and-place).

Scene differences vs Eval-1:

* **Six cubes** (one per palette color) spawned per env. Two are placed
  adjacent in the workspace per episode (the "active pair"); the other
  four are parked off-table where the wrist camera can't see them. See
  :func:`mdp.events.place_clutter_blocks` for the placement geometry.
* **No** semantic-segmentation channel on the wrist image (Eval-1's mask
  doesn't disambiguate target from distractor when both are visible).
  Wrist image is 3-channel RGB only; the policy learns to read color
  from the RGB pixels conditioned on the target-color one-hot.
* The bowl-pose command's rejection-sampling targets the *target* cube
  only (no need to keep the bowl away from the distractor).

Action / robot / control rate are inherited verbatim from Eval-1 — only
the perception + reward layer changes.
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaac_so_arm101.tasks.clutterpickplace.mdp as mdp
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


# ---------------------------------------------------------------------------
# Wrist-camera resolution — same as Eval-1.
# ---------------------------------------------------------------------------

WRIST_RGB_WIDTH = 128
WRIST_RGB_HEIGHT = 72


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class ClutterSceneCfg(InteractiveSceneCfg):
    """Scene with ground, gray table, robot, ee_frame, wrist cam, six cubes.

    Concrete cube spawn cfgs (one per color) and the robot / ee_frame /
    wrist_cam are filled in by :mod:`joint_pos_env_cfg`. Eval-1's
    contact sensors are intentionally omitted — the target-aware contact
    reward isn't worth the complexity of filter prim paths varying per
    episode (the "target cube" is decided dynamically). Reward terms
    here rely on kinematic ``_target_lifted_mask`` / ``_target_over_bowl_high_mask``
    only, which is the recipe that worked for the Eval-1 vision teacher.
    """

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    wrist_cam: TiledCameraCfg = MISSING

    # Six cubes, one per color, all filled in by joint_pos_env_cfg.
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


# ---------------------------------------------------------------------------
# Commands — bowl pose + target color
# ---------------------------------------------------------------------------


@configclass
class CommandsCfg:
    """Two commands:

    * ``bowl_pose`` — same UniformPoseCommand as Eval-1 (where to place
      the cube). We do **not** use Eval-1's :class:`BowlPoseCommand`
      rejection sampler because its ``target_asset_name`` is a single
      scene asset — here the "target cube" varies per env. We accept a
      small fraction of episodes where the bowl spawns over the active
      cluster; the place-only-target-cube reward still penalizes those.
    * ``target_color`` — see :class:`mdp.TargetColorCommand`. Samples the
      active pair + target index per episode.
    """

    bowl_pose = UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.28),
            pos_y=(-0.12, 0.12),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )

    target_color = mdp.TargetColorCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@configclass
class ActionsCfg:
    """Joint-position around home + binary gripper, same as Eval-1."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@configclass
class ObservationsCfg:
    """Three groups: deployable policy, privileged critic, wrist image."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        gripper_state = ObsTerm(func=mdp.gripper_state)
        bowl_xy = ObsTerm(func=mdp.bowl_xy, params={"command_name": "bowl_pose"})
        ee_proj_xy = ObsTerm(func=mdp.ee_proj_xy)
        ee_to_bowl_xy = ObsTerm(func=mdp.ee_to_bowl_xy, params={"command_name": "bowl_pose"})
        target_color = ObsTerm(func=mdp.target_color_onehot, params={"command_name": "target_color"})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        gripper_state = ObsTerm(func=mdp.gripper_state)
        bowl_xy = ObsTerm(func=mdp.bowl_xy, params={"command_name": "bowl_pose"})
        ee_proj_xy = ObsTerm(func=mdp.ee_proj_xy)
        ee_to_bowl_xy = ObsTerm(func=mdp.ee_to_bowl_xy, params={"command_name": "bowl_pose"})
        target_color = ObsTerm(func=mdp.target_color_onehot, params={"command_name": "target_color"})
        target_block_position = ObsTerm(func=mdp.target_block_position)
        distractor_block_position = ObsTerm(func=mdp.distractor_block_position)
        target_block_to_bowl_xy = ObsTerm(
            func=mdp.target_block_to_bowl_xy, params={"command_name": "bowl_pose"}
        )
        target_gripper_to_block = ObsTerm(func=mdp.target_gripper_to_block)
        target_is_grasped = ObsTerm(func=mdp.target_is_grasped)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class WristImageCfg(ObsGroup):
        """3-channel RGB only (no semantic mask).

        Mask doesn't disambiguate target vs distractor when both are
        tagged ``class:block`` — keeping channels = 3 forces the policy
        to read color from RGB pixels conditioned on the target-color
        one-hot. On real, the same 3-channel RGB feeds the same CNN; no
        HSV thresholding required since there's no mask to replicate.
        """

        wrist_image = ObsTerm(
            func=mdp.wrist_rgb_dr,
            params={"corrupt": True},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    wrist_image: WristImageCfg = WristImageCfg()


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@configclass
class EventCfg:
    """Reset events: clear default state, place clutter, reset target latches.

    Order matters:
      1. ``reset_all``               — defaults (clears joint state, etc.).
      2. ``reset_target_latches``    — clears per-episode lift / approach
         flags before any reward term reads them.
      3. ``place_clutter_blocks``    — teleports all six cubes per the
         command's sampled active pair + parking slots.

    Per-episode wrist-image tint (Eval-1's :func:`randomize_wrist_image_tint`)
    is included so the same DR distribution carries over to Eval-2 — the
    eval-time camera tint variability is the same as Eval-1.
    """

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_target_latches = EventTerm(func=mdp.reset_target_latches, mode="reset")

    place_clutter_blocks = EventTerm(
        func=mdp.place_clutter_blocks,
        mode="reset",
        params={
            "cluster_center_x": (0.15, 0.22),
            "cluster_center_y": (-0.10, 0.10),
            # Per-episode half-separation sampled uniformly. 0.0125–0.030
            # gives cube centers 2.5–6 cm apart → 0.5–4 cm edge-to-edge
            # margin for 2 cm cubes. "Adjacent" per the Eval-2 spec is
            # not literally touching; humans place blocks with a gap, and
            # sampling the range trains a margin-robust policy.
            "half_separation": (0.0125, 0.030),
            "table_z": 0.01,
            "command_name": "target_color",
            "cube_prefix": "cube_",
        },
    )

    # Per-channel linear tint — bumped wider (0.55-1.45 / ±0.20) since
    # this is the cheapest exposure-DR layer and the HSV jitter below
    # handles the hue dimension orthogonally.
    randomize_wrist_tint = EventTerm(
        func=mdp.randomize_wrist_image_tint,
        mode="reset",
        params={
            "rgb_scale_range": (0.55, 1.45),
            "brightness_range": (-0.20, 0.20),
        },
    )
    # Per-episode HSV jitter — key sim2real fix for color-conditioned
    # tasks. Hue rotation in linear RGB simulates white-balance / colored
    # ambient light drift that the linear tint above can't model.
    randomize_wrist_hsv = EventTerm(
        func=mdp.randomize_wrist_hsv_dr,
        mode="reset",
        params={
            "hue_shift_deg_range": (-20.0, 20.0),
            "sat_scale_range": (0.65, 1.35),
            "val_scale_range": (0.55, 1.45),
        },
    )


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


@configclass
class RewardsCfg:
    """Reward stack — direct port of Eval-1 weights, target-cube-indexed.

    Adds two distractor-aware penalties:

    * ``distractor_disturb``  (weight=-0.5) — small penalty for shoving
      the wrong cube.
    * ``wrong_block_in_bowl`` (weight=-20.0) — heavy penalty for placing
      the wrong cube in the bowl. Sized comparable to ``release_target_in_bowl``'s
      +30 so a successful target placement still wins net.

    Other weights match Eval-1's tuned recipe (reach 1, lift 15,
    transport 16, transport_fine 5, release 30, action_rate -1e-4 →
    -1e-2 via curriculum, joint_vel similar).
    """

    reaching_object = RewTerm(func=mdp.reach_target_block, params={"std": 0.05}, weight=1.0)

    lifting_object = RewTerm(
        func=mdp.target_grasp_event,
        params={"minimal_height": 0.07},
        weight=15.0,
    )

    object_goal_tracking = RewTerm(
        func=mdp.target_transport_to_bowl,
        params={"std": 0.30, "minimal_height": 0.025, "command_name": "bowl_pose"},
        weight=16.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.target_transport_to_bowl,
        params={"std": 0.05, "minimal_height": 0.025, "command_name": "bowl_pose"},
        weight=5.0,
    )

    release_in_bowl = RewTerm(func=mdp.release_target_in_bowl, weight=30.0)

    # Distractor-aware shaping (Eval-2 specific).
    distractor_disturb = RewTerm(
        func=mdp.distractor_disturb_penalty, weight=-0.5,
        params={"threshold_speed": 0.05},
    )
    wrong_block_in_bowl = RewTerm(func=mdp.wrong_block_in_bowl, weight=-20.0)

    # Penalties — ramped via curriculum, same as Eval-1.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")}
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@configclass
class TerminationsCfg:
    """Time-out plus any-cube-off-table safety.

    Following Eval-1 we do **not** terminate on task success — the
    policy gets continuous credit for staying placed, and the success
    rate is read off TB via :func:`mdp.log_target_success_metrics`.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    block_off_table = DoneTerm(func=mdp.block_off_table_any)


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


@configclass
class CurriculumCfg:
    """Action/joint-vel penalty ramps + TB success-rate logging.

    No block-xy range expand here — Eval-2's cluster sampling is already
    tight (≤ 7 cm × ≤ 20 cm), and the policy needs the full distribution
    from the start to learn target-color discrimination.
    """

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 10000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 10000},
    )
    log_success = CurrTerm(func=mdp.log_target_success_metrics, params={})


# ---------------------------------------------------------------------------
# Top-level env cfg
# ---------------------------------------------------------------------------


@configclass
class ClutterPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """SO-ARM101 targeted-pick-and-place in 2-cube clutter."""

    # num_envs default = 2048 (vs Eval-1's 4096) — 6 cubes/env makes
    # physics ~3-4× heavier; halve env count to keep step throughput
    # comparable. Override in your train script via env_cfg.scene.num_envs
    # if VRAM allows more.
    scene: ClutterSceneCfg = ClutterSceneCfg(
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
        # Episode budget: 5 s = 250 policy steps @ 50 Hz — same as Eval-1
        # since the task is "one grasp + one place" just like Eval-1.
        self.episode_length_s = 5.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        # Shared timing + PhysX buffer sizing across all multi-cube tasks.
        _multicube_sim.apply_multicube_sim_settings(self)
