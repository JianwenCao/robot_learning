# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO-ARM101 wiring for the clutter pick-and-place env.

Plugs the robot, ee_frame, wrist_cam, and the **six color-tagged cubes**
into :class:`ClutterPickPlaceEnvCfg`. Cube spawning differs from Eval-1:

* We can't use the dex_cube USD's baked-in visual material if we want
  per-cube colors, so each cube is a :class:`CuboidCfg` primitive with
  ``PreviewSurfaceCfg`` of the palette color baked in at spawn time.
* Physics material is tuned for grippability — high friction, medium
  mass — matching the dex_cube's reliability for binary-gripper grasping
  (the Eval-1 root cause of switching from CuboidCfg → dex_cube was
  hand-tuned friction; we replicate the working values here).
* Each cube is semantic-tagged with its own per-color class
  (``class:cube_red``, ``class:cube_blue``, …). The wrist TiledCamera
  emits ``semantic_segmentation``; :func:`mdp.wrist_rgb_mask_dr` reads
  it and produces a target-keyed binary instance mask for the wrist
  image's 4th channel. On the real arm, the same mask comes from
  Florence-2 prompted by the target color (HSV thresholds proved too
  brittle at the cube's working distance — see deploy/cube_detector.py).
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.utils import configclass

import isaac_so_arm101.tasks.clutterpickplace.mdp as mdp
from isaac_so_arm101.robots import SO_ARM101_CFG
from isaac_so_arm101.tasks.clutterpickplace.clutterpickplace_env_cfg import (
    ClutterPickPlaceEnvCfg,
    ClutterStateAprilTagObservationsCfg,
    WRIST_RGB_HEIGHT,
    WRIST_RGB_WIDTH,
)
from isaac_so_arm101.tasks.clutterpickplace.mdp.events import (
    BLOCK_COLORS,
    COLOR_NAMES,
    HIDDEN_PARK_XY,
)
from isaaclab.managers import EventTermCfg as EventTerm

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


def _cube_cfg(color_name: str, default_xy: tuple[float, float]) -> RigidObjectCfg:
    """Build a 2 cm CuboidCfg for one palette color.

    Default spawn xy is the cube's parking slot — the ``place_clutter_blocks``
    event teleports active cubes to the workspace and parks the rest at
    these defaults on every reset, so the spawn pose only matters until
    the first reset.
    """
    rgb = BLOCK_COLORS[color_name]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/cube_{color_name}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[default_xy[0], default_xy[1], 0.05],
            rot=[1, 0, 0, 0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.02, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=rgb,
                roughness=0.5,
            ),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            # Tuned to mimic the dex_cube's grippability (Eval-1 used the
            # NVIDIA dex_cube for reliable grasps; we replicate the
            # working friction/mass here so the 2 cm primitive grips the
            # same way under the binary gripper policy).
            mass_props=sim_utils.MassPropertiesCfg(mass=0.020),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            # Per-color semantic class so the wrist camera's
            # ``semantic_segmentation`` output can be filtered to the
            # *target* cube only (vs Eval-1's single ``class:block`` tag
            # that lumps every cube together). The class ID lookup for
            # each color is cached in :func:`mdp.wrist_rgb_mask_dr` after
            # the first render.
            semantic_tags=[("class", f"cube_{color_name}")],
        ),
    )


@configclass
class SoArm101ClutterPickPlaceEnvCfg(ClutterPickPlaceEnvCfg):
    """SO-ARM101 + 6 colored cubes + bowl-pose + target-color command."""

    def __post_init__(self):
        super().__post_init__()

        # Open-gripper home pose, same as Eval-1.
        home_state = SO_ARM101_CFG.init_state.replace(
            joint_pos={**SO_ARM101_CFG.init_state.joint_pos, "gripper": 0.5},
        )
        self.scene.robot = SO_ARM101_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=home_state,
        )

        # Actions — verbatim from Eval-1.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            open_command_expr={"gripper": 0.5},
            close_command_expr={"gripper": 0.0},
        )

        # Bowl-pose command marker.
        self.commands.bowl_pose.body_name = "gripper_link"
        from isaaclab.markers.config import SPHERE_MARKER_CFG, FRAME_MARKER_CFG as _FM
        goal_marker = SPHERE_MARKER_CFG.copy()
        goal_marker.prim_path = "/Visuals/Command/goal_pose"
        goal_marker.markers["sphere"].radius = 0.03
        goal_marker.markers["sphere"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
        self.commands.bowl_pose.goal_pose_visualizer_cfg = goal_marker
        cur_marker = _FM.copy()
        cur_marker.prim_path = "/Visuals/Command/body_pose"
        cur_marker.markers["frame"].scale = (0.04, 0.04, 0.04)
        self.commands.bowl_pose.current_pose_visualizer_cfg = cur_marker

        # Six cubes. Default spawn at the parking slot for each — the
        # reset event teleports actives into the workspace, leaving the
        # rest at their parking xy.
        for i, name in enumerate(COLOR_NAMES):
            setattr(self.scene, f"cube_{name}", _cube_cfg(name, HIDDEN_PARK_XY[i]))

        # End-effector frame transformer (same as Eval-1).
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.01, 0.0, -0.09]),
                ),
            ],
        )

        # Wrist camera — RGB + semantic_segmentation. The seg output is
        # filtered per-target-color by :func:`mdp.wrist_rgb_mask_dr` to
        # produce the 4th image channel (a binary instance mask of the
        # *target* cube only). At deploy the same mask comes from
        # Florence-2; per-step DR on the sim mask channel matches that
        # detector's noise profile (see ``mdp.wrist_rgb_mask_dr``).
        intrinsics = mdp.load_wrist_cam_intrinsics()
        self.scene.wrist_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_link/wrist_cam",
            update_period=self.sim.dt * self.decimation,
            height=WRIST_RGB_HEIGHT,
            width=WRIST_RGB_WIDTH,
            data_types=["rgb", "semantic_segmentation"],
            colorize_semantic_segmentation=False,
            spawn=PinholeCameraCfg(
                focal_length=intrinsics["focal_length"],
                horizontal_aperture=intrinsics["horizontal_aperture"],
                clipping_range=(0.01, 2.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(-0.001, 0.1, -0.04),
                rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
                convention="ros",
            ),
        )


@configclass
class SoArm101ClutterPickPlaceEnvCfg_PLAY(SoArm101ClutterPickPlaceEnvCfg):
    """Small-num_envs variant for visual inspection."""

    def __post_init__(self):
        super().__post_init__()
        from isaac_so_arm101.tasks import _multicube_sim
        self.scene.num_envs = _multicube_sim.DEFAULT_PLAY_NUM_ENVS
        self.scene.env_spacing = _multicube_sim.ENV_SPACING
        self.observations.policy.enable_corruption = False
        self.observations.wrist_image.wrist_image.params = {"corrupt": False}


@configclass
class SoArm101ClutterPickPlaceTeacherFastEnvCfg(SoArm101ClutterPickPlaceEnvCfg):
    """Camera-free env cfg for the Stage-1 state teacher.

    The default env cfg spawns a ``TiledCamera`` on every env so the
    same scene works for the vision student in Stages 2/3. The teacher
    never reads the wrist image — but PhysX + RTX still pay the full
    rendering cost each step. With 6 cubes per env Eval-2's render
    cost is ~3-4× Eval-1's (proportional to per-env prim count); the
    camera-free variant is the right default for the teacher.

    This subclass nulls the camera spawn and the ``wrist_image`` obs
    group entirely. Trade-off: the teacher's training env is no longer
    *visually* identical to the student's. That's fine because the
    teacher only consumes ``policy + critic`` ground-truth state — the
    image-side DR is added back in Stage 2 distillation and Stage 3 PPO
    where the student actually reads pixels.

    No ``--enable_cameras`` needed at launch.
    """

    def __post_init__(self):
        super().__post_init__()
        # Null the wrist camera spawn. Scene cfg manager skips entries set to None.
        self.scene.wrist_cam = None
        # Drop the wrist_image obs group too — its obs function would
        # otherwise try to read sensors["wrist_cam"] each step.
        self.observations.wrist_image = None


@configclass
class SoArm101ClutterPickPlaceTeacherFastEnvCfg_PLAY(SoArm101ClutterPickPlaceTeacherFastEnvCfg):
    """Smaller variant of the camera-free teacher env, for visual eval."""

    def __post_init__(self):
        super().__post_init__()
        from isaac_so_arm101.tasks import _multicube_sim
        self.scene.num_envs = _multicube_sim.DEFAULT_PLAY_NUM_ENVS
        self.scene.env_spacing = _multicube_sim.ENV_SPACING
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101ClutterPickPlaceTeacherFastEnvCfg(SoArm101ClutterPickPlaceEnvCfg):
    """Camera-free env for the state-only clutter teacher.

    Same scene + rewards + commands as the vision env, but the wrist
    TiledCamera spawn and the ``wrist_image`` obs group are both nulled
    so PhysX doesn't pay the RTX render cost the policy never reads.
    The accompanying agent cfg should set
    ``obs_groups = {"policy": ["policy", "critic"], "critic": [...]}``
    so the teacher's actor consumes the privileged cube positions
    (which include target / distractor xyz, ee-to-target, is-grasped).

    No ``--enable_cameras`` flag needed at launch.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.wrist_cam = None
        self.observations.wrist_image = None


@configclass
class SoArm101ClutterPickPlaceTeacherFastEnvCfg_PLAY(SoArm101ClutterPickPlaceTeacherFastEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        from isaac_so_arm101.tasks import _multicube_sim
        self.scene.num_envs = _multicube_sim.DEFAULT_PLAY_NUM_ENVS
        self.scene.env_spacing = _multicube_sim.ENV_SPACING
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101ClutterPickPlaceStateAprilTagEnvCfg(SoArm101ClutterPickPlaceTeacherFastEnvCfg):
    """State-only + AprilTag deploy path for Eval-2 (single sub-goal).

    Inherits the Teacher-Fast subclass (camera-free, no wrist_image obs)
    and:

    * Swaps in :class:`ClutterStateAprilTagObservationsCfg` so ``PolicyCfg``
      gains ``cube_positions_xy_noisy`` ``(N, 12)`` and ``cube_visible_flags``
      ``(N, 6)``. Goal group (``target_color`` one-hot) is inherited.
    * Adds ``reset_cube_positions_bias`` event listed **after**
      ``place_clutter_blocks`` so ``_cube_pos_last`` is seeded against
      the freshly-placed cube positions.

    See ``docs/STATE_APRILTAG_PLAN.md`` for the deploy-side mirror. The
    accompanying runner cfg uses
    ``obs_groups = {"policy": ["policy", "goal"], "critic": ["policy", "goal", "critic"]}``
    — actor consumes deployable obs only; critic sees privileged GT cube
    positions for stable value estimation.
    """

    observations: ClutterStateAprilTagObservationsCfg = ClutterStateAprilTagObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_cube_positions_bias = EventTerm(
            func=mdp.reset_cube_positions_bias, mode="reset"
        )


@configclass
class SoArm101ClutterPickPlaceStateAprilTagEnvCfg_PLAY(SoArm101ClutterPickPlaceStateAprilTagEnvCfg):
    """Play variant: fewer envs, no corruption, no AprilTag noise.

    ``corrupt=False`` short-circuits the bias / Gaussian / dropout / ID
    swap inside :func:`mdp.observations._compute_apriltag_obs`. The
    post-grasp target freeze still applies (physical occlusion, not
    corruption).
    """

    def __post_init__(self):
        super().__post_init__()
        from isaac_so_arm101.tasks import _multicube_sim
        self.scene.num_envs = _multicube_sim.DEFAULT_PLAY_NUM_ENVS
        self.scene.env_spacing = _multicube_sim.ENV_SPACING
        self.observations.policy.enable_corruption = False
        self.observations.policy.cube_positions_xy_noisy.params = {"corrupt": False}
        self.observations.policy.cube_visible_flags.params = {"corrupt": False}
