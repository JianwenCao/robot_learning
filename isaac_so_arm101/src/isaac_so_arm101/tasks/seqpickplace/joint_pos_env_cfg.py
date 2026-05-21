# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO-ARM101 wiring for the Eval-3 sequential pick-and-place env."""

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

import isaac_so_arm101.tasks.seqpickplace.mdp as mdp
from isaac_so_arm101.robots import SO_ARM101_CFG
from isaac_so_arm101.tasks.seqpickplace.seqpickplace_env_cfg import (
    SeqPickPlaceEnvCfg,
    SeqStateAprilTagObservationsCfg,
    WRIST_RGB_HEIGHT,
    WRIST_RGB_WIDTH,
)
from isaac_so_arm101.tasks.seqpickplace.mdp.events import (
    BLOCK_COLORS,
    COLOR_NAMES,
    HIDDEN_PARK_XY,
)
from isaaclab.managers import EventTermCfg as EventTerm

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


def _cube_cfg(color_name: str, default_xy: tuple[float, float]) -> RigidObjectCfg:
    rgb = BLOCK_COLORS[color_name]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/cube_{color_name}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[default_xy[0], default_xy[1], 0.05],
            rot=[1, 0, 0, 0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.02, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=0.5),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.020),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            # Per-color semantic class — mirrors clutterpickplace. The
            # wrist camera's ``semantic_segmentation`` is filtered to the
            # current step's target cube by
            # :func:`mdp.wrist_rgb_mask_dr` and exposed as the 4th
            # wrist-image channel. At deploy the same mask is produced by
            # Florence-2 prompted with the current target colour.
            semantic_tags=[("class", f"cube_{color_name}")],
        ),
    )


@configclass
class SoArm101SeqPickPlaceEnvCfg(SeqPickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        home_state = SO_ARM101_CFG.init_state.replace(
            joint_pos={**SO_ARM101_CFG.init_state.joint_pos, "gripper": 0.5},
        )
        self.scene.robot = SO_ARM101_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=home_state,
        )

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

        # Cubes
        for i, name in enumerate(COLOR_NAMES):
            setattr(self.scene, f"cube_{name}", _cube_cfg(name, HIDDEN_PARK_XY[i]))

        # ee frame
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

        # wrist cam — RGB + semantic_segmentation. The seg output is
        # filtered per-step-target-color by :func:`mdp.wrist_rgb_mask_dr`
        # to produce the 4th image channel; matches the clutterpickplace
        # (Eval-2) image contract one-to-one.
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
class SoArm101SeqPickPlaceEnvCfg_PLAY(SoArm101SeqPickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        from isaac_so_arm101.tasks import _multicube_sim
        self.scene.num_envs = _multicube_sim.DEFAULT_PLAY_NUM_ENVS
        self.scene.env_spacing = _multicube_sim.ENV_SPACING
        self.observations.policy.enable_corruption = False
        self.observations.wrist_image.wrist_image.params = {"corrupt": False}


@configclass
class SoArm101SeqPickPlaceTeacherFastEnvCfg(SoArm101SeqPickPlaceEnvCfg):
    """Camera-free env for the state-only sequential teacher.

    Same scene + rewards + commands as the vision env, but the wrist
    TiledCamera spawn and the ``wrist_image`` obs group are both nulled
    so PhysX doesn't pay the RTX render cost the policy never reads.
    The accompanying agent cfg should set
    ``obs_groups = {"policy": ["policy", "critic"], "critic": [...]}``
    so the teacher's actor consumes the privileged cube positions that
    the vision student infers from the image.

    No ``--enable_cameras`` flag needed at launch.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.wrist_cam = None
        self.observations.wrist_image = None


@configclass
class SoArm101SeqPickPlaceTeacherFastEnvCfg_PLAY(SoArm101SeqPickPlaceTeacherFastEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        from isaac_so_arm101.tasks import _multicube_sim
        self.scene.num_envs = _multicube_sim.DEFAULT_PLAY_NUM_ENVS
        self.scene.env_spacing = _multicube_sim.ENV_SPACING
        self.observations.policy.enable_corruption = False


@configclass
class SoArm101SeqPickPlaceStateAprilTagEnvCfg(SoArm101SeqPickPlaceTeacherFastEnvCfg):
    """State-only + AprilTag deploy path for Eval-3 (3 sequential sub-goals).

    Inherits Teacher-Fast (camera-free, no wrist_image) and:

    * Swaps in :class:`SeqStateAprilTagObservationsCfg` so the actor's
      ``PolicyCfg`` is **bitwise identical** to Eval-2's (27-D, target-only
      AprilTag stream via ``target_cube_pos_xy_noisy``). ``seq_goal``
      stays in the *critic* obs (privileged) but is dropped from the
      policy stream — the env advances ``_seq_step_idx`` internally on
      release, and :func:`mdp.target_cube_pos_xy_noisy` reads the
      current sub-goal target via ``_current_target_palette_idx`` so the
      published xy + post-grasp freeze re-key automatically.
    * Adds a ``reset_cube_positions_bias`` event after ``place_blocks`` so
      the per-episode hand-eye bias + per-cube mount + last-value seed
      live alongside the placement.

    Same actor as Eval-1/Eval-2 StateAprilTag (plain MLP via
    :class:`PickPlaceVisionActorCritic` auto-disabling its CNN). The
    sub-goal switching is realised purely through the env-side
    ``_seq_step_idx`` advancement; on real, the deploy script re-keys
    the AprilTag detector ID to mirror that advancement (see EVAL3_PLAN.md
    §8).
    """

    observations: SeqStateAprilTagObservationsCfg = SeqStateAprilTagObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_all = EventTerm(func=mdp.reset_robot_to_default, mode="reset")
        self.events.place_blocks = EventTerm(
            func=mdp.place_seq_blocks_once,
            mode="reset",
            params=self.events.place_blocks.params,
        )
        self.events.reset_cube_positions_bias = EventTerm(
            func=mdp.reset_cube_positions_bias, mode="reset"
        )


@configclass
class SoArm101SeqPickPlaceStateAprilTagEnvCfg_PLAY(SoArm101SeqPickPlaceStateAprilTagEnvCfg):
    """Play variant: smaller envs, no corruption."""

    def __post_init__(self):
        super().__post_init__()
        from isaac_so_arm101.tasks import _multicube_sim
        self.scene.num_envs = _multicube_sim.DEFAULT_PLAY_NUM_ENVS
        self.scene.env_spacing = _multicube_sim.ENV_SPACING
        self.observations.policy.enable_corruption = False
        self.observations.policy.target_cube_pos_xy_noisy.params = {"corrupt": False}
