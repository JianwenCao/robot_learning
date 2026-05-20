# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO-ARM101 wiring for the Bonus-B singulation env."""

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

import isaac_so_arm101.tasks.singulation.mdp as mdp
from isaac_so_arm101.robots import SO_ARM101_CFG
from isaac_so_arm101.tasks.singulation.singulation_env_cfg import (
    SingulationEnvCfg,
    WRIST_RGB_HEIGHT,
    WRIST_RGB_WIDTH,
)
from isaac_so_arm101.tasks.singulation.mdp.events import (
    BLOCK_COLORS,
    COLOR_NAMES,
    HIDDEN_PARK_XY,
)

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
            # Per-color semantic tag so the wrist-cam segmentation can
            # carry instance information; `wrist_rgb_union_mask_dr`
            # OR-reduces over all 6 IDs into one binary channel, and the
            # `_resolve_color_class_ids` helper (from clutterpickplace)
            # is reused as-is.
            semantic_tags=[("class", f"cube_{color_name}")],
        ),
    )


@configclass
class SoArm101SingulationEnvCfg(SingulationEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        home_state = SO_ARM101_CFG.init_state.replace(
            joint_pos={**SO_ARM101_CFG.init_state.joint_pos, "gripper": 0.5},
        )
        self.scene.robot = SO_ARM101_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=home_state,
        )

        # Bowl command anchors its xy goal to gripper_link, same as
        # Eval-1/2/3 (this only determines which body the goal pose is
        # *expressed relative to* for the upstream UniformPoseCommand —
        # SingulationBowlPoseCommand still uses robot-frame xy via
        # `pose_command_b`).
        self.commands.bowl_pose.body_name = "gripper_link"

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

        for i, name in enumerate(COLOR_NAMES):
            setattr(self.scene, f"cube_{name}", _cube_cfg(name, HIDDEN_PARK_XY[i]))

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
class SoArm101SingulationEnvCfg_PLAY(SoArm101SingulationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        from isaac_so_arm101.tasks import _multicube_sim
        self.scene.num_envs = _multicube_sim.DEFAULT_PLAY_NUM_ENVS
        self.scene.env_spacing = _multicube_sim.ENV_SPACING
        self.observations.policy.enable_corruption = False
        self.observations.wrist_image.wrist_image.params = {"corrupt": False}


@configclass
class SoArm101SingulationTeacherFastEnvCfg(SoArm101SingulationEnvCfg):
    """Camera-free env for the state-only singulation teacher.

    Same scene + rewards as the vision env, but the wrist TiledCamera
    spawn and the ``wrist_image`` obs group are both nulled so PhysX
    doesn't pay the RTX render cost the policy never reads. The
    accompanying agent cfg should set
    ``obs_groups = {"policy": ["policy", "critic"], "critic": [...]}``
    so the teacher's actor gets the privileged 6×3 cube position vector
    + active mask + arrangement / n_active one-hots (everything the
    student would otherwise have to infer from the image).

    No ``--enable_cameras`` flag needed at launch.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.wrist_cam = None
        self.observations.wrist_image = None


@configclass
class SoArm101SingulationTeacherFastEnvCfg_PLAY(SoArm101SingulationTeacherFastEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        from isaac_so_arm101.tasks import _multicube_sim
        self.scene.num_envs = _multicube_sim.DEFAULT_PLAY_NUM_ENVS
        self.scene.env_spacing = _multicube_sim.ENV_SPACING
        self.observations.policy.enable_corruption = False
