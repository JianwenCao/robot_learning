# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO-ARM101 wiring for the pick-and-place env.

Mirrors the lift-task pattern: this subclass fills in the robot, ee_frame
and object slots that ``PickPlaceBowlEnvCfg`` left as ``MISSING``, and binds
the joint-position / binary-gripper actions.

Following EVAL1_PLAN §3.2: action is **absolute-around-home** with
``scale=0.5`` (NOT delta), exactly matching ``SoArm101LiftCubeEnvCfg`` so
the policy lands cleanly on Feetech ``goal_position`` writes at deploy.

The block is an explicit 2 cm cube (``CuboidCfg``) rather than a scaled
DexCube — its size is the spec, so we encode it directly.
"""

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.manipulation.lift.mdp as lift_mdp  # noqa: F401
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

import isaac_so_arm101.tasks.pickplace.mdp as mdp
from isaac_so_arm101.robots import SO_ARM101_CFG  # noqa: F401
from isaac_so_arm101.tasks.pickplace.pickplace_env_cfg import PickPlaceBowlEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


@configclass
class SoArm101PickPlaceBowlEnvCfg(PickPlaceBowlEnvCfg):
    """SO-ARM101 + 2 cm wooden block + bowl-goal command.

    Concrete subclass plugged into the gym registry. The 2-D bowl goal is
    visualized via a frame marker referencing the gripper link, but the
    bowl itself has no rigid body — see EVAL1_PLAN §3.3.
    """

    def __post_init__(self):
        super().__post_init__()

        # SO-ARM101 articulation
        self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Joint-position action — absolute-around-home with scale=0.5,
        # exactly the lift-task wiring (EVAL1_PLAN §3.2). Setting action=0
        # returns the arm to its default home pose, which makes safety
        # bounds straightforward both in sim and on hardware.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            # Open command widened from lift's 0.5 → 1.5 because the 2 cm
            # block needs more clearance than the 2.5 cm shrunk-DexCube.
            open_command_expr={"gripper": 1.5},
            close_command_expr={"gripper": 0.0},
        )

        # Bowl command marker is parented to the gripper for debug viz only;
        # the actual goal is the (x, y) sampled in the robot frame.
        self.commands.bowl_pose.body_name = "gripper_link"

        # 2 cm wooden block. We define the cube directly with CuboidCfg so
        # the geometry matches the eval spec exactly. Mass ≈ 4.8 g
        # (density ≈ 600 kg/m³ → 8e-6 m³ × 600 = 4.8e-3 kg).
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.0, 0.01], rot=[1, 0, 0, 0]),
            spawn=sim_utils.CuboidCfg(
                size=(0.02, 0.02, 0.02),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                # Default visual color (will be overwritten per-env by DR).
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
                physics_material=RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=0.8,
                    dynamic_friction=0.6,
                ),
            ),
        )

        # End-effector frame transformer (used by reward / obs distance terms)
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
                    # Same offset as the lift task: ~9 cm out along z-local
                    # of the gripper link, plus 1 cm x correction. Keep this
                    # in sync with the FK helper used at deploy on hardware.
                    offset=OffsetCfg(pos=[0.01, 0.0, -0.09]),
                ),
            ],
        )


@configclass
class SoArm101PickPlaceBowlEnvCfg_PLAY(SoArm101PickPlaceBowlEnvCfg):
    """Smaller, low-DR variant for visual inspection / play."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable obs corruption when visualizing (DR is tested at train time).
        self.observations.policy.enable_corruption = False
