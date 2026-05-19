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
from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import (
    FrameTransformerCfg,
    OffsetCfg,
)
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg  # noqa: F401  kept for fallback paths
from isaaclab.sim.spawners.sensors.sensors_cfg import PinholeCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import isaac_so_arm101.tasks.pickplace.mdp as mdp
from isaac_so_arm101.robots import SO_ARM101_CFG  # noqa: F401
from isaac_so_arm101.tasks.pickplace.pickplace_env_cfg import (
    PickPlaceBowlEnvCfg,
    WRIST_RGB_HEIGHT,
    WRIST_RGB_WIDTH,
)

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

        # SO-ARM101 articulation. Override the home gripper joint to be
        # **open** (0.5 rad ≈ matches the lift-task ``open_command_expr``)
        # instead of the SO_ARM101_CFG default 0 rad (closed). With jaws
        # already open at start, the policy's grasp problem reduces to
        # "close gripper when ee is close to block", which is one-step
        # rather than the longer sequence (open → approach → close) that
        # plagued the closed-jaw home in earlier diagnostics.
        home_state = SO_ARM101_CFG.init_state.replace(
            joint_pos={**SO_ARM101_CFG.init_state.joint_pos, "gripper": 0.5},
        )
        self.scene.robot = SO_ARM101_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=home_state,
        )

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
            # Open command matches the lift task (0.5 rad). Earlier we used
            # 1.5 rad for "more clearance" around the 2 cm block, but with
            # identical PD gains and effort_limit=2.5 N, closing 1.5 → 0 is
            # ~3× slower than 0.5 → 0, and at 50 Hz control the gripper
            # never finishes closing before the arm moves on. Lift task
            # grips a 2.5 cm cube fine at 0.5; our 2 cm cube is smaller.
            open_command_expr={"gripper": 0.5},
            close_command_expr={"gripper": 0.0},
        )

        # Bowl command marker is parented to the gripper for debug viz only;
        # the actual goal is the (x, y) sampled in the robot frame.
        self.commands.bowl_pose.body_name = "gripper_link"

        # Make the goal-pose marker actually visible — default
        # ``FRAME_MARKER_CFG`` is a 5 cm RGB tripod that disappears against
        # the busy scene with multiple envs. Replace with a bright RED
        # SPHERE_MARKER (radius 3 cm) at the bowl xyz so the place target
        # is unmistakable in the viewport. Also shrink the current-pose
        # frame marker (gripper-tracking) so it doesn't visually clutter
        # near the goal.
        from isaaclab.markers.config import SPHERE_MARKER_CFG, FRAME_MARKER_CFG

        goal_marker = SPHERE_MARKER_CFG.copy()
        goal_marker.prim_path = "/Visuals/Command/goal_pose"
        goal_marker.markers["sphere"].radius = 0.03  # 3 cm — visible on table scale
        goal_marker.markers["sphere"].visual_material.diffuse_color = (1.0, 0.0, 0.0)  # red
        self.commands.bowl_pose.goal_pose_visualizer_cfg = goal_marker

        cur_marker = FRAME_MARKER_CFG.copy()
        cur_marker.prim_path = "/Visuals/Command/body_pose"
        cur_marker.markers["frame"].scale = (0.04, 0.04, 0.04)
        self.commands.bowl_pose.current_pose_visualizer_cfg = cur_marker

        # 2 cm cube — switched from CuboidCfg primitive to NVIDIA's
        # ``dex_cube_instanceable.usd`` (verbatim port from the upstream
        # isaac_so_arm101 lift task) after run-19 diagnostic showed our
        # primitive cube's hand-tuned physics_material wouldn't reliably
        # grasp under random gripper exploration. The dex cube ships with
        # NVIDIA-tuned mass/friction/collision shape that's known-good
        # for binary-gripper grasping (used by Franka Lift, ManiSkill3
        # PickCube, and the upstream SO-ARM101 lift task that converged
        # on this same MDP class).
        #
        # Native dex_cube is ~5 cm; scale 0.4 → ~2 cm to match the eval
        # spec (single 2×2×2 cm wooden cube). Upstream uses scale 0.5
        # (~2.5 cm); we go slightly smaller. The instanceable USD scales
        # proportionally on mass and collision via ``RigidBodyPropertiesCfg``
        # so physics stays balanced.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.0, 0.01], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.4, 0.4, 0.4),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                # Semantic class label so the wrist camera's
                # ``semantic_segmentation`` output produces a clean binary
                # mask of the block (channel 4 of ``mdp.wrist_image``).
                semantic_tags=[("class", "block")],
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

        # ------------------------------------------------------------------
        # Wrist camera — TiledCamera parented to gripper_link.
        # ------------------------------------------------------------------
        # Intrinsics are loaded from camera_intrinsics.yaml (the real wrist
        # cam's calibration) and converted to USD pinhole parameters. This
        # makes the simulated horizontal FOV match the real camera's exactly,
        # which is what we rely on for sim-to-real visual transfer.
        #
        # Render resolution is 16:9 (WRIST_RGB_WIDTH × WRIST_RGB_HEIGHT) so
        # the aspect ratio matches the real cam — preserving the wide
        # horizontal FOV (~102°) that the policy uses to search for the
        # block when it spawns at a workspace edge (EVAL1_PLAN §4).
        #
        # Extrinsic offset is a starting estimate; measure with calipers
        # before deployment and update both the sim OffsetCfg and the
        # deploy-side FK chain in lockstep.
        intrinsics = mdp.load_wrist_cam_intrinsics()
        self.scene.wrist_cam = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_link/wrist_cam",
            update_period=self.sim.dt * self.decimation,  # match policy step
            height=WRIST_RGB_HEIGHT,
            width=WRIST_RGB_WIDTH,
            # RGB + depth + semantic seg → assembled into a 5-channel
            # ``wrist_image`` obs in :func:`mdp.wrist_image`. Depth fuels
            # the geometric "step function at the cube" cue (sim-real
            # invariant); semantic seg becomes the binary cube-mask
            # channel (replicated on the real side via HSV thresholding).
            # ``colorize_semantic_segmentation=False`` keeps the seg
            # output as a single-channel int8 ID map (not RGB), which is
            # what the obs function reads.
            # v4 dropped depth from ``mdp.wrist_image``; depth was still being
            # rendered each step despite being unused. Removed from data_types
            # to skip the depth pass (~15-20% render speedup).
            data_types=["rgb", "semantic_segmentation"],
            colorize_semantic_segmentation=False,
            # ``semantic_filter="class:block"`` restricts the segmentation
            # output to only prims tagged ``("class", "block")`` — every
            # other prim (table, robot, ground) is treated as unlabeled
            # (ID 0). Without this filter, the default ``"*:*"`` labels
            # every prim with a non-zero ID and our binary mask
            # ``(seg > 0)`` ends up covering the entire image (caught
            # this in the post-implementation smoke test: mask_frac=0.935
            # before this fix).
            semantic_filter="class:block",
            spawn=PinholeCameraCfg(
                focal_length=intrinsics["focal_length"],
                horizontal_aperture=intrinsics["horizontal_aperture"],
                # Vertical aperture auto-derived to keep square pixels (we
                # already verified fx ≈ fy in the loader). Far clipping
                # tightened to 2 m — workspace is ≤ 0.5 m from cam.
                clipping_range=(0.01, 2.0),
            ),
            # Wrist-cam mounting offset on gripper_link. Verbatim from
            # LeIsaac's `single_arm_env_cfg.py`, which targets the same
            # physical robot (TheRobotStudio SO-ARM101) and the same
            # standard LeRobot/WOWROBO side-bracket wrist-cam mount we
            # use. Their USD link `Robot/gripper` and our URDF-converted
            # `Robot/gripper_link` are the same physical link with the
            # same frame, so the 7 numbers transfer directly:
            #   https://github.com/LightwheelAI/leisaac/blob/main/source/leisaac/leisaac/tasks/template/single_arm_env_cfg.py
            # The bracket sits ~10 cm out in gripper +Y (the side away
            # from the moving jaw) and tilts the lens ~48° down-and-
            # toward-fingers so the table and gripper tips are both in
            # frame at home pose.
            offset=CameraCfg.OffsetCfg(
                pos=(-0.001, 0.1, -0.04),
                rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
                convention="ros",
            ),
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
        # Turn off the per-step image corruption inside ``mdp.wrist_image``
        # too, otherwise eval frames carry the same Gaussian noise / depth
        # jitter the training run does. The per-episode tint event still
        # fires (kept it on so the eval distribution matches training) but
        # the `corrupt=False` arg short-circuits the per-frame jitters.
        self.observations.wrist_image.wrist_image.params = {"corrupt": False}


@configclass
class SoArm101PickPlaceBowlTeacherFastEnvCfg(SoArm101PickPlaceBowlEnvCfg):
    """Camera-free env cfg for the state-only teacher.

    The default env cfg spawns a ``TiledCamera`` (RGB + semantic-seg) on
    every env so the same scene works for the vision student. The teacher
    never reads the wrist image — but PhysX + RTX still pay the full
    rendering cost each step, dropping GPU util to ~30 % and capping
    iter wall-clock at ~3.2 s.

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
        # Null the wrist camera spawn. Manager skips entries set to None.
        self.scene.wrist_cam = None
        # Drop the wrist_image obs group too — its obs function would
        # otherwise try to read sensors["wrist_cam"] each step.
        self.observations.wrist_image = None


@configclass
class SoArm101PickPlaceBowlTeacherFastEnvCfg_PLAY(SoArm101PickPlaceBowlTeacherFastEnvCfg):
    """Smaller variant of the camera-free teacher env, for visual eval."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
