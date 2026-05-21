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
* **4-channel wrist image** (``RGB + target_mask``). Each cube carries
  a unique semantic class (``class:cube_<color>``), and
  :func:`mdp.wrist_rgb_mask_dr` filters the TiledCamera's
  ``semantic_segmentation`` output to the *target* cube per env. On the
  real arm the same mask comes from Florence-2 prompted by the target
  colour; the mask channel is corrupted in sim with morphology /
  dropout / wrong-colour-swap to match Florence-2's noise profile. The
  target_color one-hot is still in the policy state + FiLM head as
  belt-and-suspenders (fallback when the mask drops out).
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

    * ``bowl_pose`` — :class:`mdp.ClusterBowlPoseCommand`, a generalization
      of Eval-1's :class:`BowlPoseCommand` that rejection-samples bowl
      xy against the **two active cubes** (read from
      ``env._active_cube_indices``). 15 cm minimum separation in robot
      frame, 16 rejection attempts. Without this, ~5–10 % of episodes
      spawn the bowl on top of the cluster, giving the policy free
      "block already in bowl" reward signal.
    * ``target_color`` — see :class:`mdp.TargetColorCommand`. Samples the
      active pair + target index per episode.
    """

    bowl_pose = mdp.ClusterBowlPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        # Bumped 0.12 → 0.15 (2026-05-20) to match Eval-1's spacing —
        # the two active cubes land clearly away from the bowl so the
        # policy must actually transport (no visual ambiguity at the
        # ~20 cm wrist-cam standoff). max_attempts bumped 8 → 16 because
        # two cubes excluding two 15-cm disks in the 13×24 cm bowl
        # workspace makes rejection acceptance noticeably lower.
        min_distance=0.15,
        max_attempts=16,
        ranges=UniformPoseCommandCfg.Ranges(
            # 2026-05-20: unified to (0.18, 0.30) × (-0.15, 0.15) across
            # Eval-1/2/3 — see EVAL1 docstring for the reach-envelope
            # rationale (close-to-base bowls were forcing near-vertical
            # arm poses).
            pos_x=(0.18, 0.30),
            pos_y=(-0.15, 0.15),
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
    """Four groups: deployable policy, privileged critic, target-color goal, wrist image.

    ``goal`` carries the target_color one-hot in its own obs group so the
    vision actor-critic can route it to the CNN's FiLM head (Eval-2 §3.1).
    Stage 1 teacher includes it in ``policy``+``critic`` of ``obs_groups``
    so the MLP also reads it.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        gripper_state = ObsTerm(func=mdp.gripper_state)
        bowl_xy = ObsTerm(func=mdp.bowl_xy, params={"command_name": "bowl_pose"})
        ee_proj_xy = ObsTerm(func=mdp.ee_proj_xy)
        ee_to_bowl_xy = ObsTerm(func=mdp.ee_to_bowl_xy, params={"command_name": "bowl_pose"})
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
    class GoalCfg(ObsGroup):
        """Goal conditioning — target color one-hot, kept separate so the
        vision actor-critic can route it to the CNN's FiLM head."""

        target_color = ObsTerm(func=mdp.target_color_onehot, params={"command_name": "target_color"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class WristImageCfg(ObsGroup):
        """4-channel ``RGB + target_mask`` wrist obs.

        Each cube is tagged ``class:cube_<color>`` in
        :mod:`joint_pos_env_cfg`; :func:`mdp.wrist_rgb_mask_dr` filters
        the TiledCamera's ``semantic_segmentation`` to the *target*
        cube per env (resolved via ``env._target_cube_idx``). The mask
        channel is corrupted to mimic Florence-2's failure modes at
        deploy — see ``mdp.wrist_rgb_mask_dr`` docstring for the four
        DR axes (small-area dropout, morphology, full dropout, wrong-
        colour swap). Play cfgs override ``corrupt=False`` so the eval
        viewer sees the GT mask.
        """

        wrist_image = ObsTerm(
            func=mdp.wrist_rgb_mask_dr,
            params={"corrupt": True},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    goal: GoalCfg = GoalCfg()
    wrist_image: WristImageCfg = WristImageCfg()


@configclass
class ClutterStateAprilTagObservationsCfg(ObservationsCfg):
    """Obs variant for the state-only + AprilTag deploy path (Eval-2).

    Mirrors EVAL2_PLAN.md §2: the policy obs is **target-only** — the
    deploy script keys ``pupil-apriltags`` to a single tag ID via
    ``AprilTagDetector.set_target_id``, so the sim obs is one 2-D slot
    matching that one-tag stream (``target_cube_pos_xy_noisy``). The full
    AprilTag noise pipeline still runs under the hood inside
    :func:`mdp.target_cube_pos_xy_noisy` (shared hand-eye bias + per-cube
    mount + per-step Gaussian + Bernoulli dropout + post-grasp freeze
    keyed on the target palette idx); the policy just consumes the
    target slot.

    The previous variant exposed all-cubes (``cube_positions_xy_noisy``,
    12-D) + per-cube visibility flags (6-D) without any explicit target
    identifier — the policy had to infer which slot was the target from
    masking dynamics. That made the obs harder to learn and didn't match
    the deploy stream. The target-only design also generalises to the
    Eval-3 sub-goal advancement (just re-key the tag ID; see
    :class:`SeqStateAprilTagObservationsCfg`).

    The env cfg that wires this in typically also nulls
    ``scene.wrist_cam`` + ``observations.wrist_image`` (no rendering
    needed for the state-only PPO path).
    """

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        target_cube_pos_xy_noisy = ObsTerm(func=mdp.target_cube_pos_xy_noisy)

    policy: PolicyCfg = PolicyCfg()


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
            # Workspace ranges for cube placement. Slightly tighter than
            # the bowl ranges (0.15-0.28, -0.12-0.12) so cubes stay in
            # the manipulable region.
            "block_x": (0.13, 0.25),
            "block_y": (-0.12, 0.12),
            # 12 cm pairwise separation between the 2 active cubes
            # leaves ~10 cm edge gap (for 2 cm cubes) — comfortably wider
            # than the SO-ARM gripper finger span, so the policy can
            # approach either cube without contacting the other. Bumped
            # from 10 cm to give more margin against AprilTag noise and
            # approach-jitter (per EVAL2_PLAN.md §1).
            "min_block_separation": 0.12,
            "table_z": 0.01,
            "max_attempts": 20,
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

    Tracks two distractor-aware diagnostics at zero weight:

    * ``distractor_disturb`` — shoving the wrong cube.
    * ``wrong_block_in_bowl`` — placing the wrong cube in the bowl.
      These remain logged, but they deliberately do not shape PPO:
      wrong-object behavior gets no reward and no penalty.

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

    # Target-keyed port of Eval-1's release-pose shaping. After the target
    # has been lifted, keep the end-effector high over the bowl xy,
    # including after release. The target cube is free to fall; release_in_bowl
    # is the term that rewards it landing low inside the footprint.
    ee_release_pose_over_bowl = RewTerm(
        func=mdp.target_ee_release_pose_over_bowl,
        params={
            "ee_height": 0.08,
            "xy_std": 0.06,
            "z_std": 0.04,
            "r_safe": 0.06,
            "bowl_height": 0.06,
            "minimal_height": 0.07,
            "command_name": "bowl_pose",
        },
        weight=20.0,
    )

    release_in_bowl = RewTerm(func=mdp.release_target_in_bowl, weight=30.0)

    # Anti-hover lures — target-keyed port of Eval-1's 2026-05-20 fix.
    # Without these, two consecutive Eval-2 v3 runs (2026-05-20_08-41-24,
    # 09-00-17) stalled in a "reach-and-camp" basin: reach saturated near
    # 0.7/step but lift never fired, release stayed 0. Eval-1's fix was to
    # add a small open-jaws-over-bowl lure (+3) and a hover-with-grasp
    # penalty (-1); replicating that target-keyed version here removes
    # the local maximum at "park EE on cube without closing gripper".
    # Same weight ratios as Eval-1; max ~3/step lure is well below
    # release(30) so full release still strictly dominates.
    gripper_open_above_bowl_lure = RewTerm(
        func=mdp.target_gripper_open_above_bowl_lure, weight=3.0,
    )
    still_grasped_above_bowl_penalty = RewTerm(
        func=mdp.target_still_grasped_above_bowl_penalty, weight=-2.0,
    )

    # Distractor-aware diagnostics (Eval-2 specific). Keep these terms at
    # zero weight so a wrong cube gets no reward and no penalty. Penalizing
    # them during early exploration produced a "move less" basin before the
    # policy had learned reliable target lift.
    distractor_disturb = RewTerm(
        func=mdp.distractor_disturb_penalty, weight=0.0,
        params={"threshold_speed": 0.05},
    )
    wrong_block_in_bowl = RewTerm(func=mdp.wrong_block_in_bowl, weight=0.0)

    # Penalties — ramped via curriculum, same as Eval-1.
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

    # Match Eval-1's action smoothness schedule exactly. The earlier Eval-2
    # target (-1e-2 over 15k steps) let high-energy fling / smash strategies
    # survive even after release success reached ~85%; Eval-1's working
    # recipe uses the stronger -1e-1 target over 10k steps.
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000},
    )
    log_success = CurrTerm(func=mdp.log_target_success_metrics, params={})


# ---------------------------------------------------------------------------
# Top-level env cfg
# ---------------------------------------------------------------------------


@configclass
class ClutterPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """SO-ARM101 targeted-pick-and-place in 2-cube clutter."""

    # num_envs default = 4096 (matches Eval-1) — bumped from 2048 after
    # the v3 stall, since GPU usage at 2048 envs was only ~4.5 GB / 32 GB
    # despite the 6-cube physics. Override in your train script via
    # env_cfg.scene.num_envs if VRAM is tight.
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
