"""Shared real-robot driver, FK, action decode, and homing for ``deploy/``.

Imported by :mod:`deploy.deploy_real` (state-only + AprilTag closed loop)
and :mod:`deploy.calibrate_hand_eye`. Holds everything the deploy stack
needs that does not itself depend on the policy actor or the per-frame
detector: joint-name order, sim-side defaults, gripper unit conversion,
the LeRobot SO-101 follower wrapper, and pre-rollout homing.

If you change action scaling or joint ordering here, mirror it in the sim
side (``isaac_so_arm101/.../tasks/.../joint_pos_env_cfg.py``) — sim and
real must round-trip identically or the policy lands out-of-distribution.

Sim-to-real execution gap
-------------------------
The sim arm uses ``ImplicitActuatorCfg(velocity_limit_sim=1.5 rad/s)`` plus a
stiff PD, so each commanded position step is rate-limited and the joints
physically cannot slew faster than 1.5 rad/s. The Feetech bus has no such
cap by default — ``SO101Follower.send_action`` writes ``Goal_Position`` raw
and the servo rushes to the new target at near-max speed. To make execution
match what the policy was trained against:

* host-side slew clamp ``MAX_RAD_PER_STEP = 1.5 / FPS`` on the commanded
  target (mirrors sim's ``velocity_limit_sim``);
* EWMA on the finite-difference joint velocity, since the sim's
  ``joint_vel_rel`` is a clean PhysX read and the servo FD is encoder-quantized
  + bus-jittered noise that lives outside the policy's training distribution;
* pre-rollout homing to the sim reset pose ``(0, 0, 0, 1.57, 0, gripper=open)``
  by commanding the full home target directly (slew cap bypassed) so the t=0
  obs has ``joint_pos_rel ≈ 0`` (matches the sim reset distribution; otherwise
  the slew cap silently clips the policy's early actions while the arm catches
  up from whatever pose it sat in). See ``_slew_to_home`` for the full
  rationale. Homing has a timeout — on miss we warn and proceed rather than
  block the rollout.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

URDF_PATH = (
    PROJECT_ROOT
    / "isaac_so_arm101" / "src" / "isaac_so_arm101"
    / "robots" / "trs_so101" / "urdf" / "so_arm101.urdf"
)
INTRINSICS_YAML = PROJECT_ROOT / "camera_intrinsics.yaml"

# Joint order must match LeRobot's SO101 follower obs / action keys.
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# FK end link. We use ``gripper_link`` (not ``gripper_frame_link``) so the
# real ``ee_xyz`` matches the sim ``ee_frame`` exactly — sim's
# ``FrameTransformer`` is ``gripper_link + offset(0.01, 0, -0.09)`` in the
# link's local frame, while ``gripper_frame_link`` adds a different URDF
# offset (-0.0079, ~0, -0.0981) plus a 180° y-rotation. The position
# discrepancy is ~1.8 cm in x — small but shifts ``ee_proj_xy`` and
# ``ee_to_bowl_xy`` out of the trained state distribution.
EE_LINK_NAME = "gripper_link"
EE_LOCAL_OFFSET = np.array([0.01, 0.0, -0.09], dtype=np.float64)

# Joint defaults the sim's ``joint_pos_rel`` subtracts. Sim's articulation
# default_joint_pos (set in ``joint_pos_env_cfg.py``) is
# (0, 0, 0, 1.57, 0, 0.5) — gripper home is OPEN at 0.5 rad. Keep this in
# lockstep with that override; otherwise ``joint_pos_rel[5]`` is off by 0.5
# every step.
JOINT_DEFAULTS_RAD = np.array([0.0, 0.0, 0.0, 1.57, 0.0, 0.5], dtype=np.float32)
ARM_ACTION_SCALE = 0.5                  # JointPositionActionCfg.scale
GRIPPER_OPEN_RAD = 0.5                  # sim "open" gripper joint position
GRIPPER_CLOSE_RAD = 0.0                 # sim "closed" gripper joint position
# Gripper unit conversion. LeRobot's ``SO101Follower`` puts the gripper
# motor in ``MotorNormMode.RANGE_0_100`` (not DEGREES like the arm). The
# standard SO-101 calibration sweep records ``range_min``/``range_max``
# at the jaw's mechanical limits, so on the bus 0 % = jaw fully closed
# and 100 % = jaw fully open. The URDF (``so_arm101.urdf``) declares the
# matching joint range as ``lower=-0.1745, upper=1.7453`` rad — sim
# defaults the gripper to 0.5 rad ("open" per the lift task's
# ``BinaryJointPositionActionCfg(open=0.5)``), which is only ~35 % of
# that band, NOT the mechanical limit. We therefore use an affine map
# pct ↔ sim_rad over the URDF range so 100 % pct lands at the URDF upper
# (where the calibration sweep ended), and sim 0.5 rad lands at ~35.1 %
# pct — both obs and action stay in the policy's training distribution.
GRIPPER_JAW_RAD_MIN = -0.174533         # URDF gripper joint ``lower``
GRIPPER_JAW_RAD_MAX = 1.74533           # URDF gripper joint ``upper``
GRIPPER_JAW_RAD_SPAN = GRIPPER_JAW_RAD_MAX - GRIPPER_JAW_RAD_MIN


def _gripper_pct_from_sim_rad(sim_rad: float) -> float:
    return (sim_rad - GRIPPER_JAW_RAD_MIN) / GRIPPER_JAW_RAD_SPAN * 100.0


def _gripper_sim_rad_from_pct(pct: float) -> float:
    return pct / 100.0 * GRIPPER_JAW_RAD_SPAN + GRIPPER_JAW_RAD_MIN


# Pre-rollout homing target. Arm joints match the sim reset pose
# (``SO_ARM101_CFG.init_state.joint_pos`` overridden in ``joint_pos_env_cfg.py``
# with ``gripper=0.5``). The home target is commanded directly (no slew cap,
# see ``_slew_to_home``) so gravity-loaded joints actually reach it before
# the timeout fires.
HOME_POSE_RAD = np.array([0.0, 0.0, 0.0, 1.57, 0.0, GRIPPER_OPEN_RAD], dtype=np.float32)
HOME_TOLERANCE_RAD = 0.03               # ~1.7° — wider than Feetech deadband
HOMING_TIMEOUT_S = 6.0                  # raw servo response; one or two PID cycles per joint

FPS = 50                                # matches sim (decimation=2, sim.dt=0.01 → 50 Hz)
MAX_STEPS = 5 * FPS                     # 5 s episode in sim
# Per-step joint-target slew cap. Mirrors so_arm101.py ``velocity_limit_sim=1.5``
# rad/s. Applied as ``|target - q| <= MAX_RAD_PER_STEP`` before the servo write.
SIM_VEL_LIMIT = 1.5
MAX_RAD_PER_STEP = SIM_VEL_LIMIT / FPS
# EWMA factor for the finite-difference joint velocity. α=0.3 ≈ 5-sample
# trailing average at 50 Hz — fast enough to track real motion, slow enough
# to attenuate per-step encoder quantization noise.
QDOT_EWMA_ALPHA = 0.3

SERVO_PORT  = "/dev/tty.usbmodem5B140335911"
# Capture at the calibration resolution (camera_intrinsics.yaml is 1280×720).
# Required for cv2.undistort to be valid — K/dist are in 1280×720 pixel units,
# and 1280×720 is 16:9, matching the sim render aspect (128×72) so a downsize
# is a clean ×10 with no FOV squashing.
CAM_INDEX   = 0
CAM_WIDTH   = 1280
CAM_HEIGHT  = 720


# ============================================================== forward kine
class FK:
    """Wrapper around kinpy URDF FK. Caches the chain so per-step calls are cheap."""

    def __init__(self, urdf_path: Path = URDF_PATH, ee_link: str = EE_LINK_NAME):
        try:
            import kinpy as kp                                      # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError(
                "kinpy is required for ee_proj_xy. Install with `pip install kinpy`."
            ) from e
        self._kp = kp
        with open(urdf_path, "rb") as f:
            self.chain = kp.build_serial_chain_from_urdf(f.read(), end_link_name=ee_link)
        self._joint_names = self.chain.get_joint_parameter_names()  # 5 arm joints

    def ee_xyz(self, joint_pos_rad: np.ndarray) -> np.ndarray:
        """Compute ee xyz in the robot base frame, matching sim's ee_frame.

        kinpy gives us ``T = base → gripper_link``; we then apply the same
        local offset ``EE_LOCAL_OFFSET`` that the sim's
        ``FrameTransformer`` uses, rotated by the link's orientation, so
        the result equals the sim's ``ee_w`` to floating-point.
        """
        arm_vals = {n: float(v) for n, v in zip(JOINT_NAMES[:5], joint_pos_rad[:5])}
        th = [arm_vals[n] for n in self._joint_names]
        T = self.chain.forward_kinematics(th)
        ee = np.asarray(T.pos, dtype=np.float64) + T.rot_mat @ EE_LOCAL_OFFSET
        return ee.astype(np.float32)


# ============================================================== camera + undistort
def _load_intrinsics():
    """Return (K, dist) numpy arrays from camera_intrinsics.yaml, or (None, None)."""
    if not INTRINSICS_YAML.exists():
        return None, None
    import yaml
    class _Loader(yaml.SafeLoader):
        pass
    _Loader.add_multi_constructor("tag:yaml.org,2002:python/object/apply:",
                                  lambda l, t, n: None)
    _Loader.add_multi_constructor("tag:yaml.org,2002:python/object/new:",
                                  lambda l, t, n: None)
    _Loader.add_multi_constructor("tag:yaml.org,2002:python/name:",
                                  lambda l, t, n: None)
    with open(INTRINSICS_YAML, "r") as f:
        data = yaml.load(f, Loader=_Loader)
    K = np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    dist = np.array(data["distortion_coefficients"]["data"], dtype=np.float64)
    return K, dist


# ============================================================== hardware driver
class LerobotSO101Driver:
    """Thin wrapper around ``lerobot.robots.so_follower.SO101Follower`` + USB cam.

    Converts the bus's mixed unit modes (arm in DEGREES, gripper in
    RANGE_0_100) to/from the sim's pure-radian convention so the policy
    sees the same observation distribution it trained on.
    """

    def __init__(self):
        self._robot = None
        self._cap = None
        self._cv2 = None

    def connect(self):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        cfg = SO101FollowerConfig(port=SERVO_PORT, id="follower")
        self._robot = SO101Follower(cfg)
        self._robot.connect()
        import cv2
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(CAM_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        if not self._cap.isOpened():
            raise RuntimeError(f"failed to open camera index {CAM_INDEX}")

    def disconnect(self):
        if self._cap is not None:
            self._cap.release()
        if self._robot is not None:
            self._robot.disconnect()

    def read_proprio_sim_rad(self) -> np.ndarray:
        """Read all 6 joint positions and return them in the sim's units.

        Arm joints (0..4) are configured as ``MotorNormMode.DEGREES`` on
        the bus, so ``obs["{n}.pos"]`` returns degrees → convert to rad.
        The gripper (5) is configured as ``MotorNormMode.RANGE_0_100``,
        so ``obs["gripper.pos"] ∈ [0, 100]`` (% of the calibrated range)
        → linearly mapped to the sim's [0, GRIPPER_OPEN_RAD] range so the
        observation matches what the policy saw in training.
        """
        obs = self._robot.get_observation()
        out = np.empty(6, dtype=np.float32)
        for i, n in enumerate(JOINT_NAMES[:5]):
            out[i] = float(obs[f"{n}.pos"]) * (np.pi / 180.0)
        out[5] = _gripper_sim_rad_from_pct(float(obs["gripper.pos"]))
        return out

    def capture_wrist_rgb_hwc(self) -> np.ndarray:
        ok, bgr = self._cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        return rgb                                                   # original res; caller may resize

    def send_joint_targets_sim_rad(self, target_sim_rad: np.ndarray) -> None:
        """Send a 6-D joint target in sim units to the bus.

        Inverse of :meth:`read_proprio_sim_rad`: arm rad → deg, gripper
        sim_rad → pct. Without this conversion, sending ``gripper=0.5``
        (sim "open") writes ``0.5`` to the RANGE_0_100 motor — i.e.
        0.5 % open ≈ fully closed.
        """
        cmd: dict[str, float] = {}
        for i, n in enumerate(JOINT_NAMES[:5]):
            cmd[f"{n}.pos"] = float(target_sim_rad[i]) * (180.0 / np.pi)
        pct = float(np.clip(_gripper_pct_from_sim_rad(float(target_sim_rad[5])), 0.0, 100.0))
        cmd["gripper.pos"] = pct
        self._robot.send_action(cmd)


# ============================================================== control helpers
def _rad_to_deg(x): return x * (180.0 / np.pi)


def _decode_action(action6: np.ndarray) -> np.ndarray:
    """Map policy mean to joint targets in radians (same as sim action manager)."""
    arm_target = JOINT_DEFAULTS_RAD[:5] + ARM_ACTION_SCALE * action6[:5]
    grip_target = GRIPPER_OPEN_RAD if action6[5] > 0.0 else GRIPPER_CLOSE_RAD
    return np.concatenate([arm_target, [grip_target]]).astype(np.float32)


def _slew_limit(target_rad: np.ndarray, current_rad: np.ndarray,
                max_step: float = MAX_RAD_PER_STEP) -> np.ndarray:
    """Clamp |target - current| ≤ max_step per joint.

    Mirrors sim's actuator velocity cap (``velocity_limit_sim=1.5`` rad/s
    over a 1/FPS dt) so a single servo write can't outrun what the policy
    was trained to expect from one control tick.
    """
    delta = np.clip(target_rad - current_rad, -max_step, max_step)
    return (current_rad + delta).astype(np.float32)


def _slew_to_home(driver: LerobotSO101Driver, target_rad: np.ndarray = HOME_POSE_RAD,
                  tol: float = HOME_TOLERANCE_RAD,
                  timeout_s: float = HOMING_TIMEOUT_S,
                  fps: int = FPS) -> np.ndarray:
    """Drive the arm toward ``target_rad`` by commanding the home pose directly.

    Returns the last observed joint pose (rad). On timeout, prints a warning
    and returns the residual pose so the caller can proceed — a sticky joint
    or off-by-a-lot start pose shouldn't block the rollout.

    Bypasses the policy-loop slew cap (``MAX_RAD_PER_STEP``) on *all* joints
    during homing. The slew cap exists to mirror sim's
    ``velocity_limit_sim=1.5 rad/s`` during the in-distribution rollout, but
    during homing it instead keeps the per-step commanded-position error
    tiny (~0.03 rad), which produces almost no Feetech-PID torque (configured
    P=16 by LeRobot to suppress shakiness). Gravity-loaded joints —
    shoulder_lift and elbow_flex in particular — then never move from their
    pre-home pose, the 6 s timeout fires, and the rollout starts well outside
    the trained workspace (~9 cm off in ee_x in one observed run). Sending
    the full home target directly gives the PID a large position error to
    drive against, so each joint reaches home in one or two servo response
    cycles. Homing is a one-time pre-rollout setup, not an in-distribution
    policy step, so matching sim's transition rate isn't required here; the
    slew cap is still enforced during the rollout itself.
    """
    dt = 1.0 / fps
    t_start = time.time()
    deadline = t_start + timeout_s
    q = driver.read_proprio_sim_rad()
    next_tick = time.time()
    while time.time() < deadline:
        residual = q - target_rad
        if float(np.max(np.abs(residual))) < tol:
            per_joint = dict(zip(JOINT_NAMES, residual.round(3).tolist()))
            print(f"[ppo] homed in {time.time() - t_start:.2f}s "
                  f"(per-joint residual={per_joint})")
            return q
        driver.send_joint_targets_sim_rad(target_rad)
        next_tick += dt
        sleep_for = next_tick - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.time()
        q = driver.read_proprio_sim_rad()
    per_joint = dict(zip(JOINT_NAMES, (q - target_rad).round(3).tolist()))
    print(f"[ppo] WARNING: homing timed out after {timeout_s:.1f}s — "
          f"per-joint residual={per_joint}. Starting rollout anyway.")
    return q
