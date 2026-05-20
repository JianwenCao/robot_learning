"""Closed-loop PPO deployment on a real SO-ARM101.

This is the Eval-1 PPO real-robot loop. The checkpoint is the asymmetric
vision actor-critic from training; the actor side is reimplemented locally
in :mod:`bc.ppo_actor` so this PC does not need ``rsl_rl`` / ``isaaclab``
installed.

Usage::

    python -m bc.deploy_real --bowl-xy 0.20,-0.05
    # validate everything except the arm + camera:
    python -m bc.deploy_real --bowl-xy 0.20,-0.05 --dry-run

Observation pipeline (must match what training saw):

* state (25,) — joint_pos_rel(6) + joint_vel_rel(6) + gripper_state(1) +
                  bowl_xy(2) + ee_proj_xy(2) + ee_to_bowl_xy(2) + last_action(6)
  All radians / metres, robot base frame. ``*_rel`` = joint value minus the
  URDF default joint position (matches ``isaaclab.envs.mdp.joint_pos_rel`` /
  ``joint_vel_rel`` semantics). FK is computed on the host via :mod:`kinpy`.
* image (4, 72, 128) — RGB + binary block mask in ``[0, 1]``.
    * RGB from the wrist USB cam, undistorted with ``camera_intrinsics.yaml``,
      resized to 128×72.
    * Mask via HSV ``cv2.inRange``. For Task 1 the target cube can be any
      color (red / orange / yellow / green / blue / purple), so the default
      bound is a **saturation gate** (H ∈ [0, 180], S ≥ ~90) that fires on
      any colored object and rejects the near-white table. The largest
      connected component is then kept, mimicking the single-block semantic
      mask the sim policy was trained with and dropping distractors like
      cables or a tool handle in the corner. Override per-scene with
      ``--hsv-low`` / ``--hsv-high`` if lighting shifts a lot.

Action pipeline (must match training):

* Policy outputs 6-D Gaussian mean (we use the mean directly, no sampling).
* Arm targets (5 joints): ``target_rad = default_rad + 0.5 * action[:5]``
  (matches ``JointPositionActionCfg(scale=0.5, use_default_offset=True)``).
* Gripper (1 joint): ``0.5`` if ``action[5] > 0`` else ``0.0`` (matches
  ``BinaryJointPositionActionCfg(open=0.5, close=0.0)``).
* Per-motor unit conversion lives in ``LerobotSO101Driver``: the five arm
  motors are in ``MotorNormMode.DEGREES`` (so rad ↔ deg) and the gripper
  is in ``MotorNormMode.RANGE_0_100`` (so the sim's [0, 0.5] rad maps
  linearly to [0, 100] %). Writing sim-rad straight to the bus would put
  the gripper at 0.5 % open (≈ fully closed) on every "open" command.

Control rate is 50 Hz to match the sim (``decimation=2``, ``sim.dt=0.01`` →
50 Hz, 250 steps over the 5 s episode). The policy is queried *every* step
(no chunking) because PPO is reactive and the image pipeline runs in <20 ms
on CPU.

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
* pre-rollout slow-slew to the sim reset pose ``(0, 0, 0, 1.57, 0, gripper=open)``
  using the same ``MAX_RAD_PER_STEP`` cap, so the t=0 obs has ``joint_pos_rel ≈ 0``
  (matches the sim reset distribution; otherwise the slew cap silently clips the
  policy's early actions while the arm catches up from whatever pose it sat in).
  Homing has a timeout — on miss we warn and proceed rather than block the rollout.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import torch

from bc.config import PROJECT_ROOT
from bc.ppo_actor import PPOActor

# ====================================================================== config
CKPT_CANDIDATES = [
    PROJECT_ROOT / "bc" / "runs" / "deploy" / "model.pt",
]
URDF_PATH      = PROJECT_ROOT / "isaac_so_arm101" / "src" / "isaac_so_arm101" / "robots" / "trs_so101" / "urdf" / "so_arm101.urdf"
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
# motor in ``MotorNormMode.RANGE_0_100`` (not DEGREES like the arm). So
# ``obs["gripper.pos"] ∈ [0, 100]`` and goal-position writes to
# ``gripper.pos`` are interpreted as % of the calibrated range. We map
# [0, 100] linearly to the sim's [0, GRIPPER_OPEN_RAD] joint range so the
# observation and the action both arrive in the sim's units (which is the
# distribution the policy was trained on). 100 % open → 0.5 sim_rad, 0 %
# → 0 sim_rad.
GRIPPER_PCT_PER_SIM_RAD = 100.0 / GRIPPER_OPEN_RAD

# Pre-rollout homing target. Arm joints match the sim reset pose
# (``SO_ARM101_CFG.init_state.joint_pos`` overridden in ``joint_pos_env_cfg.py``
# with ``gripper=0.5``). Slew is rate-limited by ``MAX_RAD_PER_STEP``, same as
# the policy loop, so no jerk relative to what the policy expects.
HOME_POSE_RAD = np.array([0.0, 0.0, 0.0, 1.57, 0.0, GRIPPER_OPEN_RAD], dtype=np.float32)
HOME_TOLERANCE_RAD = 0.03               # ~1.7° — wider than Feetech deadband
HOMING_TIMEOUT_S = 6.0                  # 1.5 rad/s cap → ~1 s per rad; 6 s covers worst-case start

IMG_H, IMG_W = 72, 128
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
# and 1280×720 is 16:9, matching the sim render aspect (128×72) so the downsize
# to (IMG_W, IMG_H) is a clean ×10 with no FOV squashing.
CAM_INDEX   = 0
CAM_WIDTH   = 1280
CAM_HEIGHT  = 720
DEVICE      = "cpu"

# Color-agnostic mask for Task 1: any high-saturation cube on a near-white
# table. H is wide-open (0..180), S>=90 rejects the table/cables/bowl, V>=40
# rejects deep shadow. Tune via --hsv-low / --hsv-high if your lighting is
# unusually dim/bright.
HSV_LOW_DEFAULT  = (0,   90,  40)
HSV_HIGH_DEFAULT = (180, 255, 255)


# ============================================================== forward kine
class FK:
    """Wrapper around kinpy URDF FK. Caches the chain so per-step calls are cheap."""

    def __init__(self, urdf_path: Path, ee_link: str = EE_LINK_NAME):
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


def _hsv_mask(rgb_hwc_uint8: np.ndarray, low: tuple, high: tuple,
              keep_largest: bool = True, min_area: int = 6) -> np.ndarray:
    """Binary mask of colored objects on a near-white table.

    The default HSV bound is a saturation gate, so this fires on any
    high-S hue (red/orange/yellow/green/blue/purple — all Task 1 cubes).
    ``keep_largest=True`` then collapses the result to a single
    connected component (the dominant cube cluster), so distractors like
    a tool handle in the corner do not bleed into the channel the policy
    was trained to read as "the block".
    """
    import cv2
    hsv = cv2.cvtColor(rgb_hwc_uint8, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    if keep_largest:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            (m > 0).astype(np.uint8), connectivity=8
        )
        if n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]          # skip background (label 0)
            best = int(np.argmax(areas)) + 1
            if stats[best, cv2.CC_STAT_AREA] >= min_area:
                m = ((labels == best).astype(np.uint8)) * 255
            else:
                m = np.zeros_like(m)
        else:
            m = np.zeros_like(m)

    return (m.astype(np.float32) / 255.0)                            # H × W in [0, 1]


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
        out[5] = float(obs["gripper.pos"]) / GRIPPER_PCT_PER_SIM_RAD
        return out

    def capture_wrist_rgb_hwc(self) -> np.ndarray:
        ok, bgr = self._cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        return rgb                                                   # original res; we resize later

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
        pct = float(np.clip(target_sim_rad[5] * GRIPPER_PCT_PER_SIM_RAD, 0.0, 100.0))
        cmd["gripper.pos"] = pct
        self._robot.send_action(cmd)


# ============================================================== control loop
def _rad_to_deg(x): return x * (180.0 / np.pi)


def _build_state(q_rad, qdot_rad, bowl_xy, ee_xy, last_action) -> np.ndarray:
    joint_pos_rel = (q_rad - JOINT_DEFAULTS_RAD)[:6]                 # (6,)
    joint_vel_rel = qdot_rad[:6]                                     # (6,)
    gripper_state = q_rad[5:6]                                       # (1,)
    ee_to_bowl    = bowl_xy - ee_xy                                  # (2,)
    return np.concatenate([
        joint_pos_rel, joint_vel_rel, gripper_state,
        bowl_xy, ee_xy, ee_to_bowl, last_action,
    ]).astype(np.float32)                                            # 6+6+1+2+2+2+6 = 25


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


def _slew_to_home(driver, target_rad: np.ndarray = HOME_POSE_RAD,
                  tol: float = HOME_TOLERANCE_RAD,
                  timeout_s: float = HOMING_TIMEOUT_S,
                  fps: int = FPS,
                  max_step: float = MAX_RAD_PER_STEP) -> np.ndarray:
    """Drive the arm toward ``target_rad`` using the policy-loop slew cap.

    Returns the last observed joint pose (rad). On timeout, prints a warning
    and returns the residual pose so the caller can proceed — a sticky joint
    or off-by-a-lot start pose shouldn't block the rollout.
    """
    dt = 1.0 / fps
    t_start = time.time()
    deadline = t_start + timeout_s
    q = driver.read_proprio_sim_rad()
    next_tick = time.time()
    while time.time() < deadline:
        residual = q - target_rad
        if float(np.max(np.abs(residual))) < tol:
            print(f"[ppo] homed in {time.time() - t_start:.2f}s "
                  f"(residual={residual.round(3)} rad)")
            return q
        step = _slew_limit(target_rad, q, max_step=max_step)
        driver.send_joint_targets_sim_rad(step)
        next_tick += dt
        sleep_for = next_tick - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.time()
        q = driver.read_proprio_sim_rad()
    print(f"[ppo] WARNING: homing timed out after {timeout_s:.1f}s — "
          f"residual={(q - target_rad).round(3)} rad. Starting rollout anyway.")
    return q


# ============================================================== debug dump
def _open_debug_dir(args, bowl_xy: np.ndarray, ckpt_path: Path) -> Path | None:
    """Create the per-run debug dump directory and write a small meta.json.

    Returns the directory (so :func:`_debug_dump_step` knows where to
    write), or ``None`` if ``--debug-dump`` was not passed.
    """
    if not getattr(args, "debug_dump", False):
        return None
    import json
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = PROJECT_ROOT / "bc" / "runs" / "deploy" / "debug" / stamp
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "ckpt": str(ckpt_path),
        "bowl_xy": [float(bowl_xy[0]), float(bowl_xy[1])],
        "hsv_low": list(args.hsv_low),
        "hsv_high": list(args.hsv_high),
        "joint_defaults_rad": JOINT_DEFAULTS_RAD.tolist(),
        "fps": FPS,
        "max_steps": MAX_STEPS,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ppo] --debug-dump active: writing to {out}")
    return out


def _debug_dump_step(out: Path, t: int, image: np.ndarray, state: np.ndarray,
                      action: np.ndarray, q_rad: np.ndarray, ee_xy: np.ndarray,
                      target_rad: np.ndarray) -> None:
    """Save one step's RGB+mask image + a JSONL row for the rest.

    The image is the **exact** ``(4, 72, 128)`` tensor the policy receives,
    laid out side-by-side as a 72×(128+128) PNG: RGB on the left, mask on
    the right (binary mask broadcast to 3 channels). State/action/q/ee
    land in ``log.jsonl`` so they can be diffed against a sim-side
    rollout numerically.
    """
    import cv2, json
    rgb_u8 = (image[:3].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    mask_u8 = (image[3:4].repeat(3, axis=0).transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    composite = np.concatenate([rgb_u8, mask_u8], axis=1)
    bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out / f"step_{t:04d}.png"), bgr)
    row = {
        "t": int(t),
        "state": state.tolist(),
        "action": action.tolist(),
        "q_sim_rad": q_rad.tolist(),
        "ee_xy": ee_xy.tolist(),
        "target_sim_rad": target_rad.tolist(),
    }
    with open(out / "log.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def _build_image(rgb_hwc, K, dist, hsv_low, hsv_high) -> np.ndarray:
    import cv2
    if K is not None and dist is not None:
        rgb_hwc = cv2.undistort(rgb_hwc, K, dist)
    rgb_hwc = cv2.resize(rgb_hwc, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    rgb_chw = rgb_hwc.transpose(2, 0, 1).astype(np.float32) / 255.0  # (3, H, W)

    mask = _hsv_mask(rgb_hwc, hsv_low, hsv_high)                     # (H, W)

    return np.concatenate(
        [rgb_chw, mask[None]], axis=0
    ).astype(np.float32)                                             # (4, H, W)


def run(bowl_xy: np.ndarray, args) -> None:
    ckpt_path = next((p for p in CKPT_CANDIDATES if p.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(
            "PPO checkpoint not found. Looked at:\n  "
            + "\n  ".join(str(p) for p in CKPT_CANDIDATES)
            + "\nFetch the Drive zip per bc/readme.md."
        )
    policy = PPOActor.from_checkpoint(ckpt_path, map_location=DEVICE).to(DEVICE)
    print(f"[ppo] loaded {ckpt_path}")

    fk = FK(URDF_PATH)
    K_mat, dist = _load_intrinsics()
    if K_mat is None:
        warnings.warn("camera_intrinsics.yaml missing — skipping undistort.")

    if args.dry_run:
        print("[ppo] --dry-run: skipping hardware, doing a single forward with synthetic inputs.")
        q_rad = JOINT_DEFAULTS_RAD.copy()
        qdot_rad = np.zeros(6, dtype=np.float32)
        ee_xy = fk.ee_xyz(q_rad)[:2]
        state = _build_state(q_rad, qdot_rad, bowl_xy[:2], ee_xy, np.zeros(6, dtype=np.float32))
        synth_rgb = np.full((CAM_HEIGHT, CAM_WIDTH, 3), 128, dtype=np.uint8)
        image = _build_image(
            synth_rgb, K_mat, dist,
            tuple(args.hsv_low), tuple(args.hsv_high),
        )
        with torch.no_grad():
            a = policy(
                torch.from_numpy(state).unsqueeze(0).to(DEVICE),
                torch.from_numpy(image).unsqueeze(0).to(DEVICE),
            )[0].cpu().numpy()
        print(f"[ppo] dry-run image shape = {image.shape}  action mean = {a.round(3)}")
        return

    dump_dir = _open_debug_dir(args, bowl_xy, ckpt_path)

    driver = LerobotSO101Driver()
    driver.connect()
    try:
        q0 = driver.read_proprio_sim_rad()
        print(f"[ppo] pre-home pose: q_sim_rad={q0.round(3)} "
              f"(arm_deg={_rad_to_deg(q0[:5]).round(2)}, gripper_pct≈{q0[5]*GRIPPER_PCT_PER_SIM_RAD:.1f})")
        if not args.no_confirm:
            input("[ppo] arm will slow-slew to home (shoulder=0, wrist=90°, gripper=open). "
                  "Clear the workspace. Press <enter> to home, ctrl-C to abort … ")
        q_rad_prev = _slew_to_home(driver)
        ee_xyz_now = fk.ee_xyz(q_rad_prev)
        print(f"[ppo] homed: q_sim_rad={q_rad_prev.round(3)}  "
              f"ee_xyz={ee_xyz_now.round(3)} (sim home ≈ (0.247, 0.000, 0.063))")
        if not args.no_confirm:
            input("[ppo] place block within x∈(0.13,0.28), y∈(-0.12,0.12). "
                  "Press <enter> to start rollout, ctrl-C to abort … ")
        qdot_filt = np.zeros(6, dtype=np.float32)
        last_action = np.zeros(6, dtype=np.float32)
        dt = 1.0 / FPS
        next_tick = time.time()

        for t in range(MAX_STEPS):
            q_rad = driver.read_proprio_sim_rad()
            qdot_raw = (q_rad - q_rad_prev) / dt
            qdot_filt = (QDOT_EWMA_ALPHA * qdot_raw
                         + (1.0 - QDOT_EWMA_ALPHA) * qdot_filt).astype(np.float32)
            q_rad_prev = q_rad

            ee_xy = fk.ee_xyz(q_rad)[:2]
            state = _build_state(q_rad, qdot_filt, bowl_xy[:2], ee_xy, last_action)

            rgb = driver.capture_wrist_rgb_hwc()
            image = _build_image(
                rgb, K_mat, dist,
                tuple(args.hsv_low), tuple(args.hsv_high),
            )

            with torch.no_grad():
                action = policy(
                    torch.from_numpy(state).unsqueeze(0).to(DEVICE),
                    torch.from_numpy(image).unsqueeze(0).to(DEVICE),
                )[0].cpu().numpy()
            last_action = action.astype(np.float32)

            target_rad = _decode_action(action)
            target_rad = _slew_limit(target_rad, q_rad)
            driver.send_joint_targets_sim_rad(target_rad)

            if dump_dir is not None:
                _debug_dump_step(dump_dir, t, image, state, action, q_rad, ee_xy, target_rad)

            if (t + 1) % 30 == 0:
                print(
                    f"  t={t+1:4d}  action={action.round(2)}  "
                    f"ee_xy={ee_xy.round(3)}  bowl_xy={bowl_xy[:2].round(3)}"
                )

            next_tick += dt
            sleep_for = next_tick - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.time()
    finally:
        driver.disconnect()
        if dump_dir is not None:
            print(f"[ppo] debug dump written to {dump_dir}")


# ============================================================== entry
def _parse_hsv(s: str) -> tuple[int, int, int]:
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected H,S,V triple, got {s!r}")
    return tuple(parts)                                              # type: ignore[return-value]


def main() -> int:
    p = argparse.ArgumentParser(description="PPO closed-loop deploy on real SO-ARM101")
    p.add_argument("--bowl-xy", type=str, required=True,
                   help="Comma-separated 'x,y' metres, robot base frame, e.g. 0.20,-0.05")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip hardware; load model + run one forward with synthetic obs.")
    p.add_argument("--hsv-low",  type=_parse_hsv, default=HSV_LOW_DEFAULT,
                   help=f"HSV lower bound for block mask, default {HSV_LOW_DEFAULT}")
    p.add_argument("--hsv-high", type=_parse_hsv, default=HSV_HIGH_DEFAULT,
                   help=f"HSV upper bound for block mask, default {HSV_HIGH_DEFAULT}")
    p.add_argument("--no-confirm", action="store_true",
                   help="Skip the pre-rollout <enter> prompt and start immediately.")
    p.add_argument("--debug-dump", action="store_true",
                   help="Save the wrist image (rgb+mask) and a state/action JSONL "
                        "per step under bc/runs/deploy/debug/<timestamp>/, so the "
                        "real rollout can be diffed against a sim play render.")
    args = p.parse_args()

    x, y = (float(s) for s in args.bowl_xy.split(","))
    bowl_xy = np.array([x, y], dtype=np.float32)

    run(bowl_xy, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
