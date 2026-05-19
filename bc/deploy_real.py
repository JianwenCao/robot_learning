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
    * Mask via HSV ``cv2.inRange`` against the wood-tone block on the gray
      table. Defaults are a reasonable starting bound; override with
      ``--hsv-low`` / ``--hsv-high`` after one-shot calibration.

Action pipeline (must match training):

* Policy outputs 6-D Gaussian mean (we use the mean directly, no sampling).
* Arm targets (5 joints): ``target_rad = default_rad + 0.5 * action[:5]``
  (matches ``JointPositionActionCfg(scale=0.5, use_default_offset=True)``).
* Gripper (1 joint): ``0.5`` if ``action[5] > 0`` else ``0.0`` (matches
  ``BinaryJointPositionActionCfg(open=0.5, close=0.0)``).
* Targets converted to degrees before sending — LeRobot's SO101 follower API
  takes Feetech servo positions in degrees.

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
* homing trajectory at startup so the first policy obs matches the sim
  reset state (``JOINT_DEFAULTS_RAD``, ``last_action = 0``);
* EWMA on the finite-difference joint velocity, since the sim's
  ``joint_vel_rel`` is a clean PhysX read and the servo FD is encoder-quantized
  + bus-jittered noise that lives outside the policy's training distribution.
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
EE_LINK_NAME = "gripper_frame_link"     # URDF tip frame; see so_arm101.urdf

# Defaults from SO_ARM101_CFG.init_state.joint_pos (radians).
JOINT_DEFAULTS_RAD = np.array([0.0, 0.0, 0.0, 1.57, 0.0, 0.0], dtype=np.float32)
ARM_ACTION_SCALE = 0.5                  # JointPositionActionCfg.scale
GRIPPER_OPEN_RAD = 0.5
GRIPPER_CLOSE_RAD = 0.0

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

SERVO_PORT  = "/dev/tty.usbmodem5B140319261"
CAM_INDEX   = 0
CAM_WIDTH   = 640
CAM_HEIGHT  = 480
DEVICE      = "cpu"

# HSV thresholds for the wood-tone block on a gray table. These get you in the
# ballpark; calibrate once on a captured frame and pass --hsv-low / --hsv-high.
HSV_LOW_DEFAULT  = (5, 60, 50)
HSV_HIGH_DEFAULT = (30, 255, 220)


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
        """Compute ee xyz in the robot base frame.

        Args:
            joint_pos_rad: ``(6,)`` joint positions in radians, in JOINT_NAMES
                order. Only the first 5 (arm) are used for FK; gripper is skipped.
        """
        # kinpy returns the chain joints in URDF order. We must reorder our
        # 6-vector to whatever kinpy enumerated. JOINT_NAMES[:5] are the arm
        # joints; build a name → value map and pull in chain order.
        arm_vals = {n: float(v) for n, v in zip(JOINT_NAMES[:5], joint_pos_rad[:5])}
        th = [arm_vals[n] for n in self._joint_names]
        T = self.chain.forward_kinematics(th)
        return np.asarray(T.pos, dtype=np.float32)


def _hsv_mask(rgb_hwc_uint8: np.ndarray, low: tuple, high: tuple) -> np.ndarray:
    import cv2
    hsv = cv2.cvtColor(rgb_hwc_uint8, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
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

    def read_proprio_deg(self) -> np.ndarray:
        obs = self._robot.get_observation()
        return np.array([float(obs[f"{n}.pos"]) for n in JOINT_NAMES], dtype=np.float32)

    def capture_wrist_rgb_hwc(self) -> np.ndarray:
        ok, bgr = self._cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        return rgb                                                   # original res; we resize later

    def send_joint_targets_deg(self, target_deg: np.ndarray) -> None:
        self._robot.send_action(
            {f"{n}.pos": float(v) for n, v in zip(JOINT_NAMES, target_deg)}
        )


# ============================================================== control loop
def _deg_to_rad(x): return x * (np.pi / 180.0)
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


def _home_arm(driver: "LerobotSO101Driver", settle_s: float = 0.3) -> None:
    """Rate-limited move from wherever the arm is to ``JOINT_DEFAULTS_RAD``.

    Ensures the first policy obs matches the sim reset state (joint_pos_rel
    ≈ 0, joint_vel_rel ≈ 0, last_action = 0). Caps each step at
    ``MAX_RAD_PER_STEP`` so the homing trajectory is itself sim-rate.
    """
    dt = 1.0 / FPS
    next_tick = time.time()
    while True:
        q_rad = _deg_to_rad(driver.read_proprio_deg())
        err = JOINT_DEFAULTS_RAD - q_rad
        if np.max(np.abs(err)) < MAX_RAD_PER_STEP * 0.5:
            break
        step_target = _slew_limit(JOINT_DEFAULTS_RAD, q_rad)
        driver.send_joint_targets_deg(_rad_to_deg(step_target))
        next_tick += dt
        sleep_for = next_tick - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.time()
    time.sleep(settle_s)  # let the servos settle before reading proprio for the policy


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

    driver = LerobotSO101Driver()
    driver.connect()
    try:
        print("[ppo] homing arm to URDF defaults …")
        _home_arm(driver)
        q0_deg = driver.read_proprio_deg()
        q_rad_prev = _deg_to_rad(q0_deg)
        qdot_filt = np.zeros(6, dtype=np.float32)
        last_action = np.zeros(6, dtype=np.float32)
        dt = 1.0 / FPS
        next_tick = time.time()

        for t in range(MAX_STEPS):
            q_rad = _deg_to_rad(driver.read_proprio_deg())
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
            driver.send_joint_targets_deg(_rad_to_deg(target_rad))

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
    args = p.parse_args()

    x, y = (float(s) for s in args.bowl_xy.split(","))
    bowl_xy = np.array([x, y], dtype=np.float32)

    run(bowl_xy, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
