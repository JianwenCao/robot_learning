"""Closed-loop BC deployment on a real SO-ARM101.

Usage:
    python -m bc.deploy_real --bowl-xy <x>,<y>

    <x>,<y> are the bowl centre in metres, robot **base frame**.

Conventions (must match training data — LeRobot v3.0 ``so_follower`` schema):
    proprio        (6,) float32, **degrees**, joint order:
                       shoulder_pan, shoulder_lift, elbow_flex,
                       wrist_flex,   wrist_roll,    gripper
    action         (6,) float32, **degrees**, same layout — absolute joint targets
    wrist image    (3, 72, 128) uint8 RGB — captured cam is resized to this
    control rate   30 Hz (= bc.config.FPS, = demo recording rate)

Action chunking:
    The policy predicts CHUNK_K=8 future actions per query. We execute the
    first EXECUTE_K=4 open-loop at 30 Hz, then re-query — same as training.

Hardware defaults (edit here, not via CLI):
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from bc.config import CHUNK_K, EXECUTE_K, FPS, IMG_H, IMG_W, PROJECT_ROOT
from bc.model import GoalCondBCPolicy
from bc.normalize import Stats

# ------------------------------------------------------ hardware / run config
RUN_NAME       = "bc_eval1_v2"
CKPT_NAME      = "best.pt"
SERVO_PORT     = "/dev/ttyACM0"
CAMERA_INDEX   = 0
CAM_WIDTH      = 640
CAM_HEIGHT     = 480
MAX_STEPS      = 600           # 20 s @ 30 Hz
BOWL_Z         = 0.0           # demos use 0.0
DEVICE         = "cpu"         # model is ~12M params, CPU is fine

RUNS_DIR = PROJECT_ROOT / "bc" / "runs"


# ============================================================ robot driver ===
class LerobotSO101Driver:
    """Real SO101 follower via the LeRobot package + OpenCV webcam.

    Requires: `pip install lerobot opencv-python`
    """

    def __init__(self):
        self._robot = None
        self._cap = None
        self._cv2 = None

    def connect(self):
        # ---- LeRobot SO101 follower (TODO: confirm import path matches your lerobot version) ----
        try:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        except ImportError as e:
            raise RuntimeError(
                "lerobot is not installed (or import path changed). "
                "Install with: `pip install lerobot`. If your version exposes "
                "SO101 at a different path, edit LerobotSO101Driver.connect()."
            ) from e
        cfg = SO101FollowerConfig(port=SERVO_PORT, id="follower")
        self._robot = SO101Follower(cfg)
        self._robot.connect()

        # ---- OpenCV webcam (USB wrist cam) ----
        import cv2
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        if not self._cap.isOpened():
            raise RuntimeError(f"failed to open camera index {CAMERA_INDEX}")
        print(f"[hw] connected: SO101 on {SERVO_PORT}, cam {CAMERA_INDEX} "
              f"@ {CAM_WIDTH}x{CAM_HEIGHT}")

    def disconnect(self):
        if self._cap is not None:
            self._cap.release()
        if self._robot is not None:
            self._robot.disconnect()

    def read_proprio_deg(self) -> np.ndarray:
        obs = self._robot.get_observation()
        order = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                 "wrist_flex.pos",   "wrist_roll.pos",    "gripper.pos"]
        return np.array([float(obs[k]) for k in order], dtype=np.float32)

    def capture_wrist_rgb(self) -> np.ndarray:
        ok, frame_bgr = self._cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        frame_bgr = self._cv2.resize(frame_bgr, (IMG_W, IMG_H), interpolation=self._cv2.INTER_AREA)
        frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        return frame_rgb.transpose(2, 0, 1).astype(np.uint8)  # (3, H, W)

    def send_joint_targets_deg(self, target_deg: np.ndarray) -> None:
        action = {
            "shoulder_pan.pos":  float(target_deg[0]),
            "shoulder_lift.pos": float(target_deg[1]),
            "elbow_flex.pos":    float(target_deg[2]),
            "wrist_flex.pos":    float(target_deg[3]),
            "wrist_roll.pos":    float(target_deg[4]),
            "gripper.pos":       float(target_deg[5]),
        }
        self._robot.send_action(action)


# ============================================================ control loop ===
def run(bowl_xyz: np.ndarray) -> None:
    run_dir = RUNS_DIR / RUN_NAME
    stats = Stats.load(run_dir / "stats.json")
    ck = torch.load(run_dir / CKPT_NAME, map_location=DEVICE, weights_only=False)
    policy = GoalCondBCPolicy(k=CHUNK_K).to(DEVICE)
    policy.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    policy.eval()
    print(f"[bc] loaded {run_dir / CKPT_NAME}"
          + (f"  epoch={ck['epoch']}" if isinstance(ck, dict) and "epoch" in ck else ""))

    driver = LerobotSO101Driver()
    driver.connect()
    try:
        # Sanity-check shapes/ranges right after connect.
        q0 = driver.read_proprio_deg()
        img0 = driver.capture_wrist_rgb()
        assert q0.shape == (6,) and q0.dtype == np.float32, f"bad proprio: {q0.shape} {q0.dtype}"
        assert img0.shape == (3, IMG_H, IMG_W) and img0.dtype == np.uint8, \
            f"bad image: {img0.shape} {img0.dtype}"
        print(f"[bc] @reset  proprio(deg)={q0.round(2)}  bowl_xyz={bowl_xyz.round(3)}")
        print(f"[bc] @reset  img range=[{img0.min()}, {img0.max()}]")

        dt = 1.0 / FPS
        chunk: np.ndarray | None = None
        step_in_chunk = 0
        next_tick = time.time()

        for t in range(MAX_STEPS):
            if chunk is None or step_in_chunk >= EXECUTE_K:
                img = driver.capture_wrist_rgb()
                proprio = driver.read_proprio_deg()
                with torch.no_grad():
                    prop_n = stats.normalize("proprio", proprio)
                    bowl_n = stats.normalize("bowl",    bowl_xyz)
                    out_n = policy(
                        torch.from_numpy(img).to(DEVICE).unsqueeze(0),
                        torch.from_numpy(prop_n).to(DEVICE).unsqueeze(0),
                        torch.from_numpy(bowl_n).to(DEVICE).unsqueeze(0),
                    )                                   # (1, CHUNK_K, 6)
                chunk = stats.denormalize("action", out_n[0].cpu().numpy())   # (K, 6) deg
                step_in_chunk = 0

            target_deg = chunk[step_in_chunk].astype(np.float32)
            driver.send_joint_targets_deg(target_deg)
            step_in_chunk += 1

            if (t + 1) % 30 == 0:
                print(f"  t={t+1:4d}  target(deg)={target_deg.round(2)}")

            next_tick += dt
            sleep_for = next_tick - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.time()       # we fell behind; resync

    finally:
        driver.disconnect()


def main() -> int:
    p = argparse.ArgumentParser(description="BC closed-loop deploy on real SO-ARM101")
    p.add_argument("--bowl-xy", type=str, required=True,
                   help="Comma-separated 'x,y' in metres, robot base frame, e.g. 0.20,-0.05")
    args = p.parse_args()

    x, y = (float(s) for s in args.bowl_xy.split(","))
    bowl_xyz = np.array([x, y, BOWL_Z], dtype=np.float32)

    run_dir = RUNS_DIR / RUN_NAME
    if not (run_dir / CKPT_NAME).exists() or not (run_dir / "stats.json").exists():
        print(f"ERROR: missing {run_dir}/{CKPT_NAME} or {run_dir}/stats.json", file=sys.stderr)
        return 2

    run(bowl_xyz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
