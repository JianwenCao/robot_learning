"""Single source of truth for paths and a few constants shared across bc/."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEMOS_ROOT = PROJECT_ROOT / "demonstrations" / "RobotLearning-RL" / "Eval1"

PILOTS = [
    "eval1-pick-place-pilot",
    "eval1-pick-place-pilot-2",
]

# Action / observation dimensions
ARM_DOF = 5          # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
GRIPPER_DOF = 1
ACTION_DIM = ARM_DOF + GRIPPER_DOF        # 6
PROPRIO_DIM = ACTION_DIM                   # follower state has the same layout
BOWL_DIM = 3

# Image — matches sim_renders shape
IMG_C, IMG_H, IMG_W = 3, 72, 128

# Action chunking
CHUNK_K = 8           # predict this many future actions
EXECUTE_K = 4         # execute this many before re-querying at deploy

FPS = 30

RUNS_DIR = PROJECT_ROOT / "bc" / "runs"
