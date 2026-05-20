#!/usr/bin/env bash
# One-time setup for the Eval-1/2/3 PPO **real-robot** inference PC.
#
# Idempotent: safe to re-run — each step skips itself when already satisfied.
# Creates/updates the `so_arm` conda env with PyTorch + the deploy deps
# (numpy, opencv, lerobot, kinpy for host FK). Isaac Lab / Isaac Sim are NOT
# installed here — this PC drives the real arm, not a simulator.
#
# Host pre-reqs (must be in place BEFORE running this script):
#   - miniconda3 or anaconda installed and `conda` on PATH.
#     If not, install with:
#       curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
#       bash /tmp/mc.sh -b -p $HOME/miniconda3
#       eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
#       conda init bash
#       exec bash    # restart the shell so 'conda' is on PATH
#
# GPU note: PPO actor runs fine on CPU at 30 Hz. If you have a CUDA GPU and
# want to use it, swap the CPU torch wheel for the cu128 wheel from
# https://pytorch.org/get-started/previous-versions/.
#
# Usage:
#   bash deploy/setup_inference_pc.sh

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-so_arm}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ============================================================ pre-flight ===
echo "[setup] repo root: $REPO_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: 'conda' is not on PATH.

Install miniconda3 first (one-time, ~5 min):
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
  bash /tmp/mc.sh -b -p $HOME/miniconda3
  eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
  conda init bash
  exec bash    # restart the shell

Then re-run: bash deploy/setup_inference_pc.sh
EOF
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# ============================================================ env create ===
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "[setup] conda env '$CONDA_ENV' exists — updating in place."
else
  echo "[setup] creating conda env '$CONDA_ENV' (python 3.11) ..."
  conda create -y -n "$CONDA_ENV" python=3.11 pip git
fi
conda activate "$CONDA_ENV"
echo "[setup] active python: $(which python) ($(python -V 2>&1))"

# ============================================================ installs ===
# 1. PyTorch (CPU). The PPO actor is tiny (~0.2M params for the CNN + MLP).
if python -c "import torch" 2>/dev/null; then
  echo "[setup] PyTorch already installed ($(python -c 'import torch;print(torch.__version__)')) — skipping."
else
  echo "[setup] installing PyTorch + torchvision (CPU wheels) ..."
  pip install --upgrade torch torchvision \
      --index-url https://download.pytorch.org/whl/cpu
fi

# 2. NumPy + OpenCV + PyYAML.
python -c "import numpy"  2>/dev/null || pip install numpy
python -c "import cv2"    2>/dev/null || pip install opencv-python
python -c "import yaml"   2>/dev/null || pip install pyyaml

# 3. LeRobot (real SO101 follower driver) + Feetech servo SDK.
#    - lerobot 0.4.x pulls huggingface_hub 0.35; transformers must stay < 5
#      because the 5.x line moved is_offline_mode out of huggingface_hub's
#      public API (we pin transformers<5 again in step 5 in case a later
#      install transitively bumps it).
#    - scservo_sdk is a runtime dep of LeRobot's FeetechMotorsBus (used by
#      SO101Follower) but isn't declared, so we install feetech-servo-sdk
#      explicitly.
python -c "import lerobot" 2>/dev/null || pip install lerobot 'transformers<5'
python -c "import scservo_sdk" 2>/dev/null || pip install feetech-servo-sdk

# 4. kinpy (pure-Python URDF FK; used to compute ee_proj_xy on host).
python -c "import kinpy" 2>/dev/null || pip install kinpy

# 5. Pin transformers<5 LAST. lerobot 0.4.x ships with huggingface_hub 0.35,
#    which is incompatible with transformers 5.x (the 5.x line moved
#    is_offline_mode out of huggingface_hub's public API). Other transitives
#    can pull a 5.x wheel in, so we downgrade after everything else is
#    installed.
if python -c "import transformers; from packaging.version import Version; import sys; sys.exit(0 if Version(transformers.__version__) < Version('5') else 1)" 2>/dev/null; then
  echo "[setup] transformers already <5 — skipping pin."
else
  echo "[setup] pinning transformers<5 for lerobot/huggingface_hub compatibility ..."
  pip install --quiet --upgrade 'transformers<5'
fi

# 6. AprilTag detector (CPU, ~2 ms/frame). The project is pivoting toward
#    state-only + AprilTag pose-injection as the default real-robot mask
#    source (see docs/STATE_APRILTAG_PLAN.md). pupil-apriltags is the
#    detector dep; the calibration + sim-side obs work is tracked separately.
python -c "import pupil_apriltags" 2>/dev/null || pip install --quiet pupil-apriltags

# 7. gdown — used by deploy/README.md Step 5 to fetch the trained PPO
#    checkpoint from Google Drive. Tiny pure-Python dep; install once here
#    so the README's `gdown ...` line works without extra steps.
python -c "import gdown" 2>/dev/null || pip install --quiet gdown

# ============================================================ verify ===
echo
echo "[verify] importing the modules real-robot PPO inference needs ..."
python - <<'PY'
import importlib, sys
mods = [
    "torch", "torchvision", "numpy", "cv2", "yaml", "kinpy",
    "pupil_apriltags",
    "deploy.ppo_actor", "deploy.driver", "deploy.deploy_real", "deploy.cube_detector",
]
fails = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  ok   {m}")
    except Exception as e:
        fails.append((m, e))
        print(f"  FAIL {m}: {e}")

import torch
print(f"  torch                     = {torch.__version__}")
print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}  (CPU is fine for this model)")

sys.exit(1 if fails else 0)
PY

# ============================================================ artifacts ===
echo
CKPT_PRIMARY="deploy/runs/state_apriltag_model.pt"
CKPT_FALLBACK="deploy/runs/model.pt"

if [[ -f "$CKPT_PRIMARY" || -f "$CKPT_FALLBACK" ]]; then
  echo "[verify] state+AprilTag PPO checkpoint present (one of $CKPT_PRIMARY, $CKPT_FALLBACK)."
else
  cat >&2 <<EOF
[verify] WARNING: state+AprilTag PPO checkpoint missing. Looked at:
   $CKPT_PRIMARY
   $CKPT_FALLBACK

Drop a trained state-only checkpoint at $CKPT_PRIMARY (see deploy/README.md step 5).
EOF
fi

if [[ -f "camera_intrinsics.yaml" ]]; then
  echo "[verify] camera_intrinsics.yaml present."
else
  echo "WARNING: camera_intrinsics.yaml missing at repo root — undistort will be skipped." >&2
fi

echo
echo "[setup] DONE."
echo "Validate the pipeline without hardware:"
echo "   python -m deploy.deploy_real --bowl-xy 0.20,-0.05 --dry-run"
echo "Then run on real hardware:"
echo "   python -m deploy.deploy_real --bowl-xy 0.20,-0.05"
