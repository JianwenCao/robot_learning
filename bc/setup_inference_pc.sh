#!/usr/bin/env bash
# One-time setup for the Eval-1 BC **real-robot** inference PC.
#
# Idempotent: safe to re-run — each step skips itself when already satisfied.
# Creates/updates the `so_arm` conda env with a CPU-only PyTorch (the BC model
# is a ResNet-18 + small MLPs, runs at >>10 Hz on CPU). Isaac Lab / Isaac Sim
# are NOT installed here — this PC drives the real arm, not a simulator.
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
# No GPU / NVIDIA driver required. If you do have a CUDA GPU on the inference
# PC and want to use it, install the cu128 wheels from
# https://pytorch.org/get-started/previous-versions/ instead of the CPU step
# below; the rest of the script is unchanged.
#
# Usage:
#   bash bc/setup_inference_pc.sh

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

Then re-run: bash bc/setup_inference_pc.sh
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
  conda create -y -n "$CONDA_ENV" python=3.11 pip
fi
conda activate "$CONDA_ENV"
echo "[setup] active python: $(which python) ($(python -V 2>&1))"

# ============================================================ installs ===
# 1. PyTorch (CPU). The BC policy is small — ResNet-18 backbone + a couple of
#    MLPs, ~12M params — and runs comfortably at the control rate on CPU.
if python -c "import torch" 2>/dev/null; then
  echo "[setup] PyTorch already installed ($(python -c 'import torch;print(torch.__version__)')) — skipping."
else
  echo "[setup] installing PyTorch + torchvision (CPU wheels) ..."
  pip install --upgrade torch torchvision \
      --index-url https://download.pytorch.org/whl/cpu
fi

# 2. NumPy + OpenCV (image preproc on the robot side: resize + BGR→RGB).
if python -c "import numpy" 2>/dev/null; then
  echo "[setup] numpy already installed — skipping."
else
  echo "[setup] installing numpy ..."
  pip install numpy
fi
if python -c "import cv2" 2>/dev/null; then
  echo "[setup] opencv-python already installed — skipping."
else
  echo "[setup] installing opencv-python (wrist-cam capture + resize) ..."
  pip install opencv-python
fi

# 3. LeRobot (drives the real SO-ARM101 follower over USB).
#    If you have a custom servo driver, you can skip this and replace
#    LerobotSO101Driver in bc/deploy_real.py with your own RobotDriver subclass.
if python -c "import lerobot" 2>/dev/null; then
  echo "[setup] lerobot already installed — skipping."
else
  echo "[setup] installing lerobot (real SO101 follower driver) ..."
  pip install lerobot
fi

# ============================================================ verify ===
echo
echo "[verify] importing the modules real-robot BC inference needs ..."
python - <<'PY'
import importlib, sys
mods = [
    "torch", "torchvision", "numpy", "cv2",
    "bc.config", "bc.model", "bc.normalize", "bc.deploy_real",
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
BC_RUN="${BC_RUN:-bc_eval1_v2}"
BC_CKPT="${BC_CKPT:-best.pt}"
CKPT="bc/runs/${BC_RUN}/${BC_CKPT}"
STATS="bc/runs/${BC_RUN}/stats.json"

if [[ -f "$CKPT" && -f "$STATS" ]]; then
  echo "[verify] BC artifacts present: $CKPT + $STATS"
else
  cat >&2 <<EOF
[verify] WARNING: BC artifacts missing at:
   $CKPT
   $STATS

Fetch them from the Google Drive release (see bc/readme.md step 3), e.g.:
   mkdir -p bc/runs && cd bc/runs
   pip install --quiet gdown
   gdown "https://drive.google.com/uc?id=1fnlNFXiMqMZsxhbM1DPW05M7xisu9a9Q" -O bc_eval1_v2.zip
   unzip -q bc_eval1_v2.zip && rm bc_eval1_v2.zip

Or set BC_RUN=<other-run-name> if the artifacts live under a different dir.
EOF
fi

if [[ -f "camera_intrinsics.yaml" ]]; then
  echo "[verify] camera_intrinsics.yaml present — match the real wrist cam HFOV to these intrinsics, and resize/crop to 128×72 before feeding the policy."
else
  echo "WARNING: camera_intrinsics.yaml missing at repo root — needed as a reference for real-cam HFOV / framing." >&2
fi

echo
echo "[setup] DONE."
echo "Validate the BC pipeline without hardware:"
echo "   python -m bc.deploy_real --run ${BC_RUN} --bowl-xy 0.20,-0.05 --dry-run"
echo "Then run on real hardware (adjust --port and --camera-index for your setup):"
echo "   python -m bc.deploy_real --run ${BC_RUN} --bowl-xy 0.20,-0.05 \\"
echo "       --port /dev/ttyACM0 --camera-index 0"
