#!/usr/bin/env bash
# Demo-day one-rollout launcher for the Eval-1 BC policy in Isaac sim.
#
# Activates the `so_arm` conda env, accepts the Omniverse EULA, and runs
# `python -m bc.deploy_sim` with the bowl (x, y) you specify. Any extra
# args after `x y` are forwarded to deploy_sim (e.g. --headless).
#
# Usage:
#   bash bc/run_eval1_rollout.sh <x> <y> [extra deploy_sim args...]
#
# Examples:
#   bash bc/run_eval1_rollout.sh 0.20 -0.05
#   bash bc/run_eval1_rollout.sh 0.20 -0.05 --headless
#   bash bc/run_eval1_rollout.sh 0.20 -0.05 --init-gripper-closed
#   ROLLOUTS=10 bash bc/run_eval1_rollout.sh 0.20 -0.05
#   BC_RUN=bc_eval1_v1 bash bc/run_eval1_rollout.sh 0.20 -0.05
#
# (x, y) are metres in the robot base frame.

set -euo pipefail

usage() {
  echo "Usage: bash bc/run_eval1_rollout.sh <bowl_x> <bowl_y> [extra deploy_sim args...]" >&2
  echo "  bowl_x, bowl_y : floats in metres, robot base frame" >&2
  echo "  example        : bash bc/run_eval1_rollout.sh 0.20 -0.05" >&2
  exit 2
}

if [[ $# -lt 2 ]]; then
  echo "ERROR: bowl x and y are required." >&2
  usage
fi

BOWL_X="$1"; BOWL_Y="$2"; shift 2
# Validate that both args parse as numbers (accepts integers, decimals, sign, sci-notation).
_num_re='^-?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$'
if ! [[ "$BOWL_X" =~ $_num_re && "$BOWL_Y" =~ $_num_re ]]; then
  echo "ERROR: bowl x='$BOWL_X' y='$BOWL_Y' — both must be numeric." >&2
  usage
fi

# --- Resolve repo root (this script lives in <repo>/bc/) ---------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- Activate conda env ------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-so_arm}"
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: 'conda' not on PATH." >&2
  echo "       First-time on this PC? Run: bash bc/setup_inference_pc.sh" >&2
  echo "       (its header has the miniconda install snippet)." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "ERROR: conda env '$CONDA_ENV' not found." >&2
  echo "       First-time on this PC? Run: bash bc/setup_inference_pc.sh" >&2
  exit 1
fi
conda activate "$CONDA_ENV"

# --- Isaac Sim 5.1 non-interactive EULA --------------------------------------
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y

# --- Defaults (override via env vars) ----------------------------------------
TASK="${TASK:-Isaac-SO-ARM101-PickPlace-Bowl-Play-v0}"
BC_RUN="${BC_RUN:-bc_eval1_v2}"
BC_CKPT="${BC_CKPT:-best.pt}"
ROLLOUTS="${ROLLOUTS:-1}"

CKPT_PATH="bc/runs/${BC_RUN}/${BC_CKPT}"
STATS_PATH="bc/runs/${BC_RUN}/stats.json"
if [[ ! -f "$CKPT_PATH" ]]; then
  echo "ERROR: BC checkpoint not found: $CKPT_PATH" >&2
  echo "       Set BC_RUN=<dir under bc/runs/> or BC_CKPT=<file> to override." >&2
  exit 1
fi
if [[ ! -f "$STATS_PATH" ]]; then
  echo "ERROR: normalizer stats not found: $STATS_PATH" >&2
  echo "       The run dir must contain both '<ckpt>.pt' and 'stats.json'." >&2
  exit 1
fi

echo "[demo] repo root : $REPO_ROOT"
echo "[demo] conda env : $CONDA_ENV  ($(python -V 2>&1))"
echo "[demo] task      : $TASK"
echo "[demo] BC run    : $CKPT_PATH"
echo "[demo] bowl xy   : ($BOWL_X, $BOWL_Y) m  (robot base frame)"
echo "[demo] rollouts  : $ROLLOUTS"
echo "[demo] extra args: $*"
echo

# --- Launch ------------------------------------------------------------------
exec python -m bc.deploy_sim \
    --task "$TASK" \
    --run "$BC_RUN" \
    --ckpt "$BC_CKPT" \
    --rollouts "$ROLLOUTS" \
    --bowl-xy "${BOWL_X},${BOWL_Y}" \
    --enable_cameras \
    "$@"
