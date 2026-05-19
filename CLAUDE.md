# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

Two cooperating packages plus shared assets:

- `isaac_so_arm101/` — Isaac Lab / Isaac Sim 5.1 RL environment + RSL-RL training scripts for SO-ARM101 (Python 3.11, managed with **`uv`**). Vendored fork; the active task is `Isaac-SO-ARM101-PickPlace-*` (see `src/isaac_so_arm101/tasks/pickplace/`).
- `bc/` — runs on the inference / real-robot PC. Contains a goal-conditioned behavior-cloning baseline (training + sim eval) **and** the real-robot deploy stack for the PPO policy trained in `isaac_so_arm101/`. Uses a separate conda env (`so_arm`), CPU torch, no Isaac.
- `docs/EVAL1_PLAN.md` — authoritative spec for the Eval-1 task. Read first when touching the pick-and-place env or training pipeline.
- `camera_intrinsics.yaml` — real wrist-cam calibration. Loaded by both the sim camera spawn (`joint_pos_env_cfg.py`) and the real-robot deploy undistort. Keep these in lockstep.
- `demonstrations/` and `bc/runs/`, `logs/` — gitignored; large data downloaded out-of-band.

## Common commands

### Training env (`isaac_so_arm101/`)

Setup once: `cd isaac_so_arm101 && uv sync` (also installs Isaac Lab 2.3.0 + Isaac Sim 5.1.0 from the NVIDIA pip index pinned in `pyproject.toml`). All scripts below are `uv run` from inside `isaac_so_arm101/`.

```bash
uv run list_envs                                # registered tasks
uv run zero_agent --task <task>                 # zero-action smoke
uv run random_agent --task <task>               # random-action smoke

# Three-stage pickplace pipeline (see docs/EVAL1_PLAN.md §7):
# Stage 1 — state teacher (camera-free, no --enable_cameras needed):
uv run train --task Isaac-SO-ARM101-PickPlace-Bowl-Teacher-Fast-v0 --headless

# Stage 2 — vision distill (DAgger, NOT to convergence; 200–500 iters):
uv run train --task Isaac-SO-ARM101-PickPlace-Bowl-Student-v0 --headless --enable_cameras \
    --load_run from_teacher --checkpoint model_<best>.pt

# Stage 3 — vision PPO warm-start (must pass --teacher_ckpt — see "Critic warm-start" below):
uv run train --task Isaac-SO-ARM101-PickPlace-Bowl-v0 --resume --headless --enable_cameras \
    --num_envs 1024 \
    --load_run <distill_run> --checkpoint model_<best>.pt \
    --teacher_ckpt logs/rsl_rl/pickplace_bowl_teacher/<teacher_run>/model_<best>.pt

# Eval / replay a checkpoint:
uv run play --task Isaac-SO-ARM101-PickPlace-Bowl-Play-v0 \
    --load_run <run> --checkpoint model_<best>.pt --enable_cameras
```

Logs land in `logs/rsl_rl/<experiment_name>/<timestamp>/` (TB + `model_*.pt` + `params/*.yaml`). `experiment_name` is per-runner-cfg (`pickplace_bowl`, `pickplace_bowl_teacher`, etc.) — do **not** rename, the three-stage scripts reference these paths.

### BC baseline + real-robot deploy (`bc/`)

```bash
# One-time setup of the inference-PC conda env `so_arm`:
bash bc/setup_inference_pc.sh

# Train the BC policy (full epoch loop, or --overfit N for the one-batch sanity test):
conda activate so_arm
python -m bc.train --epochs 50
python -m bc.train --overfit 200            # gate: loss<0.01 in <200 steps

# Closed-loop BC in Isaac sim (training PC; needs isaac_so_arm101 env):
bash bc/run_eval1_rollout.sh 0.20 -0.05     # bowl xy in robot base frame
# wraps:  python -m bc.deploy_sim --bowl-xy 0.20,-0.05 ...

# Closed-loop PPO on the real arm (inference PC; needs `so_arm` env, /dev/ttyACM0, USB cam):
python -m bc.deploy_real --bowl-xy 0.20,-0.05
python -m bc.deploy_real --bowl-xy 0.20,-0.05 --dry-run   # no hardware; one synthetic forward
```

There is currently no test suite, lint config, or CI. Smoke-test entry points: `python -m bc.model` (`_smoke_test`), `python -m bc.train --overfit 200`, `uv run zero_agent`, `python -m bc.deploy_real --dry-run`.

## Architecture you can't see from a single file

### Single env cfg, multiple gym IDs, per-stage runner cfgs

`tasks/pickplace/pickplace_env_cfg.py` defines one `ObservationsCfg` with **three** obs groups: `policy` (deployable state), `critic` (privileged), `wrist_image` (RGB+mask, `(N,4,72,128)`). The asymmetric A-C is realised purely by each runner cfg's `obs_groups` dict — the env never changes:

```python
# Stage 1 teacher        (teacher_ppo_cfg.py):  policy/critic = ["policy","critic"]
# Stage 2 distill        (distill_cfg.py):       student=policy+wrist_image, teacher=policy+critic
# Stage 3 vision PPO     (rsl_rl_ppo_cfg.py):    actor=policy+wrist_image, critic=policy+critic
# Pretrained backbone PPO (pretrained_ppo_cfg.py, §9 alt path)
```

`tasks/pickplace/__init__.py` registers each stage as its own gym ID (`…-Teacher-Fast-v0`, `…-Student-v0`, `…-Bowl-v0`, `…-Pretrained-v0`, plus `-Play-v0` variants with 50 envs and corruption disabled). `joint_pos_env_cfg.SoArm101PickPlaceBowlTeacherFastEnvCfg` is a special subclass that **nulls** `scene.wrist_cam` and the `wrist_image` obs group so Stage 1 skips RTX rendering entirely — that's why Stage 1 uses `…-Teacher-Fast-v0`, not `…-Teacher-v0`.

### Custom policy classes are class-injected into RSL-RL at import time

RSL-RL resolves `policy.class_name` with `eval()` against the runner module's globals. To make `PickPlaceVisionActorCritic` / `PickPlaceVisionStudentTeacher` / `PickPlaceResNetActorCritic` discoverable, each `*_cfg.py` calls `setattr(rsl_rl.runners.{on_policy_runner,distillation_runner}, Cls.__name__, Cls)` at module load. Importing the cfg (which happens via `tasks.pickplace.__init__`'s gym registration) is what enables `OnPolicyRunner`/`DistillationRunner` to find the class. If you add a new actor-critic, replicate this pattern — class registration via decorator/registry is not used.

### Critic warm-start (Stage-3 must-have)

`scripts/rsl_rl/train.py` has a non-standard `--teacher_ckpt` flag that, **after** `runner.load(--load_run)` warm-starts the distilled actor, overlays the Stage-1 teacher's `critic.*` keys onto the policy via `nn.Module.load_state_dict(filtered, strict=False)`. This bypasses RSL-RL's custom `load_state_dict` (which returns a `bool` distill/resume signal, not a tuple). Skipping this flag will silently destroy Stage 3: the random critic produces O(magnitude) noisy advantages and the warm-started actor degrades within ~50 iters. See `docs/EVAL1_PLAN.md` §7.2 intervention #5.

### Sim ↔ real action contract

Action decoder lives in two places that **must** match exactly:
- Sim (`joint_pos_env_cfg.py`): `JointPositionActionCfg(joint_names=["shoulder_.*","elbow_flex","wrist_.*"], scale=0.5, use_default_offset=True)` + `BinaryJointPositionActionCfg(open=0.5, close=0.0)`.
- Real (`bc/deploy_real.py`): `arm_target_rad = JOINT_DEFAULTS_RAD[:5] + 0.5 * action[:5]`; gripper `open if action[5] > 0 else close`; finally `rad → deg` for the Feetech bus.

Home pose `JOINT_DEFAULTS_RAD = (0, 0, 0, 1.57, 0, 0)` is duplicated by hand on the real side. Same for `JOINT_NAMES` order — it must agree with the URDF and LeRobot's SO101Follower obs keys.

### Wrist image is 4-channel (RGB + mask), shape `(N, 4, 72, 128)`

- Sim: `TiledCamera` renders `["rgb","semantic_segmentation"]` filtered to `class:block`. `mdp.wrist_image` concatenates RGB/255 with the binary semantic-seg mask and applies per-step photometric DR (only when `corrupt=True`; the `_PLAY` cfgs short-circuit this).
- Real: `bc/deploy_real._build_image` undistorts via `camera_intrinsics.yaml`, resizes to 128×72, and replaces the seg-mask with an HSV `cv2.inRange` against the wood-tone block. Tune `--hsv-low/--hsv-high` per scene.
- The BC baseline (`bc/model.py`) only consumes RGB — its sim renders are stored in `demonstrations/<pilot>/sim_renders.npy` and image augmentation lives in the model module so dataloader workers can stay simple.

### Two PyTorch installs, two requirement chains

The repo intentionally splits envs because Isaac Lab demands CUDA-pinned wheels from `pypi.nvidia.com` and the inference PC must not need any of that:

- **Training PC** (`isaac_so_arm101/`): `uv sync` resolves `isaaclab[all,isaacsim]==2.3.0`, `torch==2.7.0` (cu128).
- **Inference PC** (`bc/setup_inference_pc.sh`): conda env `so_arm`, **CPU** torch from `download.pytorch.org/whl/cpu`, plus `lerobot`, `feetech-servo-sdk`, `kinpy`, `opencv-python`, `pyyaml`. Pins `transformers<5` (huggingface_hub API broke).

`bc/ppo_actor.py` is a hand-rewritten forward-only copy of `PickPlaceVisionActorCritic` so `bc/deploy_real.py` never imports `rsl_rl` / `isaaclab`. If you change the sim-side actor architecture, mirror the change here or checkpoints will load with shape mismatches.

### Curriculum, latches, and γ

Reward (`mdp/rewards.py`) uses per-episode latches stored on the env (`reset_was_grasped`, `reset_was_over_bowl_above_rim`). `release_in_bowl=30` is a long-horizon dense tail; **`gamma=0.98` is load-bearing** across Stage 1 and Stage 3 (mismatched γ between the teacher critic and Stage-3 PPO breaks the Bellman fixed point that the warm-start relies on — see EVAL1_PLAN §7.2 #3).

### `bc/deploy_sim.py` quirks

Bowl xy is normally sampled per reset; `_force_bowl_xy` reaches into `env.unwrapped.command_manager._terms["bowl_pose"].command` and overwrites env 0's slot to evaluate at a fixed `(x,y)`. The BC policy outputs **degree-absolute joint targets in 8-step chunks** (`CHUNK_K=8`, execute first `EXECUTE_K=4`); `deploy_sim` re-converts these to the env's rad-delta-around-home action space. Demos are 30 Hz, sim is 50 Hz — pass `--control-stride 2` (default) to hold each BC target for two sim ticks ≈ 25 Hz.
