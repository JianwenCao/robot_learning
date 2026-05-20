# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

Two cooperating packages plus shared assets:

- `isaac_so_arm101/` — Isaac Lab / Isaac Sim 5.1 RL environment + RSL-RL training scripts for SO-ARM101 (Python 3.11, managed with **`uv`**). Vendored fork; the active task is `Isaac-SO-ARM101-PickPlace-*` (see `src/isaac_so_arm101/tasks/pickplace/`).
- `deploy/` — runs on the inference / real-robot PC. Closed-loop PPO deploy for policies trained in `isaac_so_arm101/`. Uses a separate conda env (`so_arm`), CPU torch, no Isaac. `ppo_actor.py` holds two forward-only actor mirrors so this PC never imports `rsl_rl` / `isaaclab`: `PPOActor` (Eval-1, small-CNN, 25-D state) and `PPOActorClutter` (Eval-2/3, frozen ResNet-18 layer1-2 + FiLM + spatial-softmax, 31-D state = policy + target_color_onehot). Entry points: `deploy_real.py` (Eval-1) and `deploy_real_clutter.py` (Eval-2 single-target / Eval-3 multi-sub-goal). `cube_detector.py` is the pluggable mask-source layer (HSV or Florence-2); Florence supports `set_prompt()` so the Eval-3 outer loop can re-key per sub-goal without reloading the 1 GB model.
- `docs/EVAL1_PLAN.md` — authoritative spec for the Eval-1 task. Read first when touching the pick-and-place env or training pipeline.
- `camera_intrinsics.yaml` — real wrist-cam calibration. Loaded by both the sim camera spawn (`joint_pos_env_cfg.py`) and the real-robot deploy undistort. Keep these in lockstep.
- `deploy/runs/` and `logs/` — gitignored; checkpoints/run artefacts live here.

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

### Real-robot deploy (`deploy/`)

```bash
# One-time setup of the inference-PC conda env `so_arm`:
bash deploy/setup_inference_pc.sh

# Drop trained vision-PPO checkpoints at deploy/runs/ (see deploy/README.md):
#   Eval-1: deploy/runs/model.pt
#   Eval-2: deploy/runs/clutter_model.pt
#   Eval-3: deploy/runs/eval3_model.pt

# Closed-loop PPO on the real arm (inference PC; needs `so_arm` env, /dev/ttyACM0, USB cam):
conda activate so_arm
python -m deploy.deploy_real --bowl-xy 0.20,-0.05                                # Eval-1
python -m deploy.deploy_real_clutter --mode eval2 --target-color red --bowl-xy 0.22,0.0
python -m deploy.deploy_real_clutter --mode eval3 --colors red,blue,yellow --bowl-xy 0.22,0.0
python -m deploy.deploy_real --bowl-xy 0.20,-0.05 --dry-run                       # no hardware; synthetic forward
python -m deploy.deploy_real_clutter --mode eval2 --target-color red --bowl-xy 0.22,0.0 --dry-run
```

There is currently no test suite, lint config, or CI. Smoke-test entry points: `uv run zero_agent`, `python -m deploy.deploy_real --dry-run`, `python -m deploy.deploy_real_clutter --mode eval2 --target-color red --bowl-xy 0.22,0.0 --dry-run`.

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
- Real (`deploy/deploy_real.py`): `arm_target_rad = JOINT_DEFAULTS_RAD[:5] + 0.5 * action[:5]`; gripper `open if action[5] > 0 else close`; finally `rad → deg` for the Feetech bus.

Home pose `JOINT_DEFAULTS_RAD = (0, 0, 0, 1.57, 0, 0)` is duplicated by hand on the real side. Same for `JOINT_NAMES` order — it must agree with the URDF and LeRobot's SO101Follower obs keys.

### Wrist image is 4-channel (RGB + mask), shape `(N, 4, 72, 128)`

- **Eval-1 sim**: `TiledCamera` renders `["rgb","semantic_segmentation"]` with `class:block` on the single dex_cube. `mdp.wrist_image` concatenates RGB/255 with the binary semantic-seg mask and applies per-step photometric DR.
- **Eval-2/3 sim**: same `TiledCamera` data types, but each of the 6 cubes carries a unique `class:cube_<color>` tag. `mdp.wrist_rgb_mask_dr` filters the seg image to the per-env target colour (via `env._target_cube_idx`) and corrupts the mask channel to mimic Florence-2's noise profile at deploy: small-area dropout (cube too far), morphological erode/dilate ±2 px, Bernoulli full-frame dropout (p=0.10), wrong-colour swap (p=0.03, replaces with a distractor's instance mask). Play cfgs override `corrupt=False`.
- **Eval-1 real**: `deploy/deploy_real._build_image` undistorts → resizes → mask channel from `--mask-source` (default `florence` with the fixed prompt `"small wooden cube"`; `hsv` keeps the saturation-gate fallback).
- **Eval-2/3 real**: `deploy/deploy_real_clutter` uses the same `_build_image` helper, but the Florence prompt is set per sub-goal via `Detector.set_prompt(f"{target_color} cube")` — no model reload. Pass the target via `--target-color` (Eval-2) or `--colors red,blue,yellow` (Eval-3). HSV fallback is colour-agnostic and only useful as an A/B for Eval-2 with one cube on the table.
- The `Detector` protocol in `deploy/cube_detector.py` is the swap-in slot for other segmenters (GroundedSAM / SAM-3 / CLIPSeg) — implement `mask(rgb)` + `set_prompt(prompt)` and register in `build_detector`.

### Two PyTorch installs, two requirement chains

The repo intentionally splits envs because Isaac Lab demands CUDA-pinned wheels from `pypi.nvidia.com` and the inference PC must not need any of that:

- **Training PC** (`isaac_so_arm101/`): `uv sync` resolves `isaaclab[all,isaacsim]==2.3.0`, `torch==2.7.0` (cu128).
- **Inference PC** (`deploy/setup_inference_pc.sh`): conda env `so_arm`, **CPU** torch from `download.pytorch.org/whl/cpu`, plus `lerobot`, `feetech-servo-sdk`, `kinpy`, `opencv-python`, `pyyaml`, and Florence-2 segmenter deps (`timm`, `einops`, `accelerate`, `Pillow` — only used when `--mask-source florence`). Pins `transformers<5` (huggingface_hub API broke).

`deploy/ppo_actor.py` is a hand-rewritten forward-only copy of the training-side actors so the deploy scripts never import `rsl_rl` / `isaaclab`. `PPOActor` mirrors `PickPlaceVisionActorCritic` (Eval-1); `PPOActorClutter` mirrors `ClutterPickPlaceVisionActorCritic` (Eval-2/3: frozen ResNet-18 layer1-2, 1×1 conv head with FiLM modulated by the target_color one-hot, spatial-softmax, MLP[256,128,64]; state is 31-D = policy(25) + target_color_onehot(6), and the trailing 6 dims are sliced inside `forward()` to feed the FiLM head). If you change the sim-side actor architecture, mirror the change here or checkpoints will load with shape mismatches.

### Curriculum, latches, and γ

Reward (`mdp/rewards.py`) uses per-episode latches stored on the env (`reset_was_grasped`, `reset_was_over_bowl_above_rim`). `release_in_bowl=30` is a long-horizon dense tail; **`gamma=0.98` is load-bearing** across Stage 1 and Stage 3 (mismatched γ between the teacher critic and Stage-3 PPO breaks the Bellman fixed point that the warm-start relies on — see EVAL1_PLAN §7.2 #3).

### Sim replay / debug-dump (`scripts/rsl_rl/play.py`)

Use `play.py` (not a separate sim-deploy script) to evaluate a trained PPO checkpoint in sim. `--dump-bowl-xy x,y` reaches into `env.unwrapped.command_manager._terms["bowl_pose"].command` and overwrites env 0's slot so the rollout runs against a fixed bowl, mirroring what `deploy/deploy_real.py --bowl-xy` does on hardware. `--debug-dump` writes the same per-step `step_XXXX.png` + `log.jsonl` schema the real deploy emits, so the two folders can be diffed directly (see `deploy/README.md` §6).
