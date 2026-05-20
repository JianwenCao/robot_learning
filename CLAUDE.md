# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

Two cooperating packages plus shared assets:

- `isaac_so_arm101/` — Isaac Lab / Isaac Sim 5.1 RL environment + RSL-RL training scripts for SO-ARM101 (Python 3.11, managed with **`uv`**). Vendored fork; the active task is `Isaac-SO-ARM101-PickPlace-*` (see `src/isaac_so_arm101/tasks/pickplace/`).
- `deploy/` — runs on the inference / real-robot PC. Closed-loop PPO deploy for policies trained in `isaac_so_arm101/`. Uses a separate conda env (`so_arm`), CPU torch, no Isaac. The single entry point is `deploy_real.py` (state-only policy + AprilTag cube localisation, parallel naming with the gym task `Isaac-SO-ARM101-PickPlace-Bowl-StateAprilTag-v0`). Shared hardware / FK / action-decode helpers live in `deploy/driver.py` so the deploy script and `calibrate_hand_eye.py` import them without circular deps. `ppo_actor.py` holds the forward-only actor mirror `PPOActorState` (pure MLP, 27-D state = policy(25) + cube_pos_xy(2)) so this PC never imports `rsl_rl` / `isaaclab`. `cube_detector.py` provides `AprilTagDetector` with a `pose(rgb, T_base_ee) → ((x, y), valid)` API and a `set_target_id()` hook for per-sub-goal switching.
- `docs/EVAL1_PLAN.md` — authoritative spec for the **state-only + AprilTag** Eval-1 task (single-stage camera-free PPO, noisy `cube_pos_xy` filled by `pupil-apriltags` at deploy, gym ID `Isaac-SO-ARM101-PickPlace-Bowl-StateAprilTag-v0`). Also the foundational doc for the AprilTag pipeline shared by Eval-2 / Eval-3 / Bonus-B — read this first when touching anything in the deploy pipeline or the state-PPO env cfgs.
- `docs/EVAL2_PLAN.md`, `docs/EVAL3_PLAN.md`, `docs/BONUS_B_PLAN.md` — per-task plans (clutter, sequential, singulation). All inherit the AprilTag foundation from `EVAL1_PLAN.md` and only document their deltas. The previous 3-stage vision pipeline (state teacher → vision distill → vision PPO + teacher-critic warm-start, with CNN encoders / FiLM / Florence-2) has been retired; anything in the code still referencing those classes is legacy.
- `camera_intrinsics.yaml` — real wrist-cam calibration. Loaded by both the sim camera spawn (`joint_pos_env_cfg.py`) and the real-robot deploy undistort. Keep these in lockstep.
- `deploy/runs/` and `logs/` — gitignored; checkpoints/run artefacts live here.

## Common commands

### Training env (`isaac_so_arm101/`)

Setup once: `cd isaac_so_arm101 && uv sync` (also installs Isaac Lab 2.3.0 + Isaac Sim 5.1.0 from the NVIDIA pip index pinned in `pyproject.toml`). All scripts below are `uv run` from inside `isaac_so_arm101/`.

```bash
uv run list_envs                                # registered tasks
uv run zero_agent --task <task>                 # zero-action smoke
uv run random_agent --task <task>               # random-action smoke

# Legacy 3-stage vision pickplace pipeline (RETIRED — state-only + AprilTag is the
# default for all evals; see docs/EVAL1_PLAN.md). These commands still work if the
# corresponding cfgs are present in the code tree, but are kept only for
# reproducing older runs:
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

# State-only + AprilTag path (see docs/EVAL1_PLAN.md):
# Single-stage from-scratch PPO, camera-free, ~1-2 h wall-clock. No teacher,
# no distillation, no warm-start. Policy obs adds cube_pos_xy_noisy (filled
# from AprilTag on real). Experiment name `pickplace_bowl_state_apriltag`.
uv run train --task Isaac-SO-ARM101-PickPlace-Bowl-StateAprilTag-v0 --headless
uv run play  --task Isaac-SO-ARM101-PickPlace-Bowl-StateAprilTag-Play-v0 \
    --load_run <run> --checkpoint model_<best>.pt
```

Logs land in `logs/rsl_rl/<experiment_name>/<timestamp>/` (TB + `model_*.pt` + `params/*.yaml`). `experiment_name` is per-runner-cfg (`pickplace_bowl`, `pickplace_bowl_teacher`, etc.) — do **not** rename, the three-stage scripts reference these paths.

### Real-robot deploy (`deploy/`)

```bash
# One-time setup of the inference-PC conda env `so_arm`:
bash deploy/setup_inference_pc.sh

# Drop the trained state+AprilTag PPO checkpoint at deploy/runs/ (see deploy/README.md):
#   deploy/runs/state_apriltag_model.pt   (falls back to deploy/runs/model.pt)

# Closed-loop PPO on the real arm (inference PC; needs `so_arm` env, /dev/ttyACM0, USB cam).
# Requires deploy/hand_eye.yaml — run calibrate_hand_eye.py first.
conda activate so_arm
python -m deploy.calibrate_hand_eye                                               # interactive, ≤5 mm verify
python -m deploy.deploy_real --bowl-xy 0.20,-0.05 --target-color red              # Eval-1 (red cube; default)
python -m deploy.deploy_real --bowl-xy 0.20,-0.05 --target-color blue             # Eval-2 single-target
python -m deploy.deploy_real --bowl-xy 0.20,-0.05 --colors red,blue,yellow        # Eval-3 three sub-goals (shared bowl)
python -m deploy.deploy_real --bowl-xy 0.20,-0.05 --dry-run                       # no hardware; synthetic forward
```

There is currently no test suite, lint config, or CI. Smoke-test entry point: `uv run zero_agent`, `python -m deploy.deploy_real --bowl-xy 0.20,-0.05 --dry-run`.

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

`scripts/rsl_rl/train.py` carries a legacy `--teacher_ckpt` flag from the retired 3-stage vision pipeline (overlaid Stage-1 teacher's `critic.*` keys via `nn.Module.load_state_dict(filtered, strict=False)`). The state-only + AprilTag pipeline (current default for all evals — see `docs/EVAL1_PLAN.md`) is **single-stage** PPO from scratch and does not use this flag. It is left in `train.py` only so older vision-pipeline checkpoints remain reproducible.

### Sim ↔ real action contract

Action decoder lives in two places that **must** match exactly:
- Sim (`joint_pos_env_cfg.py`): `JointPositionActionCfg(joint_names=["shoulder_.*","elbow_flex","wrist_.*"], scale=0.5, use_default_offset=True)` + `BinaryJointPositionActionCfg(open=0.5, close=0.0)`.
- Real (`deploy/driver.py::_decode_action`, called from `deploy/deploy_real.py`): `arm_target_rad = JOINT_DEFAULTS_RAD[:5] + 0.5 * action[:5]`; gripper `open if action[5] > 0 else close`; finally `rad → deg` (arm) / `sim_rad → pct` (gripper) inside `LerobotSO101Driver.send_joint_targets_sim_rad` for the Feetech bus.

Home pose `JOINT_DEFAULTS_RAD = (0, 0, 0, 1.57, 0, 0.5)` and the `JOINT_NAMES` order live in `deploy/driver.py` and must agree with the URDF and LeRobot's SO101Follower obs keys.

### AprilTag cube localisation (state-only deploy)

- **Sim obs** (`mdp.observations.cube_pos_xy_noisy`): policy sees the cube's xy with deploy-shaped noise — Bernoulli pre-grasp dropout (held-last when invalid) plus a deterministic post-grasp freeze. `StateAprilTagObservationsCfg.PolicyCfg` appends this to the 25-D policy state, so the policy obs is 27-D.
- **Real obs** (`deploy/deploy_real.py`): every step the wrist cam undistorts → `pupil-apriltags` detects the configured tag ID → script composes `T_base_tag = T_base_ee · T_ee_cam · T_cam_tag` and feeds the resulting `(x, y)` into the same 27-D slot. `T_ee_cam` comes from `deploy/hand_eye.yaml` (`deploy/calibrate_hand_eye.py` writes it once per camera mount).
- **Grasp latch**: deploy mirrors the sim post-grasp freeze with a kinematic gate — once the gripper-close command fires AND the last detected cube was within `GRASP_XY_TOL` (4 cm) of the EE in the table plane, the held xy is frozen for the rest of the rollout (tag is occluded by the gripper anyway).
- **Pluggable detector**: `deploy/cube_detector.py` defines a `Detector` protocol; `AprilTagDetector` is the current implementation. Eval-2/3 reuses the same detector — only the `set_target_id()` argument changes between sub-goals. No FiLM head, no `target_color_onehot`, no Florence-2 — those vision paths were removed.

### Two PyTorch installs, two requirement chains

The repo intentionally splits envs because Isaac Lab demands CUDA-pinned wheels from `pypi.nvidia.com` and the inference PC must not need any of that:

- **Training PC** (`isaac_so_arm101/`): `uv sync` resolves `isaaclab[all,isaacsim]==2.3.0`, `torch==2.7.0` (cu128).
- **Inference PC** (`deploy/setup_inference_pc.sh`): conda env `so_arm`, **CPU** torch from `download.pytorch.org/whl/cpu`, plus `lerobot`, `feetech-servo-sdk`, `kinpy`, `opencv-python`, `pyyaml`, and `pupil-apriltags`. Pins `transformers<5` (huggingface_hub API broke).

`deploy/ppo_actor.py` is a hand-rewritten forward-only copy of the training-side actor so the deploy script never imports `rsl_rl` / `isaaclab`. `PPOActorState` mirrors the state-only `PickPlaceStateActorCritic` (pure MLP, 27-D input = policy(25) + cube_pos_xy(2)). If you change the sim-side actor architecture, mirror the change here or checkpoints will load with shape mismatches.

### Curriculum, latches, and γ

Reward (`mdp/rewards.py`) uses per-episode latches stored on the env (`reset_was_grasped`, `reset_was_over_bowl_above_rim`). `release_in_bowl=30` is a long-horizon dense tail; **`gamma=0.98` is load-bearing** for the 300-step credit chain (the +30 release reward fires deep in the episode; γ=0.95 collapses it). See `docs/EVAL1_PLAN.md` §7.

### Sim replay / debug-dump (`scripts/rsl_rl/play.py`)

Use `play.py` (not a separate sim-deploy script) to evaluate a trained PPO checkpoint in sim. `--dump-bowl-xy x,y` reaches into `env.unwrapped.command_manager._terms["bowl_pose"].command` and overwrites env 0's slot so the rollout runs against a fixed bowl, mirroring what `deploy/deploy_real.py --bowl-xy` does on hardware. `--debug-dump` writes a per-step `log.jsonl` (and, on the vision-era sim side, `step_XXXX.png`) matching what the real deploy emits, so the two folders can be diffed directly (see `deploy/README.md`).
