# Eval 1 — Single-Object Pick-and-Place

Goal-conditioned PPO on SO-ARM101 → zero-shot real-arm deploy. Task: `Isaac-SO-ARM101-PickPlace-Bowl-v0`. Three-stage pipeline: state teacher → vision distill → vision PPO warm-started with teacher critic.

## 1. MDP

| Item | Value |
|---|---|
| Control | 50 Hz (decimation 2, sim 100 Hz) |
| Episode | 6.0 s = 300 steps |
| Action | 5 arm joints (absolute around home, `scale=0.5`) + 1 binary gripper (`open=0.5`, `close=0.0`) |
| Workspace | `x ∈ [0.10, 0.30] m`, `y ∈ [−0.15, 0.15] m` |
| Home `q` | `(0, 0, 0, 1.5708, 0, 0)`, gripper open |
| Terminations | `time_out`, `block_off_table` |
| Table | `0.6 × 1.0 × 0.02 m` at `(0.25, 0, −0.01)`; top `z=0`, back edge `x=−0.05` |

Block: 2 cm DexCube USD (scale 0.4), xy randomized. Bowl: 2-D goal from `BowlPoseCommandCfg`, **no scene prim**; rejection-sampled with `‖block − bowl‖ ≥ 0.10 m`. Same `(x, y)` frame at deploy.

## 2. Observations (asymmetric A-C)

`ObservationsCfg` defines three groups; runner cfgs select per stage (§7).

| Group | Fields | Shape |
|---|---|---|
| `policy` (deployable) | `joint_pos_rel`, `joint_vel_rel`, `gripper_state`, `bowl_xy`, `ee_proj_xy`, `ee_to_bowl_xy`, `last_action` | 1-D |
| `critic` (privileged) | `policy` + `block_position`, `block_to_bowl_xy`, `gripper_to_block`, `is_grasped` | 1-D |
| `wrist_image` | RGB + binary block mask | `(N, 4, 72, 128)` |

Wrist `TiledCamera`: parented to `gripper_link` at `pos=(-0.001, 0.1, -0.04)`, ros quat `(-0.404379, -0.912179, -0.0451242, 0.0486914)`. Intrinsics from `camera_intrinsics.yaml` → USD pinhole (FOV ≈ 102°). Renders `["rgb", "semantic_segmentation"]` (no depth).

## 3. Network — `PickPlaceVisionActorCritic`

`agents/vision_actor_critic.py`, registered into RSL-RL runner namespace at import.

```
wrist_image (4×72×128) ──[Spatial-Softmax CNN]── 128-D kpts ─┐
state (policy) ────────────────────────── concat ── MLP[256,128,64] ── μ (σ scalar Param)
critic state (policy+critic) ────── MLP[256,128,64] ── V(s)
```

CNN: `Conv(8/4) → ELU → Conv(4/2) → ELU → Conv(3/1, K)` → per-channel spatial softmax → `(x, y)` per kpt → LayerNorm → 2K = 128-D (Levine 2016).

DrQ ±4 px replicate-pad-and-crop, training-only, in `_encode_actor` (Stage 3) AND `PickPlaceVisionStudentTeacher._encode_student` (Stage 2).

When `wrist_image` is absent from `obs_groups`, both CNNs auto-disable → pure MLP A-C (Stage 1 teacher).

## 4. Reward (`mdp/rewards.py`)

| Term | Weight | Trigger |
|---|---|---|
| `reaching_object` | 1.0 | `1 − tanh(‖ee − block‖ / 0.05)` |
| `lifting_object` | 15.0 | `𝟙[block_z > 0.07]` |
| `object_goal_tracking` | 16.0 | `(1 − tanh(‖block − goal‖ / 0.30)) · 𝟙[block_z > 0.025]` |
| `object_goal_tracking_fine_grained` | 5.0 | same at `std=0.05` |
| `release_in_bowl` | 30.0 | block near bowl ∧ `z<0.06` ∧ gripper open ∧ settled, gated on **lift latch (≥0.07 m)** AND **over-bowl-above-rim latch (≥0.08 m within 6 cm of bowl xy)** |
| `action_rate`, `joint_vel` | −1e-4 → −1e-2 | ramp at 10 k env-steps |

Two per-episode latches (cleared by `reset_was_grasped` / `reset_was_over_bowl_above_rim`):

- **Lift latch 0.07 m** — blocks drag-into-bowl; `cube_center > 0.07` ⇒ `cube_bottom` clears the real bowl's 5 cm rim.
- **Over-bowl-above-rim latch 0.08 m** — forces above-rim approach; closes the sim/real rim gap (sim has no bowl prim).

No `task_success` termination — would let "hover and hold" beat "release and stay". `release_in_bowl=30` pays every post-release step until time-out. **γ=0.98 is load-bearing** (long dense tail).

## 5. Curriculum & DR (`mdp/events.py`, `CurriculumCfg`)

Both stages:
- `reset_block_position` ±7 × ±12 cm at `(0.20, 0)`; bowl rejection-sampled in the same band.
- Action-rate / joint-vel penalty ramp −1e-4 → −1e-2 at 10 k env-steps.
- `log_success` TB metric.

Stage 3 adds:
- `expand_block_xy_range`: ±3 × ±3 cm for 5 k env-steps, then linearly → ±7 × ±12 cm over 30 k.
- DrQ ±4 px (in CNN, see §3).
- `mdp.wrist_image` per-step photometric jitter: brightness ±15 %, RGB noise σ=5/255.

Deferred to deploy: cube/table color DR, lighting DR, HDRI background, motion blur / JPEG, distractors.

## 6. PPO config

| | Stage 1 teacher (`teacher_ppo_cfg.py`) | Stage 3 vision PPO (`rsl_rl_ppo_cfg.py`) |
|---|---|---|
| `num_envs` | 2048 | 1024 (32 GB GPU; image rollouts OOM at 2048) |
| `num_steps_per_env` | 24 | 16 |
| `max_iterations` | 1500 | 2000 |
| `init_noise_std` | 1.0 | 0.5 (forced; distill saves 0.1) |
| hidden dims | `[256, 128, 64]` ELU | `[256, 128, 64]` ELU + spatial-softmax CNN |
| `entropy_coef` | 0.006 | 0.006 → 0.003 (after ~500 iters) |
| epochs / mini-batches | 5 / 4 | 8 / 16 |
| `learning_rate` / `desired_kl` | 1e-4 / 0.01 | 1e-4 / 0.005 |
| `gamma / lam / clip / max_grad_norm` | 0.98 / 0.95 / 0.2 / 1.0 | same |

```python
# Stage 1: symmetric on privileged state
obs_groups = {"policy": ["policy", "critic"], "critic": ["policy", "critic"]}
# Stage 2 (distill_cfg.py): vision student, state teacher
obs_groups = {"policy": ["policy", "wrist_image"], "teacher": ["policy", "critic"]}
# Stage 3: vision actor, privileged critic (no image to critic)
obs_groups = {"policy": ["policy", "wrist_image"], "critic": ["policy", "critic"]}
```

Stage-1 critic and Stage-3 critic take identical inputs (`policy + critic`, MLP `[59 → 256 → 128 → 64 → 1]`) → teacher critic loads layer-for-layer via `load_state_dict(strict=False)`.

## 7. Three-stage pipeline

- **Stage 1 — state teacher.** Task `…-Teacher-v0`. MLP A-C on `policy + critic`. Saves `actor.*` + `critic.*`.
- **Stage 2 — short distill.** Task `…-Student-v0`. RSL-RL `DistillationRunner`, MSE, on-policy DAgger. `PickPlaceVisionStudentTeacher` regresses teacher actions. **Not to convergence** — 200–500 iters, stop when `release_from_scratch` ~30–50 %.
- **Stage 3 — vision PPO from warm-start.** Task `…-Bowl-v0`. `PickPlaceVisionActorCritic.load_state_dict` routes distill `student_cnn.* / student.*` → `actor_cnn.* / actor.*`. Trains on §4 reward to convergence.

### 7.1 Five Stage-3 interventions

| # | Fix | Location | Solves |
|---|---|---|---|
| 1 | DrQ in `_encode_student` | `vision_student_teacher.py` | Distribution shift at Stage 2→3 boundary |
| 2 | Drop loaded `std`; reinit from `init_noise_std=0.5` | `vision_actor_critic.load_state_dict` distill branch | Distill's `std=0.1` too narrow for binary gripper + arm exploration |
| 3 | `gamma=0.98` (match teacher) | `rsl_rl_ppo_cfg.py` | `release_in_bowl=30` needs long horizon |
| 4 | `expand_block_xy_range` (§5) | `CurriculumCfg` | Low-variance returns while actor stays in imitation basin |
| 5 | **`--teacher_ckpt` overlays teacher `critic.*` after distill warm-start** | `scripts/rsl_rl/train.py` | Random critic produces O(magnitude)-noisy advantages → degrades actor in ~50 iters |

#5 is the asymmetric-AC handoff (Pinto 2018): teacher critic anchors V; PPO's value regression moves V_teacher → V_student over ~100–500 iters. Not a soft constraint — student is free to exceed teacher.

### 7.2 Workflow

```bash
# Stage 1 — camera-free env cfg, no --enable_cameras needed (~2-3× faster than Teacher-v0)
train --task Isaac-SO-ARM101-PickPlace-Bowl-Teacher-Fast-v0 --headless

# Stage 2 (NOT to convergence)
train --task Isaac-SO-ARM101-PickPlace-Bowl-Student-v0 --headless --enable_cameras \
      --load_run from_teacher --checkpoint model_<best>.pt

# Stage 3
train --task Isaac-SO-ARM101-PickPlace-Bowl-v0 --resume --headless --enable_cameras \
      --num_envs 1024 \
      --load_run <distill_run> --checkpoint model_<best>.pt \
      --teacher_ckpt logs/rsl_rl/pickplace_bowl_teacher/<teacher_run>/model_<best>.pt
```

Teacher task variants:
- `…-Teacher-Fast-v0` — `SoArm101PickPlaceBowlTeacherFastEnvCfg` nulls the camera spawn AND the `wrist_image` obs group. Skips RTX rendering entirely. **No `--enable_cameras` needed.**
- `…-Teacher-v0` — shared env cfg with the vision task; camera rendered each step but output discarded. Use only for diagnostic comparison or when an existing teacher run resumes; requires `--enable_cameras`.

Stages 2 and 3 need `--enable_cameras` — they actually read the wrist image. The `from_teacher` symlink dance is in `RUNNING.md` §5.1.

## 8. Visual modality — RGB + mask

`(N, 4, 72, 128)`:

| Ch | Sim | Real |
|---|---|---|
| 0–2 | TiledCamera `rgb` → `/255` | USB cam → `cv2.undistort` → resize → `/255` |
| 3 | `semantic_segmentation` filtered to `class:block` | `cv2.inRange` on HSV (calibrated to block color) |

Block ID looked up once from `info["idToLabels"]` and cached on the camera object.

## References

- **Code:** `tasks/pickplace/{pickplace_env_cfg,joint_pos_env_cfg}.py`, `mdp/{observations,rewards,events,terminations,commands}.py`, `agents/{vision_actor_critic,vision_student_teacher,rsl_rl_ppo_cfg,teacher_ppo_cfg,distill_cfg}.py`, `camera_intrinsics.yaml`. Run/deploy: [`RUNNING.md`](./RUNNING.md), [`DEPLOY.md`](./DEPLOY.md).
- **Pinto et al.**, *Asymmetric Actor Critic*, RSS 2018. <https://arxiv.org/abs/1710.06542>
- **Levine et al.**, *End-to-End Visuomotor Policies*, JMLR 2016 — spatial softmax.
- **Kostrikov et al.**, *DrQ*, ICLR 2021. <https://arxiv.org/abs/2004.13649>
- **RSL-RL**, Schwarke et al., 2025. <https://github.com/leggedrobotics/rsl_rl>
- **LeIsaac**, <https://github.com/LightwheelAI/leisaac> — wrist camera mount.
