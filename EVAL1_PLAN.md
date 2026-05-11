# Eval 1 — Single-Object Pick-and-Place: Method

> Goal-conditioned PPO on SO-ARM101 in Isaac Lab → zero-shot deploy on the real arm. No teleop data; the only training signal is sim rollouts. Task gym ID: `Isaac-SO-ARM101-PickPlace-Bowl-v0`. For env setup, training, evaluation, and deploy commands, see [`RUNNING.md`](./RUNNING.md) and [`DEPLOY.md`](./DEPLOY.md).
>
> **Recipe lineage.** Asymmetric actor–critic (Pinto et al. 2018, privileged critic + image actor) + Levine spatial-softmax CNN + DrQ random-shift augmentation, trained through a three-stage state-teacher → vision-distillation → vision-PPO pipeline (§7) where the teacher's *critic* carries across the BC→RL handoff to avoid the random-init advantage shock that destroys naïve warm-starts. Hyperparameter and DR envelopes anchored on peer projects on the same robot family: **ManiSkill3 PickCube → SO-100** (StoneT2000, ≈ 91.6 % zero-shot real success, 25–40 M env-steps; cold-start single-stage PPO — γ=0.9 / 8 ep × 32 mb) and **CS6341 SO-101 vision** (Evans & Hegde 2025, partial vision transfer, surfaced the visual-DR + cold-start gap). Hyperparameter divergences from ManiSkill3 are flagged in §6; the BC→RL fix-up is §7.2.

## 1. MDP

A single PPO actor–critic over thousands of randomized envs. The block (2 cm cube) is randomized in xy; the bowl is **not a scene prim** — it is a 2-D goal `(x, y)` sampled per episode by `BowlPoseCommandCfg` in the robot base frame, with rejection sampling that keeps `‖block − bowl‖ ≥ 0.10 m`. Same frame at deploy (`--bowl_xy x y`), so no coordinate transform.

| Item | Value |
|---|---|
| Control | 50 Hz (decimation 2, sim 100 Hz) |
| Episode | 6.0 s = 300 steps |
| Action | 5 absolute-around-home arm joint targets (`scale=0.5`) + 1 binary gripper (`open=0.5`, `close=0.0`) |
| Workspace | `x ∈ [0.10, 0.30] m`, `y ∈ [−0.15, 0.15] m`, both block & bowl |
| Home `q` | `(0, 0, 0, 1.5708, 0, 0)` with gripper *open* (`gripper=0.5`) |
| Terminations | `time_out`, `block_off_table`. **No** success-termination (it incentivized "hold and hover" over release). |

## 2. Observations (asymmetric A-C)

Three obs groups in `tasks/pickplace/pickplace_env_cfg.py::ObservationsCfg`. Runner cfg routes:

```
actor   = policy + wrist_image      (deployable)
critic  = policy + critic            (privileged, sim-only; no image)
```

The critic deliberately does **not** receive the image — it has ground-truth block pose, so an image encoder is redundant compute. Only the actor has a CNN.

| Group | Fields | Shape |
|---|---|---|
| `policy` (deployable) | `joint_pos_rel`, `joint_vel_rel`, `gripper_state`, `bowl_xy`, `ee_proj_xy`, `ee_to_bowl_xy`, `last_action` | 1-D, concatenated |
| `critic` (privileged) | policy fields + `block_position`, `block_to_bowl_xy`, `gripper_to_block`, `is_grasped` | 1-D, concatenated |
| `wrist_image` | RGB + depth + binary block mask, all in `[0,1]` | `(N, 5, 72, 128)` (see §8) |

`ee_proj_xy` is taken from the `ee_frame` `FrameTransformer` at `gripper_link + offset=(0.01, 0, -0.09)`. The wrist `TiledCamera` is parented to `gripper_link` at `pos=(-0.001, 0.1, -0.04)`, ros-convention quaternion `(-0.404379, -0.912179, -0.0451242, 0.0486914)` (LeIsaac's verbatim mount). Intrinsics from `camera_intrinsics.yaml` via `mdp.load_wrist_cam_intrinsics()` → USD pinhole (`focal_length = fx · horizontal_aperture / W`); horizontal FOV ≈ 102°, identical to the real cam.

## 3. Network — `PickPlaceVisionActorCritic`

Subclass of `rsl_rl.modules.ActorCritic` in `tasks/pickplace/agents/vision_actor_critic.py`. Registered into RSL-RL's runner namespace at import time so `class_name="PickPlaceVisionActorCritic"` resolves.

```
wrist_image (5×72×128) ──[Spatial-Softmax CNN]── 128-D keypoints ─┐
state (policy group) ─────────────────────────────────────────── concat ── MLP[256,128,64] ── μ
                                                                 │           σ (scalar Param)
critic state (policy + critic groups) ───── MLP[256,128,64] ── V(s)
```

CNN: `Conv(8/4)→ELU → Conv(4/2)→ELU → Conv(3/1, K channels)` → spatial softmax per channel → expected `(x, y)` per keypoint → LayerNorm. Output is `2K = 128` dims (Levine et al. 2016). The spatial softmax is the inductive bias for "small object on a flat workspace": each channel becomes a soft-argmax keypoint, which dense MLP projection failed to discover from grasp gradient alone.

**DrQ random-shift** (Kostrikov et al., ICLR 2021): 4-pixel pad-and-crop on the actor's image input during training only.

No recurrence — `is_recurrent=False`. The reach-stage curriculum (§5) keeps the cube under the home FOV from iter 0, and the wide horizontal FOV makes search unnecessary.

## 4. Reward (in `mdp/rewards.py`)

Stock Franka Lift recipe + one latched release term, identical across Stage 1 (teacher) and Stage 3 (warm-started vision PPO). Prior commits accumulated an 8-term stack (`pre_grasp_pose`, `place_in_bowl`, `block_dropped`, …) that fought itself in training — runs 11–18 showed each addition adding variance that pushed σ to inflate and PPO to oscillate. Commit `010d28c` reverted to stock; the only deliberate divergence is `release_in_bowl`, added as a fine-tune term once the stock 4-term teacher converged.

| Term | Weight | Trigger |
|---|---|---|
| `reaching_object` | 1.0 | `1 − tanh(d_ee_block / 0.05)` — dense, ungated |
| `lifting_object` | 15.0 | indicator `block_z > 0.025` (lift = grasp proxy) |
| `object_goal_tracking` | 16.0 | `(1 − tanh(d_block_goal / 0.30)) · 𝟙[block_z > 0.025]` — dense, **per-step lift gate** |
| `object_goal_tracking_fine_grained` | 5.0 | same fn at `std=0.05` |
| `release_in_bowl` | 30.0 | block xy near bowl ∧ block_z < bowl height ∧ gripper open ∧ block settled, gated on **per-episode lift latch** |
| `action_rate`, `joint_vel` | −1e-4 → −1e-2 | curriculum ramp at 10 k env-steps |

Two design choices to call out:

1. **Per-episode lift latch** (`_episode_lifted_mask`, cleared each reset by `reset_was_grasped`): `release_in_bowl` only fires once the policy has lifted the cube ≥ 2.5 cm at some prior step in the same episode. Closes the "drag the cube laterally into the bowl" exploit a dense block-to-bowl term otherwise rewards. `object_goal_tracking` uses a per-step lift gate instead (stock Franka semantics) — transport pays only while the cube is airborne, so dropping the cube costs reward immediately.

2. **No success-termination.** Removing `task_success` from `TerminationsCfg` was load-bearing: with success-termination the episode ends on first release and "hold and hover until timeout" beats "release and stay" by ~5× cumulative reward. With it removed, `release_in_bowl=30` pays every post-release step until time-out and the policy learns to actually let go.

**Reward weights were tuned for γ=0.98** (see §6). The `release_in_bowl` latch is a long-horizon term — once it fires, it pays for the remainder of the 250-step episode. A short-horizon discount (γ=0.9 → effective horizon ≈ 10 steps) chops 80 % of that tail's value and changes the optimal policy.

## 5. Curriculum & DR (in `mdp/events.py`, `CurriculumCfg`)

**Stage 1 (state teacher) — active:**
- **Cube placement randomization** (`reset_block_position`): block xy ∈ ±7 cm × ±12 cm around `(0.20, 0, 0.01)`, intentionally tightened from the original ±10×±15 cm to the comfortable reach band so all (block, bowl) pairs are solvable. The bowl is rejection-sampled within the same band with `min_distance ≥ 10 cm` from the block (`mdp/commands.py::BowlPoseCommand`). This is the *only* placement DR needed — the policy must learn the full workspace, but every sampled pose is physically reachable.
- **Action / joint-vel penalty ramp**: −1e-4 → −1e-2 at 10 k env-steps.
- **`log_success` curriculum term**: emits binary `success_rate` (fraction of ended episodes where `release_in_bowl` fired) to TB.
- **No image-side DR.** The teacher reads ground-truth state; the camera renders each step but the obs term is not in `obs_groups`. Visual DR is irrelevant to it.

**Stage 3 (warm-started vision PPO) — adds:**
- **Block-xy expand curriculum** (`expand_block_xy_range`, currently defined in `events.py` but un-wired in `CurriculumCfg`). Wire it back with a **warm-start-adapted schedule** — *not* the cold-start one the function originally documented. Initial ±3×±3 cm for 5 k env-steps (~150 PPO iters at 2048 envs × 16 steps), expanded linearly to full width (±7×±12 cm) over the next 30 k env-steps (~900 iters). The point is not "make reach easy" (the warm-started student already reaches); it is to give the freshly-initialized critic a low-variance return distribution to fit against while the actor sits inside the imitation basin. Once `V` has caught up, expand to full width. See §7.2 intervention #4.
- **DrQ random-shift on actor image** (`_random_shift_pad`, ±4 px replicate-pad-and-crop, training-only). Already wired in `PickPlaceVisionActorCritic._encode_actor`. **Must also be applied in Stage 2 distillation** (§7.2 intervention #1) so the CNN sees the same image distribution in both stages.
- **Per-step photometric jitter** in `mdp.wrist_image` (already active): brightness ±15 %, RGB Gaussian noise σ = 5/255, depth scale jitter ±5 %, depth Gaussian noise σ = 0.01. The depth-channel jitter deliberately mimics DA3's frame-to-frame artifacts so the policy sees DA3-flavored depth in sim too (§8).

**Deliberately not applied:**
- **Wrist-camera pose DR** (`randomize_camera_uniform`, ±25 mm / ±2.5°). Defined in `events.py` but intentionally not wired. WOWROBO bracket geometry is reproducible across rebuilds, and the ±4 px DrQ shift covers small calibration drift cheaply at the image level. Re-enable only if real-world deploy surfaces a calibration-sensitive gap.
- **Bootstrap-grasp** (`init_block_in_gripper`, `decay_p_grasped`). Wired in earlier runs and removed in the revert. Short decay windows caused the policy to ride the subsidy and collapse when `p` hit 0 (memory: `bootstrap-curriculum-pitfall`). Stays parked unless the warm-start path is insufficient and we need to seed grasp trajectories for an emergency cold-start fallback.
- **Per-episode RGB tint** (`randomize_wrist_image_tint`). Defined but un-wired; the per-step photometric jitter above already covers within-episode color variance.

**Deferred to real-deploy readiness (NOT on the Stage 1–3 critical path):**
- Cube color / material DR (Isaac Replicator path, requires `replicate_physics=False`).
- Table color jitter (`#B8ADA9 ± 15 RGB`).
- Lighting DR (intensity `[500, 1500]`, color temperature `[2500, 9500] K`, 1–3 light sources, optional HDRI dome — CS6341 + NVIDIA SO-101 envelopes).
- HDRI / greenscreen background overlay — ManiSkill3 PickCube's bridge for the "sim has no clutter, real has whatever's on the desk" gap.
- Heavier image corruption (motion blur 0–3 px, JPEG q `[70, 100]`).
- Distractor objects outside the workspace.

These gate real-world transfer, not sim-side success rate. They go in only after Stage 3 has demonstrated convergence under the §7 fix-up.

## 6. PPO config (`tasks/pickplace/agents/{teacher_ppo_cfg,rsl_rl_ppo_cfg}.py`)

Three columns: Stage 1 teacher, Stage 3 vision PPO (warm-started, post-fix-up), and ManiSkill3 PickCube as a sanity check against a recipe with validated zero-shot real success on the same robot family. **Stage 3's `gamma` is 0.98, not 0.9.** ManiSkill3 uses γ=0.9 with a near-sparse reward; our `release_in_bowl=30` latch is dense-per-step over the entire post-release tail, so the long horizon is load-bearing. The teacher trained with γ=0.98 — Stage 3 must match or the warm-started actor is being optimized against a different objective than the one it was distilled from.

| Hyperparam | Stage 1 (state teacher) | Stage 3 (vision PPO, warm-started) | ManiSkill3 PickCube |
|---|---|---|---|
| `num_envs` | 2048 | 2048 (1024 if VRAM-bound) | 1024–2048 |
| `num_steps_per_env` | 24 | 16 | 16 |
| `max_iterations` | 1500 | 2000 (≈ 65 M env-steps) | ≈ 25–40 M sufficed |
| `init_noise_std` | 1.0 (stock) | **0.5 — forced on warm-start** (overrides the `std=0.1` saved by the distill checkpoint; see §7.2 intervention #2) | — |
| `actor_hidden_dims` / `critic_hidden_dims` | `[256, 128, 64]`, ELU | `[256, 128, 64]`, ELU + spatial-softmax CNN on actor | Nature-CNN + small MLP |
| `clip_param` | 0.2 | 0.2 | 0.2 |
| `entropy_coef` | 0.006 | 0.006 (first ~500 iters) → 0.003 | (default) |
| `num_learning_epochs / num_mini_batches` | 5 / 4 | 8 / 16 | 8 / 32 |
| `learning_rate / schedule / desired_kl` | 1e-4 / adaptive / 0.01 | 1e-4 / adaptive / 0.005 | 3e-4 (typical) |
| `gamma / lam / max_grad_norm` | **0.98** / 0.95 / 1.0 | **0.98** / 0.95 / 1.0 | 0.9 / 0.95 / 1.0 |

Asymmetric obs wiring across the three stages:

```python
# Stage 1 (teacher) — symmetric on privileged state
obs_groups = {"policy": ["policy", "critic"], "critic": ["policy", "critic"]}

# Stage 2 (distill) — vision student, state teacher
obs_groups = {"policy": ["policy", "wrist_image"], "teacher": ["policy", "critic"]}

# Stage 3 (vision PPO) — vision actor, privileged critic, NO image to critic
obs_groups = {"policy": ["policy", "wrist_image"], "critic": ["policy", "critic"]}
```

The critic in Stage 3 deliberately does not receive the image — it has ground-truth block pose via the privileged `critic` group, so a CNN is redundant compute. **More importantly: the teacher's critic and Stage 3's critic take identical inputs** (`policy + critic`, no image, MLP `[59 → 256 → 128 → 64 → 1]`). This is what makes the Pattern-A teacher-critic warm-start in §7.2 intervention #5 work as a layer-for-layer load.

## 7. Teacher warm-start for vision PPO

Cold-start vision PPO routinely stalls before the CNN discovers grasp-relevant keypoints — peer projects on this robot family hit the same wall. **Warm-start the vision actor from a state-based teacher**: teacher is the *bootstrap to escape the cold-start regime*, not the performance ceiling. Pure DAgger-to-convergence (the DextrAH-G recipe) is rejected here — it caps the student at the teacher's privileged-state action distribution, burns compute on regression PPO would learn anyway, and that distribution isn't Bayes-optimal under the partially-observed student MDP the deployable policy actually faces.

Three stages, each a stock RSL-RL runner over the same env. Stage 3 carries the §7.2 fix-up; without it the warm-started actor degrades within ~50 PPO iterations because a fresh-random critic produces noisy advantage estimates that drag the actor out of the imitation basin. The naïve "load distill weights, run PPO" path was tried first and is what motivates the rest of this section.

**Stage 1 — state-only teacher (PPO).** Task ID `Isaac-SO-ARM101-PickPlace-Bowl-Teacher-v0`. Same MDP / reward, but `obs_groups = {"policy": ["policy", "critic"], "critic": ["policy", "critic"]}` — the CNN in `PickPlaceVisionActorCritic` auto-disables when `wrist_image` is absent from any group, so the same module collapses to a pure MLP A-C. Stock Franka-Lift hyperparams (`lr=1e-4`, `entropy=0.006`, `γ=0.98`, `num_steps=24`, `epochs=5`, `mini_batches=4`). Converges cheaply (target ~1500 iters at 2048 envs) — it sees ground-truth block pose. Save `model_*.pt`; both `actor.*` and `critic.*` are reused downstream.

**Stage 2 — short distillation (BC warm-start).** Task ID `Isaac-SO-ARM101-PickPlace-Bowl-Student-v0`. RSL-RL `DistillationRunner` with `loss_type="mse"`, on-policy DAgger. Student is `PickPlaceVisionStudentTeacher` (CNN + MLP[256,128,64] on `policy + wrist_image`); teacher is the frozen MLP loaded from Stage 1 via `StudentTeacher.load_state_dict()` (PPO `actor.*` keys auto-route to `self.teacher.*`). **Apply the same DrQ pad-and-crop augmentation here that Stage 3 will use** — fix-up intervention #1 below. Run ~200–500 iters, stop when `release_from_scratch` clears ~30–50 %. Not to convergence — we just need the CNN keypoints + actor MLP out of the cold-start regime.

**Stage 3 — vision PPO from warm-start.** Task ID `Isaac-SO-ARM101-PickPlace-Bowl-v0`, resumed from the Stage-2 student checkpoint. The warm-start path in `PickPlaceVisionActorCritic.load_state_dict()` detects distill-format keys (`student_cnn.*`, `student.*`) and routes them to `actor_cnn.*`, `actor.*`. With the §7.2 interventions applied the critic is no longer fresh-random, `std` is forced back to an exploration-meaningful scale, and `γ` matches the teacher. Trains on §4 task reward to convergence — student exceeds teacher because PPO's value-function regression continuously updates `V` against the *student's* current rollouts, so the value estimates leave the teacher's prior within ~100–500 iters and the policy gradient optimizes the real (partially-observed) student MDP.

### 7.1 Why naïve warm-start collapses

Diagnostic from the first warm-start attempt: Stage 2 produces a student that tracks the teacher fine (low distillation MSE, visible competent play-mode rollouts). Stage 3 starts PPO from that checkpoint and the policy degrades within ~50 iters — can't grasp, can't lift, can't place. Root causes, in priority:

1. **Random critic.** `critic.*` initializes fresh. The PPO advantage `A_t = R_t + γV(s') − V(s)` is dominated by the random `V(·)` term (off by orders of magnitude from the true return scale), not by the reward gradient. PPO's surrogate update then drives the actor in a corrupted direction. The `desired_kl=0.005` adaptive LR slows the damage but does not reverse it.
2. **`std=0.1` inherited from the distill checkpoint.** Too narrow for the binary gripper (effectively deterministic) and for arm exploration. PPO cannot probe out of the imitation basin to find rewarding deviations even when the advantage is occasionally informative.
3. **DrQ distribution shift at the Stage 2→3 boundary.** Stage 3 applies ±4 px pad-and-crop to actor images; the original Stage 2 distillation didn't. The CNN's keypoint statistics shift the moment Stage 3 starts.
4. **γ mismatch.** Teacher trained with γ=0.98 (effective horizon ≈ 50 steps); the prior Stage 3 config used γ=0.9 (horizon ≈ 10 steps), inherited from ManiSkill3 PickCube. But our `release_in_bowl=30` is a long-horizon latch that pays every post-release step — γ=0.9 chops ~80 % of that tail's value. The teacher's policy was a fixed point of a long-horizon Bellman equation; Stage 3 must match.
5. **Block-xy variance from iter 0.** The deleted `expand_block_xy_range` curriculum left Stage 3 facing full-width block randomization from iter 0, on top of the random-critic shock.

### 7.2 Fix-up — five interventions, applied together

| # | Fix | Cost | Addresses | Reference |
|---|---|---|---|---|
| 1 | Add DrQ pad-and-crop to `PickPlaceVisionStudentTeacher._encode_student` (Stage 2). The student CNN trains on the same image distribution it will face during Stage 3 PPO. | 1 line | (3) | Kostrikov et al., DrQ |
| 2 | In `PickPlaceVisionActorCritic.load_state_dict()` distill branch, **drop the loaded `std`** and re-initialize from the cfg's `init_noise_std=0.5`. | 2 lines | (2) | Standard BC→RL practice |
| 3 | `rsl_rl_ppo_cfg.py`: `gamma=0.9 → 0.98`. Match the teacher's discount; `release_in_bowl=30` needs the long horizon. | 1 line | (4) | — |
| 4 | Wire `expand_block_xy_range` back into `CurriculumCfg` with a **warm-start-adapted** schedule: ±3 cm for 5 k env-steps warmup, expand to ±7×±12 cm over the next 30 k env-steps (~150 → ~1000 iters at 2048 envs × 16 steps). The narrow initial range gives the fresh critic a low-variance return distribution to fit against while the actor sits in the imitation basin. | ~15 lines | (1), (5) | ManiSkill3, NVIDIA Isaac DR cookbook |
| 5 | **Load the Stage-1 teacher's critic into Stage 3's critic.** Teacher's `obs_groups["critic"] = ["policy","critic"]` matches Stage 3's `obs_groups["critic"] = ["policy","critic"]` layer-for-layer — both have `critic_cnn=None`, both have MLP shape `[59 → 256 → 128 → 64 → 1]`. Extend `load_state_dict` to accept a teacher-checkpoint path and selectively load `critic.*` keys *before* processing the distill checkpoint, or do it in the train script: `{k: v for k, v in teacher_sd.items() if k.startswith("critic.")}` filtered into `load_state_dict(strict=False)`. | ~10 lines | (1) | Pinto et al., Asymmetric A-C (RSS 2018) |

**Intervention #5 is load-bearing.** It is the canonical asymmetric-AC handoff: the privileged critic carries across the BC→RL boundary, and the value-function regression `0.5 · (V(s) − Ȓ_t)²` under PPO moves `V` away from `V_teacher` to `V_student` over the first ~100–500 iters. The teacher's value is *not* a soft constraint on the student — it is a numerical anchor that avoids the random-init advantage shock. The student is free to exceed the teacher as soon as the critic catches up, because PPO's policy gradient is computed against the student's own current value estimate and the true task reward. This is the pattern reused in DextrAH-G's teacher→student handoff (modulo their no-RL-fine-tune choice), ANYMAL locomotion distillation, and every Pinto-asymmetric-AC follow-up. Our Stage 3 critic input is *identical in shape* to the teacher's critic input — there is no remap, no fresh layers, just `nn.Module.load_state_dict(critic_sd, strict=False)`.

Total implementation surface: ≤ 30 lines across `vision_student_teacher.py`, `vision_actor_critic.py`, `rsl_rl_ppo_cfg.py`, `pickplace_env_cfg.py`. No new training script, no algorithm fork. Diagnose by Stage-3 iter ~200: `release_from_scratch` should clear the Stage-2 plateau within that budget if the fix-up is sufficient.

### 7.3 Fallback ladder (if 7.2 is insufficient)

Run 7.2 first and inspect TB by iter ~200. If `release_from_scratch` is still flat or regressing, layer the next intervention on top. The ladder is ordered by implementation cost and by what it diagnostically tells you about the failure mode.

- **F1 — Critic-only burn-in.** Set `requires_grad=False` on `actor`, `actor_cnn`, `std`; zero the PPO surrogate loss for the first ~100 iters; train only the value-function regression. Unfreezes after the critic has fit the warm-started policy's true value function. This is the diagnostic-cleanest experiment because it isolates "random-critic shock" — if F1 alone (without #5) recovers Stage 3, you have a clean attribution. ~5 lines. Source: Nair et al., AWAC (2020); also DeepMimic.
- **F2 — DAPG-style auxiliary BC loss.** Keep the Stage-1 teacher MLP in memory; add `λ(t) · MSE(μ_student(s), μ_teacher(s).detach())` to the PPO loss with `λ(t) = λ₀ · 0.999^iter` exponential decay, λ₀ ≈ 0.5. The teacher tether prevents drift while the critic catches up; the decay lets the student deviate once PPO signal is reliable. ~50-line PPO update patch. Source: Rajeswaran et al., DAPG (RSS 2018) — the seminal "BC + RL fine-tune" recipe.
- **F3 — Refined Policy Distillation (RPD).** Same shape as F2 but with constant α ≈ 0.1–0.5 on the MSE term, no decay. RPD's published advantage is specifically on **ManiSkill3 PickCube** — our direct reference task — beating both vanilla PPO and BC-to-convergence. If DAPG's decay is too aggressive (teacher tether fades before the critic has caught up), RPD is the variant to try. Source: Jülich, ICRA 2025.
- **F4 — Jump-Start RL (JSRL).** For each episode, the frozen teacher rolls out the first `k` steps and the student rolls out the remaining `H − k`. Only student-controlled steps contribute to PPO gradients. `k` decays from `H` (pure teacher rollouts) toward 0 over training. The student only sees states it is already competent at early, and gradually inherits the harder early steps as the value function matures. Heavier to implement (rollout loop modification, not a loss patch), but the only fallback that addresses *both* random-critic shock and exploration-deficit in one move. Source: Uchendu et al., ICML 2023.

### 7.4 Workflow

```bash
# Stage 1: state-only teacher
train --task Isaac-SO-ARM101-PickPlace-Bowl-Teacher-v0 --headless

# Stage 2: short distill (~200–500 iters, NOT to convergence).
# Stage 2 student's _encode_student now applies the same DrQ shift as Stage 3 (intervention #1).
train --task Isaac-SO-ARM101-PickPlace-Bowl-Student-v0 \
      --load_run <teacher_run> --checkpoint model_<best>.pt --headless

# Stage 3: warm-started vision PPO.
# --teacher_ckpt is a new flag wired in the train script (intervention #5):
# loads Stage-1 PPO critic.* keys into the Stage-3 critic BEFORE the distill
# warm-start is applied to the actor. Without it, the random critic destroys
# the warm-started actor within ~50 iters.
train --task Isaac-SO-ARM101-PickPlace-Bowl-v0 --resume \
      --load_run <distill_run>  --checkpoint model_<best>.pt \
      --teacher_ckpt logs/rsl_rl/pickplace_bowl_teacher/<teacher_run>/model_<best>.pt \
      --headless
```

## 8. Visual modality — RGB + depth + mask, same pipeline in sim and real

The wrist camera is the only physical sensor; depth and the block mask are **derived** from its RGB feed at deploy. The key sim-to-real principle: **run the same derivation pipeline in both domains.** Don't rely on Isaac's perfect `distance_to_camera` for sim depth and DA3 for real depth — that just relocates the gap. Apply DA3 to the *sim-rendered RGB* identically to the real RGB, so the policy sees DA3-flavored depth (with its smoothing and edge artifacts) in both worlds. Same convention we already use for `cv2.undistort`: sim is a perfect pinhole but we still pipe both feeds through the same undistort to keep the preprocess identical.

Wrist input is a 5-channel `(N, 5, 72, 128)` tensor stacking:

| Channel | Source (sim *and* real) | Notes |
|---|---|---|
| 0–2 | RGB | sim TiledCamera (`rgb`) → cv2.undistort (no-op in sim, real distortion in deploy) → resize 128×72 → `/255`. |
| 3 | **Depth from Depth Anything 3** | Run `depth-anything/DA3-Small` (80 M params, DINOv2-S backbone) on the same RGB above. Relative depth, calibrate to metric once per session via a linear fit against known FK distances at home pose; clip to `[0, 0.5 m]`, divide by 0.5. Run DA3 at 252×140 (DINOv2 patch is 14 px — 128×72 = 9×5 patches is below training distribution and degrades accuracy), downsample the depth map to 128×72. |
| 4 | Binary block mask | sim: pull the cube class from `semantic_segmentation`. real: HSV `cv2.inRange` calibrated once on a wood-tone block on the gray table; optional 3×3 morphological close. Resize to 128×72. For Eval 2/3 the bounds become target-color-conditioned — same channel, no network change. |

Network change: `_SpatialSoftmaxCNN` first conv goes `Conv2d(3 → 32) → Conv2d(5 → 32)`. Otherwise unchanged.

**Why DA3.** Depth Anything 3 (ByteDance-Seed, 11/2025) is SOTA on indoor monocular metric depth (ETH3D δ₁ = 0.917, SUN-RGBD AbsRel = 0.105 — beats DA-V2, UniDepth v2, DepthPro, Metric3D v2 on standard indoor benchmarks). The Small checkpoint is the right speed/quality trade-off for the 50 Hz inner loop; upgrade to `DA3-Metric-Large` (350 M, directly metric, skips the calibration step) only if timing budget allows.

**Why a mask.** ManiSkill3 PickCube went from 91.6 % (RGB) to 95 % (with object segmentation). The cube + uniform-gray table makes the segmentation easy at deploy — HSV thresholding gets a high-recall mask in 5 lines of OpenCV — and exact in sim. We keep the mask alongside RGB+depth (not as a replacement) so the policy stays robust if the real-side threshold ever misses.

**Encoder fallback.** If the from-scratch spatial-softmax CNN doesn't transfer even with this 5-channel input + §5 visual DR, swap to a **frozen pretrained backbone**: Theia-Tiny (CMU, ~6 M params, manipulation-priored, distilled from DINOv2 + SAM + Depth-Anything) or DINOv2-Small (22 M). Keep the spatial-softmax head on top of the backbone's spatial feature map for the same positional inductive bias. ~10–20 ms inference, fits 50 Hz.

**Deliberately skipped:**
- Optical flow (wrist motion swamps object motion).
- Full point cloud + PointNet (~3 % gain over RGBD-as-channels at 10× engineering cost).
- Open-vocab detector → 2D block-center coordinate (~50 ms latency, unnecessary for one cube; reconsider for Eval 2/3).
- Depth-only without RGB (loses the cube-vs-table-color cue when block sits flat).

**Rollout order.** (1) Get Stage 3 converging in sim under the §7.2 fix-up first — depth and mask channels are already in the 5-channel `mdp.wrist_image` tensor, but they're useless if the warm-start path is broken. (2) Once sim success is high, add the §5 deferred visual DR (cube/table color, lighting, HDRI overlay) — most peer-project *transfer* failures are visual-DR shortfalls, not modality shortfalls. (3) Validate that DA3-on-RGB in deploy matches the sim depth-jitter distribution; tune the per-step depth corruption envelope in `mdp.wrist_image` if needed. (4) Only if 1–3 don't transfer, swap the from-scratch CNN to frozen Theia / DINOv2 (this section). Steps 2–3 each touch only `mdp.wrist_image` and `events.py` — surgical changes, no env restructure.

## References

**This repo:** `tasks/pickplace/{pickplace_env_cfg.py, joint_pos_env_cfg.py}`, `mdp/{observations,rewards,events,terminations,commands}.py`, `agents/{vision_actor_critic.py, vision_student_teacher.py, rsl_rl_ppo_cfg.py, teacher_ppo_cfg.py, distill_cfg.py}`, `camera_intrinsics.yaml`, `robots/trs_so101/{so_arm101.py, urdf/so_arm101.urdf}`. Run/deploy guides: [`RUNNING.md`](./RUNNING.md), [`DEPLOY.md`](./DEPLOY.md).

### Method lineage (architecture & augmentation)

- **Pinto, Andrychowicz et al.**, *Asymmetric Actor Critic for Image-Based Robot Learning*, RSS 2018. <https://arxiv.org/abs/1710.06542>. The privileged-critic + image-actor pattern §2/§6/§7 are built on; §7.2 intervention #5 (teacher critic carry-over across the BC→RL boundary) is this paper's canonical handoff.
- **Levine et al.**, *End-to-End Training of Deep Visuomotor Policies*, JMLR 2016. Spatial-softmax CNN — the inductive bias for "find a small object on a flat workspace" the §3 encoder uses.
- **Kostrikov et al.**, *DrQ — Image Augmentation Is All You Need*, ICLR 2021. <https://arxiv.org/abs/2004.13649>. Pad-and-crop random shift; the augmentation that lets a from-scratch CNN train via on-policy PPO without a pretrained backbone. Fix-up intervention #1 applies it in Stage 2 *and* Stage 3.
- **Yarats et al.**, *DrQ-v2*, 2021. <https://arxiv.org/abs/2107.09645>. Sub-pixel shift variant — not used (integer pad-and-crop is sufficient for 128×72), kept here as the upgrade path if image augmentation becomes a bottleneck.
- **ByteDance-Seed**, *Depth Anything 3*, arXiv 2511.10647 (11/2025). <https://github.com/ByteDance-Seed/Depth-Anything-3>. The monocular metric depth model used in §8 to derive the depth channel from RGB at deploy, with sim-side depth perturbed to match its artifacts.

### Warm-start recipes (Stage 3 fix-up and fallback)

- **Rajeswaran et al.**, *DAPG — Learning Complex Dexterous Manipulation with Deep RL and Demos*, RSS 2018. <https://arxiv.org/abs/1709.10087>. Seminal "BC warm-start + RL fine-tune" recipe with auxiliary BC loss `λ(t) · ℒ_BC` and exponential decay — fallback F2.
- **Jülich**, *Refined Policy Distillation*, ICRA 2025. <https://arxiv.org/abs/2503.05833>. PPO + frozen-teacher MSE term, validated on ManiSkill3 PickCube — our direct reference task. Fallback F3.
- **Sun et al.**, *Proximal Policy Distillation*, 2024. <https://arxiv.org/abs/2407.15134>. KL-to-teacher constraint variant of RPD; mentioned in §7.3 commentary.
- **Uchendu et al.**, *Jump-Start Reinforcement Learning*, ICML 2023. <https://arxiv.org/abs/2204.02372>. Teacher rolls out the first `k` steps each episode, `k` decays over training. Fallback F4 — the only fallback that addresses both random-critic shock and exploration deficit in one mechanism.
- **Nair et al.**, *AWAC — Accelerating Online RL with Offline Datasets*, 2020. <https://arxiv.org/abs/2006.09359>. Critic burn-in pattern (train V with actor frozen before unfreezing the policy gradient) — fallback F1.
- **Lum, Allshire et al.**, *DextrAH-G*, 2024. <https://sites.google.com/view/dextrah-g>. DAgger-to-convergence teacher→student handoff with privileged-critic carry-over. The "no RL fine-tune" variant — explicitly rejected here in favor of Stage 3, but the critic-handoff pattern (intervention #5) is taken from their privileged-state→depth-student transition.

### Peer recipes on the same robot family

- **Tao et al.**, *ManiSkill3 / lerobot-sim2real (StoneT2000)*, 2024. <https://github.com/StoneT2000/lerobot-sim2real>, <https://github.com/haosulab/ManiSkill>. Single-stage PPO from scratch, RGB + segmentation mask, 25–40 M env-steps, 91.6 % → 95 % zero-shot real on SO-100 PickCube. Their γ=0.9 / 8 epochs × 32 mini-batches is the §6 sanity-check column; their reward is "5 lines of code" sparse-ish — *not* a long-horizon latch like our `release_in_bowl=30`, which is why our γ stays at 0.98.
- **NVIDIA**, *Train an SO-101 Robot From Sim-to-Real With NVIDIA Isaac*. <https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/01-overview.html>. IL/VLA recipe (teleop demos + co-training, not RL). Not a direct competitor, but their DR envelope (lighting 2500–9500 K, camera pose ±0.02 m / ±0.05 rad, HDRI overlay) is the right reference for the deferred visual-DR work in §5.
- **Evans & Hegde**, *Vision-Based Manipulation via Sim-to-Real RL — SO-101 with Isaac Lab*, CS6341 Fall 2025. <https://yuxng.github.io/Courses/CS6341Fall2025/project_group_15.pdf>. The negative result on cold-start vision PPO that motivates the §7 teacher warm-start.
- **LeIsaac (LightwheelAI)**. <https://github.com/LightwheelAI/leisaac>. Wrist-camera mount and intrinsics conventions reused verbatim in §2 / `joint_pos_env_cfg.py`.

### Tooling

- **RSL-RL**, Schwarke et al., 2025. <https://arxiv.org/abs/2509.10771>, <https://github.com/leggedrobotics/rsl_rl>. Documents the `DistillationRunner` + "BC warm-start then RL" workflow that maps onto our Stage 2 → Stage 3 pipeline.
