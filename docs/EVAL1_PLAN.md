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

**Table geometry (sim-to-real match).** The table cuboid is `0.6 m × 1.0 m × 0.02 m` centered at `(0.25, 0.0, −0.01)` — top surface at `z = 0`, back edge at `x = −0.05 m`, front edge at `x = +0.55 m`. The arm base sits at origin, so the table extends only ~5 cm behind the base — matching the real rig where the SO-ARM101 is clamped at the table edge. The earlier 1 m × 1 m table extended 30 cm behind the base, which the deployed wrist cam never sees on the real setup; closing that background-distribution mismatch is a free sim-to-real win at zero policy-side cost.

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
| `lifting_object` | 15.0 | indicator `block_z > 0.07` (2 cm above real bowl's 5 cm rim) |
| `object_goal_tracking` | 16.0 | `(1 − tanh(d_block_goal / 0.30)) · 𝟙[block_z > 0.025]` — dense, **per-step lift gate** |
| `object_goal_tracking_fine_grained` | 5.0 | same fn at `std=0.05` |
| `release_in_bowl` | 30.0 | block xy near bowl ∧ block_z < 0.06 ∧ gripper open ∧ block settled, gated on **per-episode lift latch at 0.07 m** AND **per-episode over-bowl-above-rim latch at 0.08 m** |
| `action_rate`, `joint_vel` | −1e-4 → −1e-2 | curriculum ramp at 10 k env-steps |

Three design choices to call out:

1. **Per-episode lift latch at 0.07 m** (`_episode_lifted_mask`, cleared each reset by `reset_was_grasped`): `release_in_bowl` only fires once the policy has lifted the cube ≥ 7 cm at some prior step in the same episode. Closes the "drag the cube laterally into the bowl" exploit a dense block-to-bowl term otherwise rewards (the original motivation). The 0.07 threshold (2 cm above rim) is chosen so cube_center > 0.07 implies cube_bottom > 0.06 (clears rim) and gripper sits at z ≈ 0.12 (well clear of rim). `object_goal_tracking` and the fine-grained variant keep the looser per-step `block_z > 0.025` gate to preserve gradient signal during the controlled descent INTO the bowl. `lifting_object` shares the 0.07 threshold so the lift-bonus reward kicks in at exactly the height band the deploy trajectory needs.

2. **Per-episode over-bowl-above-rim latch at 0.08 m** (`_episode_over_bowl_high_mask`, cleared each reset by `reset_was_over_bowl_above_rim`, added 2026-05-11): `release_in_bowl` AND `place_in_bowl` AND `task_success` additionally require the cube to have been *simultaneously* above 0.08 m AND within `r_safe=6 cm` of bowl xy at some prior step. This closes the **bowl-rim sim-to-real gap** that the lift latch alone left open: with no physical bowl prim in sim, a policy can lift the cube to 7 cm far from the bowl, descend to z ≈ 0.02 off-bowl-center, then slide laterally at low z into the bowl footprint — earning release reward at z ≈ 0.02 with a trajectory that would slam a real gripper into the 5 cm rim. Empirically observed on the 2026-05-11 student checkpoint (`pickplace_bowl_student/model_550.pt`) on playback: SR ≈ 0.72 in TB but visually the cube descends at low z while crossing the rim annulus. Adding this latch forces the descent to come from above. The 0.08 m clearance (1 cm above the lift latch's 0.07 m) is to make the gate stricter than the existing latch — the policy now needs to *actually* hover over the bowl above the rim, not just have been lifted somewhere else. Implementation: ~50 lines across `rewards.py`, `events.py`, `terminations.py`, `mdp/__init__.py`, `pickplace_env_cfg.py:EventCfg`. TB now also logs `Curriculum/log_success/over_bowl_high_rate` for diagnostic visibility.

3. **No success-termination.** Removing `task_success` from `TerminationsCfg` was load-bearing: with success-termination the episode ends on first release and "hold and hover until timeout" beats "release and stay" by ~5× cumulative reward. With it removed, `release_in_bowl=30` pays every post-release step until time-out and the policy learns to actually let go.

**Reward weights were tuned for γ=0.98** (see §6). The `release_in_bowl` latch is a long-horizon term — once it fires, it pays for the remainder of the 250-step episode. A short-horizon discount (γ=0.9 → effective horizon ≈ 10 steps) chops 80 % of that tail's value and changes the optimal policy.

## 5. Curriculum & DR (in `mdp/events.py`, `CurriculumCfg`)

**Stage 1 (state teacher) — active:**
- **Cube placement randomization** (`reset_block_position`): block xy ∈ ±7 cm × ±12 cm around `(0.20, 0, 0.01)`, intentionally tightened from the original ±10×±15 cm to the comfortable reach band so all (block, bowl) pairs are solvable. The bowl is rejection-sampled within the same band with `min_distance ≥ 10 cm` from the block (`mdp/commands.py::BowlPoseCommand`). This is the *only* placement DR needed — the policy must learn the full workspace, but every sampled pose is physically reachable.
- **Action / joint-vel penalty ramp**: −1e-4 → −1e-2 at 10 k env-steps.
- **`log_success` curriculum term**: emits binary `success_rate` (fraction of ended episodes where `release_in_bowl` fired) to TB.
- **No image-side DR.** The teacher reads ground-truth state; the camera renders each step but the obs term is not in `obs_groups`. Visual DR is irrelevant to it.

**Stage 3 (warm-started vision PPO) — adds:**
- **Block-xy expand curriculum** (`expand_block_xy_range`, wired in `CurriculumCfg` as of the 2026-05-11 fix-up). **Warm-start-adapted schedule** — *not* the cold-start one the function originally documented. Initial ±3×±3 cm for 5 k env-steps warmup, expanded linearly to full width (±7×±12 cm) over the next 30 k env-steps. At 1024 envs × 16 steps/iter the verified Stage 3 reached x=0.066 / y=0.111 (≈ 93 % of target) by iter 1999. The point is not "make reach easy" (the warm-started student already reaches); it is to give the freshly-initialized critic a low-variance return distribution to fit against while the actor sits inside the imitation basin. Once `V` has caught up, expand to full width. See §7.2 intervention #4 and §7.5 for the realized trajectory.
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
# loads Stage-1 PPO critic.* keys into the Stage-3 critic AFTER the distill
# warm-start is applied to the actor (overlay only — actor / CNN / std stay
# at their post-warm-start values). Without it, the random critic destroys
# the warm-started actor within ~50 iters.
train --task Isaac-SO-ARM101-PickPlace-Bowl-v0 --resume \
      --load_run <distill_run>  --checkpoint model_<best>.pt \
      --teacher_ckpt logs/rsl_rl/pickplace_bowl_teacher/<teacher_run>/model_<best>.pt \
      --headless --num_envs 1024
```

`--num_envs 1024` is load-bearing on a 32 GB GPU — the env cfg default of 4096 OOMs during rollout-storage allocation for the 5-channel `(72, 128)` image obs. Stage 2 and Stage 3 both use 1024. RUNNING.md §5.1 documents the exact symlink dance needed for `--load_run` (Isaac Lab requires the run dir to live under the *current* experiment's log dir, not the source stage's).

### 7.5 Implemented and verified (2026-05-11 run)

Single end-to-end run with all five §7.2 interventions wired together. No fallback ladder needed.

| Stage | Run dir | Iters | Wall time | Final SR | Notes |
|---|---|---|---|---|---|
| 1 (teacher PPO) | `pickplace_bowl_teacher/2026-05-10_22-40-19` | 700 | (carried over) | — | State-only, reused from earlier work |
| 2 (distill) | `pickplace_bowl_student/2026-05-11_13-22-12` | 550 | ~20 min | **0.66–0.73** | Stopped on plateau; behavior_loss 0.12 |
| 3 (vision PPO) | `pickplace_bowl/2026-05-11_13-58-29` | 1999 | **1 h 6 min** | **0.91** | Final ckpt `model_1999.pt` |

**SR trajectory across Stage 3** (the failure-mode-inverted shape — *gains* on the warm-start baseline rather than collapsing):

| iter | SR | reward | σ | curriculum (x, y) |
|---|---|---|---|---|
| 138 | 0.75 | 156–175 | 0.52 | 0.030, 0.030 (warmup) |
| 660 | 0.89–0.93 | 181–192 | 0.59 | 0.037, 0.047 (expanding) |
| 1590 | 0.83–0.93 | 170–173 | 0.69 | 0.057, 0.091 |
| 1999 | 0.91 | 167 | 0.72 | 0.066, 0.111 (~93 % of target) |

Code surface that actually shipped (matches the §7.2 cost column within ~10 lines):

1. **DrQ in distill** — `agents/vision_student_teacher.py::_encode_student` calls `_random_shift_pad(img, DRQ_PAD_PIXELS)` when `self.training`, reusing the helper from `vision_actor_critic.py`. 3 lines.
2. **Drop distill `std` on warm-start** — `agents/vision_actor_critic.py::load_state_dict` distill branch skips the `std` key and adds it to `expected_missing`, so the cfg's `init_noise_std=0.5` is retained. Logged at boot: `std kept at cfg init_noise_std=0.5`. 2 lines.
3. **γ = 0.98 in Stage 3** — `agents/rsl_rl_ppo_cfg.py::PickPlaceBowlPPORunnerCfg.algorithm.gamma`. 1 line.
4. **Block-xy expand curriculum** — `pickplace_env_cfg.py::CurriculumCfg.block_range_expand` wires `mdp.expand_block_xy_range` with `initial_xy=(0.03, 0.03)`, `final_xy=(0.07, 0.12)`, `warmup_steps=5_000`, `expand_steps=30_000`. ~15 lines.
5. **Teacher critic overlay** — `scripts/rsl_rl/train.py` adds the `--teacher_ckpt` flag and a block that runs *after* `runner.load()`, filters the teacher checkpoint to `critic.*` keys, and overlays them via `torch.nn.Module.load_state_dict(policy, critic_sd, strict=False)` — bypassing both `PickPlaceVisionActorCritic.load_state_dict` and the `rsl_rl.modules.ActorCritic.load_state_dict` it wraps, since both return `bool` rather than the standard `(missing, unexpected)` tuple. ~25 lines.

Also implemented in the same session, outside the §7.2 fix-up: the **table-geometry sim-to-real match** in §1 — table cuboid shrunk to `0.6 × 1.0 × 0.02 m`, back edge at `x = −0.05` so the arm base sits flush with the real-rig table-edge clamp. No policy-side change.

Verification commands (full invocations including the `--num_envs 1024` and `from_teacher`/`from_student` symlink dance) live in [`RUNNING.md`](./RUNNING.md) §5.1; the GUI playback / video-rendering recipe is §5.2. Stage 3's final checkpoint `pickplace_bowl/2026-05-11_13-58-29/model_1999.pt` is the policy that will go into the sim-to-real visual DR work in §5 (deferred bucket) and the §8 RGB→DA3 deploy pipeline.

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

## 9. Alternative path — pretrained-backbone cold-start (no teacher)

> **Status: experimental ablation, parallel to §7. The verified §7.5 Stage 1–3 pipeline is the production path and is untouched by this section.** §9 adds an entirely separate training path (new agent class + new agent cfg + new gym ID + new env-cfg subclass) to test one specific hypothesis: did the teacher warm-start fix a *perception* bottleneck (which a pretrained encoder would also fix) or an *optimization-dynamics* bottleneck (which only §7.2 fixes)? Built so a negative result is informative and a positive result is a candidate Eval-2/3 path.

### 9.1 What this experiment isolates

Single-stage cold-start PPO from a pretrained convolutional backbone, **frozen** (supervisor decision — see §9.4 update note). Same task, same reward, same γ, same env as §1–§5. The only deliberate change vs §7's Stage-3 config is the encoder; warm-start, distillation, and teacher-critic overlay are all removed. If §9 reaches comparable SR, the §7.1 attribution shifts toward "perception was load-bearing"; if it plateaus, the §7.1 attribution toward "critic shock + exploration std + γ-tuning was load-bearing" stands. Either result tightens the story.

**Update (2026-05-11, supervisor):** the original §9.1 specified fine-tuning the trunk with a decoupled encoder LR; we now freeze the ResNet trunk and train only the depth/mask CNN + fused head + MLPs. PPO's on-policy distribution shift is well-documented as destabilizing for a fine-tuned ImageNet encoder; freezing trades adaptation capacity for stability and ~5× fewer trainable params. The R3M/MVP fallback path (§9.2 "Upgrade if R18 plateaus" row) becomes the *only* upgrade path — *not* unfreezing ImageNet, which the literature consistently shows underperforms manipulation-pretrained alternatives.

### 9.2 Backbone choice — ResNet-18 over DINOv2 / Theia

**Pick: ResNet-18, ImageNet-1k pretrained (`torchvision`), trunk frozen (per §9.1 update); head + depth/mask CNN + MLPs train at LR 1e-4.** Evaluated against our spec (5-channel 128×72 wrist images, PPO with non-stationary on-policy gradient, 50 Hz at ~1024 envs):

| Candidate | Verdict | Why at our spec |
|---|---|---|
| **ResNet-18 (ImageNet)** | **Primary** | Conv translation-equivariance > ViT patches at 128×72 (DINOv2-S/v3 patch-14 gives 9×5 = 45 tokens, well below the 256-token training distribution). BN→GN conversion gives stable RL fine-tuning. ~3 ms forward fits the inner loop. 5-ch channel inflation is one-liner. |
| DINOv2-S / DINOv3 | Reserve | Stronger benchmark features but patch granularity hurts at 128×72; ViT fine-tuning under PPO is documented as fragile — published wins (e.g. `2509.17684`) keep it frozen for diffusion-policy IL, not on-policy RL with fine-tune. |
| Theia-Tiny | Already §8 deploy-side fallback | Manipulation-priored, but same patch issue (ViT-based). Better than DINOv2-S for manipulation, worse than R18 for our resolution. Reserved for sim-to-real transfer aid (§8 step 4) — orthogonal use. |
| R3M (ResNet-50, Ego4D) | Upgrade if R18 plateaus | Conv arch (good) + manipulation prior (good) but 2× params and ~5 ms forward; revisit only if R18-ImageNet underperforms by iter ~2000. Same wiring, swap the backbone constructor and conv1 inflation pattern. |

If §9 with R18 plateaus, R3M is the immediate next step — *not* DINO. Switching architectures (conv → ViT) and training prior (ImageNet → SSL) simultaneously would conflate the variable.

### 9.3 Architecture — `PickPlaceResNetActorCritic`

Dual-stream encoder: pretrained ResNet-18 on RGB, small from-scratch CNN on depth + mask, fused before the spatial-softmax head. Keeps the §3 Levine-style soft-argmax inductive bias on top of pretrained features — task-aligned regardless of backbone.

```
wrist_image (5 × 72 × 128)
 ├─ RGB (3 ch)        → ResNet-18 (BN→GN, ImageNet weights, **frozen**, truncated at layer3) → (256, 5, 8)
 ├─ Depth+Mask (2 ch) → 3-layer ELU CNN, from scratch                   → (64,  3, 4)
 └─ concat (576, 3, 4) → 1×1 Conv(576 → 64) → spatial-softmax per ch → (x, y) per kpt → LayerNorm → 128-D
state (policy group)  ─ concat ─ MLP[256, 128, 64] ─ μ
                                                    σ (scalar Param, init=1.0)
critic state (policy + critic groups) ─ MLP[256, 128, 64] ─ V(s)
```

Two non-obvious choices:

1. **`BatchNorm2d → GroupNorm(32)` at load time.** BN running statistics drift catastrophically under PPO's non-stationary rollouts (every iter the policy distribution shifts, BN's moving average chases it, encoder output distribution shifts under the head, head gradients amplify). GN sidesteps the issue. ImageNet weights stay valid — BN's affine γ/β re-map to GN's γ/β with no retraining. Standard recipe (Wu & He, GroupNorm, ECCV 2018).
2. **Channel inflation, not RGB-only.** Replace ResNet's `conv1` (`Conv2d(3, 64, k=7, s=2, p=3)`) with `Conv2d(5, 64, k=7, s=2, p=3)`; copy ImageNet weights into channels 0–2 and init channels 3–4 from the RGB-channel mean (preserves rough activation statistics for the new channels). One-line override at load. The alternative — RGB-through-ResNet-only, drop depth + mask — would discard the §8 modality work and is *not* what this experiment is testing.

Critic is unchanged from §3 / §6 Stage 3: MLP on `policy + critic` groups, no image. The image encoder is actor-only — consistent with the asymmetric A-C principle (§2).

### 9.4 PPO config — `pretrained_ppo_cfg.py`

| Hyperparam | §9 cold-start | Stage 3 (§6) | Note |
|---|---|---|---|
| `num_envs` | 1024 (512 if VRAM-bound) | 1024 | R18 + 5-ch image fits on a 32 GB GPU but is closer to the OOM line than the from-scratch CNN |
| `num_steps_per_env` | 16 | 16 | same |
| `max_iterations` | 4000 (~130 M env-steps) | 2000 | cold-start needs ~2× the warm-start budget |
| `init_noise_std` | **1.0** | 0.5 | wide exploration; no imitation basin to preserve |
| `learning_rate` (head + MLPs) | 1e-4 | 1e-4 | same |
| `learning_rate` (encoder) | **N/A — trunk frozen** | n/a | trunk has `requires_grad=False`; no param group needed. Implementation supersedes §9.1 original ("decoupled LR 1e-5") per supervisor decision. |
| `entropy_coef` | 0.006 → 0.003 over 1500 iters | 0.006 → 0.003 | same |
| `gamma / lam` | **0.98** / 0.95 | 0.98 / 0.95 | **must match** — reward shape unchanged, isolates the encoder variable |
| `epochs / mini-batches` | 8 / 16 | 8 / 16 | same |
| `clip_param / max_grad_norm` | 0.2 / 1.0 | 0.2 / 1.0 | same |
| `desired_kl` | 0.01 | 0.005 | looser — random init, less worry about drifting out of a basin |

`γ=0.98` and the §4 reward stay identical so the experiment cleanly tests one variable. Dropping to `γ=0.9` + a sparse-ish reward would test "ManiSkill3 recipe transfer" instead, which is a different hypothesis (§9.9).

### 9.5 Curriculum, rewards, and events — cold-start adaptations

All overrides live in `pretrained_env_cfg.py` so §7 paths are untouched.

**Curriculum** (`PretrainedColdStartCurriculumCfg`):
- **`expand_block_xy_range` schedule** — ±3 cm warmup for **15 k env-steps** (3× the warm-start window), expanded linearly to ±7 × ±12 cm over the next **60 k env-steps**.
- **`p_grasped_decay`** (v3, 2026-05-12) — decays the bootstrap-grasp event's `p_grasped` from **0.5 → 0** over warmup 5 k + decay 50 k env-steps. Memory `[[bootstrap-curriculum-pitfall]]` requires a long decay window (3× longer than feels needed) or the policy rides the subsidy and collapses when p hits 0.
- **`log_metrics`** — wires `mdp.log_bootstrap_metrics` so TB shows `grasp_bootstrap`, `grasp_from_scratch`, `release_bootstrap`, `release_from_scratch` split by bootstrap status. The from-scratch metric is the actual signal: it should rise toward 1 as the policy learns grasp without the subsidy.

**Rewards** (`PretrainedRewardsCfg`, v3):
- **`lifting_object.minimal_height` 0.07 → 0.025** — back to the stock Franka Lift / ManiSkill3 value. The 0.07 m threshold is fine when a teacher seeds the lift skill (§7), but it's too strict for cold-start: random PPO exploration starting from a closed gripper near the cube cannot produce a 7 cm lift trajectory at non-trivial probability, so the gradient signal for grasp is effectively zero, σ collapses around hover-near-cube, lift is never discovered. 0.025 m gives gradient for a 1.5 cm incremental lift — the same threshold every successful from-scratch IsaacGymEnvs / IsaacLab lift task uses. Rim safety (the 5 cm physical-bowl gap from §4 design choice 2) is still enforced separately on the success / release path via the `_episode_over_bowl_high_mask` latch (0.08 m clearance), which is where deploy safety actually matters.

**Events** (`PretrainedEventCfg`, v3):
- **`bootstrap_grasped`** (`mdp.init_block_in_gripper`, `p_grasped=0.5`) — half of resetting envs spawn with the block already in the closed gripper at end-effector home pose. Those episodes immediately receive `lifting_object` + `transport_to_bowl` + `release_in_bowl` reward signal, so the post-grasp reward stages have strong gradient from iter 0. From-scratch envs (the other half) learn the grasp skill from the shared policy gradient generated by the bootstrap envs. As `p_grasped` decays to 0, the policy must perform the full task on every reset. This is the **ManiSkill3 PickCube** strategy translated to Isaac-Lab plumbing: their `agent.is_grasping(cube)` contact-pair reward pays +1 per step whenever the cube is grasped without a height gate; we get the same value-landscape effect by forcing half of episodes to start grasped.

**DrQ pad-and-crop** stays — reused unchanged from `vision_actor_critic._random_shift_pad` (imported, not duplicated).

**v1 / v2 ablations that motivated the v3 changes:**
- v1 (`pickplace_bowl_pretrained/2026-05-11_17-43-35`, fine-tuned ResNet, no rim gate, lift=0.07): reach saturated 0.81 by iter 875; lift = 0 through iter 1000; killed manually.
- v2 (`pickplace_bowl_pretrained/2026-05-11_22-18-36`, frozen ResNet + rim gate, lift=0.07): reach hit 0.81 at iter 676, then regressed to 0.49 at iter 1402 as workspace expanded; σ collapsed 0.98 → 0.60; lift = 0 throughout; killed at iter 1552.
- Both runs failed at *exploration of the grasp action*, not perception (reach learned fine in both). The encoder choice — fine-tune vs frozen — was irrelevant to the failure mode. v3 changes attack the actual bottleneck: the reward landscape's grasp gradient.

**v3 ablation that motivated v4 (2026-05-13):** v3 (`pickplace_bowl_pretrained/2026-05-12_23-40-36`, frozen ResNet + lift=0.025 + bootstrap_grasped p=0.5 with 50 k decay) ran 1968 iters. Bootstrap envs hit `grasp_bootstrap` ≈ 0.48 and `release_bootstrap` ≈ 0.32, but `grasp_from_scratch` stayed pinned at exactly 0 for the full run. As `p_grasped` decayed 0.50 → 0.24, `success_rate` collapsed monotonically (0.147 → 0.148 → 0.058 → 0.040). σ widened 0.94 → 1.06 — PPO's adaptive KL loosening because no sharper policy paid off. Diagnosis: init-state bootstrap teaches the value function and the *post-grasp* action sequence, but never generates positive advantage on grasp-*leading* actions in from-scratch rollouts — random exploration alone can't reliably stumble into a successful grasp under a 5-DoF arm + binary gripper, so PPO has no gradient toward "close jaws when near cube". The §9.5 hypothesis "bootstrap teaches grasp via shared weights" is empirically wrong: bootstrap is a value-function aid, not a policy-gradient generator on the action it subsidizes.

**v4 fix** — add `mdp.closed_grasp_signal` at weight 3.0 in `PretrainedRewardsCfg`. Returns `proximity × closedness`:

- `proximity = exp(−‖ee − block‖² / 0.03²)` — sharp Gaussian, fires within ~3 cm of the cube.
- `closedness = clamp((0.3 − gripper_q) / 0.3, 0, 1)` — 1 when fully closed, 0 at the threshold. Home pose `gripper=0.5` puts the policy *outside* the payout zone by default, so the term only fires when the policy *actively* closes jaws past the open default while near the cube.

This is the kinematic proxy of ManiSkill3's `agent.is_grasping(cube)` contact reward (paying +1/step on cube contact, no height gate). Same policy-gradient signal — close jaws near cube earns reward — without needing `activate_contact_sensors=True` on the SO-ARM101 asset.

**v4 outcome (run killed iter 798): the ungated variant created a hover-with-grasp attractor.** Bootstrap envs hit 92 % grasp rate but `release_in_bowl` decayed 0.017 → 0.004 and `success_rate` fell to 0 (vs v3 at iter 818: 0.148). Diagnosis: with no gate, hover-with-cube paid `closed_grasp(3) + lift(15) + transport(16) = 34/step` indefinitely. Release pays 46/step but requires an exploration cost (open jaws, descend, settle) the policy never paid because the hover basin is locally stable. The naïve weight-budget argument (release > hover) failed because PPO is myopic to exploration cost.

**v4.1 fix:** gate `closed_grasp` on `block_z < pre_lift_height = 0.025`. Pays only while the cube is on the table; turns off the moment the cube lifts. After lift, `lifting_object` + `transport_to_bowl` carry the policy and `release_in_bowl` is the unambiguous next-stage payoff. Bonus: bootstrap envs (cube spawns at ee_home_z ≈ 0.083) earn ~0 from this term, so the gradient is *concentrated on from-scratch envs*, exactly where the bootstrap subsidy fails to provide signal.

**v4.1 outcome (run killed iter 1073): same hover-attractor at lower magnitude.** Initial signs were good — at iter 48: `release_in_bowl=0.31`, `success_rate=0.063`, `release_bootstrap=0.013`, `grasp_from_scratch=0.0058` (all firing). But the policy converged into the hover basin by iter 423: `release_in_bowl=0`, `success_rate=0`, `over_bowl_high_rate=0.39` (policy reaches the safe-approach state but won't release). At iter 1073 — 650+ consecutive iters with `release_in_bowl=0` and `success_rate=0` — `over_bowl_high_rate` dropped to 0.04, the bootstrap envs lost the transport skill, and `grasp_bootstrap` rose to 0.86 (pure hover-and-hold). The closed_grasp shaping was insufficient to break the hover attractor.

**Re-diagnosed (2026-05-13, supervisor input).** The v3/v4/v4.1 failures share a single root cause: `lifting_object=15` pays indefinitely while `block_z > 0.025`, creating a hover-with-grasp attractor at z ≈ 0.08 over the bowl. Lowering the cube past z=0.025 triggers an immediate −12/step reward cliff (lift drops to 0), and PPO is myopic about the +30/step release tail that comes 5–10 steps later. The original "negative result for §9" framing conflated *MDP design* with *encoder choice* — the encoder was fine throughout (perception learned, lift+transport learned). The reward landscape was the problem.

**v5 fix (surgical):** drop `lifting_object` (set weight=0) + lower `release_in_bowl.minimal_height` 0.07 → 0.025. This removes the cliff. The reward landscape becomes monotonically increasing from hover → lower → release: 12 → 15 → 46/step, and PPO can find release through normal exploration. Rim safety is preserved by the independent `_episode_over_bowl_high_mask` latch (`rim_clearance=0.08`) on `release_in_bowl`. The "no independent lift bonus" pattern matches every peer project that succeeds at cold-start vision PPO on pick-and-place — ManiSkill3 PickCube (StoneT2000, 91.6 % zero-shot real on SO-100), Robosuite PickPlace, IsaacGymEnvs FrankaCubeStack — none have an unconditional "cube is up" reward; lift behavior is implicit in "approach goal with cube grasped". `closed_grasp_signal` (pre-grasp-gated, weight 3) stays as the kinematic proxy of ManiSkill3's contact-grasp gradient.

**Why §7 still exists.** v5 makes §9 cold-start a viable alternative path, not a replacement. The §7 teacher warm-start path was already verified end-to-end (0.91 SR by iter 1999, §7.5) and remains the production path until v5 reaches comparable SR. The two paths now test orthogonal hypotheses: §7 attacks the optimization-dynamics gap (random-critic shock, exploration std collapse); v5 attacks the reward-landscape gap. The §9.1 attribution question — "did the teacher fix perception or optimization-dynamics?" — is now answerable: v5 either succeeds (perception was solved by the frozen encoder; the teacher was a *reward-landscape* workaround that fixed exploration as a side-effect) or fails (some other §7 mechanism was load-bearing too).

All other §5 DR (per-step photometric jitter, action-rate / joint-vel penalty ramp) unchanged. Deferred visual DR (§5 "Deferred to real-deploy readiness" bucket) stays deferred until §9 reaches comparable sim SR.

### 9.6 Implementation surface — all-new files, zero edits to §1–§7 code paths

| New file | Purpose | Lines (est.) |
|---|---|---|
| `tasks/pickplace/agents/pretrained_resnet_actor_critic.py` | `PickPlaceResNetActorCritic` subclass; `_build_resnet18_5ch()` channel-inflation helper; `_bn_to_gn()` converter; reuses `_random_shift_pad` from `vision_actor_critic` | ~250 |
| `tasks/pickplace/agents/pretrained_ppo_cfg.py` | `PickPlaceBowlPretrainedPPORunnerCfg` with §9.4 hyperparams, `class_name="PickPlaceResNetActorCritic"`, encoder-LR field consumed by the train script | ~80 |
| `tasks/pickplace/pretrained_env_cfg.py` | `PickPlaceBowlPretrainedEnvCfg(PickPlaceBowlEnvCfg)` overriding `CurriculumCfg` per §9.5 | ~40 |
| `tasks/pickplace/__init__.py` (one new `gym.register`) | new task ID `Isaac-SO-ARM101-PickPlace-Bowl-Pretrained-v0` → new env-cfg + new agent-cfg | ~10 |
| `scripts/rsl_rl/train.py` | **unchanged** — trunk-freeze removes the need for a 2-group optimizer; supersedes the original "additive 2-group branch" row (§9.1 update) | 0 |

Strictly **not modified** (so the §7.5 verified pipeline runs identically after this PR): `pickplace_env_cfg.py`, `joint_pos_env_cfg.py`, `mdp/*`, `agents/vision_actor_critic.py`, `agents/vision_student_teacher.py`, `agents/rsl_rl_ppo_cfg.py`, `agents/teacher_ppo_cfg.py`, `agents/distill_cfg.py`. The new task ID has its own env / agent / log directory and shares zero state with `Isaac-SO-ARM101-PickPlace-Bowl-{Teacher,Student}-v0` or the Stage-3 ID.

### 9.7 Workflow

```bash
# Cold-start vision PPO with ImageNet-pretrained ResNet-18 encoder.
# Distinct task ID from the §7 Stage-3 path → separate log dir, no symlink dance.
train --task Isaac-SO-ARM101-PickPlace-Bowl-Pretrained-v0 \
      --headless --num_envs 1024
```

No `--load_run`, no `--teacher_ckpt`, no Stage 2 prerequisite. Single command, single run.

### 9.8 Success criteria and diagnostic milestones

| Iter | Check | Read |
|---|---|---|
| ~200 | `lifting_object` term firing, mean reward > 0 | If flat, the encoder hasn't routed grasp-relevant features yet — wait, do *not* intervene |
| ~800 | `release_in_bowl` firing on any episode | First evidence the spatial-softmax keypoints feed the head usefully |
| ~2000 | `success_rate ≥ 0.3` | Pace check — §7.5 Stage 3 hit ~0.75 at iter 138; cold-start at iter 2000 should at minimum be off zero |
| ~4000 | `success_rate` vs §7.5's 0.91 | Experiment outcome |

**Outcome interpretation:**
- **≥ 0.85 SR by iter 4000** → pretrained encoder is a viable cold-start path. Consider promoting for Eval 2/3 where teacher distillation cost scales with task variety (each new object class would otherwise need its own state teacher).
- **0.4–0.8 SR plateau** → encoder partially helps but the §7.1 critic-shock / exploration problem remains. Diagnostic next step: keep §9 encoder, add *only* intervention #5 (teacher-critic overlay) from §7.2 — no actor warm-start, no distill. If that closes the gap, the attribution is "critic was the dominant bottleneck, encoder was secondary."
- **< 0.3 SR by iter 4000** → pretrained features alone don't bridge the §7-fixed gap. §7.1 conclusion ("optimization dynamics, not perception, was load-bearing") stands. §7 remains the production path; §9 is filed as a recorded negative result.

### 9.9 Out-of-scope by design

- **"Pretrained encoder + teacher warm-start"** — a separate experiment that would test "does pretrained accelerate the already-working pipeline." Worth running if §9 yields ≥ 0.5 SR; not part of this plan.
- **"Pretrained encoder + ManiSkill3 reward / γ = 0.9"** — a different hypothesis (recipe transfer). Cleanest version would drop `release_in_bowl=30` to a sparse success term and set `γ = 0.9`; out of scope here because we're isolating the encoder, not the MDP.
- **Real-world deploy under §9.** Defer until §9 sim SR is within ~0.05 of §7.5 and §8 visual-DR is layered in. Same gating as §8 step 2–3.

## 10. Vision BC on real-robot demos — pre-deploy baseline + actor-CNN pretrain

> **Status: parallel side-track to §7. Does not touch the §7 critical path.** §10 is a self-contained training pipeline that consumes real teleop demos (LeRobot v3 format, on-disk at `demonstrations/RobotLearning-RL/Eval1/`) and produces (a) a deployable real-arm wrist-vision BC policy as a safety-net baseline for eval day, and (b) a §3-shape-compatible image encoder checkpoint that can later be inflated into the Stage-2/Stage-3 vision actor as an additional warm-start signal. The §7.5-verified Stage 1–3 pipeline remains the production path; §10 runs in parallel and ships independently.

### 10.1 What this experiment isolates

Two distinct questions, one training pipeline:

1. **Baseline question** — *what success rate can supervised cloning of 24 teleop episodes hit on the real arm, with no sim involvement at all?* This is the floor the §7 sim→real pipeline must beat to justify itself. ManiSkill3 PickCube on SO-100 hit 91.6 % with sim PPO; on small teleop sets (≤ 100 episodes) the BC baselines in that literature sit at 0.3–0.6 SR. We expect similar — the value is the *number*, not the model.
2. **Encoder-warmstart question** — *does the §3 actor-CNN, pretrained on real images via BC, converge faster or higher in Stage 3 than the from-scratch CNN?* Single intervention, additive to §7.2 (would become "intervention #6: load `actor_cnn.*` from BC checkpoint, channel-inflated"). Tested only if §10 baseline is competent (≥ 0.4 SR) and Stage-3 baseline plateaus below §7.5 numbers — otherwise it's noise.

Both questions are addressed by the *same* trained encoder; the only difference is what we do with the checkpoint downstream.

### 10.2 Data

LeRobot v3 dataset, recorded on the real `so_follower` arm. Two pilots:

| Pilot | Episodes | Frames | Per-ep length | Distinct bowl xy |
|---|---|---|---|---|
| `eval1-pick-place-pilot`   | 13 | 6 283 | 380–782 (≈ 13–26 s) | 5 |
| `eval1-pick-place-pilot-2` | 11 | 5 176 | similar | 5 |
| **Total used by §10** | **24** | **11 459** | 30 Hz | **~10** |

Per-frame fields (only): `action[6]` (deg), `observation.state[6]` (deg), `timestamp`, `frame_index`, `episode_index`. Per-episode: `target_x, target_y, target_z` (m, robot base frame) from `meta/episode_targets.csv`. Two 720 × 1280 H.264 videos per episode: `observation.images.{wrist,top}`.

**§10 uses wrist only.** Top is left on disk; if a future iteration wants two-cam, drop the channel split into `bc/dataset.py` and the head — no other changes.

**Three properties of the data the design has to handle:**

1. **No object pose in `observation.state`.** Joint angles + a per-episode target are all the state inputs the policy sees. The cube's location is implicit in the image. State-only BC was rejected on this basis (§"vision is needed" call in conversation). The bowl xy *can* be made a state input, since `episode_targets.csv` gives it — but doing so leaks per-episode-constant info that wouldn't help at deploy (deploy knows the bowl xy too; conditioning on it is fine and we do).
2. **Actions are absolute joint position targets, not deltas.** `|a − s|` mean is 1–3°/step; the demo is essentially position control at 30 Hz with the human commanding the next-step target. BC matches: predict next absolute joint pos.
3. **Units = degrees.** Sim uses radians. Conversion is a flat `× π / 180` at load if we ever warm-start the sim actor MLP from this checkpoint (§10.3 design note). For real deploy nothing converts — the BC policy outputs deg and the real arm consumes deg.

**Train / val split.** Hold out 4 episodes as val (last 2 from each pilot), train on the remaining 20. This keeps both pilots represented in val and gives ~1900 frames of validation. No shuffle within an episode — each frame is one sample, sampling is uniform per epoch over the union of train-episode frames.

### 10.3 Architecture — §3-compatible encoder + state-conditioned MLP head

Mirror §3's `_SpatialSoftmaxCNN` *exactly* (same kernel sizes, channels, output dim, LayerNorm head) so weights can later inflate-load into `PickPlaceVisionActorCritic.actor_cnn.*`. The only deliberate deviation from §3 is the first conv's `in_channels` — RGB is 3, not 5 — handled by channel-inflation at warm-start time (`conv1[:, :3] = bc_conv1`, channels 3–4 init from the RGB mean per §9.3 choice #2).

```
wrist_image (3 × 72 × 128, real RGB, resized + /255) ──[Spatial-Softmax CNN, §3]── 128-D keypoints ─┐
state_policy (joint_pos_norm[6] + bowl_xy_norm[2]) ─────────────────────────────────── concat ── MLP[256, 128, 64] ── action_norm[6]
```

Two non-obvious points:

1. **`state_policy` includes `bowl_xy` (normalized).** This is the only per-episode condition the policy needs to *do the right thing* — the cube is found by vision, but the bowl target is not visible in the wrist FOV. The deploy script supplies bowl xy the same way the sim env does (`--bowl_xy x y`), so this is in-distribution at test time. `joint_pos_norm` uses the per-dataset min/max from `meta/stats.json`. The 6 joint positions are also part of `observation.state` for the sim actor (§2's `joint_pos_rel`), so this matches the sim policy's state input *modulo* unit and normalization scheme.
2. **No DrQ in BC.** §3 applies DrQ random-shift to the actor image *during PPO training*. For BC we skip it — the random shift is a noise-injection regularizer that helps an on-policy RL gradient escape local optima, but for an offline regression objective it just slows convergence. Apply DrQ in Stage 3 PPO (already wired in `_encode_actor`) — that's where it earns its keep.

The image head is shape-identical to §3 except for `in_channels`. If §10's encoder ever warm-starts §7's actor, the only key rewrite is `actor_cnn.conv.0.weight` (5×3×8×8 instead of 3×3×8×8) and the depth/mask init for channels 3–4. The other 6 weight tensors load with `strict=False` 1:1.

### 10.4 Training config

| Hyperparam | Value | Note |
|---|---|---|
| Optimizer | AdamW, β=(0.9, 0.999), wd=1e-4 | Standard for vision regression |
| LR | 3e-4 with 5-epoch linear warmup → cosine to 0 | Conservative; 24 episodes is small |
| Batch size | 64 | Fits comfortably with ResNet-free encoder + 128×72 input |
| Epochs | 60 | ~180 iters/epoch × 60 = ~11k steps. Validation MAE flattens by epoch ~40 in similar BC runs |
| Loss | **L1** on normalized action (`(a − a_min) / (a_max − a_min)` per-joint, scaled to `[−1, 1]`) | L1 over MSE: teleop has gripper-snap discontinuities that an MSE gradient over-weights. ACT/lerobot use L1 for the same reason |
| Image preprocess | center-crop to 720×720, resize to 128×72, `/255` | Wrist FOV is the relevant ROI; cropping discards the side bezels |
| Image aug | ColorJitter brightness ±0.15, contrast ±0.10, saturation ±0.10 | Mirrors §5 photometric envelope. No random shift (see §10.3 point 2) |
| Action repr | 6-DOF absolute joint pos target (deg). Normalize to `[−1, 1]`; output via `tanh`; unnormalize at inference | tanh keeps the action bounded so a wild gradient can't issue a 200°/step command on the real arm. Per-joint min/max from `meta/stats.json` (queried by `bc/utils/normalize.py`) |
| Compute | ~30 min / RTX-class GPU | 11.5k frames × 60 epochs ≈ 700k samples; CNN forward+back ~3 ms; ~35 min wall |

### 10.5 Evaluation

Three eval modes, in increasing cost:

1. **Held-out frame MAE.** Per-joint L1 on the 4 val episodes. Sanity floor; should reach < 2°/joint for shoulder/elbow, < 4° for gripper (gripper snaps account for the looser bound). Logged every epoch.
2. **Open-loop trajectory replay.** Step through a val episode one frame at a time, predict next action from `(state_t, image_t)`, compare against ground-truth action sequence as a whole. Per-episode RMS joint-trajectory error. Catches early-divergence failures the per-frame MAE hides.
3. **Closed-loop real-arm rollout.** Deploy the checkpoint on the real arm, 20 trials across the 10 demo-bowl positions + 10 unseen-bowl positions, report SR. This is the actual baseline number for the §10.1 question. **Gated on (1) and (2) being clean** — no closed-loop trial unless val MAE is under target.

Sim eval is *deliberately* not in this list — adding it requires either (a) running the sim env with the BC policy's degree-based output + a unit converter (works but adds a sim-side adapter that's only used here) or (b) retraining the BC in radians (changes the deploy-side path). Defer until §10's number is in.

### 10.6 Implementation surface — all-new files, zero edits to §1–§9 code paths

| New file | Purpose | Lines (est.) |
|---|---|---|
| `bc/__init__.py` | package marker | 0 |
| `bc/dataset.py` | LeRobot v3 wrist-only reader (PyAV video decode, parquet read, train/val split, action normalize) | ~180 |
| `bc/model.py` | `WristBCPolicy` — §3-compatible `_SpatialSoftmaxCNN` (RGB 3-ch) + state-MLP head with tanh action | ~120 |
| `bc/normalize.py` | load per-joint min/max from `meta/stats.json`, normalize/denormalize helpers | ~50 |
| `bc/train.py` | argparse, AdamW + cosine warmup, L1 loss, ColorJitter aug, val every epoch, ckpt to `run_logs/bc_eval1_wrist/` | ~220 |
| `bc/eval_openloop.py` | closed-form trajectory replay on a val episode, dumps per-joint RMS to TB / json | ~80 |
| `bc/inflate_to_sim_actor.py` | one-shot util: read BC ckpt, channel-inflate `conv.0`, save sim-actor-compatible state-dict for §7.2 intervention #6 | ~60 |

Strictly **not modified**: `isaac_so_arm101/**`, `EVAL1_PLAN.md` sections 1–9. The new `bc/` dir lives at repo root and depends only on `torch`, `torchvision`, `av` (PyAV), `pyarrow`, `pandas`, `numpy`.

### 10.7 Workflow

```bash
# One-time: install PyAV for H.264 decode.
pip install av

# Train.
python -m bc.train \
    --datasets demonstrations/RobotLearning-RL/Eval1/eval1-pick-place-pilot \
               demonstrations/RobotLearning-RL/Eval1/eval1-pick-place-pilot-2 \
    --output run_logs/bc_eval1_wrist \
    --epochs 60 --batch-size 64

# Open-loop val.
python -m bc.eval_openloop \
    --ckpt run_logs/bc_eval1_wrist/best.pt \
    --val-episodes 11 12 9 10

# Inflate to sim-actor compatible ckpt (only run if §10 baseline ≥ 0.4 SR and Stage-3 plateaus).
python -m bc.inflate_to_sim_actor \
    --bc-ckpt run_logs/bc_eval1_wrist/best.pt \
    --out run_logs/bc_eval1_wrist/sim_actor_init.pt
```

### 10.8 Success criteria

| Milestone | Target | Read if missed |
|---|---|---|
| Val per-joint MAE (shoulder/elbow) | < 2° | Model under-capacity or unit/normalization bug |
| Val per-joint MAE (gripper) | < 4° | Expected — gripper snaps; only diagnose if > 8° |
| Open-loop traj RMS over a val episode | < 10° on first 5 s, < 25° to episode end | Compounding error grows; full < 25° is "fine for warm-start", > 40° is broken |
| Real-arm SR (closed-loop, 20 trials) | ≥ 0.3 on seen-bowl, ≥ 0.15 on unseen-bowl | If ≥ 0.5, this is a legitimate fallback policy for eval day. If < 0.15 seen-bowl, the dataset is too small for closed-loop and §10 reverts to "encoder warm-start only" |

### 10.9 Out-of-scope by design

- **Action chunking (ACT / Diffusion Policy).** Predicting an action *chunk* would improve smoothness and may lift SR by 5–15 % on small teleop sets (lerobot ACT result), but doubles model complexity and changes the deploy-time inference loop. Skip for v1; revisit only if single-step BC is the bottleneck.
- **Top-camera input.** Adding the top cam would require either a second sim camera (§3 modification) or accepting that the BC policy and sim actor have different visual inputs (which kills the §10.1 question 2 warm-start use). Punt until wrist-only is benchmarked.
- **DA3 depth + HSV mask channels (matching §8's 5-ch sim input).** Would let the BC checkpoint load into the sim actor with no channel inflation. Adds DA3 inference + mask threshold to the offline pipeline (~10 min one-shot). Worth doing only if §10.1 question 2's warm-start helps Stage 3 — i.e., do it *second*, not *first*.
- **Fine-tuning a pretrained backbone (R18-ImageNet, R3M).** §9 already tests this for cold-start RL. For BC on 11k frames, a from-scratch spatial-softmax CNN is the right capacity match; pretrained features would overshoot.
- **Sim eval of the BC policy.** Adds a sim-side adapter that's only used by §10 (BC outputs deg, sim consumes rad; BC state-policy normalization differs from sim `joint_pos_rel`). Defer until the deploy-side number is in — closed-loop real is the metric that matters for §10.1.

## 11. Real-demo BC as actor warm-start for vision PPO (§10 ↔ §7 bridge)

> **Status: in-progress, this is what we're trying now.** Builds on the §10 BC pipeline. Replaces §7 Stage 2 (DAgger from state teacher) as the actor warm-start source. Critic warm-start still follows §7.2 intervention #5 (load teacher's critic). One delta from §7: the actor learns from teleop demos rendered through the sim camera, *not* from the state teacher. Hypothesis: that wider data distribution lets PPO recover from a basin §7 Stage 2 doesn't reach.

### 11.1 Empirical reason §11 exists (failures we burned through)

Three iterations of §10 BC, all evaluated in sim playback:

| Version | Train data | Aug | Sim playback observation |
|---|---|---|---|
| v1 | real images only | mild color jitter | arm moves slightly, no grasp, no reach |
| v2 | real images only | aggressive (hue/grayscale/blur/DrQ ±4px) | arm moves more, still no grasp, ignores cube |
| v3 | real + sim images, 50/50 | aggressive (same as v2) | arm just replays demo trajectory keyed on `bowl_xy`, ignores cube position entirely |

The diagnosis from v3 is the load-bearing finding: with **24 demos**, each pairing one `bowl_xy` with one specific cube position, the BC objective is *solvable* without using the image at all — the model can learn `action ≈ f(state, bowl_xy)` and ignore pixels. Even with sim-mix forcing the encoder to tolerate sim renders, the *behavior* is still image-blind. This shortcut isn't a bug to fix in BC; it's a property of the dataset.

A separate bug compounded v3: the sim renderer pinned the cube to its starting xy for every frame, including post-grasp frames where the real demo had the cube *held by the gripper*. So the post-grasp sim images showed "cube on table, arm at bowl" while the demo action labels described "place the cube into the bowl". BC saw inconsistent training pairs and the visual exposure was partially wasted.

§11 fixes the renderer and uses BC as a **warm-start** rather than a deployable policy — letting PPO's exploration + critic-grounded objective break the (state, bowl_xy) shortcut that BC alone can't escape.

### 11.2 Pipeline

```
real demos (LeRobot v3)
        │
        ├─ bc.render_sim_demos (FIXED — §11.3)
        │     ├─ pre-grasp: cube pinned at FK-estimated initial xy
        │     └─ post-grasp: cube tracks gripper world position
        │
        ├─ bc.train --sim-mix-prob 0.5  (v4 — corrected renders)
        │     output: best.pt    (3-ch wrist + state-MLP, §10.3 arch)
        │
        ├─ bc.inflate_to_sim_actor  (§11.5)
        │     pads conv1 3 → 5 channels
        │     RGB weights → channels 0–2
        │     channels 3–4 (depth + mask) init from RGB-mean
        │     output: actor.pt    (sim-actor-compatible state-dict)
        │
        │     ┌────────────────────────────────────────────┐
        │     │  load actor.pt into PickPlaceVisionActorCritic  │
        │     │  load Stage-1 teacher's critic.* into critic   │
        │     │  (§7.2 intervention #5, unchanged)             │
        │     └────────────────────────────────────────────┘
        │                       │
        │                       ▼
        └─ §7 Stage-3 vision PPO (warm-started from §11, NOT §7 Stage 2)
                §7.5 verified hyperparams, no changes to that path
```

The Stage-3 entry point and env are *identical* to §7.5 — only the actor checkpoint source changes. `PickPlaceVisionActorCritic.load_state_dict()` already handles inflated 5-channel `actor_cnn.*` keys; the inflation in §11.5 produces exactly that format.

### 11.3 Renderer fix (§10's `bc/render_sim_demos.py`)

Single behavioral change: cube placement during replay.

```python
# old (v3, buggy): cube pinned for all frames
for t in range(T):
    cube.write_root_pose_to_sim(initial_cube_pose, env_ids=env_ids)
    robot.write_joint_state_to_sim(...)
    env.step(zero_action)

# new (v4): pre-grasp pinned, post-grasp tracks gripper
gframe = _grasp_frame(state_deg, thresh)
for t in range(T):
    robot.write_joint_state_to_sim(state_full_t[t : t + 1], ...)
    if t < gframe:
        cube_pose_t = initial_cube_pose
    else:
        # Use the post-write FK from the previous step's body_pos_w. One-frame
        # lag is acceptable — gripper motion between frames is ≤ a few cm.
        grip_world = robot.data.body_pos_w[0, grip_body_idx]
        cube_pose_t = build_cube_pose_at(grip_world)
    cube.write_root_pose_to_sim(cube_pose_t, env_ids=env_ids)
    env.step(zero_action)
```

Why one-frame lag is OK: gripper translation per 1/30 s ≈ 3 cm peak in the lift phase, ≤ 1 cm in steady state. The cube can lag the gripper by ≤ 3 cm in one frame, which is well under the cube's ~2 cm radius — the cube still visually overlaps the gripper jaw in the image.

### 11.4 BC v4 — same architecture, corrected renders

- Architecture: unchanged (`WristBCPolicy`, §10.3, 3-ch RGB + state-MLP).
- Train data: real + sim (corrected) mix, `sim_mix_prob=0.5`.
- Aug: same as v2 (aggressive jitter / hue / grayscale / noise / DrQ ±4 px).
- Optim: same as v3 (AdamW 3e-4, cosine, 80 ep, batch 64, L1 normalized).
- Val: real images only — apples-to-apples with v1/v2/v3 metrics.

Acceptance for v4 (before proceeding to §11.5):
- Val MAE mean ≤ 4° (in line with v2/v3 — sim mix should not regress real perf).
- Sim playback shows visible **reach-toward-cube** behavior (the diagnostic that v3 failed). Grasp is a bonus, not a requirement — PPO is expected to recover the grasping skill.

### 11.5 Inflation 3 → 5 channels

The §3 `_SpatialSoftmaxCNN` expects `in_channels=5`; BC trained with `in_channels=3`. Conv weights for layers 2 + 3 (32 → 64, 64 → 64) and LayerNorm + grids are shape-identical and load directly. Only `conv.0.weight` needs reshaping:

```python
bc_w = bc_state_dict["cnn.conv.0.weight"]                    # (32, 3, 8, 8)
sim_w = torch.zeros(32, 5, 8, 8, dtype=bc_w.dtype)
sim_w[:, :3] = bc_w                                          # RGB channels: copy
sim_w[:, 3] = bc_w.mean(dim=1)                               # depth channel: init from RGB mean
sim_w[:, 4] = bc_w.mean(dim=1)                               # mask channel: init from RGB mean
```

Same recipe as §9.3 design choice #2 — preserves the activation statistics of the new channels at the BC's converged scale. State-MLP weights load 1:1 (state dim is identical: `joint_pos_norm(6) + bowl_xy_norm(2) = 8`, matching §10.3).

Output: `actor.pt` with state-dict keys remapped to `PickPlaceVisionActorCritic`'s convention (`actor_cnn.conv.0.weight`, etc.).

### 11.6 Stage-3 PPO with BC warm-start

No new gym ID, no new agent class. Run `scripts/rsl_rl/train.py` against `Isaac-SO-ARM101-PickPlace-Bowl-v0` with the §7.5 hyperparams unchanged. The only delta from §7.5 is the source of the checkpoint passed to `--load_run` (or via direct ckpt-arg path):

- §7.5: distilled student from Stage 2 DAgger.
- §11: inflated BC actor + teacher critic, merged at load time.

`PickPlaceVisionActorCritic.load_state_dict()` already detects whether keys are in `actor_cnn.* / actor.*` form (warm-start) or `student_cnn.* / student.*` form (distill); §11's checkpoint uses the former.

### 11.7 Success criteria

| Stage | Milestone | Target | If missed |
|---|---|---|---|
| Renderer v4 | sim images show cube tracking gripper post-grasp | qualitative: cube visible in gripper jaws in frame > gframe | revisit lag handling / write-order |
| BC v4 train | val MAE mean | ≤ 4° on held-out real | augmentation regression — revert to v3 aug |
| BC v4 sim playback | arm reaches toward cube (not just demo trajectory) | qualitative pass | BC shortcut still dominant — proceed to PPO anyway and rely on exploration |
| Stage-3 PPO | `lifting_object` reward firing | by iter ~200 | warm-start is hurting; try cold-start from §11 actor without critic load |
| Stage-3 PPO | `success_rate` | ≥ 0.3 by iter ~1500 | match §7.5 fallback ladder (F1–F4) |
| Stage-3 PPO | `success_rate` vs §7.5 | match or beat 0.91 | log negative result, §11 filed as "real-demo warm-start no better than DAgger" |

### 11.8 Out-of-scope by design

- **BC alone in sim** — three §10 iterations have established that 24 teleop demos cannot solve the eval-1 MDP in sim. §11's role for BC is warm-start only.
- **Mixing teacher distill + BC warm-start** — would test whether the two warm-start sources are complementary. Possible follow-up; out of scope for first §11 run.
- **Reverting to §7 Stage 2 if §11 fails** — already proven in §7.5, no need to re-validate. §11 fails → fall back to §7.5 checkpoint for the eval-1 deliverable.

## 12. Teacher-free vision PPO from real-demo BC warm-start

> **Status: this replaces §11 as the production BC pipeline.** §11 used the §7 Stage-1 state teacher's critic via `--teacher_ckpt` to avoid the random-init advantage shock. §12 **drops that dependency entirely** — no state teacher anywhere in the BC-related path. The actor warm-start stays exactly as in §11 (BC v4 inflated to 5-ch). The critic warm-start is replaced by a self-contained **critic burn-in** stage that produces an equivalent "good initial V" by rolling out the BC actor and training only the value head against on-policy returns.

### 12.1 What §12 changes vs §11

| | §11 | §12 |
|---|---|---|
| Actor warm-start | BC v4 inflated → `actor_cnn`, fresh MLP head | **same** |
| Critic warm-start | `--teacher_ckpt` loads Stage-1 teacher critic | **critic burn-in produces equivalent V^BC ckpt** |
| State teacher in pipeline | yes (provides critic) | **none** |
| Hand-off mechanism | `--teacher_ckpt` overlay in `train.py` | **same** — burn-in ckpt has identical key structure |
| §1–§5 env / §6 PPO hyperparams | unchanged | unchanged |

The hand-off path stays the same so the existing `--teacher_ckpt` flow keeps working; the only delta is the *source* of the critic weights loaded by that flag.

### 12.2 Why the critic warm-start is load-bearing

From §7.1 / §7.2 intervention #5: a freshly-randomized critic produces noisy advantage estimates that destroy a warm-started actor within ~50 PPO iters. §11 confirmed the symmetric finding — with a teacher critic, the BC warm-start trains to 0.73 final / 0.85 peak SR; conversations during the run mapped each `lifting_object`/`over_bowl_high`/`success_rate` transition to expected curves.

Without a teacher, three teacher-free bridges exist (§7.3 fallback ladder + AWAC reference):

| Bridge | What it does | Why we prefer / reject |
|---|---|---|
| **A. Critic burn-in (AWAC)** | freeze actor, roll out, train V(s) only on on-policy returns from BC actor's trajectories | **chosen** — cleanest substitute for teacher; same role (good initial V), same hand-off mechanism, deterministic compute budget |
| B. DAPG auxiliary loss | add `λ(t) · MSE(π(s), a_demo)` to PPO loss; λ decays | viable but requires modifying the rsl-rl PPO algorithm; more invasive |
| C. Conservative PPO | lower `init_noise_std`, tighten `desired_kl` | does *not* fix the critic noise — slows actor drift but doesn't anchor it. Insufficient alone |

§12 commits to **A**; B and C are recorded fallbacks if A's critic-burn doesn't transfer cleanly into Stage 3 PPO.

### 12.3 Critic burn-in pipeline

```
[demos]                                                  (§10)
   │
   ▼
BC v4 (real+sim mix, corrected renderer)                 (§10/§11)
   │   ckpt: bc_v4/best.pt
   ▼
inflate 3-ch → 5-ch                                       (§11.5)
   │   ckpt: bc_v4_warmstart/model_0.pt
   │     - student_cnn.* (channel-inflated)
   │     - student.* (random init; sim MLP shape ≠ BC MLP shape)
   ▼
critic burn-in (§12.4) — NEW
   │   freeze actor, roll out in sim, train V on on-policy returns
   │   ckpt: bc_v4_burnin/critic_K.pt
   │     - critic.* (well-fit to V^BC(s))
   ▼
Stage 3 PPO (§6 hyperparams, unchanged)
   --resume --load_run bc_v4_warmstart --checkpoint model_0.pt
   --teacher_ckpt bc_v4_burnin/critic_K.pt          # ← the only delta from §11
```

### 12.4 Critic burn-in spec (`bc/critic_burnin.py`)

Load the env, instantiate `PickPlaceVisionActorCritic`, load `bc_v4_warmstart/model_0.pt` via the existing `load_state_dict` path (recognizes the `student_cnn.*` keys), then:

1. **Freeze actor**: `for p in policy.actor.parameters() + policy.actor_cnn.parameters(): p.requires_grad = False`. Optionally freeze `std` too — burn-in shouldn't change exploration noise. The critic's parameters (`policy.critic.*`) remain trainable.
2. **Roll out** with stochastic actor (`std=init_noise_std`, NOT `act_inference`) so the burn-in critic learns V over the **same exploration distribution PPO will see at step 1**. Match §6 Stage-3 settings: `num_envs=1024`, `num_steps_per_env=16`.
3. **Compute GAE returns** with `γ=0.98`, `λ=0.95` (matches §6). Use the *current* critic to bootstrap — this is fine because we update it every step.
4. **Critic-only update**: `optim = AdamW(policy.critic.parameters(), lr=1e-3)`, loss `= MSE(V(s), G_t)`. Larger LR than PPO's 1e-4 because we're not balancing against an actor gradient.
5. **Run for `K` iters** until `value_loss` plateaus. Expected `K = 50–100` based on AWAC pattern; we'll monitor and stop early if `value_explained_var` clears 0.7.
6. **Save** as `bc_v4_burnin/critic_K.pt` with `model_state_dict` containing only `critic.*` keys (compatible with `train.py`'s `--teacher_ckpt` filter at line 235).

Implementation surface: ~180 lines, single new file `bc/critic_burnin.py`. Reuses the env-construction path from `bc/render_sim_demos.py` and `bc/play_in_sim.py`. No edits to existing agent code.

### 12.5 Stage 3 PPO entry

After burn-in:

```bash
OMNI_KIT_ACCEPT_EULA=YES train \
    --task Isaac-SO-ARM101-PickPlace-Bowl-v0 \
    --enable_cameras --headless --num_envs 1024 --resume \
    --load_run bc_v4_warmstart --checkpoint model_0.pt \
    --teacher_ckpt run_logs/bc_eval1_wrist_v4/critic_K.pt
```

Identical to §11's command except the `--teacher_ckpt` path. Boot signals should be the same three lines (`Loading model checkpoint`, `Warm-started from distillation checkpoint`, `Teacher critic loaded — overlaid 8 critic.* keys`), with the last one wording slightly stale (the loaded weights now come from burn-in, not from the state teacher — message text unchanged).

### 12.6 Success criteria

| Stage | Milestone | Target | If missed |
|---|---|---|---|
| Critic burn-in | `value_loss` plateaus | < 0.1 by iter ~80 | extend K to 150; if still high, the BC actor may not visit success states often enough — fall back to DAPG aux loss (§12.7 option B) |
| Critic burn-in | explained-var of V | ≥ 0.7 | same as above |
| Stage 3 PPO | first 20 iters | reward not collapsing (stays > BC's natural reward) | burn-in critic wasn't good enough; try B (DAPG aux loss) |
| Stage 3 PPO | `success_rate` by iter ~200 | ≥ 0.3 | same as §11 |
| Stage 3 PPO | final `success_rate` | ≥ §11's 0.73 | if below, §12 is a regression vs §11 → record as negative result, fall back to §11 with documented "needs teacher" caveat |

The fair comparison is **§12 final SR vs §11 final SR (0.73)**, not vs §7.5 (which used DAgger). If §12 matches or beats §11, the teacher-free path is validated for the BC pipeline; the state teacher can be permanently dropped from the BC-related deliverable.

### 12.7 Fallback if critic burn-in is insufficient

**B. DAPG auxiliary loss** (Rajeswaran 2018) — modify the rsl-rl PPO update to include `λ(t) · MSE(π(s_demo), a_demo)` over a buffer of demo (s, a) pairs. λ decays exponentially from 1.0 → 0.01 over 500 iters. Anchors the actor inside the imitation basin while the critic learns. Implementation: subclass `rsl_rl.algorithms.PPO` with one extra term in `update()`. ~80 lines.

> **Caveat: action-space mismatch.** BC's `action` is absolute joint-pos in deg with a tanh-bounded [-1, 1] normalization; sim's policy outputs are interpreted through `JointPositionActionCfg(scale=0.5, use_default_offset=True)` plus a binary gripper. Direct regression `π_sim(s_demo) ↔ a_demo` is not well-defined without remapping (the inflated 5-ch BC policy in `bc_v4_warmstart/model_0.pt` has a *random* MLP because shapes don't match — §11.5). DAPG-style aux loss therefore needs a state-and-action adapter, which is itself non-trivial. Recorded but not yet attempted.

**C. Conservative PPO + small std** — `init_noise_std=0.3` (down from 0.5), `desired_kl=0.005` (down from 0.01), `learning_rate=5e-5` for first 200 iters. Doesn't fix the critic noise problem fundamentally; treat as a stop-gap if B/A are also failing.

### 12.9 Results (2026-05-18)

Two §12 runs, both falling short of §11's 0.73 final / 0.85 peak SR baseline:

| Variant | Setup | Outcome |
|---|---|---|
| **§12-v1**: burn-in critic + §6 std=0.5 / LR=1e-4 / KL=0.005 | std/LR/KL from §6 Stage-3, burn-in critic from `bc_v4_warmstart` rollouts at EV=0.83 | Reward climbed 0.0 → 7.8 by iter ~50 (matched §11's curve), then collapsed: 7.8 → 4 → 1 → 0.03 by iter ~120. Classic §7.1 critic-shock failure mode despite the burn-in. SR briefly reached 0.016 then went to 0 permanently |
| **§12-v2**: same + conservative cfg (`init_noise_std=0.20`, `learning_rate=3e-5`, `desired_kl=0.003`, `entropy_coef=0.003`) via Hydra overrides | C-style mitigation on top of v1 | Reward climbed 0.0 → 10 over ~50 iters; then *stalled* in lift basin (reward oscillating 3–8) without consistent SR > 0. Two brief SR spikes ≈ 1.5%. Did NOT collapse like v1, but did NOT cross the place-phase threshold either |

**Interpretation.** The burn-in critic fits V well on the BC actor's rollout distribution (EV 0.83 at end of burn-in), but the BC actor — with its inflated CNN + *random-init* sim-shape MLP — produces a rollout distribution that **never visits successful release-into-bowl states**. So `V` is well-fit on reach + occasional lift, but is essentially unlearned on the place phase. When PPO updates the actor, the advantages it computes for "actor moved toward place region" are based on an uninformed V — producing noisy / wrong gradients that either (v1) push the actor out of the imitation basin entirely, or (v2) prevent it from converging on the place skill.

The teacher critic (§11) does not have this problem because the §7 state teacher was *itself trained to release successfully* — its V naturally encodes how good place-region states are. The burn-in critic is fundamentally limited by the rollout distribution it observes.

**Negative result, filed.** Real-demo BC + critic burn-in + PPO does not reach §11's SR on this task without a state-teacher critic. The user's stated goal — "BC and PPO only, no state teacher" — appears to require one of:

- A **better critic-bootstrap distribution**, e.g., scripted demonstrators that *do* place successfully (effectively a different teacher);
- An **action-space-aware DAPG variant** (§12.7 B) — defer until the state+action adapter is built;
- Acceptance of the lower SR ceiling for the teacher-free path, and treating §10 BC as the deployable for real-arm and §11 (with teacher critic) as the deployable for any sim-RL fine-tuning.

### 12.10 What still works without a state teacher

- **§10 BC v4 (real+sim mix, corrected renderer)** — converges cleanly on real-demo val (val MAE 3.36°, open-loop RMS 5°), runs on the real arm directly. *No teacher anywhere in this pipeline.* This is the recommended deployable for the BC-only constraint.

The state teacher was load-bearing for the BC → vision-PPO transition specifically. For the BC pipeline ending at deploy on real hardware (which is what eval-1 grades), §10 alone is the artifact.

### 12.8 Out-of-scope by design

- **Any state teacher in the BC pipeline** — explicitly removed. The §7 state teacher remains valid for the §7.5 production path; §12 is a separate track that produces a deployable policy *without* training the teacher first.
- **Distillation from any teacher** — DAgger / RPD / PPD all assume a teacher source. Skip.
- **Multi-stage curriculum changes** — keep §5's Stage-3 settings unchanged; §12 isolates the critic-warm-start variable.

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

### Pretrained backbones (§9 alternative path)

- **He et al.**, *Deep Residual Learning for Image Recognition*, CVPR 2016. <https://arxiv.org/abs/1512.03385>. ResNet-18 / `torchvision.models.resnet18(weights=IMAGENET1K_V1)` — the ImageNet-pretrained backbone in §9.2.
- **Wu & He**, *Group Normalization*, ECCV 2018. <https://arxiv.org/abs/1803.08494>. The BN→GN replacement that makes ResNet's `BatchNorm2d` running stats survive PPO's non-stationary rollouts (§9.3 design choice #1).
- **Nair et al.**, *R3M — A Universal Visual Representation for Robot Manipulation*, CoRL 2022. <https://arxiv.org/abs/2203.12601>. ResNet-50 pretrained on Ego4D with a manipulation-aligned objective. Upgrade path in §9.2 if R18-ImageNet plateaus before iter 2000.
- **Egbe et al.**, *DINOv3-Diffusion Policy*, 2025. <https://hf.co/papers/2509.17684>. The reference behind §9.2's "DINOv2/v3 wins are documented for IL/diffusion policies, not on-policy RL fine-tune" point.
- **Xiao et al.**, *Masked Visual Pre-training for Motor Control* (MVP), 2022. <https://arxiv.org/abs/2203.06173>. Pretrained ViT encoder used frozen for downstream RL motor-control tasks — the closest existing precedent for "pretrained encoder, single-stage RL" and a useful comparison data point if §9 succeeds.
