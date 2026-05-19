# Eval 3 — Sequential Multi-Step Pick-and-Place

Sequence-conditioned PPO on SO-ARM101 → zero-shot real-arm deploy. Task: `Isaac-SO-ARM101-SeqPickPlace-v0` (scaffolded under `tasks/seqpickplace/`). Four distinct-color cubes on the table; the policy receives an ordered list of three `(target_color, bowl_xy)` sub-goals and must execute them in order. Same three-stage pipeline as Eval 1 / Eval 2 (state teacher → vision distill → vision PPO + teacher critic), re-keyed to a *current* sub-goal that automatically advances on release.

This document is the third in the series; it assumes familiarity with [`EVAL1_PLAN.md`](./EVAL1_PLAN.md) (the canonical reference for the asymmetric A-C handoff, the five Stage-3 interventions, and the wrist-cam intrinsics) and [`EVAL2_PLAN.md`](./EVAL2_PLAN.md) (the ResNet-18 + spatial-softmax encoder, the color-one-hot goal-conditioning pattern, and the wrist-tint DR rationale). Where Eval 3 inherits a piece verbatim, the prose references those docs rather than re-deriving.

## 1. Spec recap (PDF §Eval 3)

Four blocks of distinct colors are placed in the workspace. The policy is given a sequence of three `(target_color, bowl_xy)` goals; bowl positions are fixed within a rollout. Per-step scoring is **4, 4, 2** (10 pts/rollout × 5 rollouts = 50 pts), partial credit is awarded, success = correct block placed in the corresponding bowl and released. **RL is required.** Policy switching / perception-module switching is allowed, and interaction with non-target blocks (pushing, rearranging) is *permitted and encouraged* if it facilitates task completion. Optional 50-pt **bonus** is awarded on either **(A) Speed** — number of all-3-step rollouts completed inside a time limit — or **(B) Singulation** — separating a cluster/stack into individually graspable configurations.

## 2. Does Eval 1 / Eval 2 carry over?

Yes — heavily, but not verbatim. Concrete reuse audit:

| Component | Eval 1 / 2 | Eval 3 | Reuse |
|---|---|---|---|
| Robot, control, action space, home `q` | ✓ | ✓ | **verbatim** |
| Workspace, table dimensions, table color | ✓ | ✓ | **verbatim** |
| Wrist `TiledCamera` mount + intrinsics | ✓ | ✓ | **verbatim** |
| RGB-only image obs (no mask) | Eval 2 | ✓ | **verbatim** (`(N, 3, 72, 128)`) |
| `ResNet-18 (frozen) → 1×1 conv → spatial-softmax` encoder | Eval 2 | ✓ | **shared class** (`_ResNetSpatialSoftmaxCNN`) |
| Reach / lift / transport / release shaping (§4 in Eval 1) | ✓ | ✓ (current-step-aware variants) | **structure preserved**, terms re-indexed |
| Two latches (lift ≥ 0.07, over-bowl-above-rim ≥ 0.08) | ✓ | ✓ (per-step, cleared on advance) | **structure preserved** |
| Asymmetric A-C with privileged critic | ✓ | ✓ | **verbatim** |
| 3-stage pipeline (teacher → distill → vision PPO + teacher critic) | ✓ | ✓ | **verbatim** |
| `--teacher_ckpt` critic overlay (Pinto 2018 handoff) | ✓ | ✓ | **verbatim** |
| Wrist RGB tint + HSV DR | Eval 2 | ✓ (kept wide) | **verbatim** |
| `place_clutter_blocks` two-cube placement | Eval 2 | replaced with `place_seq_blocks` (4 cubes spread) | **new event** |
| `TargetColorCommand` (single target) | Eval 2 | replaced with `SequentialGoalCommand` (current-of-3) | **new command** |
| State-teacher checkpoint transfer Eval 1 → Eval 3 | — | **does not transfer** | wider state schema |

Net: every infra piece survives; the deltas are (i) the command term that owns a 3-step schedule and self-advances on release, (ii) the placement event that puts 4 of 6 palette cubes on the table and samples 3 bowl positions, (iii) reward terms re-targeted via `_seq_step_idx`, and (iv) a step-skip curriculum during Stage 1 (§5). The seqpickplace scaffold (`tasks/seqpickplace/`) already contains (i)–(iii) end-to-end; what is missing is `agents/` (PPO cfg, vision A-C cfg, distill cfg) and a state-teacher curriculum to ensure every step gets training coverage.

## 3. MDP

| Item | Value |
|---|---|
| Control | 50 Hz (decimation 2, sim 100 Hz) — verbatim from Eval 1/2 |
| Episode | **15.0 s = 750 steps** (3 × 5-s sub-budget; matches `episode_length_s=15.0` in `seqpickplace_env_cfg.py`) |
| Action | 5 arm joints (absolute around home, `scale=0.5`) + 1 binary gripper (`open=0.5`, `close=0.0`) — verbatim |
| Workspace | cubes: `x ∈ [0.13, 0.22]`, `y ∈ [−0.12, 0.12]`, min pairwise separation 5 cm; bowls: `x ∈ [0.15, 0.28]`, `y ∈ [−0.12, 0.12]`, min bowl separation 10 cm |
| Home `q` | `(0, 0, 0, 1.5708, 0, 0)`, gripper open |
| Terminations | `time_out`, `active_block_off_table` (any of the 4 active cubes off-table), **`all_steps_done`** (positive termination when step idx ≥ 3) |
| Table | `0.6 × 1.0 × 0.02 m` at `(0.25, 0, −0.01)`, top `z=0` |

**Scene composition.** Six 2 cm `CuboidCfg` cubes baked with the six palette colors (blue/yellow/purple/orange/green/red), friction tuned to dex_cube grippability — identical to Eval 2. Per reset, `place_seq_blocks` samples a length-4 random permutation of the palette, places those four in the workspace with rejection-sampled pairwise separation, and teleports the other two to `HIDDEN_PARK_XY` slots off-table where the wrist camera can't see them. Three bowls are 2-D goals from `SequentialGoalCommand` (**no scene prim** — same trick as Eval 1/2; the bowls only exist as targets in the command tensor), rejection-sampled to be ≥ 10 cm apart.

**3-step schedule.** Sampled once per episode in `place_seq_blocks`:

- `active_indices: (N, 4) long` — palette indices of the 4 active cubes.
- `goal_color_pos: (N, 3) long ∈ [0, 4)` — for each step, which slot in `active_indices` is the target. Sampling with replacement is allowed (a step can re-target a cube already moved). For evaluation alignment, distractor-free runs (no repeat) are sampled by changing `randint` → `argsort` in `place_seq_blocks`; the spec implies distinct colors per step so we should flip this to **distinct** before final training (one-line fix).
- `goal_bowl_idx: (N, 3) long ∈ [0, 3)` — distinct by default (`distinct_bowls=True`); each step has a unique bowl.
- `bowl_positions: (N, 3, 2) float` — the three bowl xy positions in the robot frame, fixed within an episode (spec: "bowl positions remain fixed within each rollout").

**Step-advancement contract.** `SequentialGoalCommand._update_command` reads `env._seq_step_release_indicator` each tick. When the release reward fires for an env at step k, the command increments `_seq_step_idx[env] += 1` *and* clears the per-step `_seq_was_grasped` / `_seq_was_over_bowl_above_rim` latches so step k+1's gating starts fresh. The command tensor returned by `cmd.command` is then automatically re-composed against the new step.

**Distractor interaction is permitted.** The spec encourages it; correspondingly, the reward stack does **not** punish disturbing non-target cubes outside of the current bowl. Only `wrong_cube_in_current_bowl` (weight −15) penalizes the specific failure mode of "drop the wrong block into the active bowl", which is a real attractor when bowl 2 is near a cube of an unrelated color.

## 4. Observations (asymmetric A-C)

`ObservationsCfg` defines three groups; runner cfgs select per stage (§8). Schema matches `tasks/seqpickplace/seqpickplace_env_cfg.py` verbatim.

| Group | Fields | Shape |
|---|---|---|
| `policy` (deployable) | `joint_pos_rel`, `joint_vel_rel`, `gripper_state`, **`seq_goal_vector` (11-D)**, `ee_proj_xy`, `last_action` | 1-D |
| `critic` (privileged) | `policy` + `all_active_block_positions` (4×3), `current_target_block_position` (3), `current_target_gripper_to_block` (3), `current_target_block_to_bowl_xy` (2) | 1-D |
| `wrist_image` | **RGB only** | `(N, 3, 72, 128)` |

**The `seq_goal_vector` (11-D) is the policy's only sequence-awareness signal.** Composition: `current_target_color_onehot (6) ⊕ current_target_bowl_xy (2) ⊕ current_step_onehot (3)`. After step k completes and the command advances, the same 11-D obs term now reflects step k+1's color + bowl + step-id — no per-policy memory state required, no separate goal-buffer obs. The 3-D step one-hot is *not* redundant with the color/bowl fields: it lets the policy disambiguate "step 0, blue → bowl A" from "step 2, blue → bowl A" (e.g., reach-strategy may differ near the end of the episode where time-to-time-out matters).

**No history / recurrence.** The Markov property is preserved because (i) the per-step latches reset at advancement and (ii) all bowls are visible as static fixtures in the env state via the critic's `all_active_block_positions` and the command's `bowl_positions`. The policy only ever needs to act on the *current* sub-goal — past sub-goals are observable through cube-position state (the policy will see, via the wrist camera, that one bowl now has a cube in it). A recurrent policy is **not** needed.

**No image mask.** Same rationale as Eval 2: the color-one-hot in `seq_goal_vector` carries the discrimination signal; the CNN learns to match pixel colors to the active one-hot index. Wider wrist-tint + HSV DR (§6) is the regularizer that keeps this transferable.

Wrist `TiledCamera`: same mount/intrinsics as Eval 1/2 (`pos=(-0.001, 0.1, -0.04)`, ros quat `(-0.404379, -0.912179, -0.0451242, 0.0486914)`, `["rgb"]` only — no `semantic_segmentation`, no depth).

## 5. Network — `SeqPickPlaceVisionActorCritic`

Same architecture as Eval 2's `ClutterPickPlaceVisionActorCritic` (which inherits Eval 1's pretrained-backbone path): **frozen ImageNet ResNet-18 trunk truncated at `layer3` → trainable 1×1 conv (64 ch) → per-channel softmax → soft-argmax (x, y) → 128-D keypoints**. We instantiate `_ResNetSpatialSoftmaxCNN` from `tasks/pickplace/agents/vision_actor_critic.py` verbatim with `in_channels=3`.

```
wrist_image (3×72×128) ──[ResNet-18 trunk (frozen, ImageNet)]── (256, 9, 16) feat
                       ──[1×1 conv → 64 ch]── softmax → soft-argmax (x,y) → 128-D kpts ─┐
state (policy, incl. seq_goal_vector 11-D) ──────────── concat ── MLP[256,128,64] ── μ (σ scalar Param)
critic state (policy+critic) ─────────────────────────── MLP[256,128,64] ── V(s)
```

DrQ ±4 px replicate-pad-and-crop training-only, in `_encode_actor` (Stage 3) AND `SeqPickPlaceVisionStudentTeacher._encode_student` (Stage 2). Optional BC-v1 weight overlay via `bc_v1_weights_path=` — same hook as Eval 2.

**No architectural changes from Eval 2** — the only delta is that the policy state vector grows from `[…, target_color_onehot(6), …]` to `[…, seq_goal_vector(11), …]`. This is purely an MLP input dim change (+5 ints), no new modules. The four "why frozen ResNet-18" reasons from Eval-2 §3 apply unchanged.

**Wiring note** (same as Eval 2's open item): `PickPlaceVisionActorCritic.__init__` hardcodes `_ImpalaSmallCNN`. Add `cnn_class: str = "resnet"` to the cfg and dispatch — fixing in one place benefits Eval 1 pretrained-backbone runs and Evals 2 and 3.

## 6. Reward (`tasks/seqpickplace/mdp/rewards.py`)

| Term | Weight | Trigger |
|---|---|---|
| `reach_current_target` | 1.0 | `(1 − tanh(‖ee − current_target‖ / 0.05)) · step_active` |
| `lift_current_target` | 15.0 | `𝟙[current_target_z > 0.07] · step_active` |
| `transport_current_target_to_bowl` (coarse) | 16.0 | `was_lifted · (1 − tanh(‖target_xy − current_bowl_xy‖ / 0.30)) · step_active` |
| `transport_current_target_to_bowl` (fine) | 5.0 | same at `std=0.05` |
| `release_current_target_in_bowl` | 30.0 | current target near current bowl ∧ `z<0.06` ∧ gripper open ∧ settled, gated on per-step lift + over-bowl-above-rim latches |
| **`step_completion_bonus`** | **1.0** | one-shot per step, with per-step weights `(4.0, 4.0, 2.0)` matching grading |
| `wrong_cube_in_current_bowl` | −15.0 | any non-target cube sits inside the *current* bowl (xy < 6 cm ∧ `z < 0.06`) |
| `action_rate`, `joint_vel` | −1e-4 → −1e-2 | ramp at 10 k env-steps |

Two per-episode latches (`_seq_was_grasped`, `_seq_was_over_bowl_above_rim`) are **cleared whenever the step advances** (in `SequentialGoalCommand._update_command`), so step k+1's gating is fresh and unaffected by step k's manipulation. The `_seq_step_release_indicator` is also consumed (zeroed) at advance time so we don't double-credit.

**Why a per-step bonus with `(4, 4, 2)` and not just stronger dense terms?** Two reasons. First, it makes the reward signal *grading-aligned*: the value function is regressing the per-rollout score the human grader will write down, which removes a subtle confounder where the policy might prefer high dense-reward trajectories that don't translate to step-completion. Second, the bonus is a one-shot event tied to the same release predicate that advances the step, so it cannot be farmed by hover-and-hold strategies — it pays only at the moment of state transition. Note that the third step's weight is 2 (not 4): step 2 is short and the dense `release_in_bowl=30` already dominates; the spec's scoring rationale (4/4/2) likely reflects that step 0 is hardest from a cold start.

**Why `release_in_bowl` is fired *every* step the predicate holds, not just on the transition.** Inherited from Eval 1: a `task_success` termination would let "hover above bowl with gripper open over the target" beat "release and stay" by an arbitrary margin in the post-release tail. Eval 3 has the same pathology *per sub-goal*; the dense `release_in_bowl=30 · step_active` reward pays every post-release step until either `_seq_step_idx` advances on the same tick or the next step's gating kicks in. With the step advancing on the same tick the release predicate latches, the practical effect is **one large impulse per sub-goal**, then the dense reward repoints at the next target.

**No contact sensor reward.** Same reasoning as Eval 2: target identity varies per step, contact-sensor filtering would need per-step prim-path patching; the kinematic lift latch (`current_target_z > 0.07`) is sufficient and that recipe worked in Eval 1's vision teacher.

**Distractor-disturb sizing.** Unlike Eval 2, there is **no continuous distractor-disturb penalty** — Eval 3 explicitly encourages distractor interaction (push/rearrange to make a grasp possible). The only distractor-related penalty is `wrong_cube_in_current_bowl=-15`, which guards the specific failure of dumping the wrong cube into the active bowl. −15 is comparable to but smaller than the +30 release reward + +4 step bonus, so a correct final placement still wins net even after one misstep, but the gradient pushes away from the failure mode. If during training we observe "the policy parks distractors in the active bowl en route", widen the penalty to −25.

## 7. Curriculum & DR (`tasks/seqpickplace/mdp/events.py`, `CurriculumCfg`)

Both stages:

- `place_seq_blocks`: 4 active cubes with `min_block_separation=0.05`; bowls in `[0.15, 0.28] × [−0.12, 0.12]` with `min_bowl_separation=0.10`, `distinct_bowls=True`.
- `reset_seq_latches`: per-episode step counter + lift/over-bowl latches cleared.
- Action-rate / joint-vel penalty ramp −1e-4 → −1e-2 at 10 k env-steps.
- `log_seq_success_metrics` TB metric — emits `step0_success`, `step1_success`, `step2_success`, `all_steps_success`, `n_episodes_ended`. **These four scalars are the diagnostic surface for the whole project.**

Stage 3 adds:

- DrQ ±4 px (in CNN, see §5).
- `randomize_wrist_image_tint`: RGB scale `(0.55, 1.45)`, brightness `(−0.20, 0.20)` — **wider than Eval 2** (Eval 2 was `0.7, 1.3` / `±0.15`). Justification: with 4 colors on screen at once (not 2), color confusion is the dominant failure mode; we widen tint DR to force the CNN to discriminate by relative palette position, not absolute hue.
- `randomize_wrist_hsv_dr`: hue ±20°, saturation `(0.65, 1.35)`, value `(0.55, 1.45)`. **New vs Eval 2** — explicit hue rotation is what catches lighting that shifts the *direction* of color in HSV space (warm vs cool indoor light). Already wired in the scaffold; cfg numbers are tuned for the four-color regime.

**Step-skip curriculum (Stage 1 only).** This is the *one new piece* in the curriculum that doesn't appear in Eval 1/2. The risk: if the state teacher fails step 0 in 90 % of envs early in training, only 10 % of envs ever see step 1's gradient signal, and we never train step 1 to convergence. Fix: during the first ~30 k env-steps of teacher training, advance any env that hasn't completed its current step after a per-step time budget (e.g., 4 s) by **directly bumping `env._seq_step_idx[stalled_envs] += 1` and clearing the per-step latches** (`_seq_was_grasped`, `_seq_was_over_bowl_above_rim`). Then linearly anneal the time budget upward (4 s → 8 s) over 30 k–60 k env-steps and finally disable the auto-advance entirely past 80 k. **Critical:** the auto-advance must **not** touch `_seq_step_release_indicator` — that buffer is what `step_completion_bonus` reads, and triggering it would pay the policy +4 / +4 / +2 for *failing* to release. The release reward itself is gated on the physical predicate (in_xy ∧ low ∧ opened ∧ settled ∧ was_lifted ∧ was_over_high) computed from a local var in `release_current_target_in_bowl`, so directly bumping the step counter avoids both rewards — exactly the desired "free skip" semantics. **Implementation:** new event term `auto_advance_stalled_steps` in `EventCfg`, mode `"interval"` at e.g. 50 Hz, tracking `env._seq_step_start_time` (set when step advances or on reset); **not yet in the scaffold** — write it under `tasks/seqpickplace/mdp/events.py` alongside `place_seq_blocks`. Toggle the budget via `CurriculumCfg` and `modify_event_param`.

No xy-expand curriculum (Eval 1 used one). Eval 3's placement band is already tight (5 cm separation, ±12 cm), and the policy needs the full 4-cube distribution from step 0 to learn discrimination. A separation curriculum is a fallback if Stage 1 fails to converge.

## 8. PPO config

Same skeleton as Eval 2, with longer training to absorb 3× the step-conditioning entropy.

| | Stage 1 teacher (`teacher_ppo_cfg.py`) | Stage 3 vision PPO (`rsl_rl_ppo_cfg.py`) |
|---|---|---|
| `num_envs` | 4096 | **1024** (same as Eval 2 — PPO stores `num_envs × num_steps_per_env=16` rollout windows, not full episodes; the 15-s episode length doesn't increase VRAM. Critic state grows by only ~6 dims vs Eval 2, immaterial. Drop to 768 only if OOM observed in practice.) |
| `num_steps_per_env` | 24 | 16 |
| `max_iterations` | **3000** (vs Eval 2's 2000 — 3-step value horizon adds variance) | **3500** (vs Eval 2's 2500) |
| `init_noise_std` | 1.0 | 0.5 (forced; distill saves 0.1) |
| hidden dims | `[256, 128, 64]` ELU | same + ResNet-18 spatial-softmax CNN |
| `entropy_coef` | 0.006 | 0.006 → 0.003 (after ~1000 iters; later than Eval 2 because step-conditioning exploration matters longer) |
| epochs / mini-batches | 5 / 4 | 8 / 16 |
| `learning_rate` / `desired_kl` | 1e-4 / 0.01 | 1e-4 / 0.005 |
| `gamma / lam / clip / max_grad_norm` | 0.98 / 0.95 / 0.2 / 1.0 | same |

**`gamma=0.98` is even more load-bearing here.** A 15-s episode at 50 Hz is 750 steps; a step-2 release fires +30 + (+2) ~600 steps after action 0. `γ^600 = 5e-6` at γ=0.98 vs `5e-19` at γ=0.95 — only the former lets the early-episode actions see the late reward through TD(λ). Practically, the value function learns step 2's contribution through Stage 1's teacher critic (which has direct state access and a privileged advantage), then Stage 3 inherits via `--teacher_ckpt`.

```python
# Stage 1: symmetric on privileged state
obs_groups = {"policy": ["policy", "critic"], "critic": ["policy", "critic"]}
# Stage 2 (distill_cfg.py): vision student, state teacher
obs_groups = {"policy": ["policy", "wrist_image"], "teacher": ["policy", "critic"]}
# Stage 3: vision actor, privileged critic (no image to critic)
obs_groups = {"policy": ["policy", "wrist_image"], "critic": ["policy", "critic"]}
```

Stage-1 critic and Stage-3 critic take identical inputs (`policy + critic` MLP) → teacher critic loads layer-for-layer via `load_state_dict(strict=False)`. State schema is *wider* than Eval 2 (4 cubes × 3 pos = 12 dims vs Eval 2's 2 × 3 = 6 dims; seq_goal 11-D vs Eval 2's color one-hot 6-D), so **Eval-2 teacher checkpoints do not transfer**; Eval 3 teacher is trained from scratch.

## 9. Three-stage pipeline

- **Stage 1 — state teacher.** Task `Isaac-SO-ARM101-SeqPickPlace-Teacher-v0` (to register). MLP A-C on `policy + critic`. **Pure PPO on privileged state in sim, no teleop.** The teacher has direct access to all 4 cube positions, the current target's pose, the current bowl xy, and the step one-hot — there is no perception bottleneck and PPO on the §6 reward stack is the right tool. With the step-skip curriculum (§7) ensuring all 3 steps see gradient early, the teacher should reach `all_steps_success ≥ 0.6` in ~1500 iters and `≥ 0.85` by 3000 iters. Saves `actor.*` + `critic.*`.
- **Stage 2 — short distill.** Task `Isaac-SO-ARM101-SeqPickPlace-Student-v0` (to register). RSL-RL `DistillationRunner`, MSE, on-policy DAgger. `SeqPickPlaceVisionStudentTeacher` regresses teacher actions on `policy + wrist_image`. **Not to convergence** — 200–500 iters, stop when `step0_success ~ 30–50 %`. Optional teleop seeding (3-step demos via lerobot) **encouraged** per the spec; useful exactly because Stage 2 is the modality bridge and color-grounding warm-up benefits most from human-quality grasp sequences.
- **Stage 3 — vision PPO from warm-start.** Task `Isaac-SO-ARM101-SeqPickPlace-v0` (already registered). `SeqPickPlaceVisionActorCritic.load_state_dict` routes distill `student_cnn.* / student.*` → `actor_cnn.* / actor.*`. Trains on §6 reward to convergence with `--teacher_ckpt` critic overlay.

### 9.1 Five Stage-3 interventions (inherited from Eval 1)

| # | Fix | Location | Solves |
|---|---|---|---|
| 1 | DrQ in `_encode_student` | `vision_student_teacher.py` | Distribution shift at Stage 2 → 3 boundary |
| 2 | Drop loaded `std`; reinit from `init_noise_std=0.5` | distill branch of `load_state_dict` | Distill's `std=0.1` too narrow for 3-step exploration |
| 3 | `gamma=0.98` (match teacher) | `rsl_rl_ppo_cfg.py` | Step-2 release reward needs 600-step horizon |
| 4 | Wider wrist HSV DR (§7) | `EventCfg.randomize_wrist_hsv_dr` | 4-color discrimination must be lighting-invariant |
| 5 | **`--teacher_ckpt` overlays teacher `critic.*` after distill warm-start** | `scripts/rsl_rl/train.py` | Random critic → O(magnitude)-noisy advantages → degrades actor in ~50 iters |

#5 remains the load-bearing piece (Pinto 2018 asymmetric-AC handoff). For Eval 3, this is **especially load-bearing** because the value function carries information about a 600-step credit chain through `gamma=0.98`; a randomly-initialized critic has no chance of bootstrapping that signal in early Stage-3 iterations.

### 9.2 Workflow

```bash
# Stage 1 (no teleop, no warm-start — train from scratch with step-skip curriculum)
train --task Isaac-SO-ARM101-SeqPickPlace-Teacher-v0 --headless --enable_cameras

# Stage 2 (NOT to convergence; teleop seed optional but encouraged)
train --task Isaac-SO-ARM101-SeqPickPlace-Student-v0 --headless --enable_cameras \
      --load_run <teacher_run> --checkpoint model_<best>.pt

# Stage 3
train --task Isaac-SO-ARM101-SeqPickPlace-v0 --resume --headless --enable_cameras \
      --num_envs 1024 \
      --load_run <distill_run> --checkpoint model_<best>.pt \
      --teacher_ckpt logs/rsl_rl/seqpickplace_teacher/<teacher_run>/model_<best>.pt
```

`--enable_cameras` is mandatory on all stages — the scene cfg spawns `TiledCamera` even for the teacher (camera prim instantiation requires the flag; output discarded when `wrist_image` is absent from `obs_groups`). Same `from_teacher` symlink dance as Eval 1 `RUNNING.md` §5.1.

## 10. Deploy

Per the spec, the policy must operate on visual observation of blocks; the target locations are (x, y, z) in the robot frame. We expose two CLI args at deploy:

- `--goal-sequence "red:0.20,0.10 blue:0.25,-0.05 yellow:0.18,0.00"` — three `color:x,y` triples parsed into the 3-step schedule.
- `--bowl-z 0.0` — single shared z (bowls sit on the table top; deploy frame matches sim).

The deploy script (`bc/deploy_real.py` equivalent for Eval 3; *to add*) constructs the `seq_goal_vector` from the CLI args at each tick, mirroring exactly the sim observation. Step-advancement happens on the operator-confirmed release (manual "press space when the cube lands in the bowl") rather than the sim's kinematic release predicate, because the real arm has no contact-pose oracle. Alternative: detect release with a wrist-cam HSV match between the released cube and the target bowl position — cleaner but adds a per-color HSV calibration step. **Default: manual step-advance**; investigate vision-based detection as a deploy-time upgrade if rollouts are bottlenecked by operator response time (relevant for the speed bonus).

## 11. Bonus options

### 11.1 Option A — Speed

Among all-3-step completions, how many finish inside the time limit. The §3 episode length is 15 s = 3 × 5-s sub-budget; the spec time limit will likely be in the 15–25 s range. **No model change required** — the reward already optimizes for completion, and `all_steps_done` is a positive termination so the env releases compute the moment all 3 steps finish (the policy can't farm time-out). To push faster: (a) add a small `time_remaining_at_completion_bonus` proportional to `episode_length_s − current_time` paid only when `all_steps_done` fires; (b) tighten `episode_length_s` to 12 s in a *speed-optimized* finetune branch — keep the 15-s training run as the primary and only swap to 12 s for the speed-tuned head. Predicted gain: 20–40 % faster median runtime, with the obvious risk of trading a few percentage points of all-step success for speed.

### 11.2 Option B — Singulation

Separate scaffold under `tasks/singulation/` (already in the repo). Outside the Eval-3 pipeline — different MDP (no bowls, reward is "all cubes separated by ≥ X cm"). **Out of scope of this plan** beyond noting that the singulation task should reuse the same wrist-cam pipeline + ResNet-18 encoder + vision distill recipe; the only deltas are the placement event (start cubes in a stack or cluster) and the reward (pairwise separation, no bowl/goal terms). Designing it gets its own plan once we lock down the singulation success criteria.

## 12. Why we don't use teleop in Stage 1 — and where it does help

Same argument as [`EVAL2_PLAN.md`](./EVAL2_PLAN.md) §9. The Stage 1 teacher sees `current_target_block_position`, `current_target_bowl_xy`, all 4 active cube positions, and the step one-hot in its observation — there is no perception bottleneck, and PPO on the §6 reward stack is the right tool. Teleop helps where the input modality is hard to map to actions, i.e., Stage 2's image-to-action regression, especially because 3-step sequencing puts more demand on the modality bridge than the single-target Eval 2 case. **Recommended:** record ~50 teleop trajectories of 3-step rollouts across diverse `(active_cubes, bowl_layout, goal_sequence)` and inject them into the Stage 2 DAgger replay buffer as off-policy expert data. The flip side, again: if Stage 1 fails to converge, the diagnosis is reward shape or the step-skip curriculum, **not** demo deficit. Adding teleop to Stage 1 would mask the real problem.

## 13. What still needs to be built

The seqpickplace scaffold (`tasks/seqpickplace/`) already contains the env cfg, the `SequentialGoalCommand`, the `place_seq_blocks` event, the per-step reward stack with `(4, 4, 2)` bonus, the per-step terminations, and the success-metric curriculum. Verified open items, in build order:

1. **Step-skip curriculum** (§7) — add `auto_advance_on_timeout` event term + curriculum-modulated time budget. New code in `tasks/seqpickplace/mdp/events.py` (~60 LoC) + a `CurrTerm` entry.
2. **Distinct-color sampling per step** — `place_seq_blocks` currently uses `torch.randint(0, N_ACTIVE_BLOCKS, …)` for `goal_color_pos`; flip to `argsort(rand)[:, :3]` if the spec requires distinct colors per step. One-line fix.
3. **`tasks/seqpickplace/agents/`** — new directory mirroring `tasks/pickplace/agents/` and `tasks/clutterpickplace/agents/`: `teacher_ppo_cfg.py`, `rsl_rl_ppo_cfg.py`, `distill_cfg.py`, `vision_actor_critic.py`, `vision_student_teacher.py`. Most are 30-line subclasses of the Eval-2 variants with `obs_groups` updated.
4. **Gym registrations for Teacher / Student variants** — `Isaac-SO-ARM101-SeqPickPlace-Teacher-v0`, `Isaac-SO-ARM101-SeqPickPlace-Student-v0` in `tasks/seqpickplace/__init__.py` (currently only `-v0` and `-Play-v0` are registered).
5. **CNN-class dispatch** in `PickPlaceVisionActorCritic.__init__` — `cnn_class: str = "resnet"` cfg arg, dispatch to `_ResNetSpatialSoftmaxCNN`. Shared with Eval 2 (same gap).
6. **Deploy script** — `bc/deploy_real_seq.py` (or extend `bc/deploy_real.py` with `--seq-goal` mode). CLI: `--goal-sequence`. Implements the seq_goal_vector at each tick + manual step-advancement keystroke.
7. **Teacher-side binding-C shortcut** (recommended; §14.1) — add `current_target_gripper_to_block` to a `TeacherPolicyCfg(PolicyCfg)` subclass and wire it into the `Isaac-SO-ARM101-SeqPickPlace-Teacher-v0` env cfg only. Single-line addition; speeds up Stage-1 convergence on the 4-color discrimination.
8. **Three sanity tests** (§14.4) — `test_eval3_scene_colors.py`, `test_eval3_command_color_trace.py`, and a Play-time wrist-image dump script. Each < 1 hr; catches binding bugs before training burns wall-clock.

Estimated effort: ~1.5 days for items 1–5 + 7 + 8, ~0.5 day for item 6, then 3 stages of training (Stage 1 ~2 hrs on a 32 GB GPU at 4096 envs × 3000 iters; Stage 2 ~1 hr; Stage 3 ~10 hrs at 1024 envs × 3500 iters).

## 14. Correctness audit — color binding, tick ordering, verification

This section answers two questions: (a) how does the command know which color is the target, and (b) is the manager pipeline race-free given the per-tick reads/writes of `_seq_step_idx` and the release indicator. Both are pipeline-level invariants — getting them wrong silently inverts the training signal, so they're documented here as contracts the implementation must keep.

### 14.1 Three-level color binding

The phrase "the command knows the color" actually spans **three independent bindings**, each owned by a different layer of the stack:

**Binding A — scene prim ↔ visual color (immutable, owned by scene cfg).** `joint_pos_env_cfg.py` iterates `COLOR_NAMES` and calls `_cube_cfg(name, …)` for each. `_cube_cfg` reads `BLOCK_COLORS[name]` (the RGB triple) and bakes it into `PreviewSurfaceCfg.diffuse_color`, then registers the prim at path `cube_<name>`. This binding is set **once at scene construction** and never changes. There is no scene-level remap; prim `cube_red` is always rendered with `BLOCK_COLORS["red"] = (0.78, 0.14, 0.17)` from spawn until shutdown. **Verification:** assert `env.scene[f"cube_{name}"].cfg.spawn.visual_material.diffuse_color == BLOCK_COLORS[name]` for all 6 colors — a one-liner unit test.

**Binding B — palette index k ↔ scene prim ↔ one-hot bit k (immutable encoding, owned by `events.py`).** `COLOR_NAMES = ("blue", "yellow", "purple", "orange", "green", "red")` is the canonical encoding. Three places must agree:

| Producer | Where | Uses |
|---|---|---|
| Scene asset registration | `joint_pos_env_cfg.py` `for i, name in enumerate(COLOR_NAMES): setattr(self.scene, f"cube_{name}", _cube_cfg(name, HIDDEN_PARK_XY[i]))` | Spawns prim `cube_<COLOR_NAMES[k]>` for each palette index k |
| Active-set sampling | `place_seq_blocks` produces `active_indices ∈ [0, NUM_COLORS)` via `torch.argsort(rand)[:, :4]` | Stores palette indices, not prim names |
| Cube-pose lookup | `_all_cube_pos_w(env)` iterates `for name in COLOR_NAMES: env.scene[f"cube_{name}"]` and stacks in palette order | Slot k in the stack ⇔ palette index k ⇔ prim `cube_<COLOR_NAMES[k]>` |
| Command one-hot | `SequentialGoalCommand._compose_command` writes `self._cmd[:, palette_idx] = 1.0` for the current step's target | Bit k in the 6-D one-hot ⇔ palette index k |

So binding B is enforced by *iteration order of `COLOR_NAMES`*. **If `COLOR_NAMES` is ever reordered, every cached buffer holding palette indices becomes invalid.** Guard: define `COLOR_NAMES` in one place (`events.py`) and import everywhere; never inline the tuple. Already done in the scaffold.

**Binding C — visual color ↔ one-hot bit (learned, owned by the CNN+MLP).** The CNN sees raw RGB; the MLP sees `seq_goal_vector[:6]`. Through the dense reward, gradients align "one-hot bit k is hot" with "find the pixel cluster whose color matches `BLOCK_COLORS[COLOR_NAMES[k]]`". This binding is what Stage 2 distillation actually trains — the teacher already has direct access to `current_target_block_position` via the privileged critic obs, so the value side is trivially correct; the *actor* side of the teacher only sees the 11-D `seq_goal_vector` and must learn binding C from the rollout reward, the same way the Eval-2 teacher does (Eval-2 PolicyCfg also excludes target position from the actor input, and the teacher solves it fine). Eval-3 has 4 cubes visible (vs Eval-2's 2), which makes the discrimination harder but not categorically different; the wider HSV DR (§7) is the regularizer.

**Optional teacher-side shortcut (Stage 1 only, recommended).** Adding `current_target_gripper_to_block` to the **Teacher-v0** policy obs gives the teacher actor a 3-D vector pointer to the current target — a state-level shortcut for binding C. The student does *not* see this (Student-v0 / vision-v0 policy obs is unchanged), so deployability is unaffected; the teacher just converges faster on Stage 1, and Stage 2 then teaches the student to recover the same target-direction signal from pixels. Single-line addition to a `TeacherObsCfg` subclass. Low-cost defensive measure; ship it. **Already added to §13 build list (item 7).**

**Why binding C is what makes or breaks Eval 3.** The whole point of `seq_goal_vector` is to compress an abstract "target color" into a vector the policy can condition on. The CNN must learn, from rollouts, that when bit k flips on, attention/keypoints should re-route to the pixel region whose color matches `BLOCK_COLORS[COLOR_NAMES[k]]`. With 4 colors visible (vs Eval-2's 2), this is a 4-way discrimination problem inside the CNN feature map. ResNet-18's ImageNet trunk already encodes color-friendly features at `layer3`, so the trainable 1×1 conv → soft-argmax head is sufficient; the wide HSV DR (§7) is what makes this transfer to the real arm. **If Stage 3 plateaus at `step0_success < 50 %` after 1500 iters, binding C is the suspect** — the fallback is to add a 4th image channel that's an HSV-thresholded target-color mask (per-step recomputed from `current_target_color_idx`), which short-circuits binding C and converts it back into Eval-1's mask-channel problem.

### 14.2 Tick-ordering invariants

`ManagerBasedRLEnv.step` executes managers in the order: **action → physics → reward → termination → command update → observation**. The Eval-3 reward and command both touch shared env state; the order matters. Trace, for a tick where env i releases the current target into the current bowl at step k:

| t | t.5 | t+1 |
|---|---|---|
| Reward stage (step_idx still = k): `reach_current_target`, `lift_current_target`, `transport_*` compute against cube `_target_cube_idx_per_step[i, k]` and bowl `bowl_positions[i, goal_bowl_idx[i, k]]`. `release_current_target_in_bowl` evaluates the predicate → True → **(1)** writes `_seq_step_release_indicator[i] = True`, **(2)** writes `_seq_success_per_step_latch[i, k] = True`, returns `1.0`. `step_completion_bonus` reads `_seq_step_release_indicator[i] = True` AND `_seq_step_idx[i] = k` → returns `weight_per_step[k]` (i.e., +4 for steps 0 / 1, +2 for step 2). | Termination stage: `all_steps_done` checks `_seq_step_idx[i] ≥ 3` — still False (step still = k). `block_off_table` checks active cubes — False. `time_out` — False. No termination. | Command update: `_update_command` reads `_seq_step_release_indicator[i] = True` AND `_seq_step_idx[i] < 3` → True → bumps `_seq_step_idx[i] = k+1`, clears `_seq_was_grasped[i] = False`, clears `_seq_was_over_bowl_above_rim[i] = False`, consumes indicator (`= False`). Obs stage: `seq_goal_vector` calls `cmd.command` which re-composes against the NEW `_seq_step_idx[i] = k+1` → policy obs reflects sub-goal k+1. |

This is the happy path. Three subtle invariants are doing work here:

**Invariant 1 — write-then-consume of `_seq_step_release_indicator` within a tick.** The indicator is OR-latched by the reward (`|=`) and unconditionally cleared by the command for advancing envs. As long as command-update runs AFTER all reward terms in a single tick, no spurious double-fire is possible. **Hazard:** if a future refactor moves the release reward to a separate manager (e.g., curriculum-controlled reward swap) that runs *after* the command manager, the indicator would be cleared before being written → step never advances. **Guard:** keep the release reward inside `RewardsCfg`, period.

**Invariant 2 — `step_completion_bonus` reads pre-advancement step_idx.** It's defined in `RewardsCfg` (so it runs in the reward stage, before the command updates). `env._seq_step_idx[i] = k` at that point, so `weight_per_step[k]` is paid — the bonus is keyed to *which step was completed*, not the new step. Correct. **Hazard:** if the term is moved to an "after command" hook, it would read `k+1` and pay `weight_per_step[k+1]` — off by one and step 2's bonus (the most expensive) would never fire. **Guard:** keep `step_completion_bonus` inside `RewardsCfg`, period.

**Invariant 3 — observations are computed AFTER command update.** Standard Isaac Lab order; `seq_goal_vector` returns the post-advancement obs. The policy on tick t+1 sees the new target/bowl on its first observation. **Hazard:** if you wire the obs to use a cached copy from earlier in the tick (e.g., to "freeze" the obs at action time), the policy gets stale goal-conditioning. **Guard:** don't cache; let the obs manager call `cmd.command` afresh each tick.

**Invariant 4 — multi-step releases within a single tick (rare but real).** Could the release predicate fire for steps k AND k+1 in the same physics tick? Only if `_seq_was_grasped[k+1]` and `_seq_was_over_bowl_above_rim[k+1]` are both True at the moment step k completes — but the command clears both at advance time on the same tick. The next chance for `release_current_target_in_bowl` to fire for step k+1 is the *next* tick at the earliest, by which time the latches start False again. Correct — no skip-step pathology.

### 14.3 Step-skip curriculum interaction with the invariants

The auto-advance event (§7) must **not** write `_seq_step_release_indicator` — that would trip `step_completion_bonus` and pay the policy +4 for failing to grasp. The implementation contract:

```python
def auto_advance_stalled_steps(env, env_ids, time_budget_s: float):
    # Find envs whose current step has elapsed > time_budget without release
    now = env.sim.current_time
    elapsed = now - env._seq_step_start_time
    stalled = (elapsed > time_budget_s) & (env._seq_step_idx < N_GOAL_STEPS)
    if not stalled.any(): return
    # Skip — bump step_idx, clear latches, update start time. Do NOT touch
    # _seq_step_release_indicator or _seq_success_per_step_latch.
    env._seq_step_idx[stalled] += 1
    env._seq_was_grasped[stalled] = False
    env._seq_was_over_bowl_above_rim[stalled] = False
    env._seq_step_start_time[stalled] = now
```

Two consequences this gives us:

1. The policy earns *no* per-step bonus on a skipped step (correct — skipping is failure).
2. The policy still earns the dense `reach_current_target` + `lift_current_target` + `transport_*` rewards on the NEXT step from action 0 of training. That's the point — early steps may be unlearned, but the gradient still flows for steps 1 and 2.

### 14.4 Verification checklist

Three concrete tests to run before the first Stage-1 launch. Each is < 1 hr to write and catches a specific binding failure:

1. **Scene-color assertion (binding A).** In a `test_eval3_scene.py`, instantiate `SoArm101SeqPickPlaceEnvCfg_PLAY` (`num_envs=16`) and assert `env.scene["cube_red"].cfg.spawn.visual_material.diffuse_color == BLOCK_COLORS["red"]` for all 6 colors. Catches: typo in `BLOCK_COLORS` keys vs `COLOR_NAMES`, future scene-cfg reorderings.

2. **Command-color trace at step boundary (binding B).** Run a Play rollout with a scripted action that releases the step-0 target (e.g., via env-level cheat: directly write the target cube's pose to the current bowl). After release, log `(env._seq_step_idx, cmd.current_target_color_idx(), env._target_cube_idx_per_step[:, env._seq_step_idx])` — these three must be consistent before and after the step advances. Catches: gather-index off-by-ones, `active_indices` cache desync.

3. **Wrist-image highlight test (binding C, sanity).** After Stage 1 converges, sample 100 rollouts at random `(active_set, target_color)` and dump pairs of (wrist RGB, target color one-hot). Verify by hand that in ≥ 95 % of frames, the cube whose pixels match the one-hot color is at least partially visible in the wrist FOV at the moment the policy chooses to descend (gripper z < 0.05). Catches: the policy learned to "always grab the closest cube" by coincidence (binding C didn't actually train) — would show up as the target color being absent from the wrist FOV when the policy commits to a grasp.

If all three pass, the binding pipeline is correct and any remaining failures are in optimization (reward shape, curriculum, exploration), not in observation/reward plumbing.

### 14.5 Efficiency notes

- **VRAM at 1024 envs, 15-s episode is the same as Eval 2 at 1024, 5-s episode.** PPO with `num_steps_per_env=16` rolls out 16 ticks per env per update — the per-update tensor is `num_envs × 16 × (obs_dim + action_dim + 1)` floats, plus the image `1024 × 16 × 3 × 72 × 128 × 4 bytes ≈ 1.5 GB`. Episode length only affects the env-side scalar buffers (cube positions, latches) which are O(num_envs × 100 floats) — < 1 MB.
- **Stage-1 wall-clock.** State-only PPO at 4096 envs, no rendering, 50 Hz × 24 steps_per_env = 1200 env-frames/iter. With ~3000 iters and ~3 iter/s on a 32 GB GPU (measured for Eval-1 teacher), Stage 1 finishes in ~17 min × 60 = ~17 min? **Actual Eval-1 teacher iter rate was ~0.5 iter/s**, so 3000 iters ≈ 100 min ≈ 1.7 hrs. Plus the step-skip curriculum is essentially free (O(num_envs) bool ops per tick).
- **Stage-3 wall-clock.** Image rollout at 1024 envs is ~5× slower than state-only. With 16 steps_per_env × 50 Hz physics, image renders dominate; measured at ~0.1 iter/s in Eval-1 Stage-3 → 3500 iters × 10 s ≈ 10 hrs. The 15-s episode doesn't change iter rate (PPO rolls 16-step windows regardless), only changes episode-completion frequency in TB metrics.
- **Eval-cycle latency.** A Play run with 16 envs at 50 Hz renders the wrist cam each tick — ~30 s for a full 750-tick episode visualization. Multiply by 5 for the 5-rollout eval → 2.5 min/policy. Fast enough to iterate.

## References

- **Code:** `tasks/seqpickplace/{seqpickplace_env_cfg,joint_pos_env_cfg}.py`, `mdp/{commands,events,observations,rewards,terminations}.py`, agents (to add). Predecessor plans: [`EVAL1_PLAN.md`](./EVAL1_PLAN.md), [`EVAL2_PLAN.md`](./EVAL2_PLAN.md).
- **Pinto et al.**, *Asymmetric Actor Critic*, RSS 2018. <https://arxiv.org/abs/1710.06542>
- **Levine et al.**, *End-to-End Visuomotor Policies*, JMLR 2016 — spatial softmax.
- **Kostrikov et al.**, *DrQ*, ICLR 2021. <https://arxiv.org/abs/2004.13649>
- **Andrychowicz et al.**, *Hindsight Experience Replay*, NeurIPS 2017. <https://arxiv.org/abs/1707.01495> — goal-conditioning template; we use an explicit step one-hot in place of relabeled goals.
- **Nasiriany et al.**, *Augmenting RL with Behavior Primitives*, CoRL 2022. <https://arxiv.org/abs/2110.03655> — closest published recipe for long-horizon manipulation with curriculum-paced sub-goals.
- **RSL-RL**, Schwarke et al., 2025. <https://github.com/leggedrobotics/rsl_rl>
- **LeIsaac**, <https://github.com/LightwheelAI/leisaac> — wrist camera mount.
