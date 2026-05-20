# Eval 3 — Multi-Step Pick-and-Place via Policy Switching

**Approach:** reuse Eval 2's single-target color-conditioned policy (retrained on the Eval-3 4-cube cluster scene) and sequence three sub-goals externally on the deploy side. The TA has confirmed that **policy switching / perception-module switching is allowed** for Eval 3; this turns a 3-sub-goal long-horizon RL problem into three independent single-target rollouts driven by a thin scheduler. The sequence-conditioned PPO pipeline that an earlier draft of this doc described is preserved as a fallback in §15.

This document is the third in the series; it assumes familiarity with [`EVAL1_PLAN.md`](./EVAL1_PLAN.md) (asymmetric A-C handoff, five Stage-3 interventions, wrist-cam intrinsics) and [`EVAL2_PLAN.md`](./EVAL2_PLAN.md) (ResNet-18 + spatial-softmax encoder, color-one-hot goal-conditioning, wrist-tint DR). Where Eval 3 inherits a piece verbatim, the prose references those docs rather than re-deriving.

## 0. Approach decision — policy switching vs sequence-conditioned PPO

| Axis | (A) Policy switching — **chosen** | (B) Sequence-conditioned PPO — fallback |
|---|---|---|
| Training cost | 1× retrain of an Eval-2-style task on the Eval-3 scene (~13 hrs) | Fresh 3-stage pipeline with new command + reward + step-skip curriculum (~13 hrs) |
| New code | ~150 LoC: placement variant (4-cube 2×2 cluster), gym IDs, outer deploy loop | ~500+ LoC: `SequentialGoalCommand`, per-step reward stack, step-skip curriculum, agents, deploy |
| MDP risk | Low — Eval 2's MDP, well understood. Only delta is the placement distribution. | Medium — new long-horizon MDP with per-step latches, tick-ordering invariants (§15.3), step-skip tuning |
| Reuse for Eval 2 | The retrained policy strictly dominates Eval 2 (handles 1 or 3 distractors) | None |
| Deployability | Trivial — invoke the policy thrice with different `(target_color, bowl_xy)` | Same (single rollout, internal advance) |
| Speed bonus | Modest — short homing between sub-goals adds wall-clock overhead | Better — sequence-aware policy can pipeline approach across sub-goals |
| Singulation bonus | Same — separate task either way (§11.2) | Same |

**Decisive factor:** risk × cost. The TA explicitly opened the door to policy switching, so we take it. Option B's only real upside is the speed bonus, which we can chase separately once Option A meets the base success bar. The original Option-B plan stays in §15 so we can pivot to it if Option A plateaus.

## 1. Spec recap (PDF §Eval 3)

Four blocks of distinct colors are placed in the workspace. The policy is given a sequence of three `(target_color, bowl_xy)` goals; the bowl is **shared across all three sub-goals** within a rollout (per the latest spec read — the scene spawns one bowl, all three placements target that same bowl). Per-step scoring is **4, 4, 2** (10 pts/rollout × 5 rollouts = 50 pts), partial credit is awarded, success = correct block placed in the bowl and released. **RL is required.** Policy switching / perception-module switching is allowed, and interaction with non-target blocks (pushing, rearranging) is *permitted and encouraged* if it facilitates task completion. Optional 50-pt **bonus** is awarded on either **(A) Speed** — number of all-3-step rollouts completed inside a time limit — or **(B) Singulation** — separating a cluster/stack into individually graspable configurations.

## 2. What carries over

Concrete reuse audit against Eval 2's `tasks/clutterpickplace/`:

| Component | Eval 2 | Eval 3 (Option A) | Reuse |
|---|---|---|---|
| Robot, control, action space, home `q` | ✓ | ✓ | **verbatim** |
| Workspace, table, palette, friction tuning | ✓ | ✓ | **verbatim** |
| Wrist `TiledCamera` mount + intrinsics | ✓ | ✓ | **verbatim** |
| 4-ch image obs `(N, 4, 72, 128)` — RGB + target instance mask | ✓ | ✓ | **verbatim** |
| `_ResNetSpatialSoftmaxCNN` encoder | ✓ | ✓ | **verbatim** |
| `policy + critic + wrist_image` 3-group asymmetric obs | ✓ | ✓ | **verbatim** |
| `TargetColorCommand` (single target, 6-D color one-hot + 2-D bowl xy) | ✓ | ✓ | **verbatim** |
| Reach / lift / transport / release / wrong-cube reward stack | ✓ | ✓ | **verbatim** |
| Two latches (lift ≥ 0.07, over-bowl-above-rim ≥ 0.08) | ✓ | ✓ | **verbatim** |
| 3-stage pipeline (teacher → distill → vision PPO + `--teacher_ckpt`) | ✓ | ✓ | **verbatim** |
| Wrist RGB tint + HSV DR | ✓ | ✓ (wider — §7) | **structure preserved** |
| `place_clutter_blocks` (2 cubes spread, ≥10 cm) | ✓ | replaced with `place_four_attached_cluster` | **new placement event** |
| Eval-2 checkpoint transfer | — | optional finetune seed (§9 alt) | **soft reuse** |

**Net:** the only required deltas are (i) a new placement event that puts 4 of 6 palette cubes in a 2×2 attached cluster, (ii) the `TargetColorCommand` extended from 2 → 4 active slots, (iii) widened wrist-tint / HSV DR for 4-color discrimination, (iv) gym IDs for the Eval-3 task, and (v) the outer deploy loop. Everything else is bitwise identical to Eval 2.

## 3. MDP

| Item | Value |
|---|---|
| Control | 50 Hz (decimation 2, sim 100 Hz) — verbatim |
| Episode | **5.0 s = 250 steps** per sub-goal rollout (matches Eval 2). The 3-sub-goal *outer* time budget lives in the deploy scheduler, not the env. |
| Action | 5 arm joints (`scale=0.5`) + 1 binary gripper (`open=0.5`, `close=0.0`) — verbatim |
| Cluster center | `x ∈ [0.16, 0.20]`, `y ∈ [−0.08, 0.08]`, 4 cubes in a 2×2 attached cluster (`half_separation=0.0105 m` → 2.1 cm center-to-center, 1 mm face gap → contact under gravity), random yaw θ ∈ [0, 2π) |
| Bowl | `x ∈ [0.18, 0.26]`, `y ∈ [−0.08, 0.08]`, rejection-sampled ≥ `min_bowl_cluster_separation = 0.08 m` from cluster center |
| Home `q` | `(0, 0, 0, 1.5708, 0, 0)`, gripper open |
| Terminations | `time_out`, `active_block_off_table` (any of the 4 active cubes off-table). **No `task_success`** — same Eval-1/2 reasoning (let the release reward pay every post-release step, avoid the hover-with-open-gripper attractor) |
| Table | `0.6 × 1.0 × 0.02 m` at `(0.25, 0, −0.01)`, top `z=0` |

**Scene composition.** Six 2 cm `CuboidCfg` cubes baked with the six palette colors (`COLOR_NAMES = ("blue", "yellow", "purple", "orange", "green", "red")` — Binding A/B from Eval 2 §14, unchanged). Per reset, `place_four_attached_cluster` (new event, §6) samples a length-4 random permutation of the palette, places those 4 as a 2×2 attached cluster (center + yaw θ; corners at `(±half_separation, ±half_separation)` rotated by θ), and teleports the other 2 to `HIDDEN_PARK_XY` slots off-table. The bowl is a 2-D goal from `TargetColorCommand` (no scene prim).

**Why an attached 2×2 cluster, not spread placement.** Two reasons. First, the spec explicitly encourages distractor interaction (push/rearrange to make a grasp possible); the attached cluster is the configuration where that behavior actually pays off. Second, this matches the real-world Eval-3 evaluation setup the TA has been describing — four cubes touching in a square. A spread-placement training distribution would leave a sim-to-real gap precisely at the moment the policy has to commit to a grasp.

**Target sampling.** `TargetColorCommand` samples the target uniformly from the 4 active slots per reset (changes `target_in_pair ∈ {0,1}` to `target_in_active ∈ {0,1,2,3}`). The full active set (`active_indices: (N, 4)`) is exposed to the critic; the policy only sees the color one-hot of the target + the bowl xy. **A single policy rollout solves one sub-goal**; the deploy-side scheduler re-issues the env with a different `target_color` for sub-goals 2 and 3.

## 4. Observations (asymmetric A-C)

Same three groups as Eval 2. Schema in `tasks/eval3clutter/eval3clutter_env_cfg.py` (new) — copy of `tasks/clutterpickplace/clutterpickplace_env_cfg.py` with `all_active_block_positions: 2 → 4` in the critic group.

| Group | Fields | Shape |
|---|---|---|
| `policy` (deployable) | `joint_pos_rel`, `joint_vel_rel`, `gripper_state`, `target_color_onehot` (6), `target_bowl_xy` (2), `ee_proj_xy`, `last_action` | 1-D |
| `critic` (privileged) | `policy` + `all_active_block_positions` (**4×3**), `target_block_position` (3), `target_gripper_to_block` (3), `target_block_to_bowl_xy` (2) | 1-D |
| `wrist_image` | RGB + current_target instance mask | `(N, 4, 72, 128)` |

**No `seq_goal_vector`, no step one-hot, no per-step latches.** The policy is a strict generalization of Eval 2's: same `target_color_onehot + target_bowl_xy` interface. Multi-step semantics live entirely outside the env.

**Color discrimination (binding C).** Eval 2 §14.1 documented this for a 2-cube scene; Eval 3 stretches it to 4-way. We ship a 4-ch image by default — channel 3 is the per-step current-target instance mask, sourced from `semantic_segmentation` (each cube tagged `class:cube_<color>`) and corrupted in sim to mimic Florence-2 at deploy (see Eval 2 §8 + §5 for the four DR axes). HSV thresholding was rejected after empirically failing at the wrist-cam working distance (no mask when far, lighting-induced false positives). The target_color_onehot still flows into the policy state + CNN FiLM head — belt-and-suspenders for the mask-dropout case.

Wrist `TiledCamera`: same mount/intrinsics as Eval 1/2 (`pos=(-0.001, 0.1, -0.04)`, ROS quat `(-0.404379, -0.912179, -0.0451242, 0.0486914)`); `data_types=["rgb", "semantic_segmentation"]` with `colorize_semantic_segmentation=False`.

## 5. Network — `Eval3ClutterVisionActorCritic`

Bitwise identical to Eval 2's `ClutterPickPlaceVisionActorCritic`: frozen ImageNet ResNet-18 trunk truncated at `layer3` → trainable 1×1 conv (64 ch) → per-channel softmax → soft-argmax → 128-D keypoints; state concat → MLP `[256, 128, 64]` ELU → μ (σ scalar Param); critic MLP `[256, 128, 64]` ELU → V(s).

```
wrist_image (4×72×128, RGB + current_target_mask) ──[ResNet-18 trunk, conv1 inflated to 4ch]── (256, 9, 16) feat
                       ──[1×1 conv → 64 ch]── softmax → soft-argmax (x,y) → 128-D kpts ─┐
state (policy, target_color_onehot 6-D + bowl_xy 2-D) ─── concat ── MLP[256,128,64] ── μ
critic state (policy+critic, 4-cube positions) ────────── MLP[256,128,64] ── V(s)
```

DrQ ±4 px replicate-pad-and-crop training-only in `_encode_actor` (Stage 3) and `_encode_student` (Stage 2).

**Implementation note.** We subclass `ClutterPickPlaceVisionActorCritic` rather than copy. The only state-dim delta is the critic's `all_active_block_positions: 6 → 12` (4 cubes × 3 instead of 2 × 3), which is an MLP input-dim change; the actor MLP input dim is **unchanged** (same `target_color_onehot + target_bowl_xy + joint_*` schema). The CNN-class dispatch (`cnn_class: str = "resnet"`) issue from Eval 2 §13 is shared — fix in `PickPlaceVisionActorCritic.__init__` once and inherit everywhere.

## 6. Reward (`tasks/eval3clutter/mdp/rewards.py`)

| Term | Weight | Trigger |
|---|---|---|
| `reach_target` | 1.0 | `1 − tanh(‖ee − target‖ / 0.05)` |
| `lift_target` | 15.0 | `𝟙[target_z > 0.07]` (per-episode latch) |
| `transport_target_to_bowl` (coarse) | 16.0 | `was_lifted · (1 − tanh(‖target_xy − bowl_xy‖ / 0.30))` |
| `transport_target_to_bowl` (fine) | 5.0 | same with `std=0.05` |
| `release_target_in_bowl` | 30.0 | target near bowl ∧ `z<0.06` ∧ gripper open ∧ settled, gated on lift + over-bowl-above-rim latches |
| `wrong_cube_in_bowl` | −15.0 | any non-target *active* cube sits inside the bowl (xy < 6 cm ∧ `z < 0.06`) |
| `action_rate`, `joint_vel` | −1e-4 → −1e-2 | ramp at 10 k env-steps (verbatim) |

**No step-completion bonus, no `step_active` gating** — this is the Eval-2 reward stack, period. The sub-goal scoring (4, 4, 2) is enforced by the *outer deploy loop's grading function*, not by the in-env reward.

**Why we keep `wrong_cube_in_bowl=-15` even though there are 3 distractors.** With a cluster of 4 cubes touching the target, a wrong-cube-drop is a real failure mode (the gripper closes around a neighbor instead). The penalty is comparable to but smaller than the +30 release + dense transport, so a correct final placement still wins net after one near-miss, but the gradient steers away. If we observe "policy parks distractors in the bowl en route," widen to −25.

**No distractor-disturb penalty.** Spec explicitly encourages distractor interaction (push/rearrange). The policy is free — and expected — to nudge neighbors aside to free the target. Only `wrong_cube_in_bowl` guards the specific "wrong cube ends up where the target should go" failure.

**`release_in_bowl` fires every step the predicate holds** — inherited from Eval 1/2 reasoning (avoid hover-with-open-gripper attractor). With a 5-s episode (250 ticks) and γ=0.98, the post-release tail pays the policy until time-out; the policy maximizes time spent in the release-predicate region.

**No contact sensor reward.** Same reasoning as Eval 2: target identity varies per episode, contact-sensor filtering would need per-target prim-path patching; the kinematic lift latch (`target_z > 0.07`) is sufficient.

## 7. Curriculum & DR (`tasks/eval3clutter/mdp/events.py`, `CurriculumCfg`)

Both stages (verbatim from Eval 2 except where noted):

- `place_four_attached_cluster` (new event — §6). 4 active cubes baked into a 2×2 cluster, cluster center sampled in `x ∈ [0.16, 0.20]`, `y ∈ [−0.08, 0.08]`, yaw θ ∈ [0, 2π) random per env.
- `reset_target_latches`: per-episode lift / over-bowl latches cleared (same as Eval 2).
- Action-rate / joint-vel penalty ramp −1e-4 → −1e-2 at 10 k env-steps (verbatim).
- `log_seq_success_metrics` TB metric — emits `step0_success`, `step1_success`, `step2_success`, `all_steps_success` (headline binary SR over the full 3-step rollout), `n_episodes_ended`. A parallel `success_rate_strict` is also emitted using the PDF-minimal gate (`in_xy ∧ low ∧ opened`, no safety latches / no `settled`), as a sanity-check upper bound on the conservative `all_steps_success`.

Stage 3 adds:

- DrQ ±4 px (in CNN, see §5).
- `randomize_wrist_image_tint`: RGB scale `(0.55, 1.45)`, brightness `(−0.20, 0.20)` — **wider than Eval 2** (`0.7, 1.3` / `±0.15`). Rationale: with 4 colors visible on screen at once (not 2), color confusion is the dominant failure mode; wider tint DR forces the CNN to discriminate by relative palette position rather than absolute hue.
- `randomize_wrist_hsv_dr`: hue ±20°, saturation `(0.65, 1.35)`, value `(0.55, 1.45)`. **New vs Eval 2.** Explicit hue rotation is what catches lighting that shifts the *direction* of color in HSV space (warm/cool indoor light). The hue ±20° band specifically guards against the failure where two palette colors (e.g., orange vs red) drift onto each other under unusual room light.

No xy-expand curriculum. The cluster center band (5 cm × 16 cm) is already tight; the policy needs the full 4-cube distribution from step 0 to learn color discrimination. **No step-skip curriculum** — irrelevant to single-target task.

**No `n_active_blocks` curriculum either.** We considered ramping `n_active ∈ {2, 3, 4}` over training to ease in the discrimination problem, but rejected it: the Eval-2 teacher converges in ~2000 iters on 2 cubes, so the gap to 4 cubes is one bit of extra discrimination per CNN channel — not a curriculum-scale jump. If Stage-1 stalls at `target_lifted < 30 %` past 1500 iters, **then** add the curriculum (n_active = 2 → 3 → 4 at 1k / 2k iters).

## 8. PPO config

Eval-2 skeleton, marginally longer iter counts for the 4-color regime.

| | Stage 1 teacher (`teacher_ppo_cfg.py`) | Stage 3 vision PPO (`rsl_rl_ppo_cfg.py`) |
|---|---|---|
| `num_envs` | 4096 | 1024 (same as Eval 2; OOM-drop to 768 only if observed) |
| `num_steps_per_env` | 24 | 16 |
| `max_iterations` | 2500 (Eval 2 used 2000; +25 % for 4-color discrimination) | 3000 (Eval 2 used 2500) |
| `init_noise_std` | 1.0 | 0.5 (forced; distill saves 0.1) |
| hidden dims | `[256, 128, 64]` ELU | same + ResNet-18 spatial-softmax CNN |
| `entropy_coef` | 0.006 | 0.006 → 0.003 after ~800 iters |
| epochs / mini-batches | 5 / 4 | 8 / 16 |
| `learning_rate` / `desired_kl` | 1e-4 / 0.01 | 1e-4 / 0.005 |
| `gamma / lam / clip / max_grad_norm` | 0.98 / 0.95 / 0.2 / 1.0 | same |

```python
# Stage 1 (teacher_ppo_cfg.py): symmetric on privileged state
obs_groups = {"policy": ["policy", "critic"], "critic": ["policy", "critic"]}
# Stage 2 (distill_cfg.py): vision student, state teacher
obs_groups = {"policy": ["policy", "wrist_image"], "teacher": ["policy", "critic"]}
# Stage 3 (rsl_rl_ppo_cfg.py): vision actor, privileged critic
obs_groups = {"policy": ["policy", "wrist_image"], "critic": ["policy", "critic"]}
```

Stage-1 critic and Stage-3 critic take identical inputs → teacher critic loads layer-for-layer via `load_state_dict(strict=False)`. Critic state is wider than Eval 2 (`all_active_block_positions: 6 → 12 dims`) — **Eval-2 teacher checkpoints do not transfer**; Eval-3 teacher trains from scratch.

**`gamma=0.98` is load-bearing** for the same reason as Eval 1 (250-step episode, release fires ~150 steps in; γ=0.95 collapses the credit chain). Mismatch between teacher and Stage-3 PPO γ silently destroys the warm-start.

## 9. Three-stage pipeline

- **Stage 1 — state teacher.** Task `Isaac-SO-ARM101-Eval3Clutter-Teacher-Fast-v0` (camera-free). MLP A-C on `policy + critic`. Pure PPO on privileged state, no teleop. Teacher sees all 4 cube positions + target pose + bowl xy directly — no perception bottleneck; PPO on the §6 reward stack is the right tool. Expect `target_in_bowl ≥ 0.6` by ~1500 iters and `≥ 0.85` by 2500 iters. Saves `actor.*` + `critic.*`.
- **Stage 2 — short distill.** Task `Isaac-SO-ARM101-Eval3Clutter-Student-v0`. RSL-RL `DistillationRunner`, MSE, on-policy DAgger. Vision student regresses teacher actions on `policy + wrist_image`. **Not to convergence** — 200–500 iters, stop when `target_in_bowl ≈ 30–50 %`. Optional teleop seeding (~30 trajectories across colors) — useful but not required.
- **Stage 3 — vision PPO from warm-start.** Task `Isaac-SO-ARM101-Eval3Clutter-v0`. Routes distill `student_cnn.* / student.*` → `actor_cnn.* / actor.*`. Trains on §6 reward to convergence with `--teacher_ckpt` critic overlay.

### 9.1 Five Stage-3 interventions (inherited verbatim from Eval 1/2)

| # | Fix | Location | Solves |
|---|---|---|---|
| 1 | DrQ in `_encode_student` | `vision_student_teacher.py` | Distribution shift at Stage 2 → 3 boundary |
| 2 | Drop loaded `std`; reinit from `init_noise_std=0.5` | distill branch of `load_state_dict` | Distill's `std=0.1` too narrow for re-exploration |
| 3 | `gamma=0.98` (match teacher) | `rsl_rl_ppo_cfg.py` | 250-step credit chain |
| 4 | Wider wrist HSV DR (§7) | `EventCfg.randomize_wrist_hsv_dr` | 4-color discrimination must be lighting-invariant |
| 5 | `--teacher_ckpt` overlays teacher `critic.*` after distill warm-start | `scripts/rsl_rl/train.py` | Random critic → O(magnitude)-noisy advantages → degrades actor in ~50 iters |

#5 remains load-bearing (Pinto 2018 asymmetric-AC handoff).

### 9.2 Workflow

```bash
# Stage 1 — camera-free state teacher (no --enable_cameras needed)
uv run train --task Isaac-SO-ARM101-Eval3Clutter-Teacher-Fast-v0 --headless

# Stage 2 — short distill (200-500 iters; NOT to convergence)
uv run train --task Isaac-SO-ARM101-Eval3Clutter-Student-v0 --headless --enable_cameras \
    --load_run from_teacher --checkpoint model_<best>.pt

# Stage 3 — vision PPO with teacher critic overlay
uv run train --task Isaac-SO-ARM101-Eval3Clutter-v0 --resume --headless --enable_cameras \
    --num_envs 1024 \
    --load_run <distill_run> --checkpoint model_<best>.pt \
    --teacher_ckpt logs/rsl_rl/eval3clutter_teacher/<teacher_run>/model_<best>.pt
```

`from_teacher` symlink follows the same convention as Eval 1 `RUNNING.md` §5.1.

### 9.3 Alternative seed — finetune from Eval 2 checkpoint

Eval 2's `ClutterPickPlace` Stage-3 checkpoint *can* warm-start Stage 3 here (same actor MLP input dim, same CNN, same policy obs schema). The critic input dim differs (`12 vs 6` for `all_active_block_positions`), so the Eval-2 critic does **not** transfer — you still need a fresh Stage-1 teacher for the `--teacher_ckpt` overlay. This is worth trying only if the Eval-2 checkpoint is solid and we're short on training wall-clock. Default path is fresh Stage 1.

## 10. Deploy — outer sequencing loop

The policy operates on visual observation of blocks (per spec). The 3 sub-goals are scheduled externally. Two deploy modes:

### 10.1 Sim deploy (extend `scripts/rsl_rl/play.py`)

```bash
uv run play --task Isaac-SO-ARM101-Eval3Clutter-Play-v0 \
    --load_run <run> --enable_cameras \
    --eval3 --bowl-xy 0.22,0.00 --colors red,blue,yellow
```

The Eval-3 outer loop is a new flag set on `play.py` (since the BC `deploy_sim.py` is gone). It reuses the existing `_force_bowl_xy` hook for the fixed-bowl override and adds an outer sub-goal scheduler that mutates `command_manager._terms["target_color"]` between sub-goals.

Behavior per sub-goal `k ∈ {0, 1, 2}`:

1. Set `target_color_onehot = onehot(colors[k])`, `target_bowl_xy = bowl-xy` in the env's `TargetColorCommand` buffer.
2. Reset the per-episode lift / over-bowl latches (`env._was_grasped[:] = False`, `env._was_over_bowl_above_rim[:] = False`).
3. Roll the policy forward for up to `sub_goal_budget_s = 5.0 s` (250 ticks).
4. On detected release (lift latch ∧ over-bowl latch ∧ gripper open ∧ target in bowl xy<6cm ∧ z<6cm — same predicate as the sim reward), advance to sub-goal `k+1`.
5. Optionally: between sub-goals, home the arm to `JOINT_DEFAULTS_RAD` for `0.5 s` to ensure the next grasp starts from a known configuration. **Skip homing if speed matters** (bonus A).
6. Score: `4·𝟙[k=0 done] + 4·𝟙[k=1 done] + 2·𝟙[k=2 done]`.

The cubes and bowl are sampled once at env reset (start of sub-goal 0) and *not* re-sampled between sub-goals — the world state persists, so sub-goal 1 starts from wherever sub-goal 0 left it. This is the realistic eval setup.

### 10.2 Real deploy (`deploy/deploy_real.py --eval3`)

```bash
python -m deploy.deploy_real --eval3 \
    --policy-ckpt <ckpt> \
    --bowl-xy 0.22,0.00 \
    --colors red,blue,yellow \
    --release-detect manual    # or "vision" — see below
```

Same loop as §10.1 with two real-world adaptations:

- **Release detection** has no kinematic oracle. Three options, in order of preference:
  - `manual`: operator presses Enter when they see the cube land in the bowl. Default. Simple and reliable.
  - `vision`: HSV-match the target cube's color against the bowl region in the wrist camera. Calibrate per-color from the same HSV table used by `deploy/deploy_real._build_image` (Eval 1). Adds ~15 min of setup per scene; pays off for the speed bonus.
  - `timed`: fixed 5-s wall-clock per sub-goal, advance unconditionally. Worst option — masks failures — but useful for unattended runs.
- **Homing between sub-goals.** Default ON for the base 50-pt rubric (reliability > speed). Add `--no-home-between-subgoals` for the speed bonus run.

**CLI design choice.** Single-bowl spec means we expose `--bowl-xy` once and `--colors` as a comma-separated list, rather than the parameterized `--goal-sequence "color:x,y …"` an earlier draft proposed. This is cleaner and matches the actual eval where the bowl moves only between *rollouts*, not between sub-goals.

### 10.3 Policy-switching contract

The same policy weights drive all 3 sub-goals. We do **not** load different checkpoints between sub-goals — that would be a different feature (true policy switching across distinct policies) and we don't need it. The "switching" the TA refers to is conceptual: we *could* swap policies if we wanted (e.g., a special "first-cube-of-cluster" policy vs an "after-disturbance" policy), but the simpler design is one policy that's been trained on the full distribution of "this is sub-goal k of an Eval-3 rollout" — and since the policy is sub-goal-agnostic (no step one-hot), it doesn't need to know which sub-goal it's on.

**Perception-module switching is also available** as an escape hatch: at deploy, `deploy/cube_detector.py`'s `Detector` protocol lets you swap Florence-2 for any other prompt-able segmenter (CLIPSeg, GroundedSAM, a small custom YOLO) by implementing one method. Useful if Florence's per-frame latency dominates the speed-bonus path (§11 bonus A) — CLIPSeg is ~5× faster on CPU at comparable quality for this palette size.

## 11. Bonus options

### 11.1 Option A — Speed

Among all-3-step completions, how many finish inside the time limit. **No model change required.** The outer scheduler is what we tune:

- Skip homing between sub-goals (`--no-home-between-subgoals`).
- Use `--release-detect vision` so the loop advances within ~50 ms of release instead of human reaction time.
- Optionally shorten each sub-goal's budget to 4.0 s (training was 5.0 s — leaves headroom).
- A separate "speed-tuned" finetune: retrain Stage 3 for 500 more iters with `episode_length_s = 4.0` to push the policy to faster grasps. Predicted gain: 20–40 % faster median runtime, with some risk of trading a few percent success.

### 11.2 Option B — Singulation

Separate task under `tasks/singulation/` (scaffold already in repo). Different MDP (no bowls, reward is "all cubes separated by ≥ X cm"). Out of scope of this plan beyond noting that singulation should reuse the same wrist-cam pipeline + ResNet-18 encoder + vision distill recipe; deltas are the placement event (stack/cluster) and the reward (pairwise separation, no bowl/goal terms). Designed in its own plan once we lock down singulation success criteria.

## 12. Why we don't use teleop in Stage 1

Same argument as Eval 2 §9: the Stage-1 teacher sees `target_block_position`, `target_bowl_xy`, all 4 active cube positions, and the color one-hot in its critic obs — there is no perception bottleneck, and PPO on the §6 reward stack is the right tool. Teleop helps where the input modality is hard to map to actions, i.e., Stage 2's image-to-action regression. **Recommended:** record ~30 teleop trajectories across diverse `(active_cubes, target_color, bowl_xy)` for Stage 2 DAgger replay — same recipe as Eval 2. If Stage 1 fails to converge, the diagnosis is reward shape or DR width, **not** demo deficit; adding teleop to Stage 1 would mask the real problem.

## 13. What needs to be built

Build order — items 1–3 unblock Stage 1, items 4–5 unblock Stage 3, item 6 unblocks deploy. All paths relative to `isaac_so_arm101/src/isaac_so_arm101/tasks/`.

1. **Placement event `place_four_attached_cluster`** in a new `tasks/eval3clutter/mdp/events.py` (or extend `tasks/clutterpickplace/mdp/events.py` with `placement_mode: "spread" | "cluster_2x2"` + `n_active_blocks: int = 2` params). 2×2 attached cluster (`half_separation=0.0105 m`), cluster center sampled in `[0.16, 0.20] × [−0.08, 0.08]`, yaw θ ∈ [0, 2π). Park the 2 unused cubes via `HIDDEN_PARK_XY`. **~80 LoC**, mirror existing `place_clutter_blocks`.
2. **`TargetColorCommand` extension** to 4 active slots. Either subclass with `n_active_blocks: int = 2 → 4`, or generalize in place — the latter is cleaner and benefits Eval 2 (no behavior change at `n_active_blocks=2`). Update `target_in_pair: (N,) ∈ {0,1}` → `target_in_active: (N,) ∈ {0,…,n_active-1}`, `active_indices: (N, 2)` → `(N, n_active)`. **~30 LoC.**
3. **New task package `tasks/eval3clutter/`** mirroring `tasks/clutterpickplace/`:
   - `eval3clutter_env_cfg.py` — copy of `clutterpickplace_env_cfg.py` with `all_active_block_positions: 2 → 4`, wider HSV DR (§7), `place_four_attached_cluster` in `EventCfg`.
   - `joint_pos_env_cfg.py` — copy with cluster placement, `TeacherFastEnvCfg` subclass that nulls `wrist_cam` + `wrist_image` obs group (same trick as Eval 1).
   - `__init__.py` — gym IDs: `…-v0`, `…-Play-v0`, `…-Student-v0`, `…-Teacher-v0`, `…-Teacher-Fast-v0` (+ `-Play-v0` variants). Mirror Eval 2's `__init__.py`.
   - `mdp/__init__.py`, `mdp/{events.py, commands.py, observations.py, rewards.py, terminations.py}` — copies from Eval 2 with the deltas above. Most files are 1-line edits.
4. **`agents/`** subdirectory in `eval3clutter/`:
   - `vision_actor_critic.py` — `Eval3ClutterVisionActorCritic(ClutterPickPlaceVisionActorCritic)` subclass; only changes `_state_dim` accounting if Eval 2's class doesn't pick it up via cfg. ~30 LoC.
   - `vision_student_teacher.py` — analogous subclass for the distill student.
   - `teacher_ppo_cfg.py`, `rsl_rl_ppo_cfg.py`, `distill_cfg.py` — copies of Eval 2's cfgs with `max_iterations` and `entropy_coef` schedule per §8. Each ~40 LoC.
   - `setattr(rsl_rl.runners.…)` class-injection lines at the bottom of each cfg (Eval 1/2 pattern — required for RSL-RL's `eval()`-based class resolution).
5. **CNN-class dispatch** in `tasks/pickplace/agents/vision_actor_critic.py` — add `cnn_class: str = "resnet"` to the cfg and dispatch in `PickPlaceVisionActorCritic.__init__`. **Shared with Eval 2** (same open item). ~15 LoC.
6. **Deploy script extension** — `deploy/deploy_real.py` (and a new `--eval3` flag set on `scripts/rsl_rl/play.py` for sim eval, since the BC `deploy_sim.py` has been removed) gain `--eval3` + `--colors red,blue,yellow` + `--release-detect {manual,vision,timed}` + `--no-home-between-subgoals`. The outer loop is ~80 LoC; the in-env per-sub-goal reset (target color + latches) needs ~20 LoC to reach into `env.unwrapped.command_manager._terms["target_color"]` the same way `_force_bowl_xy` does today.
7. **(Optional) Three sanity tests** (parallel to Eval 2 §14.4) — scene-color assertion, command-color trace under target reassignment, wrist-image highlight test after Stage 1.

**Estimated effort:** ~1 day for items 1–5, ~0.5 day for item 6, then training (~2 hrs Stage 1 + ~1 hr Stage 2 + ~10 hrs Stage 3 on a 32 GB GPU). Total wall-clock to a deployable Eval-3 policy: **~2 days build + ~13 hrs training**.

## 14. Correctness audit

The two binding-correctness sections from Eval 2 §14 (Binding A: prim ↔ visual color, Binding B: palette index ↔ prim ↔ one-hot bit) apply **verbatim** — same `COLOR_NAMES`, same `BLOCK_COLORS`, same scene-asset registration pattern. Binding C (visual color ↔ one-hot bit, learned by the CNN+MLP) stretches from a 2-way to a 4-way discrimination problem; the mask channel (sourced from `class:cube_<color>` seg + Florence-2 at deploy) carries most of the colour-grounding load, with FiLM as the fallback when the mask drops.

**No tick-ordering invariants to worry about.** The original sequence-conditioned plan needed three invariants (write-then-consume of release indicator, pre-advancement step_idx read, post-advancement obs) because the sequence command and the per-step reward shared mutable env state within a tick. Option A has no per-step state — `TargetColorCommand` writes once at reset, the reward reads it every tick, no advancement happens inside the env. Significantly simpler.

**One verification test worth running before Stage-1 launch:** dump 100 random rollouts from `Isaac-SO-ARM101-Eval3Clutter-Teacher-Fast-Play-v0`, log `(active_indices, target_in_active, target_palette_idx)` per env, and assert `active_indices.gather(1, target_in_active.unsqueeze(1)).squeeze(1) == target_palette_idx` for all envs. Catches gather-index off-by-ones in the `TargetColorCommand` 4-slot extension (item 2 of §13). 30-line test.

## 15. Fallback — sequence-conditioned PPO (Option B)

Preserved in case Option A fails to clear the success bar. **Decision rule:** if Stage-3 of Option A plateaus at `target_in_bowl < 60 %` after 1500 iters *and* none of the §9.1 interventions recover it, switch to Option B.

The full Option-B design lived in earlier revisions of this doc and is parked in the existing `tasks/seqpickplace/` scaffold (which already contains the env cfg, `SequentialGoalCommand`, `place_seq_blocks`, per-step reward stack, per-step latches, and per-step terminations). What it adds over Option A:

- **Sequence command** with `_seq_step_idx`, `_seq_step_release_indicator`, and a `seq_goal_vector (11-D) = color_onehot(6) ⊕ bowl_xy(2) ⊕ step_onehot(3)` policy obs. Step advances inside the env on detected release.
- **Per-step latches** (`_seq_was_grasped`, `_seq_was_over_bowl_above_rim`) cleared on step advance.
- **Per-step reward gating** via `step_active` multiplier + a one-shot `step_completion_bonus` with grading-aligned weights (4, 4, 2).
- **Step-skip curriculum** during Stage 1 — auto-advance stalled envs after a per-step time budget (4 s → 8 s anneal) so all 3 steps see gradient even when step 0 fails. Critical contract: do **not** write `_seq_step_release_indicator` from the skip path (would pay the bonus for failing).
- **Tick-ordering invariants** documented in the earlier revision: release indicator is OR-latched by reward and consumed by command-update within a single tick; `step_completion_bonus` reads pre-advancement `_seq_step_idx`; obs read post-advancement. All three are owned by the Isaac Lab manager order (reward → termination → command → obs).
- **Episode length** = 15.0 s = 750 steps (3 × 5-s sub-budget). Same `γ = 0.98` reasoning, much longer credit chain (600 steps between action 0 and step-2 release reward).

Build cost: ~500 LoC on top of the existing scaffold (agents directory + step-skip curriculum + Teacher/Student gym IDs + deploy refactor). Training cost: same as Option A (~13 hrs), but iter count up to 3000 / 3500 for the longer horizon.

If we end up needing Option B, the seqpickplace scaffold under `tasks/seqpickplace/` is roughly 80 % built — fill in the `agents/`, the step-skip curriculum (`auto_advance_stalled_steps` event term), and gym registrations for Teacher / Student variants.

## References

- **Code:** `tasks/eval3clutter/{eval3clutter_env_cfg,joint_pos_env_cfg}.py`, `mdp/{commands,events,observations,rewards,terminations}.py`, agents (to add). Eval-2 starting point: `tasks/clutterpickplace/`. Fallback scaffold: `tasks/seqpickplace/`. Predecessor plans: [`EVAL1_PLAN.md`](./EVAL1_PLAN.md), [`EVAL2_PLAN.md`](./EVAL2_PLAN.md).
- **Pinto et al.**, *Asymmetric Actor Critic*, RSS 2018. <https://arxiv.org/abs/1710.06542>
- **Levine et al.**, *End-to-End Visuomotor Policies*, JMLR 2016 — spatial softmax.
- **Kostrikov et al.**, *DrQ*, ICLR 2021. <https://arxiv.org/abs/2004.13649>
- **RSL-RL**, Schwarke et al., 2025. <https://github.com/leggedrobotics/rsl_rl>
- **LeIsaac**, <https://github.com/LightwheelAI/leisaac> — wrist camera mount.
