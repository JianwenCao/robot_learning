# Eval 3 — Multi-Step Pick-and-Place via Policy Switching (State-Only + AprilTag)

**Approach:** reuse the Eval-2 state-only + AprilTag policy (retrained on the Eval-3 4-cube cluster scene) and sequence three sub-goals externally on the deploy side. The TA has confirmed policy / perception-module switching is allowed — this turns a 3-sub-goal long-horizon problem into three independent single-target rollouts driven by a thin scheduler.

Assumes familiarity with [`EVAL1_PLAN.md`](./EVAL1_PLAN.md) (foundational AprilTag plan — calibration, detector, sim noise model) and [`EVAL2_PLAN.md`](./EVAL2_PLAN.md) (colour-blind state-only clutter — distractor handling, target-id detector). Where Eval 3 inherits a piece verbatim, the prose references those docs rather than re-deriving.

The previous 3-stage vision pipeline (state teacher → vision distill → vision PPO + teacher-critic warm-start, with sequence-conditioned `seq_goal_vector` and step-skip curriculum) is retired. The §15 fallback to a sequence-conditioned single rollout in the earlier draft is also retired — with policy switching plus a fast state-only policy there's no scenario where the sequence-conditioned long-horizon design wins.

---

## 1. Spec recap

Four blocks of distinct colours are placed in the workspace. The policy is given a sequence of three `(target_color, bowl_xy)` goals; the bowl is **shared across all three sub-goals** within a rollout. Per-step scoring is **4, 4, 2** (10 pts/rollout × 5 rollouts = 50 pts), partial credit awarded, success = correct block in the bowl and released. RL is required. Policy switching / perception-module switching is allowed. Distractor interaction (pushing, rearranging) is permitted and encouraged if it helps. Optional 50-pt bonus on either Speed or Singulation (Bonus B; see [`BONUS_B_PLAN.md`](./BONUS_B_PLAN.md)).

## 2. Why state-only + AprilTag is even cleaner for Eval 3

Two things compound:

1. **No colour-conditioning in the policy.** Same as Eval 2 — the AprilTag detector picks the right cube via `set_target_id`, the policy is colour-blind. For Eval 3 this means *the same policy weights drive all 3 sub-goals*; we just call `set_target_id` between sub-goals.
2. **No CNN / Florence / re-prompting latency.** Florence-2 took ~2–5 s/CPU per detection in the old plan — that dominated speed-bonus runtime and required a background-thread caching dance. AprilTag detection is ~2 ms/frame on CPU, so we can detect every frame at policy rate and there is no per-sub-goal prompt-switching cost beyond a single integer assignment.

Net: Eval 3 reduces to "run an Eval-2 policy three times in a row with different `set_target_id` calls." The scheduler is ~80 LoC and contains no perception state.

## 3. What carries over

| Component | Eval 2 (state + AprilTag) | Eval 3 (Option A, chosen) | Reuse |
|---|---|---|---|
| Robot, control, action space, home `q` | ✓ | ✓ | **verbatim** |
| Workspace, table, palette | ✓ | ✓ | **verbatim** |
| Hand-eye + `AprilTagDetector` | ✓ | ✓ | **verbatim** |
| Sim noise model for `target_cube_pos_xy_noisy` | ✓ | ✓ | **verbatim** |
| Policy obs schema (no `target_color_onehot`, no image) | 27-D | **27-D** | **verbatim** |
| MLP actor-critic `[256, 128, 64]` ELU | ✓ | ✓ | **verbatim** |
| Reach / lift / transport / release / wrong-cube reward stack | ✓ | ✓ | **verbatim** |
| Two latches (lift ≥ 0.07, over-bowl-above-rim ≥ 0.08) | ✓ | ✓ | **verbatim** |
| Single-stage state PPO with privileged critic | ✓ | ✓ | **verbatim** |
| `place_clutter_blocks` (2 cubes attached) | ✓ | replaced with `place_four_attached_cluster` | **new placement event** |
| `TargetColorCommand` (1 active, 1 distractor) | ✓ | extended to 1 active target + 3 distractors | **new active-set size** |
| Eval-2 checkpoint transfer | — | possible warm-start (§9 alt); critic dim differs so not free | **soft reuse** |

The only required deltas vs Eval 2 are: (i) a new placement event for the 4-cube 2×2 cluster, (ii) `TargetColorCommand` extended to track 4 active cubes, (iii) the critic obs widened from 1→3 distractors, (iv) gym IDs, (v) the outer deploy scheduler that calls `set_target_id` between sub-goals. Everything else is bitwise identical to Eval 2.

## 4. MDP

| Item | Value |
|---|---|
| Control | 50 Hz (verbatim) |
| Episode | 5.0 s = 250 steps **per sub-goal rollout** — the 3-sub-goal outer time budget lives in the deploy scheduler, not the env |
| Action | 5 arm joints (`scale=0.5`) + 1 binary gripper (verbatim) |
| Cluster center | `x ∈ [0.16, 0.20]`, `y ∈ [−0.08, 0.08]`, 4 cubes in a 2×2 attached cluster (`half_separation = 0.0105 m`), yaw θ ∈ U[0, 2π) |
| Bowl | `x ∈ [0.18, 0.26]`, `y ∈ [−0.08, 0.08]`, rejection-sampled `≥ 0.08 m` from cluster center |
| Terminations | `time_out`, `active_block_off_table`. **No `task_success`** (Eval-1/2 reasoning) |

**Scene composition.** Six 2 cm `CuboidCfg` cubes baked with the six palette colours. Per reset, `place_four_attached_cluster` samples a length-4 random permutation of the palette, places those 4 as a 2×2 attached cluster (center + yaw θ; corners at `(±half_separation, ±half_separation)` rotated by θ), parks the other 2 at `HIDDEN_PARK_XY`. The bowl is a 2-D goal from the per-target command (no scene prim).

**Why an attached 2×2 cluster.** Matches the real Eval-3 setup the TA has been describing; spec explicitly encourages distractor interaction in the configuration where it pays off (free a target by nudging neighbours).

**Target sampling.** `TargetColorCommand` samples the target uniformly from the 4 active slots per reset (`target_in_active ∈ {0,1,2,3}`). The full active set is exposed to the critic; the policy receives only the (noisy) `target_cube_pos_xy` of the sampled target. A single policy rollout solves one sub-goal; the deploy-side scheduler re-keys the detector for sub-goals 2 and 3 (§9).

## 5. Observations

Same two-group asymmetric A-C as Eval 2.

| Group | Fields | Shape |
|---|---|---|
| `policy` (deployable) | `joint_pos_rel`, `joint_vel_rel`, `gripper_state`, `bowl_xy`, `ee_proj_xy`, `ee_to_bowl_xy`, **`target_cube_pos_xy_noisy`** (2), `last_action` | 27-D, identical to Eval 2 |
| `critic` (privileged) | `policy` + `all_active_block_positions` (4×3 = 12), `target_block_position` (3), `target_gripper_to_block` (3), `target_block_to_bowl_xy` (2), `target_is_grasped` (1) | wider critic |

**No `target_color_onehot`, no image, no step one-hot, no `seq_goal_vector`.** Multi-step semantics live entirely outside the env. The detector picks the right cube; the policy doesn't know which colour or which sub-goal it's on. (It doesn't need to — same MDP per sub-goal.)

`target_cube_pos_xy_noisy` shares the §6-of-Eval-1 noise model verbatim, with `is_grasped` keyed on the target cube.

## 6. Network — `Eval3StateActorCritic`

Bitwise identical to Eval 2's state-only actor-critic. Pure MLP `[256, 128, 64]` ELU, σ scalar Param, standard RSL-RL `ActorCritic`. The only delta is the critic input dim widening (`all_active_block_positions` 6 → 12 dims for 4 cubes × 3). The actor MLP input dim is **unchanged** between Eval 2 and Eval 3.

No CNN, no FiLM, no DrQ — there's no image observation.

## 7. Reward

Verbatim from Eval 2 §5, except `wrong_block_in_bowl` widens from "the one distractor" to "any of the three non-target active cubes settled in the bowl." Penalty −15 (vs Eval 2's −20) because there are 3 candidate distractors and a single near-miss should still leave net reward positive after a correct final placement.

| Term | Weight | Trigger |
|---|---|---|
| `reach_target` | 1.0 | `1 − tanh(‖ee − target‖ / 0.05)` |
| `lift_target` | 15.0 | `𝟙[target_z > 0.07]` (per-episode latch) |
| `transport_target_to_bowl` (coarse) | 16.0 | `was_lifted · (1 − tanh(‖target_xy − bowl_xy‖ / 0.30))` |
| `transport_target_to_bowl` (fine) | 5.0 | same with `std = 0.05` |
| `release_target_in_bowl` | 30.0 | target near bowl ∧ `z < 0.06` ∧ gripper open ∧ settled, gated on lift + over-bowl-above-rim latches |
| `wrong_cube_in_bowl` | **−15.0** | any non-target *active* cube sits inside the bowl |
| `action_rate`, `joint_vel` | −1e-4 → −1e-2 | 10 k env-step ramp |

**No distractor-disturb penalty.** Spec explicitly encourages distractor interaction.

`release_in_bowl` fires every step the predicate holds (Eval-1/2 reasoning). γ = 0.98 load-bearing.

## 8. Curriculum & DR

- `place_four_attached_cluster` (new event) — 4 cubes in a 2×2 cluster per §4.
- `reset_target_latches` cleared per episode.
- `reset_cube_pos_bias`: per-episode U[−5, +5] mm bias on `target_cube_pos_xy_noisy`.
- Action-rate / joint-vel penalty ramp at 10 k env-steps.
- `log_seq_success_metrics` TB metric (per-sub-goal success aggregated across rollouts of the same env).

**No image DR**, no HSV jitter, no DrQ. Camera-free.

**No `n_active_blocks` curriculum.** With state obs + AprilTag, the perceptual difficulty of 4 cubes vs 2 cubes is zero — same `target_cube_pos_xy` slot. Physics dynamics are slightly heavier (4 cubes contacting), but that's the only delta and PPO handles it directly. The vision-era plan needed this curriculum because the CNN had to learn 4-way colour discrimination; we don't have a CNN.

## 9. PPO

Eval-2 skeleton, marginally longer iter count for the 4-cube physics regime.

| | State PPO (`state_apriltag_ppo_cfg.py`) |
|---|---|
| `num_envs` | 2048 |
| `num_steps_per_env` | 32 |
| `max_iterations` | 2500 (vs Eval 2's 2000) |
| `init_noise_std` | 1.0 |
| `entropy_coef` | 0.006 → 0.003 after ~1500 iters |
| `epochs / mini-batches` | 5 / 4 |
| `learning_rate / desired_kl` | 1e-4 / 0.01 |
| `γ / λ / clip / max_grad_norm` | 0.98 / 0.95 / 0.2 / 1.0 |
| `experiment_name` | `eval3clutter_state_apriltag` |
| `obs_groups` | `{"policy": ["policy"], "critic": ["policy", "critic"]}` |

Wall-clock estimate: ~3 h to ≥ 70 % sim success.

**Optional warm-start from Eval-2 checkpoint.** Same actor MLP input dim, so `actor.*` weights load directly. The critic input dim differs (12 vs 6 for `all_active_block_positions`) so the Eval-2 critic does **not** transfer — a fresh state PPO run will retrain the critic from scratch alongside. Trying this saves ~30 % of wall-clock; default path is fresh-from-scratch.

## 10. Deploy — outer sequencing loop

The detector is the per-sub-goal handle. Same policy weights for all 3 sub-goals.

### 10.1 Sim deploy (`scripts/rsl_rl/play.py --eval3`)

```bash
uv run play --task Isaac-SO-ARM101-Eval3Clutter-StateAprilTag-Play-v0 \
    --load_run <run> --eval3 \
    --bowl-xy 0.22,0.0 --colors red,blue,yellow
```

Per sub-goal `k ∈ {0, 1, 2}`:

1. Set `target_color_onehot` in the env's `TargetColorCommand` buffer to `onehot(colors[k])`. (At sim — the env still uses colour to pick the *physical* cube; the policy obs doesn't see this.)
2. Set `target_bowl_xy` to the rollout's single bowl xy.
3. Reset per-episode latches (`env._was_grasped[:] = False`, `env._was_over_bowl_above_rim[:] = False`).
4. Roll the policy forward up to `sub_goal_budget_s = 5.0 s` (250 ticks).
5. On detected release (lift latch ∧ over-bowl latch ∧ gripper open ∧ target in bowl xy < 6 cm ∧ z < 6 cm — same predicate as the sim reward), advance to sub-goal `k+1`.
6. Optionally home the arm to `JOINT_DEFAULTS_RAD` for 0.5 s between sub-goals (skip for the speed-bonus run).
7. Score: `4·𝟙[k=0 done] + 4·𝟙[k=1 done] + 2·𝟙[k=2 done]`.

Cubes and bowl are sampled once at env reset (start of sub-goal 0) and *not* re-sampled between sub-goals — world state persists, so sub-goal 1 starts from wherever sub-goal 0 left it. Matches the real eval.

### 10.2 Real deploy (`deploy/deploy_real.py --eval3`)

```bash
python -m deploy.deploy_real --eval3 \
    --policy-ckpt deploy/runs/eval3_state.pt \
    --bowl-xy 0.22,0.0 \
    --colors red,blue,yellow \
    --target-id-map "0=red,1=blue,2=yellow,3=green,4=purple,5=orange" \
    --release-detect manual
```

Per sub-goal:

1. `detector.set_target_id(id_for(colors[k]))` — one integer assignment, no model reload.
2. Run policy for up to 5 s.
3. Release detection (no kinematic oracle on real):
   - `manual` (default): operator presses Enter when the cube lands in the bowl. Simple, reliable.
   - `vision`: detect the *target* AprilTag inside the bowl region — easy because the tag stays on the cube. `tag_pose.xy` within 6 cm of `bowl_xy` and `tag_pose.z < 0.06` triggers advance. ~50 ms latency vs human reaction time, useful for the speed bonus.
   - `timed`: fixed 5-s budget, advance unconditionally. Worst — masks failures.
4. Homing default ON for reliability; `--no-home-between-subgoals` for speed.

**`--release-detect vision` is much easier here than in the vision plan.** The old plan had to HSV-match the target colour against the bowl region — fragile under lighting drift. We now have a stable, calibrated tag pose; `‖tag_xy − bowl_xy‖ < 6 cm ∧ tag_z < 6 cm` is a clean boolean.

### 10.3 Policy-switching contract

Same policy weights drive all 3 sub-goals. The "switching" is conceptually a re-keying of the perception input (`set_target_id`), not a checkpoint swap. We *could* load distinct checkpoints between sub-goals (e.g., a "first-cube-of-cluster" policy vs an "after-disturbance" policy), but it isn't needed — and would introduce a state-distribution gap between sub-goals that the current single-policy design avoids.

## 11. Bonus options

### 11.1 Speed bonus

No model change. Tune the scheduler:

- Skip homing between sub-goals (`--no-home-between-subgoals`).
- Use `--release-detect vision` (the AprilTag-based predicate is ~50 ms vs ~1-2 s for human reaction).
- Shorten per-sub-goal budget to 4.0 s (training was 5.0 s — leaves headroom).
- Optional: retrain Stage-1 with `episode_length_s = 4.0` for a "speed-tuned" finetune (~500 iters, predicted 20–40 % faster median, slight success trade).

Speed-bonus payoff is bigger here than in the vision plan because Florence-2's per-frame latency is gone — the detection step is now sub-ms.

### 11.2 Singulation bonus

Separate task — see [`BONUS_B_PLAN.md`](./BONUS_B_PLAN.md). Same state + AprilTag foundation; reads all 6 tag positions instead of one target tag.

## 12. What needs to be built

Build order — items 1–3 unblock training, items 4–5 unblock deploy. Paths relative to `isaac_so_arm101/src/isaac_so_arm101/tasks/`.

1. **Placement event `place_four_attached_cluster`** in `tasks/eval3clutter/mdp/events.py` (or extend `tasks/clutterpickplace/mdp/events.py` with `placement_mode: "spread" | "cluster_2x2"` + `n_active_blocks: int = 2` params). 2×2 attached cluster (`half_separation = 0.0105 m`), cluster center in `[0.16, 0.20] × [−0.08, 0.08]`, yaw θ ∈ U[0, 2π). Park 2 unused cubes at `HIDDEN_PARK_XY`. **~80 LoC.**
2. **`TargetColorCommand` extension** to 4 active slots. Generalize in place: `target_in_pair: (N,) ∈ {0,1}` → `target_in_active: (N,) ∈ {0,…,n_active-1}`, `active_indices: (N, 2)` → `(N, n_active)`. No behaviour change at `n_active = 2`, so Eval 2 keeps working. **~30 LoC.**
3. **New task package `tasks/eval3clutter/`** mirroring `tasks/clutterpickplace/`:
   - `eval3clutter_env_cfg.py` — copy of `clutterpickplace_env_cfg.py` with `all_active_block_positions: 2 → 4`, no image obs group (already absent in state-only), `place_four_attached_cluster` in `EventCfg`.
   - `joint_pos_env_cfg.py` — copy with cluster placement, no `wrist_cam` spawn.
   - `__init__.py` — gym IDs: `…-StateAprilTag-v0`, `…-StateAprilTag-Play-v0`.
   - `mdp/{events,commands,observations,rewards,terminations}.py` — copies from Eval 2 with the deltas above.
4. **`agents/state_apriltag_ppo_cfg.py`** — copy of Eval 2's cfg with the wider critic obs and `max_iterations = 2500`. **~40 LoC.**
5. **Deploy scheduler** — `deploy/deploy_real.py` gains `--eval3` + `--colors red,blue,yellow` + `--release-detect {manual,vision,timed}` + `--no-home-between-subgoals`. Outer loop ~80 LoC; per-sub-goal `set_target_id` is a single line. The release-detect `vision` mode is a 10-LoC predicate over the existing detector output.
6. **(Optional) Sanity test** — assert `active_indices.gather(1, target_in_active.unsqueeze(1)).squeeze(1) == target_palette_idx` over 100 rollouts of the new task; catches gather-index off-by-ones in the `TargetColorCommand` 4-slot extension.

**Estimated effort:** ~0.5 day for items 1–4, ~0.5 day for item 5, then training (~3 h state PPO on a single GPU; no Stage 2/3). Total wall-clock to a deployable Eval-3 policy: **~1 day build + ~3 h training.**

This is roughly a 10× build-cost reduction vs the old 3-stage vision plan, and the trained policy can also serve Eval 2 by setting `n_active_blocks = 2` at deploy.

## 13. Correctness audit

The two binding-correctness sections from the old Eval-2 §14 (Binding A: prim ↔ visual color, Binding B: palette index ↔ prim ↔ one-hot bit) **no longer apply** — there is no one-hot in the policy obs and no CNN to ground colour. The only remaining binding is now physical: **AprilTag ID ↔ palette colour ↔ cube prim**, enforced by:

- Print + stick: tag ID `k` goes on the cube with `BLOCK_COLORS[COLOR_NAMES[k]]`.
- Deploy: `--target-id-map` CLI populates the lookup `set_target_id(id_for(target_color))`.
- Sim: `place_clutter_blocks` / `place_four_attached_cluster` writes `env._active_cube_indices` with palette indices, and the reward stack reads `env._target_cube_idx` to select reward gates.

**One verification test worth running before any sim PPO launch**: dump 100 random rollouts from `…-StateAprilTag-Play-v0`, log `(active_indices, target_in_active, target_palette_idx)` per env, and assert `active_indices.gather(1, target_in_active.unsqueeze(1)).squeeze(1) == target_palette_idx` for all envs. 30-line test.

**One verification step at deploy bring-up**: place all 6 tagged cubes in the workspace, call `set_target_id(k)` for each `k`, and verify the detector returns the pose of the correctly-coloured cube. Catches off-by-ones in `--target-id-map` parsing.

**No tick-ordering invariants to worry about.** Same as Eval 2.

## References

- Foundation: [`EVAL1_PLAN.md`](./EVAL1_PLAN.md). Predecessor: [`EVAL2_PLAN.md`](./EVAL2_PLAN.md). Bonus B: [`BONUS_B_PLAN.md`](./BONUS_B_PLAN.md).
- Code: `tasks/eval3clutter/{eval3clutter_env_cfg,joint_pos_env_cfg}.py`, `mdp/{commands,events,observations,rewards,terminations}.py`, `agents/state_apriltag_ppo_cfg.py`. Predecessor scaffold: `tasks/clutterpickplace/`. Real-robot deploy: `deploy/{deploy_real.py,cube_detector.py}`.
- **RSL-RL**, Schwarke et al., 2025. <https://github.com/leggedrobotics/rsl_rl>
- **LeIsaac**, <https://github.com/LightwheelAI/leisaac> — wrist camera mount.
