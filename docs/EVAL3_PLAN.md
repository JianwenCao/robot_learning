# Eval 3 — Multi-Step Pick-and-Place via Policy Switching (State-Only + AprilTag)

Reuse the Eval-2 state-only + AprilTag policy (retrained on the Eval-3 4-cube cluster scene) and sequence three sub-goals externally on the deploy side. Same policy weights drive all 3 sub-goals; the detector is the per-sub-goal handle (`set_target_id`).

Inherits foundations from [`EVAL1_PLAN.md`](./EVAL1_PLAN.md) and [`EVAL2_PLAN.md`](./EVAL2_PLAN.md). Deltas only below.

---

## 1. Spec recap

Four blocks of distinct colours in the workspace. Sequence of three `(target_color, bowl_xy)` goals; bowl shared across all three. Per-sub-goal scoring 4 / 4 / 2 (10 pts/rollout × 5 = 50 pts), partial credit. Distractor interaction permitted. Optional 50-pt Speed or Singulation bonus ([`BONUS_B_PLAN.md`](./BONUS_B_PLAN.md)).

## 2. MDP

| Item | Value |
|---|---|
| Control | 50 Hz |
| Action | 5 arm joints (`scale=0.5`) + 1 binary gripper |
| Episode | 15.0 s = 750 steps per rollout (sim env steps the policy through all 3 sub-goals back-to-back, advancing `_seq_step_idx` on each release); deploy slices into ≤ 5 s budgets per sub-goal |
| Cube workspace | `x ∈ [0.13, 0.28]`, `y ∈ [−0.15, 0.15]`, rejection-sampled with **`min_block_separation = 0.08 m`** (4-cube layout, max 80 attempts) |
| Bowl | `x ∈ [0.15, 0.28]`, `y ∈ [−0.12, 0.12]`, single bowl per rollout, sampled before cubes; cubes then rejection-sampled to keep ≥ `min_bowl_block_separation = 0.15 m` from the bowl |
| Terminations | `time_out`, `active_block_off_table` |

**Scene.** Six 2 cm cubes baked palette colours. Per reset, `place_seq_blocks` samples a length-4 permutation of the palette as the 4 active cubes (the other 2 parked at `HIDDEN_PARK_XY`, env-local `x = −0.6 m`, outside the wrist-cam FOV). The 4 active cubes are placed by **sequential rejection sampling** in the cube workspace, each new cube redrawn until it is ≥ `min_block_separation` from all previously-placed cubes AND ≥ `min_bowl_block_separation` from the bowl. A length-3 permutation of the active slots is sampled as the per-step target schedule (`env._seq_goal_color_pos`, distinct across the 3 steps so a placed cube is never re-picked). Bowl is a 2-D goal (no prim).

The previous draft claimed a fixed `place_four_attached_cluster` at `half_separation = 0.0105 m` — that's a fiction; the code has always used spread rejection-sampling. The 8 cm minimum (bumped from the prior 6 cm) gives ≥ 6 cm edge gap for 2 cm cubes — wider than the SO-ARM gripper finger span — so the policy can pick any active cube without colliding with a neighbour. The **within-episode sub-goal sequencing** means the policy naturally trains on 4-cube, 3-cube, and 2-cube cluster states (one cube removed per successful sub-goal); no extra `randomize_active_count` event is needed to close the train→deploy distribution gap.

## 3. Observations

Two-group asymmetric A-C as Eval 2. Policy obs shape and semantics are **bitwise identical** to Eval-2's (`SeqStateAprilTagObservationsCfg.PolicyCfg`) — the only Eval-3-specific behaviour lives in :func:`mdp.target_cube_pos_xy_noisy`, which reads the *current sub-goal* target palette idx (via `_current_target_palette_idx`, gated by `_seq_step_idx`) so the published xy + post-grasp freeze re-key automatically when the env advances on sub-goal release.

| Group | Fields | Shape |
|---|---|---|
| `policy` | `joint_pos_rel`, `joint_vel_rel`, `gripper_state`, `bowl_xy`, `ee_proj_xy`, `ee_to_bowl_xy`, **`target_cube_pos_xy_noisy`** (2 — current sub-goal target), `last_action` | 27-D |
| `critic` | `policy` + `seq_goal_vector` (11 — full schedule), `all_active_block_positions` (4×3 = 12), `current_target_block_position` (3), `current_target_gripper_to_block` (3), `current_target_block_to_bowl_xy` (2) | wider |

No `target_color_onehot`, no `cube_positions_xy_noisy` for distractors, no `cube_visible_flags`, no step one-hot, no `seq_goal_vector` in the policy stream. The deploy scheduler re-keys the AprilTag detector per sub-goal; the sim env mirrors that re-keying via `_seq_step_idx` advancement on release — both paths drive the same single-target obs into the actor.

## 4. Network

Bitwise identical actor to Eval 2 (`[256, 128, 64]` ELU, σ scalar Param, RSL-RL `ActorCritic`). Critic input dim widens for `all_active_block_positions`.

## 5. Reward

Verbatim from Eval 2 with two changes: `wrong_cube_in_bowl` widens to "any non-target active cube settled in the bowl" at weight −15; no `distractor_disturb` term.

| Term | Weight | Trigger |
|---|---|---|
| `reach_target` | 1.0 | `1 − tanh(‖ee − target‖ / 0.05)` |
| `lift_target` | 15.0 | `𝟙[target_z > 0.07]` |
| `transport_target_to_bowl` (coarse) | 16.0 | `was_lifted · (1 − tanh(‖target_xy − bowl_xy‖ / 0.30))` |
| `transport_target_to_bowl` (fine) | 5.0 | same with `std = 0.05` |
| `release_target_in_bowl` | 30.0 | target near bowl ∧ `z < BOWL_RIM_Z` (= 0.06 m) ∧ gripper open ∧ settled, gated on lift (z ≥ 0.07 m) + over-bowl-above-rim (z ≥ 0.12 m within 6 cm xy) latches — see Eval-1 §4 |
| `wrong_cube_in_bowl` | −15.0 | any non-target active cube in bowl |
| `action_rate`, `joint_vel` | −1e-4 → −1e-2 | 10 k env-step ramp |

γ = 0.98.

## 6. Curriculum & DR

- `place_seq_blocks` per §2 (static `min_block_separation = 0.08 m`, `min_bowl_block_separation = 0.15 m`, rejection-sampled).
- `reset_seq_latches`: clears `_seq_step_idx`, per-step grasp/over-rim/success latches at episode reset.
- `reset_cube_positions_bias`: per-episode shared hand-eye bias U[−5, +5] mm + per-cube tape offset U[−2, +2] mm on `target_cube_pos_xy_noisy`; clears the post-grasp freeze latch (re-keyed per sub-goal at step-advance).
- Action-rate / joint-vel ramp at 10 k env-steps.
- `log_seq_success_metrics` TB metric (per-sub-goal success rate, mean step reached).

Within-episode step advancement handles the 4 → 3 → 2 cube count progression naturally — no separate active-count randomization event.

## 7. PPO

| | State PPO (`state_apriltag_ppo_cfg.py`) |
|---|---|
| `num_envs` | 2048 |
| `num_steps_per_env` | 32 |
| `max_iterations` | 2500 |
| `init_noise_std` | 1.0 |
| `entropy_coef` | 0.006 (constant; matches Eval-2) |
| `epochs / mini-batches` | 5 / 4 |
| `learning_rate / desired_kl` | 1e-4 / 0.01 |
| `γ / λ / clip / max_grad_norm` | 0.98 / 0.95 / 0.2 / 1.0 |
| `experiment_name` | `eval3clutter_state_apriltag` |
| `obs_groups` | `{"policy": ["policy"], "critic": ["policy", "critic"]}` |

## 8. Sub-goal sequencing

Sim training and real deploy share the same per-sub-goal contract — the only difference is *who* advances the step counter (env in sim, deploy script on real).

**Sim (training, env-internal):** the env owns `_seq_step_idx`. The seq command samples the 3-step schedule + bowl at reset; reward terms gate off `_current_target_palette_idx`; `release_current_target_in_bowl` increments `_seq_step_idx` once the strict release predicate fires (lift latch ∧ over-bowl-above-rim latch ∧ gripper-open ∧ in-bowl). Episode runs 15 s = 750 steps total so the policy sees all three sub-goals back-to-back; per-step bonus `(4, 4, 2)` aligns the value function with the human grading rubric. Cube count drops 4 → 3 → 2 across sub-goals; the policy trains on all three regimes.

**Real deploy, per sub-goal `k ∈ {0, 1, 2}`:**

1. `detector.set_target_id(tag_id_for(colors[k]))` — re-key the AprilTag pipeline to the new target. The actor obs slot is the *same* `target_cube_pos_xy_noisy` either way; only the source ID changes.
2. Reset per-sub-goal latches on the deploy state mirror (`_was_grasped`, `_was_over_bowl_above_rim`) so the lift / over-rim gates start fresh.
3. Roll the policy up to `sub_goal_budget_s = 5.0 s` (250 ticks).
4. Advance on release detection — same predicate as sim: lift latch ∧ over-bowl latch ∧ gripper-open ∧ target tag `‖xy − bowl_xy‖ < 0.06 m ∧ z < 0.06 m`. Configurable via `--release-detect {manual,vision,timed}` (operator Enter, AprilTag pose check, or 5-s budget).
5. Optionally home to `JOINT_DEFAULTS_RAD` for ~0.5 s between sub-goals (`--no-home-between-subgoals` to skip for the Speed bonus).
6. Score: `4·𝟙[k=0 done] + 4·𝟙[k=1 done] + 2·𝟙[k=2 done]`.

The cubes and bowl are physical (one reset per rollout, world state persists between sub-goals). **Same policy weights drive all 3 sub-goals** — the only state that changes between sub-goals is the AprilTag detector's target ID (real) or `_seq_step_idx` (sim).

## 9. Bonus options

**Speed.** No model change. Scheduler tuning: skip homing, `--release-detect vision`, shorten sub-goal budget to 4.0 s. Optional finetune (~500 iters) with `episode_length_s = 4.0`.

**Singulation.** Separate task — see [`BONUS_B_PLAN.md`](./BONUS_B_PLAN.md). Reads all 6 tag positions instead of one target tag.
