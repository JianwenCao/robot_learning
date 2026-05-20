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

Two-group asymmetric A-C as Eval 2.

| Group | Fields |
|---|---|
| `policy` (27-D) | identical to Eval 2 |
| `critic` | `policy` + `all_active_block_positions` (4×3 = 12), `target_block_position` (3), `target_gripper_to_block` (3), `target_block_to_bowl_xy` (2), `target_is_grasped` (1) |

No `target_color_onehot`, no image, no step one-hot, no `seq_goal_vector`. Multi-step semantics live in the deploy scheduler.

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

## 8. Outer sequencing loop

Per sub-goal `k ∈ {0, 1, 2}`:

1. `detector.set_target_id(id_for(colors[k]))` (real) or set `target_color_onehot` in `TargetColorCommand` buffer (sim).
2. Set `target_bowl_xy` to the rollout's bowl xy.
3. Reset per-episode latches (`env._was_grasped[:] = False`, `env._was_over_bowl_above_rim[:] = False`).
4. Roll policy up to `sub_goal_budget_s = 5.0 s` (250 ticks).
5. Advance on detected release (lift latch ∧ over-bowl latch ∧ gripper open ∧ target in bowl xy < 6 cm ∧ z < 6 cm).
6. Optionally home to `JOINT_DEFAULTS_RAD` for 0.5 s between sub-goals.
7. Score: `4·𝟙[k=0 done] + 4·𝟙[k=1 done] + 2·𝟙[k=2 done]`.

Cubes and bowl are sampled once at env reset (start of sub-goal 0); world state persists between sub-goals.

Release-detect modes on real: `manual` (operator Enter on bowl landing), `vision` (target AprilTag pose `‖tag_xy − bowl_xy‖ < 6 cm ∧ tag_z < 6 cm`), `timed` (5-s budget, unconditional).

Same policy weights drive all 3 sub-goals — "switching" is re-keying of the perception input, not a checkpoint swap.

## 9. Bonus options

**Speed.** No model change. Scheduler tuning: skip homing, `--release-detect vision`, shorten sub-goal budget to 4.0 s. Optional finetune (~500 iters) with `episode_length_s = 4.0`.

**Singulation.** Separate task — see [`BONUS_B_PLAN.md`](./BONUS_B_PLAN.md). Reads all 6 tag positions instead of one target tag.
