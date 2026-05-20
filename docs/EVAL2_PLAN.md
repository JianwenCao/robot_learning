# Eval 2 — Targeted Pick-and-Place in 2-Cube Clutter (State-Only + AprilTag)

Task: `Isaac-SO-ARM101-ClutterPickPlace-StateAprilTag-v0`. Single-stage camera-free PPO on privileged state plus a noisy `target_cube_pos_xy` filled at deploy by AprilTag pose. Detector selects the target cube via `set_target_id`; the policy is colour-blind.

Inherits calibration, detector, sim noise model, and training shape from [`EVAL1_PLAN.md`](./EVAL1_PLAN.md). Deltas only below.

---

## 1. MDP

| Item | Value |
|---|---|
| Control | 50 Hz |
| Episode | 5.0 s = 250 steps |
| Action | 5 arm joints (`scale=0.5`) + 1 binary gripper (verbatim Eval 1) |
| Workspace | bowl: `x ∈ [0.15, 0.28]`, `y ∈ [−0.12, 0.12]`; cluster center: `x ∈ [0.15, 0.22]`, `y ∈ [−0.10, 0.10]` |
| Home `q` | `(0, 0, 0, 1.5708, 0, 0)`, gripper open |
| Terminations | `time_out`, `block_off_table_any` |

**Scene.** Six 2 cm `CuboidCfg` cubes baked with palette colours. Per reset, `place_clutter_blocks` samples two distinct palette indices as the active pair and one of those as the target (the `TargetColorCommand` is a passive view onto these env buffers); the other four cubes parked at `HIDDEN_PARK_XY` (env-local `x = −0.6 m`, outside the table footprint and the wrist-cam FOV at any reachable joint config). The active pair is placed by **independent rejection sampling** in `[block_x] × [block_y] = [0.13, 0.25] × [−0.12, 0.12] m`, with cube B redrawn (up to `max_attempts = 20`) until pairwise distance ≥ **`min_block_separation = 0.12 m`** (12 cm centre-to-centre ≈ 10 cm edge-to-edge gap for 2 cm cubes — comfortably wider than the SO-ARM gripper finger span so the policy can approach either cube without colliding with the other). Bowl rejection-sampled vs the active pair (`ClusterBowlPoseCommandCfg`, ≥ 0.15 m from each active cube).

A previous draft of this plan described an "attached cluster at `half_separation = 0.0105 m`" — that's a fiction; the actual `events.py` has always used rejection-sampled spread placement. The 12 cm minimum (bumped from the prior 10 cm) leaves more margin for AprilTag noise + ee-approach jitter so distractor contact during the approach is rare. If the policy ever needs harder cases, dial `min_block_separation` down via a curriculum in `events.py`; do not go below 0.08 m without re-shaping the reward.

## 2. Observations

| Group | Fields |
|---|---|
| `policy` (27-D) | `joint_pos_rel`, `joint_vel_rel`, `gripper_state`, `bowl_xy`, `ee_proj_xy`, `ee_to_bowl_xy`, **`target_cube_pos_xy_noisy`** (2), `last_action` |
| `critic` | `policy` + `target_block_position` (3), `distractor_block_position` (3), `target_block_to_bowl_xy` (2), `target_gripper_to_block` (3), `target_is_grasped` (1) |

`target_cube_pos_xy_noisy` uses Eval-1 §3 noise model verbatim; post-grasp hold keyed on the *target* `is_grasped`.

## 3. Reward

Eval-1 skeleton + two clutter terms (`mdp/rewards.py`):

| Term | Weight | Trigger |
|---|---|---|
| `reaching_object` | 1.0 | `1 − tanh(‖ee − target_block‖ / 0.05)` |
| `lifting_object` | 15.0 | `𝟙[target_block_z > 0.07]` |
| `object_goal_tracking` (+ fine) | 16.0 + 5.0 | as Eval 1 |
| `release_in_bowl` | 30.0 | target near bowl ∧ `z < BOWL_RIM_Z` (= 0.06 m) ∧ gripper open ∧ settled, gated on lift (z ≥ 0.07 m) + over-bowl-above-rim (z ≥ 0.12 m within 6 cm xy) latches — see Eval-1 §4 |
| `gripper_open_above_bowl_lure` | +3.0 | target-aware port of Eval-1's lure: gated on `_target_was_over_bowl_above_rim`, pays when `action[5] > 0` (open command) |
| `still_grasped_above_bowl_penalty` | −1.0 | target-aware port of Eval-1's anti-hover: target lifted ∧ currently above rim near bowl ∧ gripper closed |
| `distractor_disturb` | −0.5 | continuous, proportional to distractor speed once `> 0.05 m/s` |
| `wrong_block_in_bowl` | **0 → −5** (20 k env-step ramp) | distractor settled in bowl |
| `action_rate`, `joint_vel` | −1e-4 → −1e-2 | 15 k env-step ramp |

γ = 0.98.

**Anti-hover terms + `wrong_block_in_bowl` ramp (2026-05-20 fix).** Two consecutive runs (`08-41-24`, `09-00-17`) of the pre-fix recipe stalled in a "reach-and-camp" basin: reach saturated near 0.7/step but `lifting_object = 0` for the entire 408-iter run, while `wrong_block_in_bowl` ramped to −1.06 weighted/episode (the distractor was drifting into the bowl during random exploration and the −5 fired before the policy had learned to discriminate "what I did" vs "ambient drift"). Two coupled fixes: (1) port Eval-1's 2026-05-20 anti-hover pair (`gripper_open_above_bowl_lure` w=+3, `still_grasped_above_bowl_penalty` w=−1) keyed on the target's latches — these break the reach-and-camp local maximum exactly as in Eval-1; (2) ramp `wrong_block_in_bowl` weight 0 → −5 over 20 k env-steps so reach+lift consolidate before the distractor penalty turns on. 20 k > the 15 k action-rate ramp so the policy has cleared all reward shaping consolidation before the wrong-block penalty becomes load-bearing.

## 4. Curriculum & DR

- `place_clutter_blocks` per §1 (static `min_block_separation = 0.12 m`, rejection-sampled).
- `reset_cube_positions_bias`: per-episode shared hand-eye bias U[−5, +5] mm + per-cube tape offset U[−2, +2] mm on `target_cube_pos_xy_noisy`; clears the post-grasp freeze latch.
- `reset_target_latches`: clears per-episode lift / over-bowl / success latches.
- Action-rate / joint-vel ramp at 15 k env-steps; `wrong_block_in_bowl` ramp 0 → −5 at 20 k env-steps (see §3).
- `log_target_success_metrics` TB metric (TC.success rate + sub-rates).

No image DR. Tightening curriculum (`min_block_separation: 0.12 → 0.08 m` over 20 k env-steps) is available as a one-line `events.py` change if the policy needs harder distributions; do not drop below 0.08 m without a corresponding reward-shaping pass — the SO-ARM gripper geometry constrains how close the distractor can be without contact during the approach.

## 5. Network & PPO

Identical actor-critic class to Eval 1. RSL-RL `ActorCritic`, `[256, 128, 64]` ELU.

| | State PPO (`state_apriltag_ppo_cfg.py`) |
|---|---|
| `num_envs` | 2048 |
| `num_steps_per_env` | 32 |
| `max_iterations` | 2000 |
| `init_noise_std` | 1.0 |
| `entropy_coef` | 0.006 |
| `epochs / mini-batches` | 5 / 4 |
| `learning_rate / desired_kl` | 1e-4 / 0.01 |
| `γ / λ / clip / max_grad_norm` | 0.98 / 0.95 / 0.2 / 1.0 |
| `experiment_name` | `clutterpickplace_state_apriltag` |
| `obs_groups` | `{"policy": ["policy"], "critic": ["policy", "critic"]}` |

