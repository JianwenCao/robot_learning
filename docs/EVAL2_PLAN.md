# Eval 2 — Targeted Pick-and-Place in 2-Cube Clutter (State-Only + AprilTag)

Color-conditioned PPO on SO-ARM101 → zero-shot real-arm deploy. Task: `Isaac-SO-ARM101-ClutterPickPlace-StateAprilTag-v0`. **Single-stage, camera-free PPO** on privileged state plus a noisy `target_cube_pos_xy` observation filled at deploy by AprilTag pose. No CNN, no FiLM, no Florence-2, no distill.

Assumes the foundational AprilTag plan from [`EVAL1_PLAN.md`](./EVAL1_PLAN.md) — tag family, calibration, sim noise model, detector protocol, training shape are inherited. This doc covers only the Eval-2 deltas: per-cube tags, target ID selection, distractor handling.

---

## 1. What you print

| Item | Quantity | Family | Printed size | Notes |
|---|---|---|---|---|
| Cube tags | 1 per cube, up to 6 | `tagStandard41h12` | 15 mm | IDs `0..5`, lookup `{0:red, 1:blue, 2:yellow, 3:green, 4:purple, 5:orange}` |
| Calibration tag | already printed for Eval 1 | — | 30 mm | Shared across all evals |

For Eval 2 only **2 of the 6** tagged cubes are on the table per rollout, but the policy is colour-agnostic at the obs level — the perception layer just hands it `target_cube_pos_xy`, and the AprilTag detector selects which physical cube is the "target" via `set_target_id(int)`.

## 2. What changes vs Eval 1

| Component | Eval 1 (state + AprilTag) | Eval 2 (state + AprilTag) | Reuse |
|---|---|---|---|
| Hand-eye calibration | `deploy/hand_eye.yaml` | same file | **verbatim** |
| Sim noise model (§6 of Eval 1) | per-axis Gaussian + per-ep bias + pre/post-grasp dropout | same | **verbatim** |
| `cube_pos_xy_noisy` obs term | single cube | **`target_cube_pos_xy_noisy`** — reads `env._target_cube_idx` to select which cube the noisy xy describes | new but mechanically identical |
| Policy obs schema | 27-D (= 25 + `cube_pos_xy`) | **27-D** — `target_cube_pos_xy` replaces `cube_pos_xy`; **no `target_color_onehot`** (the AprilTag picks the right cube; the policy doesn't need to know its colour) | almost verbatim |
| Critic obs | + `block_position`, `is_grasped`, etc. | + `target_block_position`, `distractor_block_position`, `target_is_grasped` (privileged, real cube poses; see §5) | extended |
| Detector | `AprilTagDetector(set_target_id=0)` static | **`set_target_id(id_for(target_color))` once per rollout** | extension |
| Network | MLP `[256, 128, 64]` ELU | same | **verbatim** |
| Training shape | single-stage PPO, ~1–2 h | single-stage PPO, ~2–3 h (more cubes, more dynamics variability) | **verbatim** |
| Vision pipeline (CNN, FiLM, Florence, distill) | gone | gone | **deleted** |

Net: the actor MLP input dim is **unchanged** between Eval 1 and Eval 2 (the AprilTag detector handles target identity outside the policy). The only deltas are the sim placement event, the critic obs widening for the distractor, the reward additions for the wrong-cube failure mode, and a one-line `set_target_id` call in the deploy outer loop.

This is the key payoff of switching to state+AprilTag for Eval 2: we don't need colour-conditioning in the policy at all. There's no FiLM head, no `target_color_onehot`, no per-colour CNN keypoint allocation problem. The detector picks the right cube; the policy just grasps "the cube whose xy is in my obs."

## 3. MDP

| Item | Value |
|---|---|
| Control | 50 Hz (decimation 2, sim 100 Hz) |
| Episode | 5.0 s = 250 steps |
| Action | 5 arm joints (`scale=0.5`) + 1 binary gripper — verbatim from Eval 1 |
| Workspace | bowl: `x ∈ [0.15, 0.28]`, `y ∈ [−0.12, 0.12]`; cluster center: `x ∈ [0.15, 0.22]`, `y ∈ [−0.10, 0.10]` |
| Home `q` | `(0, 0, 0, 1.5708, 0, 0)`, gripper open |
| Terminations | `time_out`, `block_off_table_any` |
| Table | `0.6 × 1.0 × 0.02 m` at `(0.25, 0, −0.01)`; top `z = 0` |

**Scene composition.** Six 2 cm `CuboidCfg` cubes baked with the six palette colours. Per reset, `TargetColorCommand` samples (i) two distinct colours from the palette as the *active pair* and (ii) one of those two as the *target*. `place_clutter_blocks` places the active pair adjacent (`half_separation = 0.0105 m` → 2.1 cm center-to-center, 1 mm spawn gap closing to contact under gravity, pair axis θ ∈ U[0, 2π)); the other four cubes are parked at `HIDDEN_PARK_XY` off-table. Bowl is rejection-sampled vs the active pair (`ClusterBowlPoseCommandCfg`, `≥ 0.15 m` from each active cube; unchanged from the vision-era plan).

**Why we still randomize colour at the sim level even though the policy doesn't see it.** The privileged critic still uses `target_block_position` vs `distractor_block_position`, and the `place_clutter_blocks` event must know which of the two active cubes is "target" so it can wire up the right reward gates. Colour is the bookkeeping handle; the policy itself is colour-blind.

## 4. Observations

```python
# Both groups; runner cfg uses obs_groups = {"policy": ["policy"], "critic": ["policy", "critic"]}.
```

| Group | Fields | Notes |
|---|---|---|
| `policy` (deployable) | `joint_pos_rel`, `joint_vel_rel`, `gripper_state`, `bowl_xy`, `ee_proj_xy`, `ee_to_bowl_xy`, **`target_cube_pos_xy_noisy`** (2), `last_action` | 27-D, no `target_color_onehot` |
| `critic` (privileged) | `policy` + `target_block_position` (3), `distractor_block_position` (3), `target_block_to_bowl_xy` (2), `target_gripper_to_block` (3), `target_is_grasped` (1) | extends Eval 1 critic with one distractor pose |

The distractor pose is in the critic so the value function understands the "wrong-cube" failure mode; the policy only sees the (noisy) target xy. The AprilTag detector at deploy holds the equivalent semantics — when you call `set_target_id(2)`, the detector returns the pose of tag-ID-2 even if tag-ID-3 is also visible.

`target_cube_pos_xy_noisy` shares the §6-of-Eval-1 noise model verbatim: 2 mm Gaussian + ±5 mm per-ep bias + Bernoulli 10 % pre-grasp dropout + post-grasp deterministic hold via `is_grasped`. The hold-on-grasp condition is keyed on the *target* `is_grasped` flag, not any cube — the distractor being occluded by the gripper doesn't matter because we never read its xy from the obs side.

## 5. Reward

Same skeleton as Eval 1, plus two clutter-specific terms (`mdp/rewards.py`):

| Term | Weight | Trigger |
|---|---|---|
| `reaching_object` | 1.0 | `1 − tanh(‖ee − target_block‖ / 0.05)` |
| `lifting_object` | 15.0 | `𝟙[target_block_z > 0.07]` |
| `object_goal_tracking` (+ fine-grained) | 16.0 + 5.0 | as Eval 1 |
| `release_in_bowl` | 30.0 | target near bowl ∧ `z < 0.06` ∧ gripper open ∧ settled, gated on lift + over-bowl-above-rim latches |
| **`distractor_disturb`** | **−0.5** | continuous penalty proportional to distractor linear speed once `> 0.05 m/s` |
| **`wrong_block_in_bowl`** | **−20.0** | distractor cube settled in bowl |
| `action_rate`, `joint_vel` | −1e-4 → −1e-2 | 10 k env-step ramp |

Sizing: `wrong_block_in_bowl = −20` is intentionally smaller than `release_in_bowl = +30` so that a correct final placement (even after disturbing the distractor en route) still wins net — the spec encourages distractor interaction when it helps. `distractor_disturb = −0.5` keeps the policy from gratuitous sweeping but doesn't hard-block contact.

γ = 0.98 load-bearing for the same reason as Eval 1 (long dense tail; 250-step episode, release fires deep).

## 6. Curriculum & DR

- `place_clutter_blocks`: cluster center in `(0.15, 0.22) × (−0.10, 0.10)`, fixed `half_separation = 0.0105 m`, pair axis θ ∈ U[0, 2π).
- `reset_target_latches`: clears per-episode lift / over-bowl latches.
- `reset_cube_pos_bias`: per-episode U[−5, +5] mm bias on `target_cube_pos_xy_noisy` (§6 of Eval 1).
- Action-rate / joint-vel penalty ramp at 10 k env-steps.
- `log_target_success_metrics` TB metric (target-correct success + wrong-block placement rate).

**No image DR**, no HSV jitter, no mask corruption, no DrQ. Camera-free training.

**No xy expand or separation curriculum** (same as the Eval-2 vision plan): cluster band is already tight and cubes are always attached. If Stage-1 stalls, the fallback is a *separation* curriculum (`half_separation` 0.025 → 0.0105 over 20 k env-steps), one-line change in `events.py`.

## 7. Network & PPO

Pure MLP actor-critic, identical class to Eval 1's state-only path. RSL-RL `ActorCritic`, `[256, 128, 64]` ELU, σ scalar Param.

| | State PPO (`state_apriltag_ppo_cfg.py`) |
|---|---|
| `num_envs` | 2048 (halved from 4096 — 6 cubes per env makes physics ~3-4× heavier) |
| `num_steps_per_env` | 32 |
| `max_iterations` | 2000 (vs Eval 1's 1500; extra cube dynamics need more samples even without colour discrimination) |
| `init_noise_std` | 1.0 |
| `entropy_coef` | 0.006 |
| `epochs / mini-batches` | 5 / 4 |
| `learning_rate / desired_kl` | 1e-4 / 0.01 |
| `γ / λ / clip / max_grad_norm` | 0.98 / 0.95 / 0.2 / 1.0 |
| `experiment_name` | `clutterpickplace_state_apriltag` |
| `obs_groups` | `{"policy": ["policy"], "critic": ["policy", "critic"]}` |

Wall-clock: ~2–3 h to ≥ 75 % sim success (estimate; tighten after first run).

## 8. Deploy

```bash
# Per-rollout: pick a target colour, look up its tag ID, set it on the detector.
python -m deploy.deploy_real \
    --mode eval2 \
    --target-color red \
    --bowl-xy 0.22,0.0 \
    --target-id-map "0=red,1=blue,2=yellow,3=green,4=purple,5=orange"
```

Per-step data flow (delta from Eval 1 §9):
1. `AprilTagDetector.set_target_id(id_for(target_color))` is called **once at outer-loop start**, not per step.
2. `pose(rgb, T_base_ee) → ((x, y), valid)` returns only the pose of the requested tag, even if other tags are in frame.
3. Everything else identical to Eval 1.

`deploy_real.py` is the single state-only deploy entry point — same script serves Eval 1 / Eval 2 / Eval 3. `--mode eval2` exposes the `--target-color` knob that maps through `--target-id-map` to a `set_target_id` call. The old `PPOActorClutter` (ResNet+FiLM) and `deploy_real_clutter.py` have been removed; `PPOActorState` serves all three eval tasks.

## 9. Risks and notes

- **Two visible tags vs one.** `pupil-apriltags` localizes each tag independently; cross-tag interference is not a failure mode at our tag sizes. The detection rate may drop slightly when two tags overlap in the wrist-cam frame (one partially covers the other), but the §6-of-Eval-1 dropout DR (Bernoulli p = 0.10 pre-grasp) covers this.
- **No colour-grounding problem at all.** This is the simplification compared to the old vision plan — the FiLM head, the wrong-colour-swap mask DR, the HSV deploy-time prompting — all gone. If the detector knows which tag-ID it should return, the colour discrimination problem doesn't exist.
- **Distractor pose isn't filled at deploy.** The policy doesn't see it (it's critic-only at train), so deploy doesn't need to detect the distractor. If you ever want to read the distractor pose at deploy for diagnostics or scheduler logic, the same `AprilTagDetector` can be queried with a second tag ID — but it's not on the policy hot path.
- **No `target_color_onehot` in obs** means Eval-2 checkpoints **cannot** be evaluated against an arbitrary colour at deploy without re-keying the detector. That's a feature: the policy is colour-blind because the detector has already picked the right cube; if you want a different cube, change the `set_target_id` call, not the policy input.

## References

- Foundation: [`EVAL1_PLAN.md`](./EVAL1_PLAN.md) — calibration, detector, sim noise model.
- Code: `tasks/clutterpickplace/{clutterpickplace_env_cfg,joint_pos_env_cfg}.py`, `mdp/{events,observations,rewards,terminations,commands}.py`, `agents/state_apriltag_ppo_cfg.py`. Real-robot deploy: `deploy/{deploy_real.py,cube_detector.py}`.
- **RSL-RL**, Schwarke et al., 2025. <https://github.com/leggedrobotics/rsl_rl>
