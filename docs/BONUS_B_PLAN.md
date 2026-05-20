# Bonus B — Singulation (two-policy modular design)

Two cooperating policies. **P1 (singulation)** takes an initial stack / pyramid / cluster of 3–4 cubes and turns it into a flat layout where every active pair is ≥ 5 cm apart on the table. **P2 (pick-and-place)** is the existing Eval-3 color-conditioned PPO policy, reused unchanged. A thin deploy-side scheduler runs P1 first; once its success indicator latches, it hands off to P2 if the rubric calls for a follow-up pick. The bonus PDF only asks for separation, so P2 chaining is insurance, not a requirement.

Assumes familiarity with [`EVAL1_PLAN.md`](./EVAL1_PLAN.md), [`EVAL2_PLAN.md`](./EVAL2_PLAN.md), [`EVAL3_PLAN.md`](./EVAL3_PLAN.md). All shared machinery (asymmetric A-C, 3-stage pipeline, wrist cam, ResNet-18 spatial-softmax encoder, DR, `--teacher_ckpt` critic overlay, γ = 0.98) is inherited verbatim.

## 1. How the policy actually solves the task

P1's emergent strategy is a **grasp-lift-place** loop with a **push** fallback. One iteration:

1. **Pick a target cube.** `reach_closest_pair` reward pulls EE toward the midpoint of the worst-violating active pair — the attractor shifts as progress is made.
2. **Grasp from above.** Gripper descends over one cube, closes. For stacks the natural target is the top cube (only one accessible from above); for flat clusters either of the closest pair works.
3. **Lift to clear the cluster** (`z ≥ 0.07 m`). This is the disassembly step that breaks stacks and pyramids.
4. **Translate to an empty xy** (no other active cube within 5 cm), open gripper. Cube lands, stays put.
5. **Loop** until `singulation_success` latches (`min_pairwise_xy ≥ 0.05` AND all active `z < 0.05`).

When gripper geometry blocks a clean grasp (4-cube flat cluster, side cubes have no vertical access), the policy falls back to **sweeping**: fingers closed at table height, lateral push. Both strategies emerge from the same reward stack; `lift_then_place` (+3) biases toward grasp-and-place because it's more sim2real-robust than sustained sliding contact (real Feetech grip force is low — sliding friction is unreliable).

For stacks, this means the policy needs to identify *which* cube to grasp from the wrist view despite occlusion (top cube hides the rest from above). §6 explains how the visual observation handles this.

## 2. MDP

| Item | Value |
|---|---|
| Control | 50 Hz (decimation 2, sim 100 Hz) |
| Episode | **12.0 s = 600 steps** — admits ~3 grasp-lift-place iterations at ~3.5 s each (per Eval-1 deploy timing) plus settle headroom. 10 s was too tight for `STACK_4`. |
| Action | 5 arm joints (`scale=0.5`) + 1 binary gripper — verbatim from Eval 1 |
| Workspace centre | `x ∈ [0.16, 0.22]`, `y ∈ [−0.08, 0.08]`, random yaw `θ ∈ [0, 2π)` per env |
| Home `q` | `(0, 0, 0, 1.5708, 0, 0)`, gripper open |
| Terminations | `time_out`, `active_cube_off_table`, **`singulation_done`** (positive — success is a stable absorbing state, so γ-discounting on early termination incentivises *fast* singulation) |
| Bowl | **Present from Stage 1 onwards**, fixed per-episode xy rejection-sampled `‖bowl − cluster_center‖ ≥ 0.15 m`. Not in any P1 reward; exists only so P1's wrist-image distribution includes the bowl prim and P2 can take over without a distribution shift. |
| Table | `0.6 × 1.0 × 0.02 m` at `(0.25, 0, −0.01)`, top `z = 0` |

**No target color.** P1 is colour-agnostic.

## 3. Initial arrangements

`sample_active_set` currently emits stacks + 2×2 clusters; we extend to 11 families covering the plausible eval distribution. `arrangement_onehot` widens to 8-D (mixed and pyramid variants get distinct bits).

| ID | n_active | Geometry | Notes |
|---|---|---|---|
| `STACK_3` | 3 | 3 cubes z-stacked, ±5 mm lateral jitter | Top cube goes down first |
| `STACK_4` | 4 | 4 cubes z-stacked | PhysX-fragile; wider jitter mandatory |
| `CLUSTER_LINE_3` | 3 | 3 in a row, 1 mm spawn gap → contact under gravity | Easiest — ~2 cm displacement per outer cube |
| `CLUSTER_TRI_3` | 3 | Equilateral triangle, each touching the other two | Common real-world clutter |
| `CLUSTER_SQUARE_4` | 4 | 2×2 attached cluster (`half_separation = 0.0105 m`) | Already implemented |
| `CLUSTER_LINE_4` | 4 | 4 in a row | Stretches reach (~6.3 cm spawn span) |
| `CLUSTER_L_4` | 4 | 3-in-line + 1 perpendicular at one end | Tests against symmetric prior |
| `PYRAMID_3` | 3 | 2 bottom + 1 on top straddling the seam | Top cube spawns at `z ≈ table_z + 0.027` |
| `PYRAMID_4` | 4 | 3 bottom in triangle + 1 on top at centroid | Top cube spawns at `z ≈ table_z + 0.036` |
| `MIXED_2STACK_PLUS_1` | 3 | 2-stack + 1 standalone ~3 cm away | Mixed mode |
| `MIXED_2STACK_PLUS_2` | 4 | 2-stack + 2 in flat contact ~3 cm away | Mixed mode |

**Sampling weights:** stacks 0.25, flat clusters 0.40, pyramids 0.20, mixed 0.15.

**Physics-stable spawning.** `stack_lateral_jitter = 0.005 m` (widened from scaffold's 0.003 to match real-world hand-placement imperfection). After writing poses, advance **5 physics ticks with the robot held at home** before the first policy action so contacts register and `|v_cube|` settles below `0.01 m/s` — without this the first policy frame sees inflated velocities and over-triggers `overspeed`. `HIDDEN_PARK_XY` slots (already in scaffold) park the 2 inactive palette cubes off-camera.

## 4. Observations (asymmetric A-C)

| Group | Fields | Shape |
|---|---|---|
| `policy` (deployable) | `joint_pos_rel`, `joint_vel_rel`, `gripper_state`, `ee_proj_xy`, `bowl_xy` (2), `n_active_onehot` (2), `arrangement_onehot` (8), `last_action` | 1-D |
| `critic` (privileged) | `policy` + `active_block_mask` (6), `all_cube_positions` (6×3), `min_pairwise_xy_active`, `mean_pairwise_xy_active`, `n_cubes_off_table` | 1-D |
| `wrist_image` | RGB + **union active-cube mask** (binary OR over all active-cube instance masks) | `(N, 4, 72, 128)` |

**`arrangement_onehot` is privileged-at-train, operator-known-at-deploy.** During training it's written by `sample_active_set`; at deploy it comes from a CLI flag (`--arrangement stack4`) since the operator/TA places the cubes and knows the configuration. **No vision-based arrangement classifier needed** — major sim2real simplification.

**`bowl_xy` in the policy obs even though P1 ignores the bowl.** Needed so P2 (which reads `bowl_xy` to plan its grasp-to-bowl trajectory) sees the same state schema after handoff. P1 will learn to ignore it (gradient flows only through the reward terms, which don't reference bowl).

## 5. Reward (`mdp/rewards.py`)

All scaffold-implemented except the two intermediates marked **NEW**.

| Term | Weight | Trigger |
|---|---|---|
| `min_pairwise_xy` | +5.0 | `clamp(min_pairwise_xy_active, 0, 0.10) / 0.10` — primary dense signal |
| `mean_pairwise_xy` | +2.0 | Same on the mean — stops gaming `min` by sacrificing the rest |
| `all_on_table` | +3.0 | `𝟙[all active cube z < 0.05]` — dominant for `STACK_*` / `PYRAMID_*` |
| `reach_closest_pair` | +1.0 | `1 − tanh(‖ee_xy − midpoint(closest_active_pair)‖ / 0.10)` — shifts as progress is made |
| `lift_then_place` **NEW** | +3.0 | `𝟙[any active cube z ∈ [0.07, 0.20] AND gripper_state = closed]` — biases toward grasp-and-place over push |
| `bowl_avoidance` **NEW** | −5.0 | `𝟙[any active cube xy within 6 cm of bowl_xy]` — stops P1 from accidentally singulating into the bowl (would confuse the P2 handoff) |
| `singulation_success` | +50 | All active pairs ≥ 5 cm AND all on table; latched, triggers `singulation_done` termination |
| `overspeed` | −3.0 | `Σ_active max(0, ‖v_cube‖ − 0.30) / 0.30`, clipped per-cube |
| `action_rate`, `joint_vel` | −1e-4 → −1e-2 | 10 k env-step ramp |

γ = 0.98 (load-bearing — `0.98^500 ≈ 4e-5`, so the +50 success bonus still dominates dense terms for `STACK_4`'s ~500-step horizon; γ = 0.95 collapses this).

## 6. Vision — mask channel, occlusion handling, and Florence-2 prompting

**Do we still need a mask channel?** Yes, but for different reasons than Eval 1/2/3:

- Eval 1/2/3 used target-keyed masks because color-conditioned grasping requires "which cube is THE target" to be visually unambiguous. Singulation has no target, so a target-keyed mask is meaningless.
- A **union mask** ("cube vs not-cube") still helps because it (a) makes the foreground/background decision robust to lighting and table-color drift, (b) lets the spatial-softmax CNN allocate keypoints to cube pixels instead of bowl/table/arm pixels, and (c) gives the policy a fallback when RGB color is degraded.

**Mask source — follows the Eval 1/2/3 recipe verbatim, only the per-cube filter differs:**

| Sim | Real |
|---|---|
| Identical pipeline to `clutterpickplace/mdp/observations.py::wrist_rgb_mask_dr` (RGB DR → concat semantic-seg mask as 4th channel → mask-channel DR). The only delta is the seg filter: instead of `class:cube_<target>` for a single target colour, OR-reduce over **all six `class:cube_*` IDs** to one binary channel. New helper `mdp.wrist_rgb_union_mask_dr` shares the RGB DR codepath; only the filter loop changes (~80 LoC, mostly copy-paste from `wrist_rgb_mask_dr`). | Plugs into the existing `deploy/cube_detector.py` `Detector` protocol. Two implementations parallel to Eval 2/3: **`UnionFlorenceDetector`** (Florence-2 with prompt `"any cube"`, single inference per frame, ~2-5 s/frame on CPU with background-thread caching — same caching path as Eval 2/3's `FlorenceDetector`); **`UnionHsvDetector`** (`cv2.cvtColor → HSV, saturation > 0.4 → largest connected components`, ~ms/frame, kept for sim↔real A/B tests on clean scenes). Selected via `--mask-source {florence,hsv}`, default `florence`, exactly mirroring Eval 1's `deploy_real.py` flag. |

**Occlusion handling for stacks (the hard case).** In a 4-stack, the wrist-cam top-down view sees the top cube fully but only sees thin edge-stripes of the lower 3. The union mask is a single tall blob; instance-level localization from the mask alone is impossible. We resolve this three ways:

1. **High-resolution spatial-softmax** — ResNet-18 truncated at `layer2` (per Eval 2 §3) gives a `(128, 9, 16)` feature map → ~3 mm precision per keypoint. The 64 trainable keypoints can latch onto cube *edges* and *corners* visible in RGB, not just blob centroids.
2. **`arrangement_onehot` + `n_active_onehot`** in the state tells the policy "this is a 4-stack" — so it knows to keep grasping the topmost visible cube and not get confused when one grasp reveals another cube below.
3. **Retry-on-failure** is implicit: if the policy commands a grasp at xy where there's no cube (because it mis-targeted), the reward stays flat and the policy tries a different xy next step. Stage 3 PPO discovers this in-distribution.

**Mask-channel DR — same three axes as Eval 2 §5**, minus the wrong-colour-swap (N/A for union mask) — gated by the same `corrupt=True` obs param the Play cfgs disable:

| Failure mode at deploy | DR term | Default |
|---|---|---|
| Florence loses cubes when small/far in frame | Small-area dropout — zero the mask when `< mask_min_pixel_area` | 8 px |
| Florence edge jitter | Per-env erode/dilate radius `U[-R, R]` | R = 2 px |
| Florence total miss | Per-env Bernoulli full-frame mask dropout | p = 0.10 |

The full-dropout term forces the policy to keep working from RGB alone when the mask channel is zero — critical insurance for Florence misfires, and the reason `randomize_wrist_image_tint` + `randomize_wrist_hsv_dr` (§9) are kept at full Eval-3 width even though colour discrimination isn't load-bearing here.

**Conv1 inflation** identical to Eval 2/3 (RGB filters keep ImageNet weights; mask channel initialized to RGB-channel mean).

## 7. Network — `SingulationVisionActorCritic`

Same architecture as Eval 3's vision actor-critic **minus FiLM**: frozen ImageNet ResNet-18 trunk truncated at `layer2`, conv1 inflated to 4 ch, trainable 1×1 conv (64 ch) → spatial softmax → 128-D keypoints; state concat → MLP `[256, 128, 64]` ELU → μ (σ scalar Param); critic MLP `[256, 128, 64]` ELU → V. DrQ ±4 px training-only in `_encode_actor` and `_encode_student`.

No FiLM because colour is irrelevant — the 10-D `n_active + arrangement` onehot is small enough that early-concat into the MLP is fine. If after Stage 3 the policy applies the same motion to stacks vs clusters (diagnostic: per-arrangement success in TB), add FiLM conditioned on `arrangement_onehot`.

## 8. PPO config

| | Stage 1 teacher | Stage 3 vision PPO |
|---|---|---|
| `num_envs` | 2048 | 1024 (drop to 768 if OOM with ResNet trunk) |
| `num_steps_per_env` | 32 | 16 |
| `max_iterations` | 3000 | 3500 |
| `init_noise_std` | 1.0 | 0.5 (forced; distill saves 0.1) |
| `entropy_coef` | 0.008 → 0.004 after 1500 iters | 0.008 → 0.004 after 1500 iters |
| `epochs / mini-batches` | 5 / 4 | 8 / 16 |
| `learning_rate / desired_kl` | 1e-4 / 0.01 | 1e-4 / 0.005 |
| `γ / λ / clip / max_grad_norm` | 0.98 / 0.95 / 0.2 / 1.0 | same |

```python
# Stage 1: symmetric on privileged state
obs_groups = {"policy": ["policy", "critic"], "critic": ["policy", "critic"]}
# Stage 2 (distill): vision student, state teacher
obs_groups = {"policy": ["policy", "wrist_image"], "teacher": ["policy", "critic"]}
# Stage 3: vision actor, privileged critic
obs_groups = {"policy": ["policy", "wrist_image"], "critic": ["policy", "critic"]}
```

Stage-1 and Stage-3 critics share input shape → `--teacher_ckpt` overlay works layer-for-layer. Eval-2/3 teacher checkpoints don't transfer (different state schema).

## 9. Sim2real considerations

Catalogued failure modes from Eval 1/2/3 plus singulation-specific ones, with the corresponding sim-side mitigation.

| Concern | Sim mitigation |
|---|---|
| **Cube mass/friction unknown at deploy** | Per-episode physics DR: cube mass `U(0.016, 0.024) kg` (±20%), cube↔cube friction `U(0.7, 1.3)`, cube↔table friction `U(0.7, 1.3)`. Add to `EventCfg` as `randomize_cube_physics` (mode="reset"). ~40 LoC. |
| **Real cubes don't spawn perfectly aligned** | `stack_lateral_jitter = 0.005 m` (widened from 0.003); pyramid corner placement gets `cluster_position_jitter = 0.004 m`. |
| **Real grasps occasionally slip** | DR-induced grasp failures already present (jitter on cube xy at gripper closure) — augment with `gripper_close_xy_noise = 0.003 m` per close event so Stage 3 sees and learns to retry. |
| **Florence-2 mask is noisy / drops out** | Mask-channel DR per §6 (small-area dropout, erode/dilate, full Bernoulli dropout). |
| **Lighting / white-balance drift** | Wrist RGB tint (`scale 0.55–1.45`, `brightness ±0.20`) + HSV (`hue ±20°`, `sat 0.65–1.35`, `val 0.55–1.45`). Inherited verbatim from Eval 3. |
| **Real arm slew-rate cap (~50°/s/joint) is slower than sim** | Match in deploy via `arm_slew_rate_cap_rad_per_s` (already in `deploy/deploy_real.py`); train with `init_noise_std = 0.5` Stage 3 → policy output is smooth enough to survive slewing. |
| **Operator places exact arrangement type** | `arrangement_onehot` from CLI flag at deploy (`--arrangement stack4`) — matches sim's privileged signal exactly. No vision classifier needed. |
| **Real wrist cam intrinsics ≠ sim** | Already calibrated in `camera_intrinsics.yaml`, loaded by both sim spawn and real undistort (verbatim from Eval 1). |
| **P2 handoff requires bowl in wrist view from start** | Bowl prim spawned in Stage 1 onwards (§2). P1 reward ignores it; `bowl_avoidance` (-5) keeps cubes out of it. P1's vision sees a bowl every episode. |
| **Real-world singulated cubes are 5+ cm apart, Eval-3 trained on ≤ 4 cm spacing** | After P1 succeeds, run a 200-iter Eval-3 finetune on `half_separation = 0.030` placements before deploying chained mode. ~1 hr. |
| **Real grasp validation** | Deploy scheduler adds a grasp-success check: if gripper closes and the union-mask area doesn't drop within 5 ticks (cube didn't leave the table), abort the lift and re-issue the policy. ~30 LoC in deploy. |

## 10. Three-stage pipeline + workflow

Same shape as Eval 2/3: state teacher → short distill (200–500 iters, not to convergence) → vision PPO with critic overlay. Five Stage-3 interventions from Eval 1 §7.1 carry verbatim.

```bash
# Stage 1 — camera-free state teacher
uv run train --task Isaac-SO-ARM101-Singulation-Teacher-Fast-v0 --headless

# Stage 2 — short distill
uv run train --task Isaac-SO-ARM101-Singulation-Student-v0 --headless --enable_cameras \
    --load_run from_teacher --checkpoint model_<best>.pt

# Stage 3 — vision PPO with teacher critic overlay
uv run train --task Isaac-SO-ARM101-Singulation-v0 --resume --headless --enable_cameras \
    --num_envs 1024 \
    --load_run <distill_run> --checkpoint model_<best>.pt \
    --teacher_ckpt logs/rsl_rl/singulation_teacher/<teacher_run>/model_<best>.pt
```

## 11. Two-policy deploy scheduler

P1 → P2 handoff trigger (sustained ≥ 100 ms = 5 control steps):

```
min_pairwise_xy(active) ≥ 0.05 m   AND   all active z < 0.05 m
                                    AND   gripper open
                                    AND   ‖ee_vel‖ < 0.05 m/s
```

On real hardware `min_pairwise_xy` is estimated from the Florence-2 union mask + connected-components centroids; occlusion-prone during the policy's grasp motions, so the production fallback is operator-confirm (`--release-detect manual`).

Sim:
```bash
uv run play --task Isaac-SO-ARM101-Singulation-Play-v0 \
    --arrangement stack4 --n-active 4 \
    --singulation-ckpt logs/rsl_rl/singulation/<run>/model_<best>.pt \
    --pickplace-ckpt   logs/rsl_rl/eval3clutter/<run>/model_<best>.pt \
    --enable_cameras [--bonus-only | --chained --target-color red --bowl-xy 0.22,0.0]
```

Real:
```bash
python -m deploy.deploy_real --bonus-b \
    --arrangement stack4 --n-active 4 \
    --singulation-ckpt deploy/runs/singulation.pt \
    [--chained --pickplace-ckpt deploy/runs/eval3.pt --target-color red --bowl-xy 0.22,0.0]
```

`--arrangement` and `--n-active` populate the policy's `arrangement_onehot` + `n_active_onehot` directly (CLI → privileged conditioning, see §4). `--bonus-only` (default) scores P1 alone; `--chained` swaps to P2 after the handoff trigger, clearing P2's per-episode lift / over-bowl latches on swap (same hooks the Eval-3 sub-goal scheduler uses).

## 12. Build list

Most plumbing exists under `tasks/singulation/`. Order: 1–5 unblock Stage 1, 6–8 unblock Stage 3, 9–11 unblock deploy.

1. **`mdp/events.py`** — extend `sample_active_set` from 2 to 11 arrangement families per §3 (replace `stacked_prob` with `arrangement_weights: dict[str, float]`); add 5-physics-tick settling step at the end. Add **`randomize_cube_physics`** event (mass / friction DR per §9). Add **`spawn_bowl_xy`** event (per-episode bowl xy rejection-sampled vs cluster centre). **~250 LoC.**
2. **`mdp/observations.py`** — widen `arrangement_onehot` to 8-D; add `bowl_xy` to policy obs; expose `min_pairwise_xy_active` / `mean_pairwise_xy_active` / `n_cubes_off_table` as critic obs; **add new `wrist_rgb_union_mask_dr`** (4-channel: RGB + OR over `class:cube_*` semantic mask) with the §6 mask-dropout DR. **~150 LoC.**
3. **`mdp/rewards.py`** — add `lift_then_place` and `bowl_avoidance` (NEW intermediates, §5); extend `log_singulation_metrics` to split by 8-way arrangement. **~50 LoC.**
4. **`singulation_env_cfg.py`** — add bowl `RigidObjectCfg` to the scene; swap `WristImageCfg.wrist_image` from `wrist_rgb_dr` (3-ch) to `wrist_rgb_union_mask_dr` (4-ch); bump `episode_length_s = 12.0`; wire the two new events into `EventCfg`. **~40 LoC.**
5. **`joint_pos_env_cfg.py`** — verify wiring (robot articulation, action terms, 6 `CuboidCfg` cubes with `PreviewSurfaceCfg(diffuse_color=BLOCK_COLORS[name])` AND `semantic_tags=[("class", f"cube_{name}")]`, `TiledCamera` with `data_types=["rgb", "semantic_segmentation"]`, `FrameTransformerCfg` ee_frame, `_TeacherFast` subclass nulling `wrist_cam` + `wrist_image`). **~60 LoC delta.**
6. **`agents/`** subdirectory mirroring `tasks/clutterpickplace/agents/`: `vision_actor_critic.py` (subclass with `cnn_class="resnet"`, `truncate_at="layer2"`, conv1 inflated to 4 ch, no FiLM), `vision_student_teacher.py` (analogous distill student), `teacher_ppo_cfg.py` / `rsl_rl_ppo_cfg.py` / `distill_cfg.py` per §8 with `setattr(rsl_rl.runners.…)` class-injection lines. **~230 LoC.**
7. **`tasks/pickplace/agents/vision_actor_critic.py`** — add `cnn_class: str = "resnet"` cfg arg and dispatch (shared item with Eval 2 / Eval 3). **~15 LoC.**
8. **`tasks/singulation/__init__.py`** — register `-Student-v0`, `-Student-Play-v0`, `-Chained-Play-v0` (the chained sim deploy variant). Base, `-Play-v0`, `-Teacher-Fast(-Play)-v0` exist. **~30 LoC.**
9. **`deploy/singulation_actor.py`** — forward-only mirror of `SingulationVisionActorCritic` (analogous to `deploy/ppo_actor.py`); state schema matches §4 policy obs. **~130 LoC.**
10. **`deploy/cube_detector.py`** — add `UnionFlorenceDetector` (prompt `"any cube"`) and `UnionHsvDetector` (`saturation > 0.4` + largest-CC pick) implementing the existing `Detector` protocol. Both wrap the existing Eval-2/3 detector helpers; ~80 LoC total. The `Detector` protocol itself doesn't change.
11. **`deploy/deploy_real.py`** — `--bonus-b` flag, `--arrangement` / `--n-active` CLI population of the onehots, `--mask-source {florence,hsv}` reuses the existing flag, §11 chained mode (P1→P2 policy swap with detector swap if P2 needs a colour-specific Florence prompt), grasp-success retry check (union-mask-area drop check, §9). **~180 LoC.**
11. **(Optional) verification tests** — per-arrangement settling check (assert `|v_cube| < 0.01` after 5-tick settle, per family), `arrangement_onehot` round-trip vs CLI flag at deploy, union-mask coverage check (≥ 95 % of true active-cube pixels, < 5 % of parked-cube pixels). **~180 LoC.**

**Effort:** ~3 days build (mask helper + bowl-in-scene push effort up vs prior estimate) + ~16 hrs training (~3 hrs Stage 1 + ~1 hr Stage 2 + ~12 hrs Stage 3) on a 32 GB GPU.

## References

- Code: `tasks/singulation/{singulation_env_cfg, joint_pos_env_cfg}.py`, `mdp/{events, observations, rewards, terminations}.py`. Predecessor plans: [`EVAL1_PLAN.md`](./EVAL1_PLAN.md), [`EVAL2_PLAN.md`](./EVAL2_PLAN.md), [`EVAL3_PLAN.md`](./EVAL3_PLAN.md). Real deploy: [`../deploy/README.md`](../deploy/README.md).
- **Eitel et al.**, *Learning to Singulate Objects using a Push Proposal Network*, ISRR 2017. <https://arxiv.org/abs/1707.08101>
- **Zeng et al.**, *Learning Synergies between Pushing and Grasping with Self-supervised Deep Reinforcement Learning*, IROS 2018. <https://arxiv.org/abs/1803.09956>
- **Pinto et al.**, *Asymmetric Actor Critic*, RSS 2018. <https://arxiv.org/abs/1710.06542>
- **Levine et al.**, *End-to-End Visuomotor Policies*, JMLR 2016 — spatial softmax.
- **Kostrikov et al.**, *DrQ*, ICLR 2021. <https://arxiv.org/abs/2004.13649>
