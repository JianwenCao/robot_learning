# Eval 2 — Targeted Pick-and-Place in 2-Cube Clutter

Color-conditioned PPO on SO-ARM101 → zero-shot real-arm deploy. Task: `Isaac-SO-ARM101-ClutterPickPlace-v0`. Same three-stage pipeline as Eval 1 (state teacher → vision distill → vision PPO + teacher critic), re-keyed to the *target* cube and with a goal-conditioned color one-hot. **No teleop is used in Stage 1** — the teacher is pure PPO on privileged state in sim. Teleop is an optional Stage-2 DAgger seed only.

## 1. MDP

| Item | Value |
|---|---|
| Control | 50 Hz (decimation 2, sim 100 Hz) |
| Episode | 5.0 s = 250 steps (same as Eval 1) |
| Action | 5 arm joints (absolute around home, `scale=0.5`) + 1 binary gripper (`open=0.5`, `close=0.0`) — verbatim from Eval 1 |
| Workspace | bowl: `x ∈ [0.15, 0.28] m`, `y ∈ [−0.12, 0.12] m`; cluster center: `x ∈ [0.15, 0.22]`, `y ∈ [−0.10, 0.10]` |
| Home `q` | `(0, 0, 0, 1.5708, 0, 0)`, gripper open |
| Terminations | `time_out`, `block_off_table_any` (any of the six cubes off-table) |
| Table | `0.6 × 1.0 × 0.02 m` at `(0.25, 0, −0.01)`; top `z=0` |

**Scene composition.** Six 2 cm `CuboidCfg` primitives, one per palette color: **blue, yellow, purple, orange, green, red**. The Eval-1 NVIDIA `dex_cube` USD can't be re-tinted per env, so cubes are primitives with `PreviewSurfaceCfg` colors baked in. Friction is tuned (`static=dynamic=1.0`, `mass=0.020 kg`) to replicate the dex_cube's grippability under the binary gripper.

**Active-pair sampling.** Per reset, `TargetColorCommand` samples (i) two distinct colors from the palette as the *active pair* and (ii) one of those two as the *target*. The active pair is placed adjacent in the workspace by `place_clutter_blocks`; the other four cubes are teleported to `HIDDEN_PARK_XY` slots off-table where the wrist camera can't see them. **Cubes are attached (no margin)** — `half_separation=0.0105 m`, i.e. cube centers 2.1 cm apart, leaving a 1 mm face-to-face gap at spawn that closes to true contact under gravity within a sim step. The 1 mm exists only to avoid PhysX initial-interpenetration artifacts; the real-eval setup places the two blocks touching, so we match that. Pair axis `θ ∈ U[0, 2π)` so orientation is fully randomized.

**Bowl.** 2-D goal from `ClusterBowlPoseCommandCfg` — **no scene prim**, **rejection sampling against the active cube pair**. Generalizes Eval 1's `BowlPoseCommand` (single-asset rejection) to the two-cube case: reads `env._active_cube_indices` written by `place_clutter_blocks`, then resamples bowl xy up to 16× until `‖bowl_xy − cube_xy‖ ≥ 0.15 m` for *both* active cubes in robot frame. Same `(x, y)` frame at deploy. (Without rejection, ~5–10 % of resets land the bowl on top of the cluster, giving the policy free "block already in bowl" reward signal.)

## 2. Observations (asymmetric A-C)

`ObservationsCfg` defines three groups; runner cfgs select per stage (§7).

| Group | Fields | Shape |
|---|---|---|
| `policy` (deployable) | `joint_pos_rel`, `joint_vel_rel`, `gripper_state`, `bowl_xy`, `ee_proj_xy`, `ee_to_bowl_xy`, **`target_color_onehot`** (6-D), `last_action` | 1-D |
| `critic` (privileged) | `policy` + `target_block_position`, `distractor_block_position`, `target_block_to_bowl_xy`, `target_gripper_to_block`, `target_is_grasped` | 1-D |
| `wrist_image` | **RGB + target_mask** (per-color instance mask, target-keyed) | `(N, 4, 72, 128)` |

Wrist `TiledCamera`: same mount and intrinsics as Eval 1 — parented to `gripper_link` at `pos=(-0.001, 0.1, -0.04)`, ros quat `(-0.404379, -0.912179, -0.0451242, 0.0486914)`. Renders `["rgb", "semantic_segmentation"]` (`colorize_semantic_segmentation=False`). Each cube carries a unique `class:cube_<color>` tag; `mdp.wrist_rgb_mask_dr` filters the seg image to the *target* cube per env (via `env._target_cube_idx`) and concatenates that binary mask as the 4th channel. The class-id lookup happens once on first render and is cached on the camera object. At deploy the same mask comes from **Florence-2** prompted by the target color (`f"{color_name} cube"`) — empirical HSV thresholding proved brittle at the cube's working distance (no mask when far + lighting drift). The mask channel is corrupted in sim to match Florence-2's noise profile (small-area dropout, morphology, full dropout, wrong-color swap) — see §5.

When `wrist_image` is absent from `obs_groups` (Stage 1 teacher), the CNN auto-disables → pure MLP A-C.

## 3. Network — `ClutterPickPlaceVisionActorCritic`

Same architecture as Eval 1's `pretrained-backbone PPO path` (commit `903ef6b`): **frozen ImageNet ResNet-18 trunk → trainable 1×1 conv → Levine-style spatial-softmax head → 128-D keypoints**. We reuse `_ResNetSpatialSoftmaxCNN` from `tasks/pickplace/agents/vision_actor_critic.py` verbatim with `in_channels=4` (RGB + target_mask). The class auto-inflates `conv1` to 4 input channels at construction — RGB filters keep the ImageNet weights; the mask channel is initialized to the RGB-channel mean so the trunk's activation statistics are preserved. `img_in_shape` is derived from the observation manager's `wrist_image` group shape, so the channel switch is picked up automatically; no agent-cfg change is required.

```
wrist_image (3×72×128) ──[ResNet-18 trunk (frozen, ImageNet)]── (128, 9, 16) feat
                       ──[1×1 conv → 64 ch]── per-channel softmax → soft-argmax (x,y) → 128-D kpts ─┐
state (policy, incl. target_color_onehot) ──────────── concat ── MLP[256,128,64] ── μ (σ scalar Param)
critic state (policy+critic) ─────────────────── MLP[256,128,64] ── V(s)
```

**Truncation choice — `layer2`, not `layer3`.** Current `_ResNetSpatialSoftmaxCNN` truncates at `layer3`, which for our 72×128 input gives `(256, 4, 8)` — 32 spatial cells, each covering ~18×16 input pixels. A 2 cm cube at the wrist-cam working distance projects to ~12–18 pixels (≈ 1 cell), so soft-argmax precision is ~1.5 cm physical — same order as the cube edge. For Eval 2 we **switch the trunk to end at `layer2`** → output `(128, 9, 16)` = 144 cells, ~3 mm precision, and the 1×1 conv input dim drops from 256 → 128 (cheaper). Layer3 features were tuned for ImageNet's 224×224; at 72×128 they're over-downsampled. One-line change in `_ResNetSpatialSoftmaxCNN.__init__` (drop `backbone.layer3` from the trunk Sequential).

DrQ ±4 px replicate-pad-and-crop training-only, in `_encode_actor` (Stage 3) AND `ClutterPickPlaceVisionStudentTeacher._encode_student` (Stage 2).

### 3.1 Goal conditioning — FiLM at the trainable head

The naïve approach — concat the target_color one-hot to the *MLP* input only, after the CNN — leaves the CNN blind to the goal. The CNN produces 64 unlabeled "blob center" keypoints, and the MLP has to implicitly learn a 6-way gated indexing ("if target=red, attend to keypoint 17"). Learnable from PPO gradient, but brittle and slow.

Instead we condition the CNN itself via **FiLM** (Perez et al., AAAI 2018), injected at the only trainable image-path layer — the 1×1 conv head:

```python
gamma, beta = self.film_mlp(target_color_onehot)   # MLP[6 → 256], output split → two (N, 256)
feat = self.trunk(rgb)                              # frozen, (N, 256, 9, 16)
feat = gamma[..., None, None] * feat + beta[..., None, None]
heat = self.head(feat)                              # trainable 1×1 conv to 64 ch
# spatial-softmax as before → 128-D kpts
```

Effect: each of the 64 keypoint channels becomes *color-conditional*. The same physical keypoint can lock onto "red blob center" when target=red and "yellow blob center" when target=yellow, instead of allocating one keypoint per color in a static unsupervised partition. Cost is ~3 k extra params (one tiny MLP) — negligible against the 64-ch 1×1 conv.

The one-hot **also** flows into the actor MLP as a normal state input — belt-and-suspenders. The MLP can still use it for non-visual decisions (e.g., bowl approach angle differs by target's relative bowl distance).

Why FiLM at the head, not early concat or unfrozen-conv1:

- Early concat (tile 6-D one-hot → 9-channel input) requires `conv1` to learn to use the new modality, which means unfreezing conv1 — and the binary `{0,1}` one-hot channels have wrong scale vs ImageNet-RGB. Loses the "frozen trunk = stable PPO" property.
- FiLM is the standard goal-conditioning recipe in language-conditioned manipulation (CLIPort) and goal-conditioned RL — well-validated for exactly this categorical-goal-over-image setup.

If FiLM plateaus in Stage 3 (e.g., < 60 % target-correct after 1500 iters), switch to §8's HSV mask channel — modular perception bypasses the CNN's color-grounding step entirely.

**Why frozen ResNet-18 + spatial-softmax head (not from-scratch small CNN).** Four reasons inherited from the Eval-1 class docstring:

1. **ImageNet features are domain-general.** Real-world lighting and color variation that sim DR can't perfectly cover are inside ImageNet's pretraining distribution. Critical for Eval 2 because color discrimination is now load-bearing.
2. **Frozen trunk = stable PPO.** The most common visual-PPO failure mode in this repo is encoder gradients fighting RL gradients. With `requires_grad=False` on the trunk, only the 1×1 conv + spatial softmax + downstream MLPs see PPO gradients. Also keeps BatchNorm running stats from drifting under PPO's non-stationary rollouts (Wu & He, GroupNorm ECCV 2018).
3. **Spatial-softmax inductive bias kept.** ResNet features by themselves are not localization-friendly; the 1×1 conv re-projects to per-keypoint heatmaps and the soft-argmax extracts (x, y) coords. So we get ImageNet color/texture features **and** the Levine-2016 keypoint geometry head — not ResNet → flatten → MLP.
**Wiring note.** As of writing, `PickPlaceVisionActorCritic.__init__` still hardcodes `_ImpalaSmallCNN` at the two `self.actor_cnn` / `self.critic_cnn` instantiation sites (lines ~456 / 460). For Eval 2 we add a `cnn_class: str = "resnet"` cfg arg to `ClutterPickPlaceVisionActorCriticCfg` and dispatch to `_ResNetSpatialSoftmaxCNN` when set. Same gap exists for Eval 1's pretrained-backbone runs — fixing it in one place benefits both.

**GPU budget note.** ResNet-18 layer1–layer2 is ~700 k params (down from ~11 M at layer3). Frozen → no gradient memory, but activation tensors through 64-ch (layer1) and 128-ch (layer2) feature maps at the rollout batch are still real. With `num_envs=1024` and rollout `num_steps=16`, that's 16 384 images per forward pass — ResNet activations at the larger spatial resolution will pressure VRAM more than Eval 1's 50 k-param CNN. Plan to drop to 768 envs if OOM; only fall back to a smaller backbone if 768 still doesn't fit.

## 4. Reward (`mdp/rewards.py`)

| Term | Weight | Trigger |
|---|---|---|
| `reaching_object` (reach_target_block) | 1.0 | `1 − tanh(‖ee − target_block‖ / 0.05)` |
| `lifting_object` (target_grasp_event) | 15.0 | `𝟙[target_block_z > 0.07]` |
| `object_goal_tracking` (target_transport_to_bowl) | 16.0 | `(1 − tanh(‖target − goal‖ / 0.30)) · 𝟙[target_z > 0.025]` |
| `object_goal_tracking_fine_grained` | 5.0 | same at `std=0.05` |
| `release_in_bowl` (release_target_in_bowl) | 30.0 | target block near bowl ∧ `z<0.06` ∧ gripper open ∧ settled, gated on lift latch + over-bowl-above-rim latch (same recipe as Eval 1) |
| **`distractor_disturb`** | **−0.5** | continuous penalty proportional to distractor linear speed once `> 0.05 m/s` |
| **`wrong_block_in_bowl`** | **−20.0** | distractor cube settled in bowl (heavy enough that net reward of misplacement < net reward of correct placement at `release=+30`) |
| `action_rate`, `joint_vel` | −1e-4 → −1e-2 | ramp at 10 k env-steps |

Two per-episode latches (cleared by `reset_target_latches`):

- **Lift latch 0.07 m** — `target_block_z > 0.07` ⇒ cube_bottom clears the real bowl's 5 cm rim.
- **Over-bowl-above-rim latch 0.08 m** — forces above-rim approach; closes the sim/real rim gap.

No `task_success` termination — same logic as Eval 1: success termination would let "hover the right cube" beat "release and stay". `release_target_in_bowl=30` pays every post-release step until time-out. `γ=0.98` remains load-bearing (long dense tail).

**Distractor-disturb sizing rationale.** The wrong-block penalty of −20 is comparable to but smaller than the +30 release reward, so a *correct* final placement (even after disturbing the distractor en route) still wins net. This is intentional — adjacency means a clean grasp sometimes requires light contact with the distractor, and Eval 3's spec explicitly encourages non-target interaction. Hard-blocking distractor motion would inject a structural local optimum (hover-without-grasping).

**No contact sensors.** Eval 1 used a `ContactSensor` filtered to the dex_cube prim path. Here the "target" varies per episode, so the filter prim paths would have to be patched at reset; we drop the contact reward and rely on the kinematic lift latch (`target_block_z > 0.07`) instead — that recipe worked for Eval 1's vision teacher.

## 5. Curriculum & DR (`mdp/events.py`, `CurriculumCfg`)

Both stages:
- `place_clutter_blocks`: cluster center in `(0.15, 0.22) × (−0.10, 0.10)`, **`half_separation=0.0105` (fixed; cubes attached, 1 mm spawn gap closing to contact under gravity)**, pair axis `θ ∈ U[0, 2π)`.
- `reset_target_latches`: clears per-episode lift / over-bowl latches.
- Action-rate / joint-vel penalty ramp −1e-4 → −1e-2 at 10 k env-steps.
- `log_target_success_metrics` TB metric (target-correct success rate + wrong-block placement rate).

DR (in env cfg — applies on every reset, including Stage 1 teacher; teacher just doesn't read the image so the wasted compute is one CPU/GPU write of a few floats per env per reset):
- `randomize_wrist_image_tint`: per-channel **linear scale `U(0.55, 1.45)`** + **brightness shift `U(−0.20, +0.20)`**. Much wider than Eval 1's ±15 %; widened because color discrimination is now load-bearing.
- `randomize_wrist_hsv_dr`: **hue rotation `±20°`** in linear RGB, saturation scale `U(0.65, 1.35)`, value scale `U(0.55, 1.45)`. **This is the sim2real-critical DR term for color-conditioned tasks** — hue rotation simulates white-balance / colored-ambient drift that linear tint can't model. Without HSV jitter, the policy memorizes the sim renderer's exact color rendition and fails on any USB-cam white-balance offset.

Stage 3 adds (training-only, in the model):
- DrQ ±4 px (in CNN, see §3).

Per-step mask-channel DR (in `mdp.wrist_rgb_mask_dr`, gated by the obs `corrupt=True` param; Play cfgs disable):

| Failure mode at deploy | DR term | Default |
|---|---|---|
| Florence loses cube when small/far in frame | Small-area dropout — zero the mask when total mask pixels `< mask_min_pixel_area` | 8 px |
| Edge-pixel jitter on Florence output | Per-env erode/dilate with radius uniform in `[-R, R]` | `R = 2 px` |
| Florence occasional total miss | Per-env Bernoulli full-frame mask dropout | `p = 0.10` |
| Florence misclassifies under bad WB → wrong cube masked | Per-env Bernoulli swap to distractor's instance mask | `p = 0.03` |

The wrong-color-swap term is the load-bearing one for multi-cube robustness: without it, a single Florence misfire that masks the distractor is a catastrophic single-step failure; with it, recovery is in-distribution.

**No xy expand or separation curriculum.** Eval 1 used `expand_block_xy_range` (±3 cm → ±7×±12 cm). For Eval 2 the cluster band is already tight (≤ 7 cm × ≤ 20 cm) and cubes are always attached (no margin to schedule). If Stage 1 fails to converge on the attached case, the fallback is a *separation* curriculum: start `half_separation=0.025` (4 cm face-gap, Eval-1-like difficulty) for the first 5 k env-steps and ramp down to `0.0105` over the next 20 k. One-line change in `events.py` to make `half_separation` curriculum-driven.

Deferred to deploy (not in sim DR yet): cube/table material DR, HDRI background, motion blur / JPEG, distractors beyond the active pair.

## 6. PPO config

Same skeleton as Eval 1 — small deltas to account for color discrimination needing more samples.

| | Stage 1 teacher (`teacher_ppo_cfg.py`) | Stage 3 vision PPO (`rsl_rl_ppo_cfg.py`) |
|---|---|---|
| `num_envs` | **2048** (`_multicube_sim.DEFAULT_TRAIN_NUM_ENVS`; halved from Eval 1's 4096 because 6 cubes/env makes physics ~3–4× heavier) | 1024 (image rollouts; drop to 768 if OOM with ResNet trunk) |
| `num_steps_per_env` | 32 (matches Eval 1 teacher) | 16 |
| `max_iterations` | 1500 (matches Eval 1 — env-step budget ~98 M, same as Eval 1) | 2500 (vs Eval 1's 2000; color discrimination needs more samples) |
| `init_noise_std` | 1.0 | 0.5 (forced; distill saves 0.1) |
| hidden dims | `[256, 128, 64]` ELU | `[256, 128, 64]` ELU + spatial-softmax CNN |
| `entropy_coef` | 0.006 | 0.006 → 0.003 (after ~700 iters; later than Eval 1 because target-color exploration matters longer) |
| epochs / mini-batches | 5 / 4 | 8 / 16 |
| `learning_rate` / `desired_kl` | 1e-4 / 0.01 | 1e-4 / 0.005 |
| `gamma / lam / clip / max_grad_norm` | 0.98 / 0.95 / 0.2 / 1.0 | same |

```python
# Stage 1: symmetric on privileged state (includes target_color_onehot)
obs_groups = {"policy": ["policy", "critic"], "critic": ["policy", "critic"]}
# Stage 2 (distill_cfg.py): vision student, state teacher
obs_groups = {"policy": ["policy", "wrist_image"], "teacher": ["policy", "critic"]}
# Stage 3: vision actor, privileged critic (no image to critic)
obs_groups = {"policy": ["policy", "wrist_image"], "critic": ["policy", "critic"]}
```

Stage-1 critic and Stage-3 critic take identical inputs → teacher critic loads layer-for-layer via `load_state_dict(strict=False)`. The state schema is wider than Eval 1 (one-hot adds 6 dims, distractor pose adds 3 dims), so Eval-1 teacher checkpoints **do not** transfer; the Eval 2 teacher is trained from scratch.

## 7. Three-stage pipeline

- **Stage 1 — state teacher.** Task `…-ClutterPickPlace-Teacher-v0` (to register). MLP A-C on `policy + critic`. Pure PPO on privileged state in sim. **No teleop, no demos** — the teacher has direct access to the target block pose via `target_block_position`, and the color one-hot is in the state, so the MDP is fully learnable from PPO on the dense reward stack. Saves `actor.*` + `critic.*`.
- **Stage 2 — short distill.** Task `…-ClutterPickPlace-Student-v0` (to register). RSL-RL `DistillationRunner`, MSE, on-policy DAgger. `ClutterPickPlaceVisionStudentTeacher` regresses teacher actions on `policy + wrist_image`. **Not to convergence** — 200–500 iters, stop when target-correct success ~30–50 %. Optional: seed the DAgger buffer with teleop trajectories of grasping each color in adjacent-cluster scenes (spec encourages teleop "for training efficiency"; useful precisely because Stage 2 is the modality bridge and demos cheapen the color-grounding warm-up). **Strictly optional** — pipeline works without it.
- **Stage 3 — vision PPO from warm-start.** Task `…-ClutterPickPlace-v0` (registered). `ClutterPickPlaceVisionActorCritic.load_state_dict` routes distill `student_cnn.* / student.*` → `actor_cnn.* / actor.*`. Trains on §4 reward to convergence with `--teacher_ckpt` critic overlay.

### 7.1 Five Stage-3 interventions (inherited from Eval 1)

| # | Fix | Location | Solves |
|---|---|---|---|
| 1 | DrQ in `_encode_student` | `vision_student_teacher.py` | Distribution shift at Stage 2→3 boundary |
| 2 | Drop loaded `std`; reinit from `init_noise_std=0.5` | `vision_actor_critic.load_state_dict` distill branch | Distill's `std=0.1` too narrow for binary gripper + color exploration |
| 3 | `gamma=0.98` (match teacher) | `rsl_rl_ppo_cfg.py` | `release_target_in_bowl=30` needs long horizon |
| 4 | Wider wrist-tint DR (`scale 0.7–1.3`) | `EventCfg.randomize_wrist_tint` | Color discrimination must be lighting-invariant |
| 5 | **`--teacher_ckpt` overlays teacher `critic.*` after distill warm-start** | `scripts/rsl_rl/train.py` | Random critic produces O(magnitude)-noisy advantages → degrades actor in ~50 iters |

#5 is the asymmetric-AC handoff (Pinto 2018): teacher critic anchors V; PPO's value regression moves V_teacher → V_student over ~100–500 iters. Same load-bearing piece as Eval 1.

### 7.2 Workflow

```bash
# Stage 1 (no teleop, no warm-start — train from scratch)
train --task Isaac-SO-ARM101-ClutterPickPlace-Teacher-v0 --headless --enable_cameras

# Stage 2 (NOT to convergence; teleop seed optional)
train --task Isaac-SO-ARM101-ClutterPickPlace-Student-v0 --headless --enable_cameras \
      --load_run <teacher_run> --checkpoint model_<best>.pt

# Stage 3
train --task Isaac-SO-ARM101-ClutterPickPlace-v0 --resume --headless --enable_cameras \
      --num_envs 1024 \
      --load_run <distill_run> --checkpoint model_<best>.pt \
      --teacher_ckpt logs/rsl_rl/clutterpickplace_teacher/<teacher_run>/model_<best>.pt
```

`--enable_cameras` mandatory on all stages — env scene cfg spawns `TiledCamera` even for the teacher (camera prim instantiation requires the flag; output discarded when `wrist_image` is absent from `obs_groups`). Same symlink dance for `--load_run` as in Eval 1 `RUNNING.md` §5.1.

## 8. Visual modality — RGB + target instance mask

`(N, 4, 72, 128)`:

| Ch | Sim | Real |
|---|---|---|
| 0–2 | TiledCamera `rgb` → `/255`, per-episode tint + HSV DR | USB cam → `cv2.undistort` → resize → `/255` |
| 3 | Per-color instance mask (`semantic_segmentation` filtered to `class:cube_<target>`), corrupted to mimic Florence-2 | Florence-2 prompted with `f"{target_color} cube"` (via `deploy/cube_detector.py`'s `Detector` protocol) |

**Why mask channel.** HSV thresholding at the cube's working distance was empirically too brittle (no mask when the cube is far in frame, lighting-induced false positives on bowl/table). Florence-2 with a color prompt is robust to those failure modes; the cost is per-frame inference latency (~2–5 s on CPU at deploy), which is mitigated by running detection on a background thread at ~0.3–0.5 Hz while the policy runs at 25 Hz. The target_color one-hot still flows into the actor state + CNN FiLM head (§3.1) so that when the mask channel is zero (detector miss or distance-gated dropout) the policy can still operate from RGB alone — belt-and-suspenders, not redundant.

**Why this beats "FiLM-only colour grounding under DR."** The FiLM-only path forces PPO to learn the colour discrimination from sparse RL gradient under lighting DR — measured Stage-3 budget ~2500 iters. With the mask channel the visual problem collapses to "go to the white pixels" (same shape as Eval 1), at the cost of training Stage 2/3 against Florence-quality mask quality rather than GT. The §5 mask DR is what closes that gap.

If a particular colour pair plateaus despite the mask channel (Florence misclassifies under deploy lighting), the next escalation is a learned color-classifier head on the policy CNN; not implemented by default.

## 9. Why we don't use teleop in Stage 1

Eval 2's spec says "expert data collected from teleoperation is encouraged to use for training efficiency." This is about **vision** efficiency, not state-MDP efficiency. The Stage 1 teacher sees the full target pose, distractor pose, and target one-hot in its observation — there is no perception bottleneck to bridge with demos, and PPO on the §4 reward stack is the right tool. Teleop demos are only useful where the input modality is hard to map to actions (i.e., Stage 2's image-to-action regression), which is why the Eval-1 §7 distill is also the natural place to inject them here.

The flip side: if Stage 1 fails to converge (e.g., the color-conditioned grasp doesn't emerge from PPO alone), the diagnosis is reward shape or curriculum, **not** demo deficit. Adding teleop to Stage 1 would mask the real problem.

## 10. Implementation status

All wiring TODOs (items 1–5) are implemented as of the latest commit:

1. ✅ **CNN dispatch** — `PickPlaceVisionActorCritic.__init__` now accepts `cnn_class: str = "small"` and `cnn_kwargs: dict | None = None`. `"small"` (default) preserves Eval-1 behavior; `"resnet"` instantiates `_ResNetSpatialSoftmaxCNN(**cnn_kwargs)`.
2. ✅ **ResNet truncation** — `_ResNetSpatialSoftmaxCNN.__init__` accepts `truncate_at: str = "layer3"`; passing `"layer2"` drops `backbone.layer3` and yields a `(128, 9, 16)` feature map at 72×128 input.
3. ✅ **FiLM head** — `_ResNetSpatialSoftmaxCNN.__init__` accepts `film_cond_dim: int = 0`; when > 0, a small `film_mlp` (cond → 64 → 2C) modulates the trunk output before the 1×1 conv. γ-init=1 / β-init=0 means identity at init. `forward()` takes an optional `film_cond` kwarg (None → no modulation).
4. ✅ **Task registrations** — `tasks/clutterpickplace/__init__.py` registers `…-v0`, `…-Play-v0`, `…-Teacher-v0`, `…-Teacher-Play-v0`, `…-Teacher-Fast-v0`, `…-Teacher-Fast-Play-v0`, `…-Student-v0`, `…-Student-Play-v0`. All carry `rsl_rl_cfg_entry_point`.
5. ✅ **Agent module** — `tasks/clutterpickplace/agents/` contains `teacher_ppo_cfg.py`, `rsl_rl_ppo_cfg.py`, `distill_cfg.py`, `vision_actor_critic.py` (subclass), `vision_student_teacher.py` (subclass). Subclasses read the `goal` obs group and pass its one-hot to the CNN's FiLM head.
6. ⏳ **`reset_scene_to_default` ordering** — verification step. Run the `Teacher-Fast-Play-v0` zero-agent rollout to confirm cubes stay at the `place_clutter_blocks` xy and don't snap back to CuboidCfg `init_state.pos` after the event.

The `ObservationsCfg` was restructured: `target_color` moved out of `policy`/`critic` into a separate `goal` group, so the vision A-C can route it to FiLM without slicing-by-position.

## References

- **Code:** `tasks/clutterpickplace/{clutterpickplace_env_cfg,joint_pos_env_cfg}.py`, `mdp/{observations,rewards,events,terminations,commands}.py`, agents (to add): `{vision_actor_critic,vision_student_teacher,rsl_rl_ppo_cfg,teacher_ppo_cfg,distill_cfg}.py`. Eval-1 plan: [`EVAL1_PLAN.md`](./EVAL1_PLAN.md).
- **Pinto et al.**, *Asymmetric Actor Critic*, RSS 2018. <https://arxiv.org/abs/1710.06542>
- **Levine et al.**, *End-to-End Visuomotor Policies*, JMLR 2016 — spatial softmax.
- **Kostrikov et al.**, *DrQ*, ICLR 2021. <https://arxiv.org/abs/2004.13649>
- **Zhou et al.**, *Versatile and Generalizable Manipulation via Goal-Conditioned RL with Grounded Object Detection*, 2025. <https://arxiv.org/abs/2507.10814> — goal-conditioned RL with explicit target representation (we use a color one-hot in their place).
- **Danielczuk et al.**, *Visuomotor Mechanical Search*, ICRA 2021. <https://arxiv.org/abs/2008.06073> — teacher-guided RL for retrieving a known target from clutter; closest published analog.
- **RSL-RL**, Schwarke et al., 2025. <https://github.com/leggedrobotics/rsl_rl>
- **LeIsaac**, <https://github.com/LightwheelAI/leisaac> — wrist camera mount.
