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

**Active-pair sampling.** Per reset, `TargetColorCommand` samples (i) two distinct colors from the palette as the *active pair* and (ii) one of those two as the *target*. The active pair is placed adjacent in the workspace by `place_clutter_blocks`; the other four cubes are teleported to `HIDDEN_PARK_XY` slots off-table where the wrist camera can't see them. **Half-separation is sampled per episode** from `U(0.0125, 0.030) m` — cube centers 2.5–6 cm apart, edge-to-edge margin 0.5–4 cm for the 2 cm cubes. The spec calls the configuration "adjacent (flat cluster)" but in practice a human evaluator places blocks with a small gap, so sampling the range trains a margin-robust policy rather than overfitting to a single canonical spacing. Pair axis `θ ∈ U[0, 2π)` so orientation is fully randomized.

**Bowl.** 2-D goal from `UniformPoseCommandCfg` — **no scene prim**, **no rejection sampling**. Eval 1's `BowlPoseCommand` rejection sampler targets a single scene asset; here the target cube varies per env, so we accept a small fraction of episodes where the bowl spawns over the cluster (the place-only-target reward still penalizes those). Same `(x, y)` frame at deploy.

## 2. Observations (asymmetric A-C)

`ObservationsCfg` defines three groups; runner cfgs select per stage (§7).

| Group | Fields | Shape |
|---|---|---|
| `policy` (deployable) | `joint_pos_rel`, `joint_vel_rel`, `gripper_state`, `bowl_xy`, `ee_proj_xy`, `ee_to_bowl_xy`, **`target_color_onehot`** (6-D), `last_action` | 1-D |
| `critic` (privileged) | `policy` + `target_block_position`, `distractor_block_position`, `target_block_to_bowl_xy`, `target_gripper_to_block`, `target_is_grasped` | 1-D |
| `wrist_image` | **RGB only** (no mask) | `(N, 3, 72, 128)` |

Wrist `TiledCamera`: same mount and intrinsics as Eval 1 — parented to `gripper_link` at `pos=(-0.001, 0.1, -0.04)`, ros quat `(-0.404379, -0.912179, -0.0451242, 0.0486914)`. Renders `["rgb"]` only — no `semantic_segmentation`, no depth. Reasoning: with both blocks tagged `class:block`, a class-id mask can't disambiguate target from distractor, and a per-prim instance mask would still need a sim→real bridge (HSV thresholding on the real arm with per-color calibration). Going 3-channel RGB + color one-hot puts the color discrimination inside the learned CNN, removes the HSV calibration step, and lets the same 3-channel pipeline run at deploy.

When `wrist_image` is absent from `obs_groups` (Stage 1 teacher), the CNN auto-disables → pure MLP A-C.

## 3. Network — `ClutterPickPlaceVisionActorCritic`

Same architecture as Eval 1's `pretrained-backbone PPO path` (commit `903ef6b`): **frozen ImageNet ResNet-18 trunk → trainable 1×1 conv → Levine-style spatial-softmax head → 128-D keypoints**. We reuse `_ResNetSpatialSoftmaxCNN` from `tasks/pickplace/agents/vision_actor_critic.py` verbatim, only re-instantiated with `in_channels=3` (RGB only, no mask).

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
4. **Optional BC-v1 weight overlay.** `_ResNetSpatialSoftmaxCNN.__init__` accepts `bc_v1_weights_path=` to load BC v1's ResNet-18 weights (saved under `img_enc.backbone.*`) on top of the ImageNet init. If the Eval-1 BC pipeline produces good encoder weights, those overlay further — strictly better than raw ImageNet for our wrist-cam pixel distribution.

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
- `place_clutter_blocks`: cluster center in `(0.15, 0.22) × (−0.10, 0.10)`, **half-separation sampled per episode from `U(0.0125, 0.030)` m** (0.5–4 cm edge-to-edge margin), pair axis `θ ∈ U[0, 2π)`. No separation curriculum — see below.
- `reset_target_latches`: clears per-episode lift / over-bowl latches.
- Action-rate / joint-vel penalty ramp −1e-4 → −1e-2 at 10 k env-steps.
- `log_target_success_metrics` TB metric (target-correct success rate + wrong-block placement rate).

DR (in env cfg — applies on every reset, including Stage 1 teacher; teacher just doesn't read the image so the wasted compute is one CPU/GPU write of a few floats per env per reset):
- `randomize_wrist_image_tint`: per-channel **linear scale `U(0.55, 1.45)`** + **brightness shift `U(−0.20, +0.20)`**. Much wider than Eval 1's ±15 %; widened because color discrimination is now load-bearing.
- `randomize_wrist_hsv_dr`: **hue rotation `±20°`** in linear RGB, saturation scale `U(0.65, 1.35)`, value scale `U(0.55, 1.45)`. **This is the sim2real-critical DR term for color-conditioned tasks** — hue rotation simulates white-balance / colored-ambient drift that linear tint can't model. Without HSV jitter, the policy memorizes the sim renderer's exact color rendition and fails on any USB-cam white-balance offset.

Stage 3 adds (training-only, in the model):
- DrQ ±4 px (in CNN, see §3).

**No xy expand or separation curriculum.** Eval 1 used `expand_block_xy_range` (±3 cm → ±7×±12 cm). For Eval 2 the cluster band is already tight and the margin is *already* randomized per episode (§1), so the policy sees the full distribution from touching-ish to clearly-separated from step 0 — there's no easy/hard regime to schedule between. If Stage 1 fails to converge on the small-margin end, swap the half-separation range to `(0.025, 0.030)` for the first 5 k env-steps and ramp the lower bound down to 0.0125 over the next 20 k — that's a one-line `events.py` change.

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

## 8. Visual modality — RGB only

`(N, 3, 72, 128)`:

| Ch | Sim | Real |
|---|---|---|
| 0–2 | TiledCamera `rgb` → `/255`, per-episode tint DR | USB cam → `cv2.undistort` → resize → `/255` |

**No mask channel.** This is the key departure from Eval 1. Justification:

1. Both blocks tagged `class:block` — semantic mask can't disambiguate.
2. An instance mask would require HSV thresholding on the real arm, with per-color HSV calibration — adds a sim/real bridge that the 3-channel path avoids.
3. Color is now in the *policy state* (one-hot), not in the *image channel* — the CNN learns "match my pixels to the one-hot" rather than "find the masked region". Lighting DR (§5) is the regularizer that keeps this transferable.

If Stage 3 plateaus at low target-correct rate (< 60 %), the fallback is to add a target-color HSV mask channel as a 4th image channel, calibrate HSV on the real cam, and retrain Stage 3 only (Stage 1 unchanged). Don't commit unless needed.

## 9. Why we don't use teleop in Stage 1

Eval 2's spec says "expert data collected from teleoperation is encouraged to use for training efficiency." This is about **vision** efficiency, not state-MDP efficiency. The Stage 1 teacher sees the full target pose, distractor pose, and target one-hot in its observation — there is no perception bottleneck to bridge with demos, and PPO on the §4 reward stack is the right tool. Teleop demos are only useful where the input modality is hard to map to actions (i.e., Stage 2's image-to-action regression), which is why the Eval-1 §7 distill is also the natural place to inject them here.

The flip side: if Stage 1 fails to converge (e.g., the color-conditioned grasp doesn't emerge from PPO alone), the diagnosis is reward shape or curriculum, **not** demo deficit. Adding teleop to Stage 1 would mask the real problem.

## 10. Unblocking Stage 1 — wiring TODOs

The plan above describes the target architecture; the following pieces are *not yet in the code* and block the §7.2 workflow:

1. **`PickPlaceVisionActorCritic` CNN dispatch.** `__init__` still hardcodes `_ImpalaSmallCNN` at lines ~456 / 460. Add `cnn_class: str = "resnet"` cfg arg + dispatch table; defaults to ResNet for Eval 2, leaves Eval 1 free to opt in. Fixes Eval 1 pretrained-backbone path too.
2. **`_ResNetSpatialSoftmaxCNN` truncation.** Drop `backbone.layer3` from the trunk Sequential (§3, risk A); update the BC-v1 weight overlay key remap accordingly (one fewer Sequential index).
3. **FiLM head.** Add the `film_mlp` (6 → 512 → split into γ/β) and modulate trunk output before the 1×1 conv (§3.1). Touches `_ResNetSpatialSoftmaxCNN.__init__` + `forward` only.
4. **Task registrations.** Add `Isaac-SO-ARM101-ClutterPickPlace-Teacher-v0` and `…-Student-v0` to `tasks/clutterpickplace/__init__.py`. The first drops `wrist_image` from `obs_groups`; the second wires `DistillationRunner`-compatible cfg.
5. **Agent module.** Create `tasks/clutterpickplace/agents/` with `vision_actor_critic.py` (`ClutterPickPlaceVisionActorCritic` subclass passing `target_color` to the FiLM head), `teacher_ppo_cfg.py`, `distill_cfg.py`, `rsl_rl_ppo_cfg.py`, and `vision_student_teacher.py`.
6. **`reset_scene_to_default` ordering.** `EventCfg.reset_all` runs `mdp.reset_scene_to_default` *before* `place_clutter_blocks` overwrites cube poses. Confirm `reset_scene_to_default` doesn't reset cubes to their CuboidCfg spawn-time `init_state.pos` *after* `place_clutter_blocks` has teleported them (would defeat the active-pair geometry).

Items 1–5 are blocking; 6 is a verification step that can be done with one `Play-v0` rollout watching the cube xy.

## References

- **Code:** `tasks/clutterpickplace/{clutterpickplace_env_cfg,joint_pos_env_cfg}.py`, `mdp/{observations,rewards,events,terminations,commands}.py`, agents (to add): `{vision_actor_critic,vision_student_teacher,rsl_rl_ppo_cfg,teacher_ppo_cfg,distill_cfg}.py`. Eval-1 plan: [`EVAL1_PLAN.md`](./EVAL1_PLAN.md).
- **Pinto et al.**, *Asymmetric Actor Critic*, RSS 2018. <https://arxiv.org/abs/1710.06542>
- **Levine et al.**, *End-to-End Visuomotor Policies*, JMLR 2016 — spatial softmax.
- **Kostrikov et al.**, *DrQ*, ICLR 2021. <https://arxiv.org/abs/2004.13649>
- **Zhou et al.**, *Versatile and Generalizable Manipulation via Goal-Conditioned RL with Grounded Object Detection*, 2025. <https://arxiv.org/abs/2507.10814> — goal-conditioned RL with explicit target representation (we use a color one-hot in their place).
- **Danielczuk et al.**, *Visuomotor Mechanical Search*, ICRA 2021. <https://arxiv.org/abs/2008.06073> — teacher-guided RL for retrieving a known target from clutter; closest published analog.
- **RSL-RL**, Schwarke et al., 2025. <https://github.com/leggedrobotics/rsl_rl>
- **LeIsaac**, <https://github.com/LightwheelAI/leisaac> — wrist camera mount.
