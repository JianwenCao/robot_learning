# Eval 1 — Single-Object Pick-and-Place: Method

> Goal-conditioned PPO on SO-ARM101 in Isaac Lab → zero-shot deploy on the real arm. No teleop data; the only training signal is sim rollouts. Task gym ID: `Isaac-SO-ARM101-PickPlace-Bowl-v0`. For env setup, training, evaluation, and deploy commands, see [`RUNNING.md`](./RUNNING.md) and [`DEPLOY.md`](./DEPLOY.md).
>
> **Recipe lineage.** Asymmetric actor–critic (Pinto et al. 2018, privileged critic + image actor) + Levine spatial-softmax CNN + DrQ random-shift augmentation. Hyperparameter and DR envelopes anchored on two peer projects on the same robot family: **ManiSkill3 PickCube → SO-100** (StoneT2000, ≈ 91.6 % zero-shot real success, 25–40 M env-steps) and **CS6341 SO-101 vision** (Evans & Hegde 2025, partial vision transfer, surfaced the visual-DR gap). Hyperparameters that diverge from ManiSkill3 are flagged in §6.

## 1. MDP

A single PPO actor–critic over thousands of randomized envs. The block (2 cm cube) is randomized in xy; the bowl is **not a scene prim** — it is a 2-D goal `(x, y)` sampled per episode by `BowlPoseCommandCfg` in the robot base frame, with rejection sampling that keeps `‖block − bowl‖ ≥ 0.10 m`. Same frame at deploy (`--bowl_xy x y`), so no coordinate transform.

| Item | Value |
|---|---|
| Control | 50 Hz (decimation 2, sim 100 Hz) |
| Episode | 6.0 s = 300 steps |
| Action | 5 absolute-around-home arm joint targets (`scale=0.5`) + 1 binary gripper (`open=0.5`, `close=0.0`) |
| Workspace | `x ∈ [0.10, 0.30] m`, `y ∈ [−0.15, 0.15] m`, both block & bowl |
| Home `q` | `(0, 0, 0, 1.5708, 0, 0)` with gripper *open* (`gripper=0.5`) |
| Terminations | `time_out`, `block_off_table`. **No** success-termination (it incentivized "hold and hover" over release). |

## 2. Observations (asymmetric A-C)

Three obs groups in `tasks/pickplace/pickplace_env_cfg.py::ObservationsCfg`. Runner cfg routes:

```
actor   = policy + wrist_image      (deployable)
critic  = policy + critic            (privileged, sim-only; no image)
```

The critic deliberately does **not** receive the image — it has ground-truth block pose, so an image encoder is redundant compute. Only the actor has a CNN.

| Group | Fields | Shape |
|---|---|---|
| `policy` (deployable) | `joint_pos_rel`, `joint_vel_rel`, `gripper_state`, `bowl_xy`, `ee_proj_xy`, `ee_to_bowl_xy`, `last_action` | 1-D, concatenated |
| `critic` (privileged) | policy fields + `block_position`, `block_to_bowl_xy`, `gripper_to_block`, `is_grasped` | 1-D, concatenated |
| `wrist_image` | RGB + depth + binary block mask, all in `[0,1]` | `(N, 5, 72, 128)` (see §8) |

`ee_proj_xy` is taken from the `ee_frame` `FrameTransformer` at `gripper_link + offset=(0.01, 0, -0.09)`. The wrist `TiledCamera` is parented to `gripper_link` at `pos=(-0.001, 0.1, -0.04)`, ros-convention quaternion `(-0.404379, -0.912179, -0.0451242, 0.0486914)` (LeIsaac's verbatim mount). Intrinsics from `camera_intrinsics.yaml` via `mdp.load_wrist_cam_intrinsics()` → USD pinhole (`focal_length = fx · horizontal_aperture / W`); horizontal FOV ≈ 102°, identical to the real cam.

## 3. Network — `PickPlaceVisionActorCritic`

Subclass of `rsl_rl.modules.ActorCritic` in `tasks/pickplace/agents/vision_actor_critic.py`. Registered into RSL-RL's runner namespace at import time so `class_name="PickPlaceVisionActorCritic"` resolves.

```
wrist_image (5×72×128) ──[Spatial-Softmax CNN]── 128-D keypoints ─┐
state (policy group) ─────────────────────────────────────────── concat ── MLP[256,128,64] ── μ
                                                                 │           σ (scalar Param)
critic state (policy + critic groups) ───── MLP[256,128,64] ── V(s)
```

CNN: `Conv(8/4)→ELU → Conv(4/2)→ELU → Conv(3/1, K channels)` → spatial softmax per channel → expected `(x, y)` per keypoint → LayerNorm. Output is `2K = 128` dims (Levine et al. 2016). The spatial softmax is the inductive bias for "small object on a flat workspace": each channel becomes a soft-argmax keypoint, which dense MLP projection failed to discover from grasp gradient alone.

**DrQ random-shift** (Kostrikov et al., ICLR 2021): 4-pixel pad-and-crop on the actor's image input during training only.

No recurrence — `is_recurrent=False`. The reach-stage curriculum (§5) keeps the cube under the home FOV from iter 0, and the wide horizontal FOV makes search unnecessary.

## 4. Reward (in `mdp/rewards.py`)

Stage-gated; weights in `RewardsCfg`:

| Term | Weight | Condition |
|---|---|---|
| `pre_grasp_pose` | 2.0 | `proximity * jaw_openness * (1 − is_grasped)` |
| `reach_block` | 1.0 | `1 − tanh(d_ee_block / 0.05)`, ungated |
| `grasp_event` | 15.0 | indicator: `block_z > 0.025` (lift = grasp proxy) |
| `transport_to_bowl` | 4.0 | `(1 − tanh(d_ee_bowl / 0.15))` × `is_grasped` |
| `place_in_bowl` | 5.0 | `block_xy near bowl AND block_z < 0.08`, gated on per-episode lift latch |
| `release_in_bowl` | 20.0 | place AND gripper open AND block settled, same lift latch |
| `action_l2`, `action_rate_l2`, `joint_vel_l2` | −1e-4 → −1e-2 | curriculum ramp at 10 k env-steps |
| `block_dropped` | −2.0 | block on table away from bowl |

The per-episode `_was_grasped` latch (in `_episode_lifted_mask`) closes the "drag-the-cube laterally" exploit — `place`/`release` only pay out after the policy has lifted the block ≥ 2.5 cm at some prior step that episode.

## 5. Curriculum & DR (in `mdp/events.py`, `CurriculumCfg`)

**Currently shipped:**
- **Block-xy expansion** (`expand_block_xy_range`): ±2 cm × ±2 cm → ±10 cm × ±15 cm linearly over 12 k warm-up + 180 k expand env-steps.
- **Bootstrap grasp** (`init_block_in_gripper`, p=0.10 *constant floor*): 10 % of resets start with the block in a closed gripper. Floor not decay — the decay version collapsed when p hit 0 (memory: `bootstrap-curriculum-pitfall`).
- **Wrist-cam pose DR** (`randomize_camera_uniform`, ros): ±25 mm / ±2.5° per reset on `gripper_link`. LeIsaac envelope.
- **Action / joint-vel penalty ramp**: −1e-4 → −1e-2 at 10 k env-steps.
- **DrQ random-shift** on actor image during training (`vision_actor_critic.py::_random_shift_pad`).

**Required visual DR before real-robot deploy** (currently missing — peer-project failures point to this as the actual transfer blocker):
- **Cube color / material DR** — uniform diffuse RGB biased away from `#B8ADA9`, roughness `[0.4, 0.9]`. Per-env at reset.
- **Table color jitter** — `#B8ADA9 ± 15 RGB` per env.
- **Lighting** — 1–3 light sources, intensity `[500, 1500]`, color temperature `[2500, 9500] K` (CS6341 + NVIDIA SO-101 envelopes). Random HDRI dome if available.
- **Image corruption** — Gaussian noise σ ∈ `[0, 5/255]`, brightness ±15 %, motion blur 0–3 px, JPEG q `[70, 100]`. Apply *after* the camera read, identically replicated in deploy preprocess.
- **Greenscreen / HDRI background overlay** — composite a random HDRI behind the table in sim. ManiSkill3 PickCube's bridge for the "sim has no clutter / real has whatever's on the desk" gap.
- **Distractors** — 0–3 random small boxes outside the workspace.

Per-bootstrap-status TB metrics (`log_bootstrap_metrics`): `release_from_scratch`, `grasp_from_scratch`, `release_bootstrap`, `grasp_bootstrap`, `p_bootstrapped`. **`*_from_scratch` are the only success curves to watch** — they exclude the bootstrapped 10 %.

## 6. PPO config (`tasks/pickplace/agents/rsl_rl_ppo_cfg.py`)

Side-by-side with the **ManiSkill3 PickCube** recipe (validated to 91.6 % zero-shot real success on SO-100). If convergence stalls past ~50 M env-steps, **try the right column verbatim before adding more reward terms.** Lower `gamma=0.9` for short-horizon manipulation is the most striking divergence — 300-step episodes don't need a 0.98 horizon, and a tighter discount sharpens credit assignment on the grasp event.

| Hyperparam | Ours | ManiSkill3 PickCube |
|---|---|---|
| `num_envs` | 2048 (1024 if VRAM-bound) | 1024–2048 |
| `num_steps_per_env` | 24 | 16 |
| `max_iterations` | 4000 (≈ 200 M env steps) | ≈ 25–40 M sufficed |
| `init_noise_std` | 0.5 (scalar) | — |
| `actor_hidden_dims` / `critic_hidden_dims` | `[256, 128, 64]`, ELU | Nature-CNN + small MLP |
| `clip_param` | 0.2 | 0.2 |
| `entropy_coef` | 0.003 | (default) |
| `num_learning_epochs` / `num_mini_batches` | 5 / 8 | **8 / 32** |
| `learning_rate` / `schedule` / `desired_kl` | `1e-4` / `adaptive` / `0.005` | `3e-4` (typical) |
| `gamma` / `lam` / `max_grad_norm` | `0.98` / 0.95 / 1.0 | **0.9** / 0.95 / 1.0 |

Asymmetric obs wiring:
```python
obs_groups = {"policy": ["policy", "wrist_image"],
              "critic": ["policy", "critic"]}
```

## 7. Fallback: teacher–student distillation

If end-to-end vision PPO doesn't converge after `max_iterations=4000` *with* §5 visual DR wired in, fall back to the DextrAH-G recipe:

1. Train a *state-based teacher* — PPO over the privileged `critic` group only (block pose, distances, grasp flag) to high success in sim. Cheap; state-only PPO already solved this MDP at the Day-3 milestone.
2. Distill into a *vision student* via DAgger or BC on teacher rollouts, student reading only `policy + wrist_image`. The student inherits the optimal action distribution rather than discovering it from grasp gradient through pixels.

Only invoke if vision PPO has clearly stalled — adds a second training stage.

## 8. Visual modality — RGB + depth + mask, same pipeline in sim and real

The wrist camera is the only physical sensor; depth and the block mask are **derived** from its RGB feed at deploy. The key sim-to-real principle: **run the same derivation pipeline in both domains.** Don't rely on Isaac's perfect `distance_to_camera` for sim depth and DA3 for real depth — that just relocates the gap. Apply DA3 to the *sim-rendered RGB* identically to the real RGB, so the policy sees DA3-flavored depth (with its smoothing and edge artifacts) in both worlds. Same convention we already use for `cv2.undistort`: sim is a perfect pinhole but we still pipe both feeds through the same undistort to keep the preprocess identical.

Wrist input is a 5-channel `(N, 5, 72, 128)` tensor stacking:

| Channel | Source (sim *and* real) | Notes |
|---|---|---|
| 0–2 | RGB | sim TiledCamera (`rgb`) → cv2.undistort (no-op in sim, real distortion in deploy) → resize 128×72 → `/255`. |
| 3 | **Depth from Depth Anything 3** | Run `depth-anything/DA3-Small` (80 M params, DINOv2-S backbone) on the same RGB above. Relative depth, calibrate to metric once per session via a linear fit against known FK distances at home pose; clip to `[0, 0.5 m]`, divide by 0.5. Run DA3 at 252×140 (DINOv2 patch is 14 px — 128×72 = 9×5 patches is below training distribution and degrades accuracy), downsample the depth map to 128×72. |
| 4 | Binary block mask | sim: pull the cube class from `semantic_segmentation`. real: HSV `cv2.inRange` calibrated once on a wood-tone block on the gray table; optional 3×3 morphological close. Resize to 128×72. For Eval 2/3 the bounds become target-color-conditioned — same channel, no network change. |

Network change: `_SpatialSoftmaxCNN` first conv goes `Conv2d(3 → 32) → Conv2d(5 → 32)`. Otherwise unchanged.

**Why DA3.** Depth Anything 3 (ByteDance-Seed, 11/2025) is SOTA on indoor monocular metric depth (ETH3D δ₁ = 0.917, SUN-RGBD AbsRel = 0.105 — beats DA-V2, UniDepth v2, DepthPro, Metric3D v2 on standard indoor benchmarks). The Small checkpoint is the right speed/quality trade-off for the 50 Hz inner loop; upgrade to `DA3-Metric-Large` (350 M, directly metric, skips the calibration step) only if timing budget allows.

**Why a mask.** ManiSkill3 PickCube went from 91.6 % (RGB) to 95 % (with object segmentation). The cube + uniform-gray table makes the segmentation easy at deploy — HSV thresholding gets a high-recall mask in 5 lines of OpenCV — and exact in sim. We keep the mask alongside RGB+depth (not as a replacement) so the policy stays robust if the real-side threshold ever misses.

**Encoder fallback.** If the from-scratch spatial-softmax CNN doesn't transfer even with this 5-channel input + §5 visual DR, swap to a **frozen pretrained backbone**: Theia-Tiny (CMU, ~6 M params, manipulation-priored, distilled from DINOv2 + SAM + Depth-Anything) or DINOv2-Small (22 M). Keep the spatial-softmax head on top of the backbone's spatial feature map for the same positional inductive bias. ~10–20 ms inference, fits 50 Hz.

**Deliberately skipped:**
- Optical flow (wrist motion swamps object motion).
- Full point cloud + PointNet (~3 % gain over RGBD-as-channels at 10× engineering cost).
- Open-vocab detector → 2D block-center coordinate (~50 ms latency, unnecessary for one cube; reconsider for Eval 2/3).
- Depth-only without RGB (loses the cube-vs-table-color cue when block sits flat).

**Rollout order.** (1) Wire §5 visual DR on the current RGB pipeline first — most peer-project transfer failures are DR shortfalls, not modality shortfalls. (2) Add depth via DA3 in both sim and real. (3) Add the mask channel. (4) Only if 1–3 don't transfer, swap to frozen Theia / DINOv2 (this section) or fall back to teacher–student (§7). Steps 1–3 each touch only `mdp.wrist_image` and the CNN's first conv — surgical changes, no env restructure.

## References

**This repo:** `tasks/pickplace/{pickplace_env_cfg.py, joint_pos_env_cfg.py}`, `mdp/{observations,rewards,events,terminations,commands}.py`, `agents/{vision_actor_critic.py, rsl_rl_ppo_cfg.py}`, `camera_intrinsics.yaml`, `robots/trs_so101/{so_arm101.py, urdf/so_arm101.urdf}`. Run/deploy guides: [`RUNNING.md`](./RUNNING.md), [`DEPLOY.md`](./DEPLOY.md).

**Method lineage / peer recipes:**
- Pinto, Andrychowicz et al., *Asymmetric Actor Critic for Image-Based Robot Learning*, RSS 2018.
- Levine et al., *End-to-End Training of Deep Visuomotor Policies*, JMLR 2016 — spatial-softmax CNN.
- Kostrikov et al., *DrQ*, ICLR 2021 — random-shift augmentation.
- ByteDance-Seed, *Depth Anything 3*, arXiv 2511.10647 (11/2025). <https://github.com/ByteDance-Seed/Depth-Anything-3>.
- Tao et al., *ManiSkill3*, 2024. <https://github.com/StoneT2000/lerobot-sim2real>, <https://github.com/haosulab/ManiSkill>.
- Evans & Hegde, *Vision-Based Manipulation via Sim-to-Real RL — SO-101 with Isaac Lab*, CS6341 Fall 2025. <https://yuxng.github.io/Courses/CS6341Fall2025/project_group_15.pdf>.
- Lum, Allshire et al., *DextrAH-G*, 2024. <https://sites.google.com/view/dextrah-g>.
- LeIsaac (LightwheelAI). <https://github.com/LightwheelAI/leisaac>.
