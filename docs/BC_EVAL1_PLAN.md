# BC for Eval 1: Plan & Implementation

Behavior-cloning approach to Project 3 Eval 1 (single-block pick-and-place into a specified bowl). Re-implementation; the previous `bc/` directory has been deleted.

## 1. Problem analysis

### 1.1 Task spec (Eval 1)
- Single wooden block, randomly placed on a light-gray table (#B8ADA9). Color is unconstrained.
- A bowl is placed at a known `(x, y, z)` position in the robot base frame; the position is given as input at eval time.
- Success = block placed inside the bowl and released. 5 rollouts × 10 pts.
- BC is **explicitly allowed** for Eval 1.
- The policy must operate on visual observation of the block; the target location is provided as coordinates. The wrist camera (RGB) is the visual sensor on SO-101.

### 1.2 What we have
- **Hardware-style robot:** SO-ARM101 follower, 6 joints (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`).
- **Sim:** Isaac Lab task `Isaac-PickPlace-SoArm101-...` (already in `isaac_so_arm101/`), wrist + top cameras at 128×72.
- **Demonstrations** at `demonstrations/RobotLearning-RL/Eval1/` (LeRobot v3.0 format):
  - `eval1-pick-place-pilot/` — 13 episodes, 6 283 frames, 30 fps.
  - `eval1-pick-place-pilot-2/` — 11 episodes, 5 176 frames, 30 fps.
  - **Total: 24 successful episodes, ≈ 11.5k frames (≈ 6.4 min of teleop).**
  - Per frame: `action` (6-D joint positions, **degrees**, absolute), `observation.state` (6-D joint positions, degrees), wrist video (720×1280 RGB MP4, packed per chunk), top video (same).
  - Per episode: bowl target `(x, y, z)` in robot base frame in `meta/episode_targets.csv`. Bowl z is constant 0.030 m; bowl xy ranges roughly `x ∈ [0.26, 0.40]`, `y ∈ [-0.16, 0.30]` (8 distinct bowl placements across the two pilots).
  - Sim-rendered companion array `sim_renders/<pilot>/wrist_images.npy` shape `(T, 3, 72, 128)` uint8, indexed by `episode_offsets.npy`. These are wrist-camera re-renders of the demo joint trajectories inside the Isaac sim — i.e. in-sim-distribution wrist views aligned 1:1 with the parquet frames. **This is what we should train against if we deploy in sim**, because real-teleop wrist video has a different camera intrinsic / lighting / table texture than the eval sim env.

### 1.3 Subtleties / risks (this is where naive BC breaks)

1. **Action-space mismatch between demos and Isaac sim env.** Demo actions are absolute joint targets in **degrees** with a continuous gripper value (0–52). Sim's RL action layer (`JointPositionActionCfg(scale=0.5, use_default_offset=True)` + `BinaryJointPositionActionCfg`) expects **radians-delta around home** for the 5 arm joints and a binary command for the gripper. We must NOT round-trip BC actions through the RL action layer — we should drive the articulation directly. (See §5.2.)
2. **Visual domain gap.** Teleop demos were recorded on real hardware (720p wrist video). Isaac sim's wrist is 128×72 with synthetic lighting, table, and a `DexCube` mesh. The `sim_renders/` arrays exist specifically to bridge this. For sim deployment, train on `sim_renders` images; for real-robot deployment, train on the MP4-decoded teleop images. We pick the deployment target first and train accordingly.
3. **Goal conditioning is required.** Each episode has a different bowl `(x, y, z)`, so a policy that just regresses (image, state) → action will learn an *averaged* place location. The bowl target must be an explicit input. We feed the 3-vector `bowl_xyz` (robot base frame, m) as a goal channel concatenated to the proprio MLP head.
4. **Compounding error / covariate shift.** Only 24 episodes is small. Mitigations: action chunking (predict next k actions, execute k', re-plan), heavy image augmentation, predicting *next-step target joint angles* rather than deltas (closer to the teleop labels and easier to chain).
5. **Sparse / asymmetric coverage of bowl positions.** Only 8 distinct bowls observed. The policy will not extrapolate well to bowls outside that envelope. We will (a) log this limit, and (b) optionally synthesize extra coverage by base-frame-relative augmentation: rotate the wrist image + joint angles + bowl vector by a small azimuth around the base z-axis, since the task is approximately rotation-equivariant in the horizontal plane. *Stretch goal — only if vanilla BC fails on held-out bowls.*
6. **Gripper continuous → binary.** The sim's gripper is binary. We threshold the BC's continuous gripper output (e.g. `> 25` → closed) when deploying in sim. On real robot we can keep it continuous.

### 1.4 Success criterion

- **Open-loop:** held-out validation MSE on actions ≤ 5° per arm joint and gripper error ≤ ~10 units.
- **Closed-loop (primary):** ≥ 3/5 successful rollouts in Isaac sim on the existing `Isaac-PickPlace-SoArm101-Joint-v0` env with bowl positions sampled from the demo distribution. ≥ 1/5 on held-out bowls is a stretch target.

## 2. Method overview

Single-stage supervised BC. Goal-conditioned, image-based, with action chunking.

```
Inputs at time t:                                         Output:
  wrist RGB (1, 3, 72, 128) uint8 → CNN encoder           predicted next k=8 actions
  proprio q_t (1, 6) degrees     → MLP encoder            shape (k, 7) = (k, 6 joints + 1 gripper)
  bowl_xyz (1, 3) meters         → linear embedding
  → concat → trunk MLP → action chunk head
```

- **Action chunk:** predict 8 next-step **absolute joint targets** (in degrees) + continuous gripper. At deploy, execute the first 4, then re-query. This is the ACT/Diffusion-Policy trick at its simplest — no transformer, just an MLP head with `8 × 7 = 56` outputs.
- **Loss:** `L1` (Huber-1) on actions, with per-joint normalization to unit variance. L1 was chosen over MSE because action labels have small spikes (gripper close/open events) that L2 smears.
- **Normalization:** fit `mean / std` per joint on the training split; store in `stats.json`. Same for `proprio` and `bowl_xyz` (bowl: z is constant, so we just standardize x, y).
- **Architecture (small, ~3 M params):**
  - Image encoder: ResNet-18 with `torchvision` ImageNet weights, head replaced with a 256-D linear projection, mean-pooled spatial features. (Pretrained weights are important given how few demos we have.)
  - Proprio encoder: `Linear(6→128) → GELU → Linear(128→128)`.
  - Goal encoder: `Linear(3→64) → GELU → Linear(64→64)`.
  - Trunk: concat (256+128+64) → `Linear(448→512) → GELU → Linear(512→512) → GELU` → head `Linear(512→56)`.
- **Image augmentation (train only):** random crop (pad 6 then crop back to 72×128), brightness/contrast/saturation jitter ±0.2, hue jitter ±0.05, color drop with p=0.1 (to be color-invariant — block color is unconstrained per spec), small Gaussian noise σ=0.01 on normalized pixels. **No horizontal flip** — that would break joint-direction semantics.

## 3. Repo layout (to create)

```
bc/
├── __init__.py
├── config.py            # all hyperparams + paths (one file, no YAML)
├── dataset.py           # LeRobot v3.0 → torch Dataset, builds (img, proprio, bowl, action_chunk)
├── model.py             # GoalCondBCPolicy (ResNet18 + MLP heads, action-chunk output)
├── train.py             # supervised training loop, AdamW, cosine schedule, eval on val split
├── normalize.py         # fit + apply per-dim stats; save/load stats.json
├── eval_openloop.py     # rollout BC on held-out demo prefixes → action-MSE / trajectory dist
├── deploy_sim.py        # closed-loop in Isaac sim; directly drives articulation joint targets
└── README.md            # how to train and eval
```

Total target: ~1k lines, no premature abstractions. One model class, one dataset class, one training loop, one deploy script.

## 4. Data pipeline (`bc/dataset.py`)

**Source of truth:** the two parquet directories. For each row we need:
- `action` (np.ndarray (6,) float32, degrees) — from parquet.
- `observation.state` (np.ndarray (6,) float32, degrees) — from parquet.
- `wrist_image` (np.ndarray (3, 72, 128) uint8) — from `sim_renders/<pilot>/wrist_images.npy` mmapped, indexed by global frame index. (For "real-camera" mode we would decode the MP4 instead; we are not doing that for v1.)
- `bowl_xyz` (np.ndarray (3,) float32, meters) — from `episode_targets.csv`, looked up by `episode_index`.

**Dataset behavior:**
- Flatten all 24 episodes from both pilots into one global list of frames, with a per-row mapping `(pilot_id, episode_idx, frame_idx_in_episode, frame_idx_global_for_npy)`.
- On `__getitem__(i)`:
  1. Load wrist image from mmapped npy.
  2. Load proprio and current-action from parquet (cache parquets in memory — they're tiny, < 5 MB each).
  3. Load action chunk: `actions[i : i+k]`, padding the tail with the last action of the episode if the chunk runs past the episode end. Also produce a `mask` for valid chunk positions (used to skip padded targets in the loss).
  4. Look up `bowl_xyz` from the episode → target table.
  5. Apply image aug if `train=True`; normalize image to ImageNet stats.
  6. Normalize proprio + action + bowl using the saved stats.
- **Train/val split:** by episode (not by frame!), 80/20, deterministic seed. Both pilots' episodes are pooled before splitting. This validation is for tuning only; the closed-loop sim eval is the real verdict.

**No shuffling within episode required** — BC is i.i.d. over `(s, a)` pairs.

## 5. Training (`bc/train.py`)

**Hyperparameters (v1, will tune):**
- batch size 128, AdamW lr 3e-4, weight decay 1e-4.
- LR schedule: cosine to 0 over 50 epochs after 1k warmup steps.
- Action chunk k = 8 (= 0.27 s at 30 Hz), execute first k'=4 at deploy.
- Image-encoder lr multiplier 0.1× (freeze BN, fine-tune everything else).
- Best-model selection on val action L1.
- Loss: mean L1 with chunk mask, summed across action dims, mean across batch.
- Outputs: checkpoint `bc/runs/<timestamp>/best.pt`, training log, stats.json.

**Expected wall-clock:** with ~11.5k frames × 50 epochs = ~575k samples on one GPU, this is < 1 hr on a single RTX-class card.

### 5.1 Verification gates (per Karpathy guidelines: each step has a check)

1. **Dataset shapes** → assert `(B, 3, 72, 128) uint8`, `(B, 6)`, `(B, 3)`, `(B, k, 7)`; run `next(iter(loader))` once and print shapes/ranges before training. Catches mismatches early.
2. **Stats sanity** → after fitting, normalized action distribution should have per-dim mean ≈ 0, std ≈ 1 on the training set.
3. **Overfit a single batch** → loss should reach < 0.01 in < 200 steps. If it can't, the model is broken; do not proceed to full training.
4. **First-epoch loss curve** → val L1 should drop monotonically for ~5 epochs. If it diverges, lr is too high.
5. **Open-loop eval** → on val episodes, predict actions step-by-step using teacher-forced proprio and compare to ground truth. Median per-joint L1 < 5° for the arm and < 10 units for the gripper.

## 5.2 Action-space conversion (CRITICAL)

The demo action is `(shoulder_pan_deg, shoulder_lift_deg, elbow_flex_deg, wrist_flex_deg, wrist_roll_deg, gripper_unit)`. The Isaac sim env's RL action layer expects radian-delta + binary gripper. We do **not** want to route BC through that layer.

**At deploy:** open the env once, then bypass `env.step(action)`. Each control tick, do:
```python
q_rad = deg2rad(predicted_arm_targets)            # (5,)
gripper_rad = 0.0 if predicted_gripper < 25.0 else 0.5  # binary thresholding
target = torch.cat([q_rad, gripper_rad], -1)      # (6,)
robot.set_joint_position_target(target, joint_ids=arm_and_gripper_ids)
sim.step()
robot.update()
```
This avoids the JointPositionAction scaling entirely. If we still want to use `env.step` for camera capture, we can pass `action=torch.zeros_like(...)` and then *overwrite* the joint targets between `pre_step` and `physics_step` — but the cleaner path is to bypass `env.step` and call the underlying ArticulationView APIs.

## 6. Closed-loop deployment (`bc/deploy_sim.py`)

Pseudocode:
```python
env = gym.make("Isaac-PickPlace-SoArm101-Joint-v0", num_envs=1, headless=False)
policy = GoalCondBCPolicy.load("bc/runs/.../best.pt")
stats = Stats.load(...)

for rollout in range(5):
    obs, _ = env.reset()
    bowl_xyz = sample_or_fix_bowl()      # set via env config; also fed to policy
    set_env_bowl(env, bowl_xyz)
    chunk = None; step_in_chunk = 0
    for t in range(MAX_STEPS):
        if chunk is None or step_in_chunk >= EXECUTE_K:
            wrist = grab_wrist_image(env)        # (3, 72, 128) uint8
            proprio = grab_joint_pos_deg(env)    # (6,)
            chunk = policy.predict_chunk(wrist, proprio, bowl_xyz)   # (k, 7)
            step_in_chunk = 0
        a = chunk[step_in_chunk]; step_in_chunk += 1
        drive_articulation(env, a)               # see §5.2
    log_success(env)
```

Success detection: reuse the existing `mdp/terminations.py` success-with-release term (block in bowl + gripper open). We do not need this to train, only to score rollouts.

## 7. Evaluation protocol

1. **Open-loop**: action L1 on the held-out 20 % of episodes (≈ 5 episodes). Should land in v1 training.
2. **In-distribution closed-loop**: run with the 8 bowl positions seen in training, 1 rollout each. Target: ≥ 5/8 success.
3. **Eval-1 standard closed-loop**: run with 5 random bowl positions sampled inside the convex hull of the training bowls. Target: ≥ 3/5 success. This is the proxy for the actual Eval 1 score.
4. **Held-out-bowl closed-loop (stretch)**: 5 bowls outside the training hull. No specific target — informational.

If (2) is < 5/8, debug before doing more rollouts (likely: vision aug too weak, or bowl not actually being fed to the policy).

## 8. Implementation order (small, verifiable steps)

| # | Step | Verify by |
|---|------|-----------|
| 1 | `bc/dataset.py` + a `__main__` smoke test that dumps one sample's shapes/min/max | Run script, eyeball numbers |
| 2 | `bc/normalize.py` fit on train split, save/load stats | Round-trip → reconstruction error 0 |
| 3 | `bc/model.py` instantiate, forward random tensors, check output shape `(B, k, 7)` | Print, no error |
| 4 | `bc/train.py` overfit a single batch | Loss < 0.01 in < 200 steps |
| 5 | Full training (50 epochs) | Val L1 trends down, best ckpt saved |
| 6 | `bc/eval_openloop.py` on val episodes | Median per-joint L1 < 5° |
| 7 | `bc/deploy_sim.py` integration — first just no-policy zero-action rollout to confirm env + articulation driver works | Robot stays still without crashing |
| 8 | Closed-loop rollout with trained policy on a seen bowl | At least one successful pick attempt visually |
| 9 | Full 5-rollout closed-loop eval | Success count logged |

Each step is checked before moving to the next. If step N fails, fix step N — don't pile on more layers.

## 9. Known things we are NOT doing in v1 (and why)

- **No transformer / ACT-style temporal attention.** k=8 chunk + MLP is enough at this scale; transformers help when you have 10× more data.
- **No diffusion policy.** Same reason; also harder to debug, and BC for Eval 1 should not need it.
- **No top-camera input.** The wrist camera is the only one available at real eval; training on top would be inadvertently using privileged info. (We could use top as auxiliary loss, but not in v1.)
- **No state-only model.** The block position is unknown without vision, so a state-only baseline is uninformative for this task.
- **No DAgger / on-policy correction.** That requires an expert query loop we don't have. If vanilla BC stalls, the next step would be sim-state-based scripted-expert DAgger, but we cross that bridge later.
- **No real-camera deployment in v1.** We aim at Isaac sim eval first because that's where the existing infrastructure (env, sim_renders, success detector) lives.

## 10. Risk / fallback table

| Risk | Symptom | Fallback |
|------|---------|----------|
| Sim-rendered wrist images don't actually match the sim env at eval time | Closed-loop policy "stares" at wrong region | Re-render `sim_renders` from the *current* env config before training; verify by overlaying a sample sim frame with a sim_render frame |
| Policy ignores `bowl_xyz` (regresses to average) | Same place-location regardless of bowl | Add an auxiliary loss: predict bowl_xyz from `(image, proprio)` — sanity that the goal channel is being used; also try a larger goal-encoding MLP |
| Gripper never closes / never opens | Block dropped or never grasped | Increase L1 weight on gripper dim 3–5× (the gripper transitions are sparse in time; class imbalance) |
| Compounding error past mid-trajectory | Reaches block then drifts | Decrease execute_k (1 → 2), retrain with stronger image aug |
| Out-of-distribution bowl positions | Held-out bowls fail systematically | Synthesize ±θ azimuthal augmentations of (image, joints, bowl) around base z-axis — task is approximately rotation-equivariant |

## 11. Glossary of files we will write (one-line each)

- `bc/config.py` — single dict of hyperparams and dataset paths.
- `bc/dataset.py` — `Eval1BCDataset(pilots: list[str], split: str, k: int)` returning `(img, proprio, bowl, action_chunk, mask)`.
- `bc/normalize.py` — `Stats.fit(loader)`, `Stats.save/load`, `Stats.normalize/denormalize`.
- `bc/model.py` — `GoalCondBCPolicy(k=8)` with `forward(img, proprio, bowl)` and `predict_chunk(img, proprio, bowl) -> denormalized actions`.
- `bc/train.py` — argparse → loop → checkpoint.
- `bc/eval_openloop.py` — load best ckpt → val set → per-joint L1 table.
- `bc/deploy_sim.py` — minimal closed-loop loop in Isaac sim with the action-conversion shim of §5.2.
- `bc/README.md` — exact commands to reproduce.

---

End of plan. Next turn: implement step 1 (`bc/dataset.py` + smoke test) and verify, then move to step 2.
