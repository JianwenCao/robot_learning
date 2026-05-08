# Eval 1 — Single-Object Pick-and-Place: Sim-to-Real RL Plan

> Goal-conditioned PPO on SO-ARM101 in Isaac Lab → zero-shot deploy on the real arm. No teleop data; the only "data" is what the policy generates inside sim.

Status: state-only MDP sanity check (Day 3) is done. **Now pivoting to vision.** This doc is the working spec for the rest of the work.

---

## 1. TL;DR

A single PPO actor over thousands of randomized envs. Actor (deployable): wrist RGB + proprio + bowl `(x, y)` in robot base frame + projected ee_xy → 6-D action (5 arm joint offsets around home + 1 binary gripper). Critic (sim-only): same + privileged block pose / distances. Heavy DR transfers it to the real arm. Bowl is **not modeled** — it is a 2-D goal coord, success judged geometrically. Same policy runs at 50 Hz on hardware via Feetech `goal_position`. Eval-time bowl input is already in the robot base frame, so no coordinate conversion is needed between sim and real.

---

## 2. Eval 1 spec

| Item | Spec | Implication |
|---|---|---|
| Object | 1 wooden block, 2×2×2 cm, color free | Aggressive block-color DR; size jitter ±5–10% |
| Target | Bowl Ø 15.5 cm, h ≈ 5 cm, pose given as `(x, y)` in **robot base frame** per rollout | Goal-condition on `bowl_xy`; no perception of bowl. Frame matches sim — no coordinate transform at deploy |
| Block init | Random | Heavy initial-state DR |
| Observation | Wrist-cam RGB | CNN encoder; bowl comes from CLI arg, not pixels |
| Action | Pick → place → **release** | Reward must include release |
| Success | Block in bowl AND released | Geometric check, §3.3 |
| Method | BC or RL | Pure RL (no teleop) |
| Eval table | `#B8ADA9` gray, 5 rollouts | Match in sim; randomize around it |

Two non-obvious points: (i) the bowl is **given**, not perceived — the camera only needs to find the *block*; (ii) "easy to modify target" at eval = the network must be conditioned on `bowl_xy`, not have it baked in.

---

## 3. Sim env design — `Isaac-SO-ARM101-PickPlace-Bowl-v0`

### 3.1 Scene

| Prim | Type | Notes |
|---|---|---|
| Table | thin cuboid, kinematic, `z=0` top | `#B8ADA9` ± 15 RGB DR; friction DR |
| SO-ARM101 | `SO_ARM101_CFG` at `{ENV_REGEX_NS}/Robot` | Joint order: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`. Home: `wrist_flex=1.57` (gripper down), `gripper=0` (closed) |
| Block | `RigidObjectCfg` + `CuboidCfg(size=(0.02,0.02,0.02))`, ~4.8 g | Per-env color, mass, friction DR |
| Bowl | **No prim** | `UniformPoseCommandCfg("bowl_pose")`, sampled per reset |
| Wrist cam | `TiledCameraCfg` parented to `gripper_link` | Real intrinsics from `camera_intrinsics.yaml` (§3.5) |
| `ee_frame` | `FrameTransformer(target=gripper_link, offset=[0.01,0,-0.09])` | Reuse from lift task |

Workspace (robot frame, m): `bowl_xy ∈ x[0.10, 0.30] × y[-0.15, 0.15]`. `block_xy` same box, with `‖block − bowl‖ ≥ 0.10`. Narrow after measuring real reachable workspace.

### 3.2 Action — same as `SoArm101LiftCubeEnvCfg`

```python
arm = JointPositionActionCfg(joints=[shoulder_pan, …, wrist_roll],
                             scale=0.5, use_default_offset=True)   # absolute around home
gripper = BinaryJointPositionActionCfg(joints=["gripper"],
                                       open_command_expr={"gripper": 1.5},
                                       close_command_expr={"gripper": 0.0})
```

`scale=0.5` covers the workspace from home; bump to `1.0` only if reach fails. Gripper range `[-0.17, 1.75]` rad — open=1.5 gives extra clearance for the 2 cm block. Action vector: 5 continuous in `[-1,1]` + 1 binary-thresholded at 0.

### 3.3 Bowl as goal — geometric success

```
in_bowl  = ‖block_xy − bowl_xy‖ < 0.06  AND  block_z < 0.06  AND  steady k=5 frames
released = gripper_cmd==OPEN AND no block↔gripper contact
success  = in_bowl AND released
```

`r_safe = 0.06 m` is well inside the 0.0775 m bowl radius. We don't need bowl-rim dynamics in sim because the real bowl (15.5 cm, shallow) catches anything released near `bowl_xy` at table height.

### 3.4 Observations

**Policy (deployable):**

| Field | Dim | Sim source | Real source |
|---|---|---|---|
| `wrist_rgb` | 3×72×128 (single frame; 3-stack deferred) | TiledCamera output, undistort+resize+normalize | USB cam, **same** undistort+resize+normalize |
| `joint_pos` | 6 | `mdp.joint_pos_rel` | Feetech `present_position` − home |
| `joint_vel` | 6 | `mdp.joint_vel_rel` | Feetech `present_velocity` |
| `last_action` | 6 | `mdp.last_action` | rolled buffer |
| `bowl_xy` | 2 | `command_manager.get_command("bowl_pose")[:,:2]` | `--bowl_xy x y` |
| `ee_proj_xy` | 2 | `(ee_w − base_w)[:, :2]` from `ee_frame` | URDF FK on `qpos` |
| `ee_to_bowl_xy` | 2 | `bowl_xy − ee_proj_xy` | same |

`ee_proj_xy` is a deliberate inductive bias — gives the policy a 2-D Cartesian "where is my gripper" feature so the CNN specializes in *block* localization. 3-frame stack at 50 Hz gives parallax (free monocular-depth signal during descent).

**Critic (sim-only, asymmetric):** policy obs + `block_xyz` + `block_quat` + `gripper_xyz` + `block_to_bowl_xy` + `gripper_to_block` + `is_grasped` (approximated from `gripper_cmd==CLOSE` AND `block_z>thresh` until contact sensors enabled in `SO_ARM101_CFG`).

Wired via `obs_groups={"actor":["policy"], "critic":["policy","critic"]}` in RSL-RL 3.0.

### 3.5 Wrist camera — real intrinsics baked in

`camera_intrinsics.yaml` is the wrist cam:

```
fx=509.236, fy=508.176,  cx=656.214, cy=356.472
W=1280, H=720
distortion = [0.063, -0.113, -0.0024, 0.00127, 0.0356]
```

Convert to Isaac `CameraCfg`:

- `horizontal_aperture` (free choice; pick 20.955 mm — Isaac default)
- `focal_length = fx * horizontal_aperture / W = 509.236 * 20.955 / 1280 ≈ 8.336 mm`
- `vertical_aperture = horizontal_aperture * H / W = 11.787 mm` (verify `fy*va/H ≈ focal_length`)
- Render the wrist `TiledCamera` at **128×72** (1/10 of native 1280×720 — exact 16:9 aspect ratio, no letterbox needed). Real frames undistort → resize to 128×72 to match.
- Isaac is a perfect pinhole (no distortion). Real-side `preprocess()` runs `cv2.undistort` first so both inputs match.
- Principal point: `(cx,cy) ≈ (656, 356)` — slightly off-center vs ideal `(640, 360)`. Negligible (≤ 4 px); leave Isaac centered.
- Extrinsic on `gripper_link`: ported verbatim from LeIsaac's `single_arm_env_cfg.py` — `pos=(-0.001, 0.1, -0.04), rot=(-0.404379, -0.912179, -0.0451242, 0.0486914)`, `convention="ros"`. The bracket sits ~10 cm out in gripper +Y (the side away from the moving jaw) and tilts the lens ~48° down-and-toward-fingers so the table and gripper tips are both in frame at home pose. Earlier on-axis placements at `(0, 0, ≤0.025)` were buried in the wrist-roll coupling / sts3215 motor body and saw nothing of the table; the real WOWROBO mount uses the same side-bracket pattern for the same reason. Verified visually via `scripts/probe_wrist_cam.py` (output at `outputs/wrist_cam_probe.png`). Replace with caliper measurement of the real WOWROBO mount on day 6 and update the deploy-side FK chain in lockstep. Per-reset DR is ±25 mm / ±2.5° (see `randomize_wrist_cam_pose` in `pickplace_env_cfg.py`).

Add a small `load_intrinsics(yaml_path)` helper in `tasks/pickplace/mdp/observations.py` that emits the `CameraCfg` kwargs — single source of truth for sim and the deploy preprocess.

### 3.6 Reward shaping (`mdp/rewards.py`)

```
r_reach   = (1 - is_grasped) * (1 - tanh(d_ee_block / 0.05))         # w=1.0
r_grasp   = is_grasped_now AND not_before                            # w=5.0  (edge)
r_transp  = object_goal_distance(std=0.20, minimal_height=0.025,
                                 command="bowl_pose")                # w=2.0
r_place   = (‖block_xy − bowl_xy‖ < 0.06) AND (block_z < 0.06)       # w=5.0
r_release = r_place AND gripper_open AND ‖block_vel‖ small           # w=10.0
p_action  = -1e-4 * action_rate_l2
p_jvel    = -1e-4 * joint_vel_l2
p_drop    = -2.0  if was_grasped AND block_z<0.005 AND not in_bowl
p_offtab  = -2.0  if block_xy outside workspace
```

Two pre-empts: `r_transp` must keep `minimal_height ≥ 0.025` so it can't fire pre-grasp; `r_release` requires `r_place` first or the policy opens everywhere. Inherit the lift-task curriculum that decays `action_rate`/`joint_vel` weights `1e-4 → 1e-1` over 10k steps.

### 3.7 Terminations

- `success` — `r_release` condition (terminate with success).
- `time_out` — 8.0 s = 400 steps @ 50 Hz (longer than lift's 5.0 s to allow search + release).
- `block_off_table` — `mdp.root_height_below_minimum(-0.05, "object")`.
- `joint_limit_violation` — penalize, don't terminate.

### 3.8 Domain randomization (`mdp/events.py`)

| Category | Knob | Range |
|---|---|---|
| Visual | block color | uniform RGB, biased away from table ±25 |
| | table color | `#B8ADA9` ± 15 RGB |
| | PBR | roughness [0.4, 0.9], metallic [0, 0.1] |
| | lights | 1–3, 500–2000 lux, full hemisphere |
| | cam intrinsics | fx,fy ±5%; cx,cy ±2 px |
| | cam extrinsic | ±25 mm, ±2.5° on `gripper_link` (matches LeIsaac `randomize_camera_uniform` envelope; `mdp.events.randomize_camera_uniform`) |
| | image noise | Gauss σ ∈ [0, 5/255], blur 0–3 px, JPEG q ∈ [70, 100] |
| | distractors | 0–3 random boxes outside workspace |
| Physical | block size | ±5–10% per axis |
| | block mass | ±50% of 4.8 g |
| | friction (table/block/pads) | μ ∈ [0.4, 1.2] |
| | bowl_xy / block_xy | full workspace (§3.1) |
| | robot base | ±2 mm, ±1° |
| Dynamics | servo PD gains | ±30% around `SO_ARM101_CFG.actuators` |
| | action latency | 1–5 sim steps (20–100 ms) |
| | obs latency | 1–3 sim steps |
| | action noise | Gauss σ=0.01 |
| Initial | joint config | small jitter around home (tighter than reach task; keep wrist down) |

If only one is shipped well, **make it visual block randomization.** Wrist-cam policies most often fail to transfer there.

### 3.9 Network

```
wrist_rgb (3×72×128, single frame) ─CNN──┐
joint_pos, joint_vel, last_action  ─┤
bowl_xy, ee_proj_xy, ee_to_bowl_xy ─┼──concat──GRU(128)──MLP[256,128,64]──μ,σ──action(6)
                                     │
                            critic  ─┴──+ block pose, distances, contact ──V(s)
```

CNN: `Conv(8/4)→Conv(4/2)→Conv(3/1)→FC(128)→LN→ELU` (matches the upstream MLP activation choice). **GRU** (or LSTM) on the post-CNN features — required for the search behavior in §4. Drop-in via RSL-RL 3.0's recurrent ActorCritic.

**Aux loss:** small head on the visual latent regressing `block_xy` (privileged in sim, MSE ≈ 0.1× policy loss). Forces the encoder to localize the block instead of summarizing pixels. Drop at deploy. **Mask the loss when `block_xy` is outside the wrist FOV** so the encoder doesn't try to hallucinate off-screen positions.

### 3.10 PPO config

`rsl_rl` 3.0.1, fork of `LiftCubePPORunnerCfg`:

```python
num_steps_per_env  = 24
max_iterations     = 5000     # 5–8× lift; vision is slower
save_interval      = 100
init_noise_std     = 1.0
actor_hidden       = [256, 128, 64]      # behind GRU
critic_hidden      = [256, 128, 64]
clip_param         = 0.2
entropy_coef       = 0.006
num_learning_epochs= 5
num_mini_batches   = 4
lr                 = 1e-4 (adaptive, KL=0.01)
gamma=0.98, lam=0.95, max_grad_norm=1.0
```

Scale: 2048 envs (drop to 1024 if cameras blow VRAM on the 5090). Budget 100–300 M env steps, ~8–24 h on the 5090.

---

## 4. Block out of camera sight — RL handles this

Since `bowl_xy` is given but block position is not, the policy may start with the block outside the wrist frame. Yes, RL can learn to **search** — but only with the right ingredients:

1. **Wide FOV gets us most of the way.** Real cam horizontal FOV ≈ `2·atan(640/509) ≈ 102°`. At home pose with the gripper ~0.20 m above the table, that's a ~0.5 m patch — the entire `[0.10,0.30] × [-0.15,0.15]` workspace is visible. So in the steady state, "block out of frame" mostly happens *during* motion, not at rollout start. Plan §3.5 already preserves this 16:9 aspect ratio end-to-end, so don't square-crop the real input — you'd lose ~40° of horizontal FOV.

2. **Asymmetric critic carries the gradient.** The critic sees `block_xyz` regardless. So `r_reach = 1 - tanh(d/0.05)` shapes V(s) even when the actor sees nothing useful in pixels. The actor learns "scan toward where V is higher" via the advantage signal, without the encoder ever localizing the block in that frame. This is the single biggest reason asymmetric A-C is the right choice here.

3. **Recurrent policy (GRU) for memory.** A 3-frame stack at 50 Hz is 60 ms — too short to remember "I scanned left, didn't see it, try right." Add a GRU on the post-CNN features (RSL-RL 3.0 supports this directly). Hidden state = persistent search state.

4. **Visibility curriculum.** Early training: spawn block in a small patch under the home view (always visible) — encoder learns to localize first. Mid training: full workspace. Late training: block can spawn at workspace edge (forces search). Implement via `mdp.modify_event_term(num_steps=…)` shrinking → expanding the block-pose ranges.

5. **Episode budget.** Bumped to 8.0 s (400 steps @ 50 Hz) — leaves headroom for a 1–2 s scan before grasping.

6. **Aux block-xy head must mask off-frame samples.** Otherwise the head is pushed to predict random off-screen positions and corrupts the encoder.

7. **Optional scan-home.** If after items 1–6 search still fails, tilt the home pose (`wrist_flex=1.3` instead of `1.57`) so the cam sees a wider patch at rollout start. Keep `scale=0.5` — the tilt is cheap insurance.

This is exactly the recipe used in OpenAI Solving Rubik's Cube and in active-vision papers — wide FOV + asymmetric critic + recurrence + curriculum.

---

## 5. Real-robot deployment

### 5.1 Camera

Intrinsics already calibrated → `camera_intrinsics.yaml` (this is the wrist cam). Plug the same values into the sim `CameraCfg` (§3.5). Real-side preprocess: `cv2.undistort` (using YAML distortion coeffs) → BGR→RGB → resize to 128×72 (native 16:9 — no crop / letterbox needed) → `float/255`. **Identical preprocess in sim and on robot, period.**

Measure the wrist-cam mounting offset on `gripper_link` once with calipers (the WOWROBO 32×32 module on top of the wrist-roll motor); mirror on the sim prim's `OffsetCfg`. Current sim offset is the LeIsaac verbatim port (§3.5) — replace with the caliper measurement on day 6 and update the deploy-side FK chain in lockstep. Disable USB auto-focus / auto-exposure with `v4l2-ctl` before each run.

### 5.2 Servo I/O (`so101_io.py`)

| Joint | URDF limit (rad) | Notes |
|---|---|---|
| shoulder_pan | ±1.91986 | |
| shoulder_lift | ±1.74533 | |
| elbow_flex | ±1.69 | |
| wrist_flex | ±1.65806 | home `1.57` |
| wrist_roll | [-2.74, 2.84] | asymmetric |
| gripper | [-0.17, 1.75] | **asymmetric**, 0=closed, larger=open |

Things to pin down once and unit-test: servo-ID → URDF joint-index map; counts↔rad scaling (Feetech default 4096/360°); per-joint sign vs URDF; servo home offset (sim `q=0` ≠ servo zero counts); soft limits (clip every command before write). This layer is the last line of defense against the policy commanding into the floor.

### 5.3 Control loop

```python
policy = torch.jit.load("policy.pt").eval().cuda()                # exported by play
cam, bus = WristCam(...), FeetechBus(...)
home_q = [0,0,0,1.57,0,0]; ACTION_SCALE = 0.5
hidden = None                                                     # GRU state
img_buf = deque(maxlen=3)
bus.go_home(); rate = Rate(50)

while not stop:
    img_buf.append(preprocess(cam.read()))
    qpos, qvel = bus.read_pos_rad(), bus.read_vel_rad()
    obs = pack(wrist_rgb=stack(img_buf), joint_pos=qpos-home_q, joint_vel=qvel,
               bowl_xy=args.bowl_xy, ee_proj_xy=fk_proj(qpos),
               ee_to_bowl_xy=args.bowl_xy-fk_proj(qpos), last_action=last_a)
    action, hidden = policy(obs, hidden)                          # recurrent
    arm_cmd     = home_q[:5] + ACTION_SCALE*action[:5]            # absolute around home
    gripper_cmd = 1.5 if action[5] > 0 else 0.0
    bus.write_pos_rad(clip(cat(arm_cmd, gripper_cmd), q_min, q_max))
    last_a = action
    rate.sleep()
```

**Five things that must match sim exactly:** control rate (50 Hz), action semantics (absolute-around-home, scale 0.5, binary gripper, gripper open=0.5 / close=0.0), joint order/signs (URDF order), image preprocess (undistort + 16:9 + 128×72 + [0,1] float, single frame), `fk_proj` (URDF FK + same `[0.01, 0, -0.09]` `gripper_link` offset, drop z). A mismatch on any one of these silently breaks transfer.

### 5.4 Safety

- Esc-key user_stop → loop exit → bus go_home.
- Workspace box check on every commanded `target_q` (FK to ee_xyz; reject if outside `x[0,0.35]×y[-0.20,0.20]×z[0,0.30]`).
- Per-step joint-velocity cap in IO (Feetech STS3215 ~1.5 rad/s — matches `velocity_limit_sim` in `SO_ARM101_CFG`).
- First runs: gripper open, hand on power switch.

### 5.5 Pre-flight diagnostic

1. Hold real arm at known pose → dump `qpos` from bus, compare to sim with same target → ≤ 1° agreement.
2. Render sim wrist-cam frame at that pose; compare visually to real frame (FOV, object scale, perspective). If off, fix intrinsics/extrinsics.
3. Sweep gripper through a small XY square; overlay `ee_proj_xy` from bus FK vs sim — should agree within 5 mm.

This catches 90% of "worked in sim, didn't on robot" failures before you burn rollouts.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Block out of frame at start | Wide FOV (102°), GRU memory, visibility curriculum, asymmetric critic shapes V (§4) |
| Monocular depth ambiguity at grasp | 3-frame stack (parallax during descent); aux block-xy head |
| 2 cm block ungraspable | Verify finger geometry vs cube; lift task grasps a 2.5 cm cube — should be fine. Print finger pads if not |
| Action latency mismatch | Step-response test on real, add measured latency + margin to sim DR |
| Policy releases too high | Add soft penalty on gripper z at release moment |
| Bowl pose outside training distribution | Sample `bowl_xy` over full reachable workspace, not "expected" region |
| Vision encoder collapse | Aux block-xy head from start; fallback to frozen R3M / DINOv2-small |
| Cam intrinsics drift | Disable USB auto-focus/exposure via `v4l2-ctl`; recalibrate if lens is bumped |
| Servo overheat | Cooldown sleeps between rollouts; monitor temperature register |
| Contact sensors disabled in `SO_ARM101_CFG` | Approximate `is_grasped` from `gripper_cmd==CLOSE` AND `block_z > thresh`; flip flag back on once Isaac Lab supports it on capsule-replaced links |

---

## 7. Execution plan — what's left

| Day | Goal | Deliverable |
|---|---|---|
| ~~1–3~~ | ~~Scaffold, register, state-only PPO~~ | **Done** — MDP solves with privileged block pose |
| 4a (in progress) | Wire `TiledCameraCfg` with real intrinsics; add `wrist_rgb` obs group; custom `PickPlaceVisionActorCritic` (CNN encoder + asymmetric critic) | **Done** — `wrist_rgb` is `(N, 3, 72, 128)`, smoke test green |
| 4b | Long vision PPO run; basic DR (visual block color, lighting); no GRU / no aux head yet | First overnight run, ≥ partial reach success |
| 5 | Iterate DR ranges from per-stage success curves; add aux block-xy head if encoder collapses; add GRU + visibility curriculum if search fails; tune entropy coef if exploration stalls | ≥ 60% success in sim play, checkpoint saved |
| 6 | Measure wrist-cam extrinsic; write `so101_io.py`; run §5.5 diagnostic | Sim and real frames overlay; `ee_proj_xy` agrees ≤ 5 mm |
| 7 | First real-robot rollouts; widen whichever DR axis was too narrow; retrain delta if needed | Non-zero successes on hardware |
| 8 | Tune; record 5 evaluation rollouts + video | Submission-ready policy + video |

If a stage stalls, instrument first (per-stage success in TB), don't blindly scale DR.

---

## 8. Open decisions (things to settle as we hit them)

1. Render resolution for sim — 1280×720 native then downsample (preferred, matches real) vs. lower-res render for speed.
2. CNN end-to-end vs frozen R3M / DINOv2-small — start end-to-end; switch if encoder collapses.
3. Visibility-curriculum schedule — when to widen `block_xy` range. Start: full at iter 0; only narrow if reach stalls.
4. Recurrent unit — GRU(128) by default; LSTM if GRU underfits.
5. `scan-home` tilt — only if §4 items 1–6 don't produce search behavior.
6. Continuous vs binary gripper — stay binary unless thresholding becomes a bottleneck.

---

## 9. References

- Project doc: `Project 3_ Reinforcement Learning – Final Details.pdf`
- Daily run instructions: `RUNNING.md`
- Upstream task we forked: `isaac_so_arm101/src/isaac_so_arm101/tasks/lift/`
- Robot config: `isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/so_arm101.py`
- URDF (joint limits, link names): `isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf`
- Wrist-cam intrinsics: `camera_intrinsics.yaml`
- RSL-RL 3.0 asymmetric / recurrent A-C: `rsl_rl.modules.ActorCritic` (`obs_groups`, `rnn`)
