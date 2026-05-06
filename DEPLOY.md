# Eval 1 — Deployment Guide (Sim → Real SO-ARM101)

> Companion to [`EVAL1_PLAN.md`](./EVAL1_PLAN.md). After a PPO checkpoint converges in sim, this is how we run it on the physical SO-ARM101.

## TL;DR

`play` exports `policy.pt` (TorchScript) under `logs/rsl_rl/pickplace_bowl/<run>/exported/`. Copy it plus a small `deploy_meta.json` and `wrist_intrinsics.yaml` to the deploy host, install `torch + opencv + feetech-servo-sdk + pinocchio`, and run a 50 Hz loop that builds the same obs the policy saw in sim and writes joint targets to the Feetech bus. Bowl `(x, y)` comes from `--bowl_xy`. Isaac Lab is **not** needed at deploy time.

## 1. Prerequisites

Don't start until: vision checkpoint ≥ 60 % success in sim play, `play` produced `exported/policy.pt`, wrist cam calibrated, pre-flight diagnostic (plan §5.5) shows sim ↔ real `ee_proj_xy` agree within ~5 mm, and `feetech-servo-sdk` can read servo positions on `/dev/ttyUSB0`.

## 2. Hardware

SO-ARM101 with Feetech STS3215 servos (URDF order: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`); USB wrist cam mounted on `gripper_link`, calibrated at the actual capture resolution (save as `wrist_intrinsics.yaml`); a 2 cm wooden block; a Ø 15.5 cm bowl with a known `(x, y)` in the robot base frame; a power switch within reach.

## 3. Export the policy

```bash
conda activate so_arm
export OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y
play --task Isaac-SO-ARM101-PickPlace-Bowl-Play-v0 \
     --num_envs 16 --enable_cameras \
     --checkpoint isaac_so_arm101/logs/rsl_rl/pickplace_bowl/<run>/model_*.pt
```

Then write `deploy_meta.json` next to `exported/policy.pt`. This file pins the values that must match sim — there's no other way to guarantee runtime numerics line up:

```json
{
  "joint_order":  ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"],
  "home_q":       [0.0, 0.0, 0.0, 1.5708, 0.0, 0.0],
  "arm_action_scale":  0.5,
  "gripper_open_cmd":  1.5,
  "gripper_close_cmd": 0.0,
  "control_dt":   0.02,
  "image_size":   [96, 96],
  "frame_stack":  3,
  "joint_lower":  [-1.91986,-1.74533,-1.69,-1.65806,-2.74385,-0.17453],
  "joint_upper":  [ 1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533],
  "ee_frame_offset_in_gripper_link": [0.01, 0.0, -0.09]
}
```

Verify the exported `.pt` reproduces the training graph: feed a saved sim obs in, compare action — must match to ~1e-6.

## 4. Deploy host setup

Same Ubuntu training box (simplest): `pip install feetech-servo-sdk pyserial opencv-python pin` into the `so_arm` env. Or a clean laptop venv with `torch numpy opencv-python feetech-servo-sdk pyserial urdfpy pyyaml` — no CUDA / Isaac needed. On Linux: `usermod -a -G dialout,video $USER` for serial + camera permissions.

## 5. IO layer (`scripts/deploy/so101_io.py`)

Wraps the Feetech bus and FK so the rest of the deploy code looks exactly like sim. Three things to **unit-test before plugging into the policy**:

1. **Servo signs and zero offsets** — manually pose the arm at `q = (0,0,0,1.5708,0,0)`, capture raw counts, set those as `SERVO_ZERO_RAW`. Push each joint individually and confirm sign matches URDF (positive `shoulder_pan` rotates base CCW from above, etc.).
2. **`fk_proj_xy`** — at home should return ~`(0.10, 0.00)`. Sweep through a known XY square and confirm.
3. **Joint clipping** — `write_pos_rad` must `np.clip(q, JOINT_LOWER, JOINT_UPPER)` before every bus write. This is the last barrier between the policy and the table.

FK must mirror sim: `pinocchio.forwardKinematics` on `so_arm101.urdf`, then `gripper_link` translation + rotation @ `[0.01, 0, -0.09]`, take XY.

## 6. Control loop (`scripts/deploy/run_pickplace.py`)

50 Hz fixed-rate loop (skeleton — see plan §5.3 for the long form):

```python
bus.go_home(); time.sleep(1.5)
img_buf = deque(maxlen=meta["frame_stack"])
last_action = np.zeros(6, dtype=np.float32)
next_t = time.perf_counter()

while not STOP:
    img  = preprocess(cap.read()[1])              # undistort, BGR→RGB, resize 96², /255
    qpos = bus.read_pos_rad();  qvel = bus.read_vel_rad()
    img_buf.append(img)
    while len(img_buf) < meta["frame_stack"]: img_buf.append(img)

    ee_xy      = fk_proj_xy(qpos)
    obs = build_obs(img_buf, qpos - home_q, qvel, last_action,
                    bowl_xy, ee_xy, bowl_xy - ee_xy)

    with torch.inference_mode():
        action = policy(obs).squeeze(0).cpu().numpy()

    arm_q   = home_q[:5] + arm_scale * action[:5]
    grip_q  = g_open if action[5] > 0 else g_close
    target  = np.clip(np.r_[arm_q, grip_q], JOINT_LOWER, JOINT_UPPER)

    if not in_workspace(fk_proj_xy(target)):       # safety
        target = qpos
    bus.write_pos_rad(target);  last_action = action

    next_t += 0.02
    sleep = next_t - time.perf_counter()
    if sleep > 0: time.sleep(sleep)
    else:         next_t = time.perf_counter()    # missed deadline; resync

bus.go_home(); bus.torque_off(); cap.release()
```

The exact `policy(obs)` signature (dict vs flat tensor) depends on what the RSL-RL exporter wrote — inspect `torch.jit.load(...).code` once and pack accordingly. Disable camera autofocus and auto-exposure (`cap.set(CAP_PROP_AUTOFOCUS, 0)`) so the runtime image stats match calibration.

Workspace box (robot base frame, m): `x ∈ [0.00, 0.35]`, `y ∈ [-0.20, 0.20]`. Reject any commanded ee-xy outside it.

## 7. Pre-flight (run before every session)

(1) Home pose round-trip: `bus.go_home()` then `read_pos_rad()` should be `(0,0,0,1.5708,0,0) ± 0.02`. (2) `fk_proj_xy(home) ≈ (0.10, 0.00)`. (3) Capture one wrist frame, eyeball against a sim render at the same pose — FOV, table color, block size should match. (4) Step each joint ±0.05 rad, confirm URDF-positive direction. (5) 10 s torque-off dry run of the policy loop, confirm measured period ≤ 25 ms (camera read or FK is the usual bottleneck).

## 8. Running an Eval 1 rollout

```bash
python scripts/deploy/run_pickplace.py \
    --policy   exported/policy.pt \
    --meta     exported/deploy_meta.json \
    --cam_intr exported/wrist_intrinsics.yaml \
    --bus_port /dev/ttyUSB0 --cam_index 0 \
    --bowl_xy 0.20 -0.05
```

Place the bowl with its center at the commanded `(x, y)`, place the block somewhere in the workspace, hand on the power switch, run. Log `qpos`, `action`, wrist frames, and success flag into `runs/<timestamp>/` — re-watching the wrist video is the fastest debug tool.

## 9. Triage

| Symptom | Cause | Fix |
|---|---|---|
| Arm reaches past the block | Action scale too big or a servo sign flipped | Re-verify §5 sign test; try `arm_action_scale=0.3` |
| Block kicked aside, not grasped | Friction / mass mismatch | Widen friction DR, retrain |
| Gripper closes early / late | Binary-threshold drift on action[5] | Add gripper-stage gating in reward, retrain |
| Releases above the bowl | No z-penalty at release | Add z-penalty term (plan §3.5), retrain |
| Loop < 50 Hz | Blocking camera read or slow FK | Thread the camera; use `pinocchio` (C++) FK |
| Servos overheat mid-session | Long high-torque holds | 30 s cooldown between rollouts; check temp register |

If it's a sim-to-real distribution gap (visual or dynamics), the answer is **widen the corresponding DR axis and retrain** — not patch the deploy script.

## 10. Values that MUST match sim

| Quantity | Value |
|---|---|
| Control rate | 50 Hz (`dt = 0.02 s`) |
| Home `q` | `(0, 0, 0, 1.5708, 0, 0)` |
| Action semantics | `target = home + 0.5 · action[:5]` (absolute-around-home, not delta) |
| Gripper | `1.5` open / `0.0` close, threshold `action[5] > 0` |
| Joint order | `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` |
| Wrist image | 96 × 96 RGB, `float/255`, 3-frame stack |
| `ee_proj_xy` | FK on `gripper_link` + offset `[0.01, 0, -0.09]`, take XY |
| Bowl input | `(x, y)` in robot base frame, via `--bowl_xy` |

When in doubt: save one `obs` from sim `play` and one from the deploy loop at the same arm pose, `np.allclose` every field. Anything off by > 1 % is a bug.

## 11. Submission

5 rollout videos (wrist + external phone), `exported/{policy.pt, deploy_meta.json, wrist_intrinsics.yaml}`, `scripts/deploy/{so101_io.py, run_pickplace.py, preflight.py}`, a short `RESULTS.md` (bowl `(x,y)` + success/fail per rollout), and the TensorBoard run dir.

## References

[`EVAL1_PLAN.md`](./EVAL1_PLAN.md) §3 (env) §5 (real robot) §6 (risks); `RUNNING.md`; robot cfg `robots/trs_so101/so_arm101.py`; URDF `robots/trs_so101/urdf/so_arm101.urdf`.
