# Real-Robot Deploy (Eval-1 / Eval-2 / Eval-3)

State-only AprilTag PPO checkpoint (trained in `isaac_so_arm101/`) → closed-loop on a real SO-ARM101. No `rsl_rl` / `isaaclab` on this PC; forward-only actor in [`deploy/ppo_actor.py`](ppo_actor.py). Single entry point [`deploy/deploy_real.py`](deploy_real.py) covers all three evals (Eval-1: `--target-color`, Eval-2: same + clutter, Eval-3: `--colors red,blue,yellow` cycles tag IDs via `detector.set_target_id`).

## Step 1 — clone

```bash
git clone -b rui https://github.com/RuiZhou-cn/robot-learning.git project3 && cd project3
```

## Step 2 — install

```bash
bash deploy/setup_inference_pc.sh
```

## Step 3 — print AprilTag tags

PNGs are vendored at `deploy/apriltags/tag41_12_*.png`. Family is **`tagStandard41h12`** (5×5 bits — fits a 2 cm cube face; `tag36h11` is too dense at this size). The detector loads this family by default in `AprilTagDetector` ([`deploy/cube_detector.py`](cube_detector.py)); foundational spec in [`docs/EVAL1_PLAN.md`](../docs/EVAL1_PLAN.md).

| ID | Use | Size | Where it goes | Required for |
|---|---|---|---|---|
| `0` | Cube — red | 15 mm | Top face, red cube | Eval-1, Eval-2/3 |
| `1`–`5` | Cubes — blue, yellow, green, purple, orange | 15 mm | Top face, matching cube | Eval-2/3 |
| `99` | **Calibration** | 30 mm | Flat on table during hand-eye calibration only | All evals |

Colour ↔ ID mapping is hard-coded in `_COLOR_TO_ID` at the top of [`deploy/deploy_real.py`](deploy_real.py); if you re-label a cube physically, update that dict in lockstep.

**For Eval-1 only**, print just `0` and `99`. For new object classes use IDs `≥ 10`; keep `0..9` reserved for cube colours and `99` reserved for calibration.

**How to print**:

1. Open `deploy/apriltags/tag41_12_00000.png` → set print size to **15 × 15 mm** (ID 99 → **30 × 30 mm**). At 600 DPI a 15 mm tag is `15 / 25.4 × 600 ≈ 354 px`.
2. **Matte** paper at **600 DPI** or higher. Glossy paper → specular highlights kill detection under room lighting.
3. Cut keeping the **white border intact** — the detector treats the white quiet zone as part of the pattern. Never cut into the black border.
4. Laminate flat with **matte** clear film (optional for first tests, required for durability — never glossy).
5. Stick to the cube's top face with double-sided tape **under** the laminate.

**Verify a print** before mounting. Hold each tag flat under the wrist cam (~20 cm standoff) and run [`deploy/verify_apriltag_print.py`](verify_apriltag_print.py) — it prints detected IDs, `decision_margin` (≥ 30 healthy, < 15 marginal), and on-image side length in pixels (≥ 25 px for a 15 mm tag at typical standoff):

```bash
conda activate so_arm
python -m deploy.verify_apriltag_print                 # /dev/video0
python -m deploy.verify_apriltag_print --cam-index 2   # other USB cam
```

If a tag is missing or has `margin < 15`: smudged black border (reprint), tag too small relative to camera distance (reprint at 30 mm or move closer), glossy paper (reprint matte), or motion blur (hold still). Re-run until every tag reports `margin > 30` and `side > 25 px`.

For background: upstream [pupil-labs/apriltags](https://github.com/pupil-labs/apriltags) (Python binding) and [AprilRobotics/apriltag](https://github.com/AprilRobotics/apriltag) (core algorithm + `tagStandard41h12` family).

## Step 4 — hand-eye calibration

Computes `T_ee_cam` (rigid transform from end-effector frame → wrist-cam frame) and writes it to `deploy/hand_eye.yaml`. **Load-bearing**: a 1 cm error here shifts every tag pose by 1 cm in base frame → policy can't grasp. Re-run any time the camera mount is bumped, removed, or reseated.

[`deploy/calibrate_hand_eye.py`](calibrate_hand_eye.py) runs the whole procedure interactively — wrist-cam preview, SPACE-capture each sample, solves `cv2.calibrateHandEye(method=CALIB_HAND_EYE_TSAI)` from the joint-FK + tag-pose pairs, prints cross-sample tag-position std (≤ 5 mm = good), writes `deploy/hand_eye.yaml`.

Procedure (~1 hour, one-time per camera mount):

1. Place the **30 mm ID-99 tag** flat on the table, anywhere visible to the wrist cam.
2. Start the script:
   ```bash
   conda activate so_arm
   python -m deploy.calibrate_hand_eye
   ```
   Optional flags: `--tag-id 99 --tag-size 0.030 --n-poses 12`.
3. Move the arm to **≥ 12 distinct poses** keeping the tag in view. Vary EE position **and** orientation — pure translation gives a degenerate solve.
4. In the OpenCV preview window: **SPACE** = capture, **D** = drop last, **Q** = finish + solve (≥ 4 samples; 12+ recommended), **ESC** = abort.
5. After solving, the script prints `T_base_tag` std across samples. If `max std > 5 mm`, collect more / better-distributed samples and re-run.
6. **Verify physically**: command the EE to the projected tag centre, move it there with leader-arm teleop, confirm offset ≤ 5 mm by ruler. Larger → redo step 3.

`camera_intrinsics.yaml` lives at the repo root, not in `deploy/` — both files are loaded by [`deploy/cube_detector.py`](cube_detector.py) at `AprilTagDetector` construction.

## Step 5 — AprilTag end-to-end check

Before pulling a checkpoint, confirm the full detection chain (wrist cam → undistort → AprilTag → `T_base_ee · T_ee_cam · T_cam_tag`) returns a sensible cube xy in the **robot base frame**. Catches a wrong `hand_eye.yaml`, swapped intrinsics file, or misprinted tag size before you load the policy. Driver: [`deploy/verify_apriltag_chain.py`](verify_apriltag_chain.py).

Place a cube on the table where you can measure it with a ruler from the robot base, then:

```bash
conda activate so_arm
python -m deploy.verify_apriltag_chain                                # red cube (id 0), 15 mm
python -m deploy.verify_apriltag_chain --tag-id 1                     # blue cube
python -m deploy.verify_apriltag_chain --tag-id 99 --tag-size 0.030   # calibration tag
```

`valid=True` and `(x, y)` within **≤ 1 cm** of where you placed the cube → chain is healthy. Common failures:

- `valid=False` — tag not in view, or print quality too low. Re-check with Step 3.
- `valid=True` but xy off by 5–10 cm — almost always a stale `hand_eye.yaml`. Re-run Step 4.
- xy off by exactly the tag size (≈ 1.5 cm) in one axis — `--tag-size` mismatch between print and detector. Defaults: 15 mm cubes, 30 mm calibration.
- xy off by ~5 % proportional to distance — undistort skipped (missing `camera_intrinsics.yaml`).

## Step 6 — checkpoint

Per-eval checkpoint (not interchangeable). Eval-1 Drive link below; train Eval-2 yourself (Eval-3 reuses Eval-2).

| Eval | Checkpoint | Notes |
|---|---|---|
| Eval-1 | `deploy/runs/model.pt` (Drive link below) | This step |
| Eval-2 | own retrained model | `Isaac-SO-ARM101-ClutterPickPlace-Bowl-StateAprilTag-v0`; pass via `--ckpt` |
| Eval-3 | **reuses Eval-2 model** | Only the AprilTag `target_id` changes between sub-goals — see [`docs/EVAL3_PLAN.md`](../docs/EVAL3_PLAN.md) |

```bash
pip install --quiet gdown && mkdir -p deploy/runs
gdown "https://drive.google.com/uc?id=1WvYqySV75dJEwXsASdPdB8LpevbpLkox" -O deploy/runs/model.pt
```

## Step 7 — replay the checkpoint in Isaac Sim (GUI)

Sanity-check on the **training PC** before going to the real arm. Entry: [`isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py`](../isaac_so_arm101/src/isaac_so_arm101/scripts/rsl_rl/play.py).

```bash
cd isaac_so_arm101
uv run play --task Isaac-SO-ARM101-PickPlace-Bowl-StateAprilTag-Play-v0 \
    --load_run <run> --checkpoint model_<best>.pt --num_envs 1
```

`--dump-bowl-xy 0.22,0.0` mirrors `deploy_real.py --bowl-xy`. `--debug-dump` writes a `log.jsonl` matching the deploy script's format for sim ↔ real diffs.

## Step 8 — real-arm rollout

Tag ↔ colour: `0`=red, `1`=blue, `2`=yellow, `3`=green, `4`=purple, `5`=orange. Entry: [`deploy/deploy_real.py`](deploy_real.py). Default ckpt search (`runs/state_apriltag_model.pt` → `runs/model.pt`) is **Eval-1 only**; pass `--ckpt` for Eval-2/3.

```bash
conda activate so_arm

# Eval-1 (red cube; default).
python -m deploy.deploy_real --bowl-xy 0.30,0.0

# Eval-2 — own model.
python -m deploy.deploy_real --bowl-xy 0.22,0.0 --target-color blue \
    --ckpt deploy/runs/eval2_model.pt

# Eval-3 — reuses Eval-2 model. Knobs: --release-detect {manual,vision,timed},
# --no-confirm, --no-home-between-subgoals.
python -m deploy.deploy_real --bowl-xy 0.22,0.0 --colors red,blue,yellow \
    --ckpt deploy/runs/eval2_model.pt
```

## How AprilTag is wired into the closed loop

Per env step (~30 Hz, in `deploy_real.py`):

1. Read servo joints → URDF FK → `T_base_ee`.
2. If **not grasped**: capture wrist RGB → `cv2.undistort` with `camera_intrinsics.yaml` → `AprilTagDetector.pose(rgb, T_base_ee)` → returns `((x, y), valid)` in base frame, filtering to the current `target_id`. Detector composes `T_base_tag = T_base_ee · T_ee_cam · T_cam_tag` internally.
3. **Detection contract** (mirrors sim-side `mdp.observations.cube_pos_xy_noisy`):
   - `valid=True` → publish `(x, y)`, store as `last_cube_xy`.
   - `valid=False` → hold `last_cube_xy`. Policy is trained against Bernoulli dropouts, so a few invalid frames are fine.
   - **Grasp latch** — once the gripper-close command fires AND the last detection was within `GRASP_XY_TOL = 4 cm` of the EE, detection is skipped for the rest of the sub-goal and `last_cube_xy` is frozen (tag is under the gripper).
4. 27-D state vector (`_build_state_27`) appends `last_cube_xy` to the 25-D proprio/bowl/action obs. Policy is a pure MLP — see [`deploy/ppo_actor.py`](ppo_actor.py).
5. For Eval-3, between sub-goals the script calls `detector.set_target_id(new_id)` — same weights, different tag.

**Debugging at runtime**: `--debug-dump` writes per-step `log.jsonl` (`cube_xy`, `cube_valid`, `grasped`, `state`, `action`) under `deploy/runs/debug_state/<timestamp>/`. If the cube is reachable but the policy never grasps, grep `log.jsonl` for `"cube_valid": true` rows — if their `cube_xy` is wrong by > 1 cm, redo Step 5 and likely Step 4.
