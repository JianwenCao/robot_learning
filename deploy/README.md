# Real-Robot Deploy (Eval-1 / Eval-2 / Eval-3)

PPO checkpoint (trained in `isaac_so_arm101/`) → closed-loop on a real SO-ARM101. No `rsl_rl` / `isaaclab` needed on this PC — the actor is reimplemented forward-only in `deploy/ppo_actor.py` (`PPOActorState`). All evals share the **same state-only AprilTag policy**: `cube_xy` is injected into the state vector via `pupil-apriltags` reading the per-cube tag, and Eval-2/3 (once their outer loops land) just route which tag ID the detector tracks.

Single entry point — `python -m deploy.deploy_real …` covers all three evals:

* **Eval-1** (single cube into bowl) — default; `--target-color red` (or just omit; red is the default).
* **Eval-2** (single target in clutter) — same script, `--target-color blue|yellow|green|purple|orange` selects which cube to grasp. The AprilTag ID is what disambiguates the target from the distractors, so no additional flag or policy change is needed.
* **Eval-3** (three sub-goals, shared bowl) — `--colors red,blue,yellow` (comma-separated sequence) runs the same policy three times in a row, calling `detector.set_target_id(...)` between sub-goals. Knobs: `--release-detect {manual,vision,timed}` (default `manual`) and `--no-home-between-subgoals`.

## Step 1 — clone

```bash
git clone -b rui https://github.com/RuiZhou-cn/robot-learning.git project3
cd project3
```

## Step 2 — install

```bash
bash deploy/setup_inference_pc.sh
```

## Step 3 — print AprilTag tags

PNGs are vendored at `deploy/apriltags/tag41_12_*.png`. Family is **`tagStandard41h12`** (5×5 bits — fits a 2 cm cube face; `tag36h11` is too dense at this size). The detector loads this family by default in `AprilTagDetector` ([`deploy/cube_detector.py`](cube_detector.py)); the foundational pipeline spec is [`docs/EVAL1_PLAN.md`](../docs/EVAL1_PLAN.md).

| ID | Use | Size | Where it goes | Required for |
|---|---|---|---|---|
| `0` | Cube — red | 15 mm | Top face, red cube | Eval-1, Eval-2/3 |
| `1`–`5` | Cubes — blue, yellow, green, purple, orange | 15 mm | Top face, matching cube | Eval-2/3 |
| `99` | **Calibration** | 30 mm | Flat on table during hand-eye calibration only | All evals |

The colour ↔ ID mapping is hard-coded in `_COLOR_TO_ID` at the top of [`deploy/deploy_real.py`](deploy_real.py); if you re-label a cube physically, update that dict in lockstep.

**For Eval-1 only**, print just `0` and `99` (two pieces total). For new object classes use IDs `≥ 10` and add a row above in the same PR; keep `0..9` reserved for cube colours and `99` reserved for calibration.

**How to print**:

1. Open `deploy/apriltags/tag41_12_00000.png` → set print size to **15 × 15 mm** (ID 99 → **30 × 30 mm**). If your print dialog won't accept mm, at 600 DPI a 15 mm tag is `15 / 25.4 × 600 ≈ 354 px`.
2. **Matte** paper at **600 DPI** or higher. Glossy paper → specular highlights kill detection under room lighting.
3. Cut keeping the **white border intact** — the detector treats the white quiet zone as part of the pattern. Never cut into the black border.
4. Laminate flat with **matte** clear film (optional for first tests, required for durability — never glossy).
5. Stick to the cube's top face with double-sided tape **under** the laminate.

Tip: print all required IDs on a single A4 sheet — they're tiny. Make 2-3 spares in case cutting goes wrong.

**Verify a print** before mounting. Hold each tag flat under the wrist cam (~20 cm standoff) and run the bundled smoke script — it prints detected IDs, the AprilTag library's `decision_margin` (≥ 30 healthy, < 15 marginal) and the on-image tag side length in pixels (should be ≥ 25 px for a 15 mm tag at typical standoff):

```bash
conda activate so_arm
python -m deploy.verify_apriltag_print                 # /dev/video0 by default
python -m deploy.verify_apriltag_print --cam-index 2   # other USB cam
```

If a tag you expect is missing or has `margin < 15`: common causes are smudged black border (reprint), tag too small relative to camera distance (reprint at 30 mm or move closer), glossy paper (reprint matte), or motion blur (hold still). Re-run until every tag you printed reports `margin > 30` and `side > 25 px`.

For background on the detector + tag families used here, see upstream [pupil-labs/apriltags](https://github.com/pupil-labs/apriltags) (the Python binding we install) and [AprilRobotics/apriltag](https://github.com/AprilRobotics/apriltag) (the core algorithm + `tagStandard41h12` family definition).

## Step 4 — hand-eye calibration

Computes `T_ee_cam` (rigid transform from end-effector frame → wrist-cam frame) and writes it to `deploy/hand_eye.yaml`. **Load-bearing**: a 1 cm error here shifts every tag pose by 1 cm in base frame → policy can't grasp. Re-run any time the camera mount is bumped, removed, or reseated.

The script `deploy/calibrate_hand_eye.py` runs the whole procedure interactively — it spins up the wrist cam preview, lets you SPACE-capture each sample, solves `cv2.calibrateHandEye(method=CALIB_HAND_EYE_TSAI)` from the joint-FK + tag-pose pairs, prints the cross-sample tag-position std (≤ 5 mm = good) and writes `deploy/hand_eye.yaml`.

Procedure (~1 hour, one-time per camera mount):

1. Place the **30 mm ID-99 tag** flat on the table, anywhere visible to the wrist cam.
2. Start the script:
   ```bash
   conda activate so_arm
   python -m deploy.calibrate_hand_eye
   ```
   Optional flags: `--tag-id 99 --tag-size 0.030 --n-poses 12`. Defaults match the table above.
3. Move the arm to **≥ 12 distinct poses** that all keep the tag in view. Vary EE position **and** orientation — pure translation gives a degenerate solve. Use leader-arm teleop in another terminal, or any other means; the script just snapshots when you press SPACE.
4. At each pose, focus the OpenCV preview window:
   - **SPACE** — capture (records `joint_pos` → FK → `T_base_ee`, and the tag detection → PnP → `T_cam_tag`).
   - **D** — drop the last sample.
   - **Q** — finish + solve (needs ≥ 4 samples; 12+ is recommended).
   - **ESC** — abort without writing.
5. After solving, the script prints the std of `T_base_tag` across samples. If `max std > 5 mm`, it warns; collect more / better-distributed samples and re-run.
6. **Verify physically**: command the EE to the projected tag centre (use the std-of-positions mean printed by the script as the tag's true base-frame xy), move the EE there with leader-arm teleop, and confirm the offset is **≤ 5 mm** by ruler. Larger → redo step 3 with more / better-distributed poses.

The script writes `deploy/hand_eye.yaml`. `camera_intrinsics.yaml` lives at the repo root, not in `deploy/` — both are loaded by [`deploy/cube_detector.py`](cube_detector.py) at `AprilTagDetector` construction.

## Step 5 — AprilTag end-to-end check

Before pulling a checkpoint, confirm the full detection chain (wrist cam → undistort → AprilTag → `T_base_ee · T_ee_cam · T_cam_tag`) actually returns a sensible cube xy in the **robot base frame**. This catches a wrong `hand_eye.yaml`, a swapped intrinsics file, or a misprinted tag size before you ever load the policy.

Put a cube with the red tag (ID 0) somewhere on the table you can measure with a ruler from the robot base, then:

```bash
conda activate so_arm
python -m deploy.verify_apriltag_chain                          # red cube (id 0), 15 mm
python -m deploy.verify_apriltag_chain --tag-id 1               # blue cube
python -m deploy.verify_apriltag_chain --tag-id 99 --tag-size 0.030   # calibration tag
```

`valid=True` and `(x, y)` within **≤ 1 cm** of where you placed the cube → the chain is healthy. Common failures:

- `valid=False` — tag not in view, or print quality too low. Re-check with the Step 3 snippet.
- `valid=True` but xy off by 5–10 cm — almost always a stale `hand_eye.yaml`. Re-run Step 4.
- xy off by exactly the cube tag size (≈ 1.5 cm) in one axis — likely a `--tag-size` mismatch between the print and the detector. Defaults are 15 mm for cubes, 30 mm for calibration.
- xy off by ~5 % proportional to distance — undistort skipped (missing `camera_intrinsics.yaml`).

## Step 6 — checkpoint

Download a known-good state-only + AprilTag PPO checkpoint to `deploy/runs/model.pt`:

```bash
pip install --quiet gdown
mkdir -p deploy/runs
gdown "https://drive.google.com/uc?id=1WvYqySV75dJEwXsASdPdB8LpevbpLkox" -O deploy/runs/model.pt
```

## Step 7 — replay the checkpoint in Isaac Sim (GUI)

Before going to the real arm, watch the policy execute in the Isaac Sim GUI. This sanity-checks the checkpoint, the action contract, and the bowl-xy override that hardware-rollouts also use — without any physical risk. Runs on the **training PC** (not the inference PC), since it needs Isaac Lab / Isaac Sim.

```bash
cd isaac_so_arm101
uv run play \
    --task Isaac-SO-ARM101-PickPlace-Bowl-StateAprilTag-Play-v0 \
    --load_run <run-folder-under-logs/rsl_rl/pickplace_bowl_state_apriltag/> \
    --checkpoint model_<best>.pt \
    --num_envs 1
```

Notes:

* Omit `--headless` (the default omits it) — the Isaac Sim GUI window opens and you can scrub the camera to watch the cube/bowl/arm.
* `--num_envs 1` keeps the GUI uncluttered; bump to 16+ for parallel eval if you want statistics.
* `--checkpoint` takes a bare filename when paired with `--load_run`; if you skip `--load_run`, pass the **absolute** path to `--checkpoint` instead (per the auto-memory note for this repo).
* `--dump-bowl-xy 0.22,0.0` overrides env 0's bowl pose to match a fixed real-table setup — useful when you want the sim replay to mirror exactly what you'll then run on hardware (`deploy_real.py --bowl-xy 0.22,0.0`).
* `--debug-dump` (optional) writes a per-step `log.jsonl` under `logs/rsl_rl/.../play_dump/<timestamp>/` with the **same layout** as `deploy_real.py --debug-dump`, so you can diff the two folders side-by-side to confirm sim ↔ real behaviour matches.
* In sim the cube xy is the privileged ground truth (no AprilTag detector runs in-sim during play); on the real arm the same slot is filled by the detector chain. If sim looks good but real fails, the bug is almost always in Steps 3–5 above.

If the policy fails in sim (cube knocked away, no grasp, oscillation around the bowl): the checkpoint or training is the problem, not the deploy pipeline — fix that before touching hardware.

## Step 8 — real-arm rollout

Tag ↔ colour: `0`=red, `1`=blue, `2`=yellow, `3`=green, `4`=purple, `5`=orange.

```bash
conda activate so_arm

# Eval-1 (red cube; --target-color red is the default)
python -m deploy.deploy_real --bowl-xy 0.30,0.0

# Eval-2 (--target-color picks the cube colour)
python -m deploy.deploy_real --bowl-xy 0.22,0.0 --target-color blue

# Eval-3 (three sub-goals, shared bowl). Defaults: manual release-detect
# between sub-goals, homing on between sub-goals. Add --no-confirm to
# skip the manual prompts; add --no-home-between-subgoals for speed.
python -m deploy.deploy_real --bowl-xy 0.22,0.0 --colors red,blue,yellow
```

## How AprilTag is wired into the closed loop

So a user can debug runtime detection issues, here's what `deploy_real.py` does every step (~30 Hz):

1. Read servo joints → URDF FK → `T_base_ee`.
2. If **not grasped**: capture wrist RGB → `cv2.undistort` with `camera_intrinsics.yaml` → `AprilTagDetector.pose(rgb, T_base_ee)` → returns `((x, y), valid)` in base frame, filtering to the current `target_id`. The detector composes `T_base_tag = T_base_ee · T_ee_cam · T_cam_tag` internally.
3. **Detection contract** (mirrors the sim-side noise model — same as `mdp.observations.cube_pos_xy_noisy`):
   - `valid=True` → publish `(x, y)`, store as `last_cube_xy`.
   - `valid=False` (occlusion, motion blur, hand briefly in front) → hold `last_cube_xy`. The policy is trained against Bernoulli dropouts, so a few invalid frames are fine.
   - **Grasp latch** — once the gripper-close command fires AND the last detection was within `GRASP_XY_TOL = 4 cm` of the EE, detection is skipped entirely for the rest of the sub-goal and `last_cube_xy` is frozen. The tag is under the gripper at that point.
4. The 27-D state vector (`_build_state_27`) appends `last_cube_xy` to the 25-D proprio/bowl/action obs. The policy is a pure MLP — see [`deploy/ppo_actor.py`](ppo_actor.py).
5. For Eval-3, between sub-goals the script calls `detector.set_target_id(new_id)` — same weights, different tag.

**Debugging at runtime**: add `--debug-dump` to write per-step `log.jsonl` (`cube_xy`, `cube_valid`, `grasped`, `state`, `action`) under `deploy/runs/debug_state/<timestamp>/`. If the cube is reachable but the policy never grasps, grep `log.jsonl` for `"cube_valid": true` rows — if their `cube_xy` is wrong by > 1 cm, redo Step 5 and likely Step 4.

If the AprilTag pipeline degrades (cube moved, lighting changed) re-run the Step 3 snippet (print quality) and Step 5 snippet (end-to-end xy); both are non-destructive.
