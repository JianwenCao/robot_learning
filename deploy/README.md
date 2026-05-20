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

PNGs are vendored at `deploy/apriltags/tag41_12_*.png` — no download needed. Family is **`tagStandard41h12`** (5×5 bits — fits a 2 cm cube face; `tag36h11` is too dense at this size). See [`docs/STATE_APRILTAG_PLAN.md`](../docs/STATE_APRILTAG_PLAN.md) for the full plan.

| ID | Use | Size | Where it goes | Required for |
|---|---|---|---|---|
| `0` | Cube — red | 15 mm | Top face, red cube | Eval-1, Eval-2/3 |
| `1`–`5` | Cubes — blue, yellow, green, purple, orange | 15 mm | Top face, matching cube | Eval-2/3 |
| `99` | **Calibration** | 30 mm | Flat on table during hand-eye calibration only | All evals |

**For Eval-1 only**, print just `0` and `99` (two pieces total). For new object classes use IDs `≥ 10` and add a row above in the same PR; keep `0..9` reserved for cube colours and `99` reserved for calibration.

**How to print**:

1. Open `deploy/apriltags/tag41_12_00000.png` → set print size to **15 × 15 mm** (ID 99 → **30 × 30 mm**). If your print dialog won't accept mm, at 600 DPI a 15 mm tag is `15 / 25.4 × 600 ≈ 354 px`.
2. **Matte** paper at **600 DPI** or higher. Glossy paper → specular highlights kill detection under room lighting.
3. Cut keeping the **white border intact** — the detector treats the white quiet zone as part of the pattern. Never cut into the black border.
4. Laminate flat with **matte** clear film (optional for first tests, required for durability — never glossy).
5. Stick to the cube's top face with double-sided tape **under** the laminate.

Tip: print all required IDs on a single A4 sheet — they're tiny. Make 2-3 spares in case cutting goes wrong.

**Verify a print** before mounting. Hold it flat under the wrist cam and run:

```bash
conda activate so_arm
python - <<'PY'
import cv2
from pupil_apriltags import Detector
det = Detector(families="tagStandard41h12")
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print("Detected IDs:", [t.tag_id for t in det.detect(gray)])
PY
```

Expected IDs missing? Common causes: smudged black border (reprint), tag too small relative to camera distance (reprint at 30 mm or move closer), glossy paper (reprint matte).

## Step 4 — hand-eye calibration

Computes `T_ee_cam` (rigid transform from end-effector frame → wrist-cam frame) and writes it to `deploy/hand_eye.yaml`. **Load-bearing**: a 1 cm error here shifts every tag pose by 1 cm in base frame → policy can't grasp. Re-run any time the camera mount is bumped, removed, or reseated.

Procedure (~1 hour, one-time per camera mount):

1. Place the **30 mm ID-99 tag** flat on the table, anywhere visible to the wrist cam.
2. Manually teleop the arm to **≥ 12 distinct poses** that all keep the tag in view. Vary EE position **and** orientation. At each pose, record:
   - `joint_pos` (servo read-back)
   - RGB frame → `pupil-apriltags` → `T_cam_tag` (4×4 pose)
3. Compute `T_base_ee` per sample via URDF FK (`kinpy`).
4. Solve `cv2.calibrateHandEye(R_base_ee, t_base_ee, R_cam_tag, t_cam_tag, method=CALIB_HAND_EYE_TSAI)` → `T_ee_cam`.
5. Write to `deploy/hand_eye.yaml`. (Note `camera_intrinsics.yaml` lives at the repo root, not in `deploy/` — they're loaded from different paths by `deploy/cube_detector.py`.)
6. **Verify**: command the EE to the projected tag center (project `T_base_tag` to xy, move EE there). Offset must be **≤ 5 mm** by ruler. If larger, redo step 2 with more / better-distributed poses.

`deploy/calibrate_hand_eye.py` runs steps 2–6 interactively (prompts each pose, captures the frame, solves at the end). *Script lands with the AprilTag pipeline — until then, this section is forward-looking; see [`docs/STATE_APRILTAG_PLAN.md`](../docs/STATE_APRILTAG_PLAN.md) §3 for the canonical procedure.*

## Step 5 — checkpoint

Download a known-good state-only + AprilTag PPO checkpoint to `deploy/runs/model.pt`:

```bash
pip install --quiet gdown
mkdir -p deploy/runs
gdown "https://drive.google.com/uc?id=1dUobpj8qXZc4PL7ASod_kWhum0y_xwwq" -O deploy/runs/model.pt
```

## Step 6 — real-arm rollout

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
