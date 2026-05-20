# Eval 1 — Single-Object Pick-and-Place (State-Only + AprilTag)

Goal-conditioned PPO on SO-ARM101 → zero-shot real-arm deploy. Task: `Isaac-SO-ARM101-PickPlace-Bowl-StateAprilTag-v0`. **Single-stage, camera-free PPO** on privileged state plus a noisy `cube_pos_xy` observation. At deploy, that slot is filled by AprilTag pose estimation in the wrist cam — no vision distill, no Stage-3 vision PPO, no CNN, no Florence-2.

This doc is the foundational state-only + AprilTag plan for all evals (1, 2, 3, Bonus B). Eval-2 / Eval-3 / Bonus-B docs inherit the calibration, detector, sim noise model, and training shape from here and only describe their deltas.

The previous 3-stage vision pipeline (state teacher → vision distill → vision PPO + teacher-critic warm-start) is retired. Anything still referencing it in code or comments is legacy and should be considered stale.

---

## 1. Why state-only + AprilTag

The 3-stage vision pipeline was solving a perception problem we don't have. The wrist cam at deploy can localize a cube in the base frame at sub-cm precision using a printed AprilTag — the same signal the privileged state teacher already trains against. Once we fill the `cube_pos_xy` policy obs from AprilTag pose at deploy, the entire vision distill + Stage-3 critic warm-start chain becomes redundant. Concretely we drop: `wrist_image` obs group, CNN encoders, Florence-2 dep, DAgger distill, the `--teacher_ckpt` overlay, ~1 day of Stage-2/3 wall-clock. We keep: one state PPO at ~1–2 h wall-clock to ≥ 80 % sim success.

The single risk this trade introduces is hand-eye calibration drift — a 1 cm error in `T_ee_cam` shifts every tag pose by 1 cm in base frame, which the policy cannot recover. §3 covers calibration; §8 the verification gate.

## 2. What you print

| Item | Quantity | Family | Printed size | Notes |
|---|---|---|---|---|
| Cube tag (Eval 1) | 1 | `tagStandard41h12` | **15 mm** square, 1-bit white border | ID `0`, stuck on the top face of the cube |
| Calibration tag | 1 | same | **30 mm** (bigger = more accurate corners) | Used only during hand-eye calibration |

Print at ≥ 600 DPI on **matte** paper. Laminate **flat** with matte clear film — glossy laminate kills detection. Cut to size leaving the white border intact. Stick to the cube top face with double-sided tape under the laminate.

Why `tagStandard41h12` and not `tag36h11`: a 2 cm cube face only fits a ~15 mm tag, and at that size `tag36h11`'s 6×6 bits go below the detector's reliable resolution. `tagStandard41h12` is 5×5 bits — drops in cleanly.

ID assignments are reserved palette-wide so Eval-2 / Eval-3 / Bonus-B share the same physical cubes: `{0:red, 1:blue, 2:yellow, 3:green, 4:purple, 5:orange}`. For Eval 1 only ID 0 (red) is on the table.

## 3. Hand-eye calibration (one-time, ~1 h)

Hand-eye = `T_ee_cam` (rigid transform from end-effector frame to camera frame). **This is the load-bearing real-side step.** A 1 cm error here propagates as 1 cm error on every tag-derived `cube_pos_xy` and the policy cannot grasp.

1. Place the **30 mm calibration tag** flat on the table, anywhere visible to the wrist cam.
2. Manually teleop the arm to **≥ 12 distinct poses** that all keep the tag in view. Vary EE position *and* orientation. At each pose, record `joint_pos` (servo read-back) and a wrist RGB frame → run `pupil-apriltags` → `T_cam_tag` (4×4 pose).
3. Compute `T_base_ee` at each sample via URDF FK (`kinpy` — already a dep).
4. Solve hand-eye: `cv2.calibrateHandEye(R_base_ee, t_base_ee, R_cam_tag, t_cam_tag, method=CALIB_HAND_EYE_TSAI)` → `T_ee_cam`.
5. Write to `deploy/hand_eye.yaml`.
6. **Verify**: command the EE to the projected tag center (project `T_base_tag` to xy, move EE there). Measure the offset by eye / ruler. **Must be ≤ 5 mm. If larger, redo step 2 with more / better-distributed poses before training touches the table.**

`deploy/calibrate_hand_eye.py` (in repo) does steps 2–6 interactively — prompts you to move the arm, captures each frame, then solves and writes the yaml.

## 4. MDP

| Item | Value |
|---|---|
| Control | 50 Hz (decimation 2, sim 100 Hz) |
| Episode | 6.0 s = 300 steps |
| Action | 5 arm joints (absolute around home, `scale=0.5`) + 1 binary gripper (`open=0.5`, `close=0.0`) |
| Workspace | `x ∈ [0.10, 0.30] m`, `y ∈ [−0.15, 0.15] m` |
| Home `q` | `(0, 0, 0, 1.5708, 0, 0)`, gripper open |
| Terminations | `time_out`, `block_off_table` |
| Table | `0.6 × 1.0 × 0.02 m` at `(0.25, 0, −0.01)`; top `z=0`, back edge `x=−0.05` |

Block: 2 cm DexCube USD (scale 0.4), xy randomized. Bowl: 2-D goal from `BowlPoseCommandCfg`, **no scene prim**; rejection-sampled with `‖block − bowl‖ ≥ 0.15 m` (up to 16 attempts). Same `(x, y)` frame at deploy.

## 5. Observations (asymmetric A-C, **no image group**)

`ObservationsCfg` defines two groups; the runner cfg uses both.

| Group | Fields | Shape |
|---|---|---|
| `policy` (deployable) | `joint_pos_rel`, `joint_vel_rel`, `gripper_state`, `bowl_xy`, `ee_proj_xy`, `ee_to_bowl_xy`, **`cube_pos_xy_noisy`** (2), `last_action` | 1-D, 27-D total |
| `critic` (privileged) | `policy` + `block_position`, `block_to_bowl_xy`, `gripper_to_block`, `is_grasped` | 1-D |

`cube_pos_xy_noisy` (new term in `tasks/pickplace/mdp/observations.py`) returns `block_pos.xy + per-episode bias + Gaussian noise + dropout mask` — the noise model is in §6. The privileged critic still sees the clean full block pose so the value function is unbiased.

No `wrist_image` group, no `TiledCamera` spawn in the scene. The env cfg used for this path is `SoArm101PickPlaceBowlStateAprilTagEnvCfg` (a subclass of `SoArm101PickPlaceBowlTeacherFastEnvCfg`), which already nulls the wrist cam.

## 6. Sim noise model for `cube_pos_xy_noisy`

Sim σ must be ≥ 1.5× measured real σ. Measure real σ from 60 s of static AprilTag pose during bring-up; if measured σ is bigger than the sim σ here, widen sim before re-evaluating real performance.

| Source | Sim distribution |
|---|---|
| Tag corner localization | Per-step Gaussian σ = 2 mm per axis |
| Hand-eye residual | Per-episode bias U[−5, +5] mm per axis (reset by `reset_cube_pos_bias` event) |
| Pre-grasp dropout (motion blur, partial cover) | Bernoulli p = 0.10 → hold last value |
| Post-grasp dropout (gripper occludes tag) | **`is_grasped=True` → freeze value for rest of episode** |

The post-grasp freeze mirrors real exactly: once the gripper closes on the cube, the tag is occluded with probability 1 for the rest of the lift+place phase. The policy must learn to ignore `cube_pos_xy` once grasped and steer with `gripper_state` + `ee_to_bowl_xy`. Pre-grasp dropout is Bernoulli (the tag can briefly disappear from motion blur or grazing-angle reflection); post-grasp is deterministic hold.

## 7. Reward, curriculum, network

Verbatim from the old state-teacher of `tasks/pickplace` — nothing else about the MDP changes when you swap the image group for a noisy xy slot.

Reward (`mdp/rewards.py`): `reaching_object` 1.0, `lifting_object` 15.0, `object_goal_tracking` 16.0 (+ fine-grained 5.0 at `std=0.05`), **`release_in_bowl` 30.0** gated on lift latch (≥ 0.07 m) AND over-bowl-above-rim latch (≥ 0.08 m within 6 cm of bowl xy), `action_rate` / `joint_vel` −1e-4 → −1e-2 ramp at 10 k env-steps. No `task_success` termination — `release_in_bowl=30` pays every post-release step until time-out. **γ = 0.98 is load-bearing** (long dense tail).

Curriculum (`CurriculumCfg`): `reset_block_position` ±10 × ±15 cm at `(0.20, 0)`; bowl rejection-sampled in the same band; `log_success` TB metric. No Stage-3 expand band, no DrQ, no photometric jitter — we're not training a CNN.

Network: pure MLP actor-critic, `[256, 128, 64]` ELU, σ scalar Param, no CNN. Standard RSL-RL `ActorCritic`. PPO: `num_envs=4096`, `num_steps_per_env=24`, `max_iterations=1500`, `init_noise_std=1.0`, `entropy_coef=0.006`, `learning_rate=1e-4`, `desired_kl=0.01`, `γ=0.98`, `λ=0.95`, `clip=0.2`, `max_grad_norm=1.0`. Runner cfg lives at `tasks/pickplace/agents/state_apriltag_ppo_cfg.py` with `obs_groups = {"policy": ["policy"], "critic": ["policy", "critic"]}`. `experiment_name = "pickplace_bowl_state_apriltag"`.

## 8. Workflow

```bash
# Sim training (no --enable_cameras — camera-free)
uv run train --task Isaac-SO-ARM101-PickPlace-Bowl-StateAprilTag-v0 --headless

# Sim eval / debug
uv run play --task Isaac-SO-ARM101-PickPlace-Bowl-StateAprilTag-Play-v0 \
    --load_run <run> --checkpoint model_<best>.pt

# Real deploy — requires deploy/hand_eye.yaml (run calibrate_hand_eye first)
python -m deploy.calibrate_hand_eye                                   # interactive, verify ≤ 5 mm
python -m deploy.deploy_real --bowl-xy 0.20,-0.05 --target-color red
python -m deploy.deploy_real --bowl-xy 0.20,-0.05 --dry-run           # no hardware; synthetic forward
```

Bring-up sequence:
1. Print tags (§2) + calibration tag.
2. `pip install pupil-apriltags` on inference PC (handled by `deploy/setup_inference_pc.sh`).
3. `python -m deploy.calibrate_hand_eye` → `deploy/hand_eye.yaml`. **Verify ≤ 5 mm; if not, redo before training.**
4. `uv run zero_agent --task Isaac-SO-ARM101-PickPlace-Bowl-StateAprilTag-v0` smoke test (camera-free, no `--enable_cameras` flag).
5. Train (~1–2 h wall-clock to ≥ 80 % sim success by iter ~1000).
6. `play.py --debug-dump` sanity check (per-step dump matches the real-deploy schema in `deploy/README.md` §6 for sim↔real diff).
7. 60-s static AprilTag stability test on the real cube; confirm sim σ ≥ 1.5× measured σ (retrain if not).
8. `deploy_real --dry-run` → check obs vector populated.
9. Real bring-up, low speed, E-stop ready.

## 9. Deploy data flow

`deploy/deploy_real.py` per step:
1. Read `joint_pos` (servo bus) → `T_base_ee` via `kinpy` FK.
2. Grab RGB → undistort with `camera_intrinsics.yaml`.
3. If `gripper_closed` AND we believe we hold the cube (latched at last successful grasp) → skip detection, hold last `cube_pos_xy`.
4. Else: run `AprilTagDetector.pose(rgb, T_base_ee)` → `((x, y), valid)`. If valid, update last; if invalid (no tag in frame) hold last value.
5. Build obs vector in `PolicyCfg` order (joint state → gripper → bowl_xy → ee_proj_xy → ee_to_bowl_xy → cube_pos_xy → last_action).
6. Forward through `PPOActorState` (pure MLP, mirrors the training-side state actor; lives in `deploy/ppo_actor.py`).
7. Decode action → joint targets + binary gripper, send to Feetech bus (sim↔real action contract verbatim from CLAUDE.md). Shared FK / action-decode helpers live in `deploy/driver.py` so `calibrate_hand_eye.py` and `deploy_real.py` share them.

`deploy/cube_detector.py::AprilTagDetector` exposes `pose(rgb, T_base_ee) → ((x, y), valid)` and `set_target_id(int)` (the latter unused for Eval 1, see Eval 2 / Eval 3).

## 10. Risks and mitigations

1. **Hand-eye drift** — bumping the camera mount silently breaks everything. Re-run the §3 step-6 verify after every reassembly; abort if > 5 mm. There is no in-policy recovery from this.
2. **Sim noise underestimated** — measure 60 s of static real tag pose σ before training; sim σ must be ≥ 1.5× that. If real σ exceeds sim σ, the policy will encounter out-of-distribution obs and miss grasps.
3. **Tag flatness** — bent paper → non-coplanar corners → garbage PnP. Laminate flat first; reject any tag that visibly bows under finger pressure.
4. **Lighting** — specular highlight on glossy cube around the tag kills detection. Matte laminate + diffuse light.
5. **Tag occlusion before grasp** — if the gripper or arm shadows the tag during approach, detection rate drops and the policy starts seeing many dropout-hold frames. The Bernoulli p = 0.10 pre-grasp dropout in sim trains against this, but only up to that rate; if real dropout exceeds 30 % during approach, the cam mount or approach trajectory is wrong, not the policy.

## References

- **Code**: `tasks/pickplace/{pickplace_env_cfg,joint_pos_env_cfg}.py`, `mdp/{observations,events,rewards,terminations,commands}.py`, `agents/state_apriltag_ppo_cfg.py`. Real-robot deploy: `deploy/{deploy_real.py,driver.py,cube_detector.py,calibrate_hand_eye.py,ppo_actor.py}`, `deploy/README.md`.
- **Olson 2011**, *AprilTag: A robust and flexible visual fiducial system*, ICRA. <https://april.eecs.umich.edu/papers/details.php?name=olson2011tags>
- **Wang & Olson 2016**, *AprilTag 2: Efficient and robust fiducial detection*, IROS. — `tagStandard41h12` family.
- **Tsai & Lenz 1989**, *A new technique for fully autonomous and efficient 3-D robotics hand-eye calibration*, IEEE T-RA. — `CALIB_HAND_EYE_TSAI`.
- **RSL-RL**, Schwarke et al., 2025. <https://github.com/leggedrobotics/rsl_rl>
- **LeIsaac**, <https://github.com/LightwheelAI/leisaac> — wrist camera mount.
