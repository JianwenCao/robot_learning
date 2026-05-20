# Eval 1 — Single-Object Pick-and-Place (State-Only + AprilTag)

Task: `Isaac-SO-ARM101-PickPlace-Bowl-StateAprilTag-v0`. Single-stage camera-free PPO on privileged state plus a noisy `cube_pos_xy`. At deploy, that slot is filled by AprilTag pose in the wrist cam.

Foundational state-only + AprilTag plan; Eval-2 / Eval-3 / Bonus-B inherit calibration, detector, sim noise model, and training shape from here.

---

## 1. MDP

| Item | Value |
|---|---|
| Control | 50 Hz (decimation 2, sim 100 Hz) |
| Episode | 6.0 s = 300 steps |
| Action | 5 arm joints (absolute around home, `scale=0.5`) + 1 binary gripper (`open=0.5`, `close=0.0`) |
| Workspace | `x ∈ [0.10, 0.30] m`, `y ∈ [−0.15, 0.15] m` |
| Home `q` | `(0, 0, 0, 1.5708, 0, 0)`, gripper open |
| Terminations | `time_out`, `block_off_table` |
| Table | `0.6 × 1.0 × 0.02 m` at `(0.25, 0, −0.01)`; top `z=0`, back edge `x=−0.05` |
| Bowl rim | `BOWL_RIM_Z ≈ 0.06 m` (real bowl ≈ 6 cm; sim assumes this; measure and update in lockstep if a different bowl is used) |

Block: 2 cm DexCube USD (scale 0.4), xy randomized. Bowl: 2-D goal from `BowlPoseCommandCfg`, no scene prim; rejection-sampled `‖block − bowl‖ ≥ 0.15 m` (up to 16 attempts).

## 2. Observations

`ObservationsCfg` defines two groups; runner cfg uses both via `obs_groups = {"policy": ["policy"], "critic": ["policy", "critic"]}`.

| Group | Fields | Shape |
|---|---|---|
| `policy` (deployable) | `joint_pos_rel`, `joint_vel_rel`, `gripper_state`, `bowl_xy`, `ee_proj_xy`, `ee_to_bowl_xy`, **`cube_pos_xy_noisy`** (2), `last_action` | 27-D |
| `critic` (privileged) | `policy` + `block_position`, `block_to_bowl_xy`, `gripper_to_block`, `is_grasped` | wider |

No `wrist_image` group, no `TiledCamera` spawn. Env cfg `SoArm101PickPlaceBowlStateAprilTagEnvCfg` subclasses `SoArm101PickPlaceBowlTeacherFastEnvCfg` (wrist cam nulled).

## 3. Sim noise model for `cube_pos_xy_noisy`

Sim σ must be ≥ 1.5× measured real σ. Run `deploy/calibrate_apriltag_noise.py` once per camera mount (stationary tag for ~30 s, then tag swept across the workspace by hand-jogging) to measure per-axis std and bias; re-tune the table below if real σ > 1.5 mm or real bias > 3 mm.

AprilTag spec: 36h11 family, edge ≥ 14 mm (fits a 20 mm cube face with a 3 mm white border). One tag per top face — wrist cam looks down, so top tags stay visible up to grasp.

| Source | Sim distribution |
|---|---|
| Tag corner localization | Per-step Gaussian σ = 2 mm per axis |
| Hand-eye residual | Per-episode bias U[−5, +5] mm per axis (`reset_cube_pos_bias` event) |
| Pre-grasp dropout | Bernoulli p = 0.10 → hold last value |
| Post-grasp dropout | `is_grasped=True` → freeze value for rest of episode |

## 4. Reward, curriculum, network

Reward (`mdp/rewards.py`): `reaching_object` 1.0, `lifting_object` 15.0, `object_goal_tracking` 16.0 (+ fine-grained 5.0 at `std=0.05`), `ee_release_pose_over_bowl` 6.0, and **`release_in_bowl` 30.0** gated on lift latch (cube z ≥ 0.07 m) AND over-bowl-above-rim latch (cube z ≥ 0.08 m, within 6 cm of bowl xy), with release firing once z < `BOWL_RIM_Z` (= 0.06 m) ∧ gripper open ∧ settled. `ee_release_pose_over_bowl` pays after lift and continues after release, preferring the end-effector high over the target xy; the cube is free to fall, and `release_in_bowl` is what rewards the cube landing low at the target. Anti-hover / anti-smash shaping is deliberately small relative to release: `gripper_open_above_bowl_lure` +3 pays only when the cube is currently over the bowl and still above release height, while `still_grasped_above_bowl_penalty` −2 applies when a lifted cube is near the bowl but the gripper is still commanded closed. This makes the local optimum over the target "open and let the cube fall" rather than "keep clamping and push the cube down to the target." `action_rate` / `joint_vel` −1e-4 → −1e-2 ramp at 10 k env-steps. No `task_success` termination.

Curriculum (`CurriculumCfg`): `reset_block_position` ±10 × ±15 cm at `(0.20, 0)`; bowl rejection-sampled in the same band; `log_success` TB metric.

Network: pure MLP actor-critic `[256, 128, 64]` ELU, σ scalar Param, standard RSL-RL `ActorCritic`.

PPO: `num_envs=4096`, `num_steps_per_env=24`, `max_iterations=1500`, `init_noise_std=1.0`, `entropy_coef=0.006`, `learning_rate=1e-4`, `desired_kl=0.01`, `γ=0.98`, `λ=0.95`, `clip=0.2`, `max_grad_norm=1.0`. Runner cfg `tasks/pickplace/agents/state_apriltag_ppo_cfg.py`; `experiment_name = "pickplace_bowl_state_apriltag"`.
