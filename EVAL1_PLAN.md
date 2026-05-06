# Eval 1 — Single-Object Pick-and-Place: Sim-to-Real RL Plan

> Target: solve Eval 1 of Project 3 (50 pts) by training a goal-conditioned RL policy in Isaac Lab on the SO-ARM101 wrist-cam setup, then deploying the same policy on the real arm. **No teleoperation rollouts are recorded** — the only "data" is what the policy generates inside the simulator.

This plan is grounded in the actual code under `isaac_so_arm101/src/isaac_so_arm101/`. Where it deviates from the upstream Lift template (`tasks/lift/`), the deviation is called out explicitly. Robot facts (joint names, body names, joint limits, default pose) come straight from `robots/trs_so101/so_arm101.py` and `robots/trs_so101/urdf/so_arm101.urdf`.

---

## 0. TL;DR

We train a single PPO actor in Isaac Lab over thousands of randomized envs. The actor consumes the SO-ARM101 wrist RGB plus proprioception plus the bowl target `(x, y)` and outputs joint position offsets around the home pose. Heavy domain randomization (visual + dynamics) makes the policy transfer zero-shot to the real arm. The bowl is **not modeled as a mesh** — it lives as a 2-D goal coordinate (a `UniformPoseCommandCfg`), with success judged geometrically (block xy near bowl xy, block low, gripper opened). On the real robot the same coordinate is provided through a CLI / config arg, and the policy is run at the same fixed rate (50 Hz, matching `sim.dt=0.01, decimation=2` from the upstream Lift cfg) it was trained at.

The plan below is the source of truth for design decisions, file layout, training schedule, and deployment steps. Edit it as the project progresses.

---

## 1. Eval 1 spec — what we are optimizing for

| Item | Spec | Design implication |
|---|---|---|
| Object | One wooden block, **fixed 2 × 2 × 2 cm**, color **unconstrained** | Train with broad block-color randomization; size jitter only ±5–10 % |
| Target | One bowl, Ø 15.5 cm, h ≈ 5 cm, position **specified per rollout** as `(x, y, z)` in robot frame | Bowl is a goal, not an object — see §3.3. Goal-condition the policy on `(x, y)` |
| Block initial pose | Random across rollouts | Heavy initial-state randomization in sim |
| Observation | "Visual observation of blocks" — wrist camera RGB | Wrist cam encoded by a CNN → latent; bowl comes from the input arg, not perception |
| Action | Pick block, place into bowl, **release** | Reward must include the release. Action = joint position offsets around home (cleanest sim-to-real with the existing `JointPositionActionCfg`) |
| Success | Block inside bowl AND released | Geometric success check (§3.3) |
| Method | "Both BC and RL methods are allowed" | Pure RL is fine. We pick RL because (a) Eval 2 / 3 require it anyway and (b) we explicitly want zero teleop data |
| Eval env | Light gray table `#B8ADA9`, bowl pose given, 5 rollouts | Match the table color in sim and randomize *around* it |

Two non-obvious points the spec implies:

1. **The bowl is given as `(x, y, z)`**, never perceived. The image only needs to find the *block*. This drastically simplifies the perception problem — the policy doesn't need to detect bowl color or shape, it only needs to localize the block in the wrist frame.
2. **"Easy to modify target locations" at eval** means we change a config value or pass `--bowl_xy x y` at run time. The network must therefore be conditioned on the bowl coordinate, not have it baked in.

---

## 2. High-level approach

We use **on-policy PPO with an asymmetric Actor-Critic** trained on massively parallel Isaac Lab envs:

- **Actor** (deployable): wrist RGB → CNN → latent ⊕ proprio ⊕ bowl_xy ⊕ ee_proj_xy → MLP → action (5-d arm + 1-d gripper).
- **Critic** (sim-only): same inputs **plus** privileged ground-truth (block pose, gripper-to-block vector, contact flags). The critic accelerates training but is discarded at deploy.
- **Massive domain randomization** at every reset (visuals, physics, dynamics) so the actor's input distribution covers the real robot.
- **Joint-position control around the home pose** at 50 Hz (`JointPositionActionCfg` with `use_default_offset=True`), the same scheme used in the upstream `tasks/lift` task. This maps cleanly to Feetech `goal_position` writes on hardware.

This is a well-established sim-to-real recipe (OpenAI Cube, ANYmal, IndustReal, RobotPearl); the components below pin it to the SO-ARM101 + Eval 1 setting.

---

## 3. Sim environment design — `Isaac-SO-ARM101-PickPlace-Bowl-v0`

### 3.1 Scene

The scaffold mirrors `tasks/lift/lift_env_cfg.py::ObjectTableSceneCfg`, with three changes: (a) the existing `SeattleLabTable` USD is replaced by a flat gray cuboid so the visual matches the eval table `#B8ADA9`; (b) the `dex_cube_instanceable.usd` (5 cm) used by lift is swapped for a `sim_utils.CuboidCfg(size=(0.02, 0.02, 0.02))` so the block is exactly 2 cm; (c) a wrist `CameraCfg` is parented to `gripper_link`.

| Prim | Type | Notes |
|---|---|---|
| Ground / table | `RigidObjectCfg` (thin box, kinematic) at `z = 0` with `VisualMaterialCfg` | Base color `#B8ADA9` jittered ±15 RGB during DR. Friction randomized. |
| SO-ARM101 | `SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")` from `isaac_so_arm101.robots` | Joints (URDF order): `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`. Home pose has `wrist_flex=1.57` (pointing the gripper down) and `gripper=0` (closed). |
| Block | `RigidObjectCfg` with `sim_utils.CuboidCfg(size=(0.02, 0.02, 0.02))`, density ≈ 600 kg/m³ (~4.8 g) | Per-env randomized visual material; mass / friction DR |
| Bowl | **No prim** | Lives only as a `UniformPoseCommandCfg` named `bowl_pose`, sampled at reset (§3.4) |
| Wrist camera | `CameraCfg` (or `TiledCameraCfg` for batched rendering) parented to `gripper_link` with a small extrinsic offset | RGB only. Intrinsics & extrinsics matched to real wrist cam (§5.1) |
| `ee_frame` | `FrameTransformerCfg(prim_path=".../base_link", target=".../gripper_link", offset=[0.01, 0, -0.09])` | Reuse the exact frame already defined in `tasks/lift/joint_pos_env_cfg.py::SoArm101LiftCubeEnvCfg`. Gives us a precomputed fingertip-frame world pose for free. |

Workspace bounds (robot base frame, meters): bowl_xy sampled in `x ∈ [0.10, 0.30]`, `y ∈ [-0.15, 0.15]`. Block_xy sampled in the same box, but enforce `‖block_xy − bowl_xy‖ ≥ 0.10` so they don't overlap. These ranges are the planned starting point — narrow them after measuring the actual reachable workspace of the SO-ARM101 (§5.5 diagnostic). The lift task currently spawns its block at `[0.2, 0.0, 0.015]` and samples a goal in `pos_x ∈ [-0.1, 0.1], pos_y ∈ [-0.3, -0.1], pos_z ∈ [0.2, 0.35]`, which confirms this region is reachable.

### 3.2 Action

We use the same action wiring as `SoArm101LiftCubeEnvCfg`:

```python
arm_action = mdp.JointPositionActionCfg(
    asset_name="robot",
    joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
    scale=0.5,                # action ∈ [-1, 1] → ±0.5 rad around home
    use_default_offset=True,  # target = default_q + scale * action  (NOT delta)
)
gripper_action = mdp.BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["gripper"],
    open_command_expr={"gripper": 1.5},   # near upper limit 1.745 rad
    close_command_expr={"gripper": 0.0},  # ≈ closed
)
```

**Important**: this is *absolute-around-home*, not delta. Setting `action=0` returns the arm to its home pose. The full reachable-from-home cone is `home ± scale` rad per joint. With `scale=0.5` and the default home (`wrist_flex=1.57`), this is comfortably enough to descend onto the table and lift back. If the policy can't cover the workspace, bump to `scale=1.0` rather than switching to delta — delta accumulation makes safety bounds harder to enforce on hardware.

The gripper has joint range `[-0.17453, 1.74533]` rad with `0` ≈ closed and positive = open (the moving jaw rotates open as the joint angle increases). The lift task's `open_command_expr` uses `0.5`; we widen to `1.5` because the 2 cm block needs more open clearance than DexCube's 2.5 cm did when it was scale-shrunk. Tune in the diagnostic.

The 6-dim action vector emitted by the policy is therefore: 5 continuous in `[-1, 1]` for the arm, plus 1 channel that the binary action term thresholds at 0 to pick `open`/`close`.

### 3.3 Bowl as a goal, not a prim — geometric success

At each reset we sample `bowl_pose` with `mdp.UniformPoseCommandCfg`. There is no bowl mesh. The success criterion is purely geometric:

```
in_bowl   = (‖block_xy − bowl_xy‖ < r_safe)   AND
            (block_z < bowl_height + ε)        AND
            (block "steady" for k frames)

released  = (gripper_cmd == OPEN)              AND
            (no contact between gripper and block)

success   = in_bowl AND released
```

with `r_safe = 0.06 m` (slightly under the real bowl radius 0.0775 m, leaves margin), `bowl_height = 0.05 m`, `ε = 0.01 m`, `k = 5` (≈ 0.10 s at 50 Hz).

Why this is fine for sim-to-real: the real bowl is wide (15.5 cm Ø) and shallow (5 cm). If the policy releases the block above `bowl_xy` with the gripper roughly at table height, the physical bowl will catch it. We don't need bowl-rim contact dynamics in sim to make that happen.

### 3.4 Observations

#### Policy group (deployable on real robot)

Modeled after `tasks/lift/lift_env_cfg.py::ObservationsCfg.PolicyCfg`, with the visual stream and the bowl goal added:

| Field | Dim | How in sim | How on real robot |
|---|---|---|---|
| `wrist_rgb` | 3 × 96 × 96 (or 128²) | `CameraCfg` output, then resize + normalize | OpenCV USB capture, same resize + normalize |
| `joint_pos` | 6 | `mdp.joint_pos_rel` (relative to default home) | Feetech `present_position` → rad, then subtract home |
| `joint_vel` | 6 | `mdp.joint_vel_rel` | Feetech `present_velocity` → rad/s |
| `last_action` | 6 | `mdp.last_action` | rolled buffer |
| **`bowl_xy`** | 2 | `mdp.generated_commands(command_name="bowl_pose")[:, :2]` | from `--bowl_xy x y` |
| **`ee_proj_xy`** | 2 | `(ee_frame.data.target_pos_w[:,0,:] − robot.data.root_pos_w)[:, :2]` | URDF FK on `qpos`, drop z |
| **`ee_to_bowl_xy`** | 2 | `bowl_xy − ee_proj_xy` | same |

`mdp.joint_pos_rel`, `mdp.joint_vel_rel`, `mdp.last_action`, and `mdp.generated_commands` come from `isaaclab.envs.mdp` and are already used by the lift task. The new helpers (`ee_proj_xy`, `ee_to_bowl_xy`, `wrist_rgb`) live in `tasks/pickplace/mdp/observations.py` (§4.1).

Including `ee_proj_xy` is a deliberate inductive bias: it gives the policy a 2-D Cartesian feature ("where is my gripper on the table plane") so it doesn't have to learn forward-kinematics through its own MLP. The visual stream then specializes in *block* localization, which is what wrist-RGB is actually good for. Because we already have `ee_frame` for the lift reward, we get this almost for free.

`ee_to_bowl_xy` is redundant given the previous two, but the explicit subtraction shortcut speeds up reach-stage learning by a large margin.

Frame stacking: stack 3 wrist-RGB frames at the policy rate. The parallax across frames during gripper descent gives free monocular-depth signal that helps the grasp stage.

#### Critic group (asymmetric, sim-only)

Everything in the policy group, plus:

- `block_xyz` (3) via `mdp.object_position_in_robot_root_frame` — already defined in `tasks/lift/mdp/observations.py`, can be reused verbatim.
- `block_quat` (4)
- `gripper_xyz` (3) (full 3-D from `ee_frame`, not projected)
- `block_to_bowl_xy` (2)
- `gripper_to_block` (3)
- contact flags: `block↔gripper`, `block↔table` (2). NOTE: `SO_ARM101_CFG` sets `activate_contact_sensors=False` ("waiting for capsule implementation"). Until it's flipped on we approximate `is_grasped` from gripper command + `block_z > threshold`, as the existing lift reward does via `mdp.object_is_lifted`.
- `is_grasped` boolean derived from gripper command + block_z (see above)

This is what `rsl_rl` 3.0 calls a `critic_observations` group; you register it as a second `ObsGroup` named `critic` next to `policy`, and the PPO trainer wires it up via `policy.class_name`'s `obs_groups` argument.

### 3.5 Reward shaping

All terms in `mdp/rewards.py`. We reuse the upstream primitives where they fit and add stage gating. Stage gating is critical — without it, a single dense reach term will dominate.

```python
# Stage 1 — reach (active until block is grasped)
# Reuse mdp.object_ee_distance(std=0.05): returns 1 - tanh(d / std)
r_reach   = (1 - is_grasped) * mdp.object_ee_distance(std=0.05)
                                                              # weight 1.0

# Stage 2 — grasp (sparse-ish event)
# is_grasped from contact flags OR block_z > 0.025 + gripper_cmd==CLOSE
r_grasp   = is_grasped_now_and_not_before                     # weight 5.0

# Stage 3 — transport (only after grasp)
# Reuse mdp.object_goal_distance(std=0.20, minimal_height=0.025, command_name="bowl_pose")
r_transp  = mdp.object_goal_distance(std=0.20, minimal_height=0.025,
                                     command_name="bowl_pose")
                                                              # weight 2.0
# (the minimal_height factor already gates this on the block being lifted —
# no extra is_grasped multiplier needed if minimal_height is set)

# Stage 4 — place (block xy near bowl, block low)
r_place   = (||block_xy - bowl_xy|| < 0.06) AND (block_z < 0.06)
                                                              # weight 5.0

# Stage 5 — release (terminal)
r_release = r_place AND gripper_open AND ||block_vel|| small  # weight 10.0

# Penalties (reuse upstream)
p_action  = 1e-4 * mdp.action_rate_l2                         # weight tuned at 1e-4
p_jvel    = 1e-4 * mdp.joint_vel_l2
p_drop    = -2.0 if (was_grasped AND block_z < 0.005 AND not in bowl)
p_offtab  = -2.0 if block_xy outside workspace
```

Reuse vs new: `r_reach`, `r_transp`, `p_action`, `p_jvel` map directly onto existing functions in `tasks/lift/mdp/rewards.py` and `isaaclab.envs.mdp`. `r_grasp`, `r_place`, `r_release`, `p_drop`, `p_offtab` need to be written. The lift task also exposes `object_ee_distance_and_lifted` (a multiplicative reach×lift) that's a useful template for combining gates.

Two failure modes to pre-empt:

- **`r_transp` leaking before grasp**: `mdp.object_goal_distance` already multiplies by `(object.z > minimal_height)`; keep `minimal_height ≥ 0.025` so it can't fire while the block is on the table. If we ever drop the gating, the policy learns to fly the gripper over the bowl without the block.
- **`r_release` without place gating**: the release reward must require the place condition first; otherwise the policy opens the gripper everywhere.

The lift task's curriculum decays `action_rate` and `joint_vel` weights from `-1e-4` to `-1e-1` over 10 000 steps via `mdp.modify_reward_weight`. Inherit the same curriculum.

### 3.6 Terminations

`mdp/terminations.py`:

- `success` — the `r_release` condition above; terminate, mark success. Pattern after `tasks/lift/mdp/terminations.py::object_reached_goal`.
- `time_out` after `episode_length_s = 6.0` → 300 steps at 50 Hz. (The lift task uses 5.0 s, but pickplace has an extra release stage.)
- `block_off_table`: reuse `mdp.root_height_below_minimum(minimum_height=-0.05, asset_cfg=SceneEntityCfg("object"))` — already used by lift.
- `joint_limit_violation` (soft) — penalize, do not terminate.

### 3.7 Domain randomization — the single most important section

Implemented in `mdp/events.py` as `EventTermCfg(mode="reset")` and `mode="interval"` items. The existing lift task only does `mdp.reset_scene_to_default` and `mdp.reset_root_state_uniform` for the object — we are adding everything else from scratch. Suggested ranges (widen if sim-to-real fails on that axis):

| Category | Knob | Distribution |
|---|---|---|
| **Visual** | Block color (RGB) | Uniform over the full cube, biased away from `#B8ADA9` ± 25 to keep contrast against the table |
| | Block texture | None / wood-grain variants |
| | Table base color | `#B8ADA9` ± 15 RGB |
| | Table material PBR | roughness ∈ [0.4, 0.9], metallic ∈ [0, 0.1] |
| | Lighting count, intensity, direction | 1–3 lights, 500–2000 lux, full hemisphere |
| | Camera intrinsics | fx, fy ± 5 %; cx, cy ± 2 px |
| | Camera extrinsic on `gripper_link` | ± 2 mm translation, ± 1° rotation |
| | Image post-noise | Gaussian σ ∈ [0, 5/255], motion blur kernel 0–3 px, JPEG q ∈ [70, 100] |
| | Background distractors (off-table) | 0–3 random-color boxes outside workspace |
| **Physical** | Block size | ± 5–10 % each axis |
| | Block mass | ± 50 % around 4.8 g |
| | Friction (table, block, gripper pads) | μ ∈ [0.4, 1.2] |
| | Bowl_xy | full workspace per §3.1 (driven by `UniformPoseCommandCfg.ranges`) |
| | Block_xy | full workspace per §3.1 (driven by `mdp.reset_root_state_uniform`) |
| | Robot base pose | ± 2 mm, ± 1° |
| **Dynamics** | Servo PD gains | ± 30 % around the values in `SO_ARM101_CFG.actuators` (e.g. shoulder_pan stiffness 200 / damping 80) |
| | Action latency | 1–5 sim steps (20–100 ms at 50 Hz) |
| | Observation latency | 1–3 sim steps |
| | Action noise | additive Gaussian σ = 0.01 |
| **Initial** | Initial joint config | small jitter around home pose. The reach task uses `mdp.reset_joints_by_scale(position_range=(0.5, 1.5))` — for our task we want a *tighter* spread so the wrist starts pointing down. |

If you only ship one of these well, **make it the visual block**. That's where wrist-cam policies most commonly fail to transfer.

### 3.8 Algorithm

`rsl_rl` 3.0.1 PPO. Start from `tasks/lift/agents/rsl_rl_ppo_cfg.py::LiftCubePPORunnerCfg` and adjust:

```python
@configclass
class PickPlaceBowlPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env  = 24       # matches lift
    max_iterations     = 5000     # 5–8× lift; vision is slower to converge
    save_interval      = 100
    experiment_name    = "pickplace_bowl"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name        = "PickPlaceActorCritic",   # custom (CNN + asymmetric critic)
        init_noise_std    = 1.0,
        actor_hidden_dims = [256, 128, 64],
        critic_hidden_dims= [256, 128, 64],
        activation        = "elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef        = 1.0,
        use_clipped_value_loss = True,
        clip_param             = 0.2,
        entropy_coef           = 0.006,
        num_learning_epochs    = 5,
        num_mini_batches       = 4,
        learning_rate          = 1.0e-4,
        schedule               = "adaptive",
        gamma                  = 0.98,
        lam                    = 0.95,
        desired_kl             = 0.01,
        max_grad_norm          = 1.0,
    )
```

Scale targets: 2048 parallel envs (drop to 1024 if cameras blow VRAM on the 5090). Total budget 100–300 M env steps. On the 5090 this is roughly 8–24 h once cameras are on. The lift task converges on state-only obs in ≤ 1500 iterations × 24 steps × 4096 envs ≈ 150 M steps — vision should need ~2× that.

### 3.9 Network architecture

```
wrist_rgb (3, 96, 96)  ──CNN──┐
joint_pos (6)                 │
joint_vel (6)                 │
bowl_xy (2)                   ├─concat─MLP(256, 128, 64)─μ, σ─actions(6)
ee_proj_xy (2)                │
ee_to_bowl_xy (2)             │
last_action (6)               │
                              └─ critic only: + block_pose, distances, contacts → V(s)
```

CNN (Option A, default — train end-to-end):

```
Conv2d( 3, 32, k=8, s=4) → ELU       # 32 × 23 × 23
Conv2d(32, 64, k=4, s=2) → ELU       # 64 × 10 × 10
Conv2d(64, 64, k=3, s=1) → ELU       # 64 × 8 × 8
Flatten → Linear(4096, 128) → LayerNorm → ELU
```

(ELU keeps the activation consistent with the existing PPO MLP.)

CNN (Option B — fall-back if PPO can't learn vision in time): freeze a pretrained R3M or DINOv2-small backbone, project to 128-d.

**Auxiliary loss (recommended)**: add a small head on the visual latent that regresses `block_xy` (privileged label, available in sim). MSE-weighted at ~ 0.1 of the policy loss. Forces the encoder to actually localize the block instead of summarizing pixels. Drop the head at deploy.

The custom `PickPlaceActorCritic` is registered with RSL-RL via `policy.class_name`. RSL-RL 3.0's `ActorCritic` accepts an `obs_groups` mapping (e.g. `{"actor": ["policy"], "critic": ["policy", "critic"]}`), which is how the asymmetric inputs are wired.

---

## 4. Isaac Lab setup plan

The repo lives at `/home/rui/Projects/Course_Code/Robot_Learning/project3/isaac_so_arm101/` and is installed editable into the `so_arm` conda env (Isaac Lab 2.3.0, Isaac Sim 5.1, RSL-RL 3.0.1, RTX 5090 + cu128). Daily-use commands live in `RUNNING.md`. Existing registered tasks (run `list_envs` to confirm): `Isaac-SO-ARM10{0,1}-{Reach,Lift-Cube}-{,-Play}-v0`. Our new IDs will be `Isaac-SO-ARM101-PickPlace-Bowl-{,-Play}-v0`.

### 4.1 Files to create

```
src/isaac_so_arm101/tasks/pickplace/
├── __init__.py                 # gym.register two ids: -v0 (train) and -Play-v0 (eval)
├── pickplace_env_cfg.py        # ManagerBasedRLEnvCfg base class (matches LiftEnvCfg pattern)
├── joint_pos_env_cfg.py        # SoArm101 subclass: wires SO_ARM101_CFG, action, ee_frame, block, camera
├── agents/
│   ├── __init__.py
│   └── rsl_rl_ppo_cfg.py       # PickPlaceBowlPPORunnerCfg + custom CNN ActorCritic class
└── mdp/
    ├── __init__.py             # `from isaaclab.envs.mdp import *`, then re-export local terms
    ├── observations.py         # ee_proj_xy, ee_to_bowl_xy, wrist_rgb, (reuse lift's object_position_in_robot_root_frame for critic)
    ├── rewards.py              # grasp/place/release events; reach & transport pulled from upstream
    ├── terminations.py         # success
    └── events.py               # reset randomizers (block, table material, lights, camera intrinsics, latency)
```

Most of this is a fork of `tasks/lift/`, with the bowl goal kept as a `UniformPoseCommandCfg`, the table swapped for a colored cuboid, the block swapped for an exact 2 cm cube, and a wrist camera added.

### 4.2 Step-by-step setup

**Step A — duplicate Lift as a starting template.**

```
cp -r src/isaac_so_arm101/tasks/lift src/isaac_so_arm101/tasks/pickplace
```

Open the new files and rename classes (`LiftEnvCfg` → `PickPlaceBowlEnvCfg`, `SoArm101LiftCubeEnvCfg` → `SoArm101PickPlaceBowlEnvCfg`, etc.). The `tasks/__init__.py` does an `import_packages(__name__, _BLACKLIST_PKGS)` that auto-picks up the new directory — no manual import needed.

**Step B — register the gym IDs.**

In `tasks/pickplace/__init__.py` (mirror `tasks/lift/__init__.py`):

```python
import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-SO-ARM101-PickPlace-Bowl-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceBowlEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceBowlPPORunnerCfg",
    },
    disable_env_checker=True,
)
gym.register(
    id="Isaac-SO-ARM101-PickPlace-Bowl-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SoArm101PickPlaceBowlEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PickPlaceBowlPPORunnerCfg",
    },
    disable_env_checker=True,
)
```

The `_PLAY` config inherits the train cfg and overrides `num_envs=50` (matching lift) and disables observation corruption, exactly as `SoArm101LiftCubeEnvCfg_PLAY` does.

**Step C — scene.**

In `pickplace_env_cfg.py`, build the scene cfg by inheriting the structure of `ObjectTableSceneCfg` from lift. Concrete swaps:

- Replace the `table = AssetBaseCfg(... SeattleLabTable ...)` line with a thin gray cuboid. The simplest is `RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Table", spawn=sim_utils.CuboidCfg(size=(1.0, 1.0, 0.02), ..., visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.722, 0.678, 0.663))))` — the diffuse color converts `#B8ADA9` to linear RGB. Set `init_state.pos = [0.5, 0.0, -0.01]` so the top is at `z=0`.
- The `object` slot stays a `RigidObjectCfg` but the spawn becomes `sim_utils.CuboidCfg(size=(0.02, 0.02, 0.02))` with explicit `mass_props` (≈ 4.8 g) and a `visual_material` that DR overwrites per env.
- Add a `wrist_cam` slot of type `CameraCfg` with `prim_path="{ENV_REGEX_NS}/Robot/gripper_link/wrist_cam"`, `data_types=["rgb"]`, `width=96, height=96`, and an `OffsetCfg` placing the camera ~3 cm along gripper_link's local -y (looking down — exact extrinsic measured in §5.1).

**Step D — bowl as a `UniformPoseCommandCfg`.**

The lift task already sets up `mdp.UniformPoseCommandCfg(asset_name="robot", body_name=["gripper_link"], ranges=...)`. Reuse it almost verbatim, renaming the term `bowl_pose`, with ranges:

```python
ranges = mdp.UniformPoseCommandCfg.Ranges(
    pos_x=(0.10, 0.30), pos_y=(-0.15, 0.15), pos_z=(0.00, 0.00),
    roll=(0,0), pitch=(0,0), yaw=(0,0),
)
```

`body_name="gripper_link"` is only used by the visualizer marker — not by the success criterion. The reward and success terms read `env.command_manager.get_command("bowl_pose")[:, :3]` directly. Set `debug_vis=True` so the bowl center renders as a frame marker during play.

**Step E — observations.**

In `mdp/observations.py`, add the helpers (the wrist_rgb fetch is the only non-trivial one):

```python
def ee_proj_xy(env, ee_frame_cfg=SceneEntityCfg("ee_frame"),
               robot_cfg=SceneEntityCfg("robot")):
    ee = env.scene[ee_frame_cfg.name]                     # FrameTransformer
    robot = env.scene[robot_cfg.name]
    ee_w = ee.data.target_pos_w[..., 0, :]                # (N, 3)
    base_w = robot.data.root_pos_w                        # (N, 3)
    return (ee_w - base_w)[:, :2]                         # (N, 2)

def bowl_xy(env, command_name="bowl_pose"):
    return env.command_manager.get_command(command_name)[:, :2]

def ee_to_bowl_xy(env, ee_frame_cfg=SceneEntityCfg("ee_frame"),
                  robot_cfg=SceneEntityCfg("robot"),
                  command_name="bowl_pose"):
    return bowl_xy(env, command_name) - ee_proj_xy(env, ee_frame_cfg, robot_cfg)

def wrist_rgb(env, sensor_cfg=SceneEntityCfg("wrist_cam"), size=(96, 96)):
    img = env.scene.sensors[sensor_cfg.name].data.output["rgb"]   # (N, H, W, 3) uint8 or float
    img = img.permute(0, 3, 1, 2).float() / 255.0
    if (img.shape[-2], img.shape[-1]) != size:
        img = torch.nn.functional.interpolate(img, size=size, mode="bilinear",
                                              align_corners=False)
    return img      # (N, 3, 96, 96)
```

Then bind these as `ObsTerm`s under two `ObsGroup`s: `policy` and `critic`. The `policy` group includes `mdp.joint_pos_rel`, `mdp.joint_vel_rel`, `mdp.last_action`, `mdp.generated_commands(command_name="bowl_pose")`, plus the four custom helpers. The `critic` group adds `mdp.object_position_in_robot_root_frame` (already in `tasks/lift/mdp/observations.py`) and any contact-flag obs (deferred until contact sensors are enabled).

**Step F — rewards.**

In `mdp/rewards.py`, define the staged reward terms. The reach and transport stages are the lift task's existing `object_ee_distance` and `object_goal_distance` (with `command_name="bowl_pose"`). Grasp/place/release are new; pattern after the lift task's `object_is_lifted` for the per-env tensor shape and `is_grasped_now_and_not_before` for edge-trigger semantics.

**Step G — events / DR.**

In `mdp/events.py`, an event term per knob in §3.7. Modes:

- `mode="reset"` for per-episode randomization (most knobs).
- `mode="interval"` (every N steps) for slow drift items like lighting if you want time-varying lights.
- `mode="startup"` for one-shot scene-build randomization (e.g. number of distractor boxes).

Use `mdp.reset_root_state_uniform` for the block (already done in lift), plus new functions for material/lighting/intrinsic randomization.

**Step H — agent (PPO + custom CNN).**

`agents/rsl_rl_ppo_cfg.py` plugs in a custom `ActorCritic` (RSL-RL 3.0 allows this via `policy.class_name`). Implement the actor as the CNN-then-concat-then-MLP in §3.9; the critic is a parallel branch consuming the privileged group.

**Step I — sanity-check progression.**

In order:

1. `zero_agent --task Isaac-SO-ARM101-PickPlace-Bowl-Play-v0 --enable_cameras` → a single env opens, scene renders, arm sits still, no errors. (Cameras must be enabled explicitly because RSL-RL `train.py` only does it when `--video` is set.)
2. `random_agent --task Isaac-SO-ARM101-PickPlace-Bowl-Play-v0 --enable_cameras` → arm flails, block can be displaced, no crashes during 60 s.
3. **State-only training run** (temporarily replace `wrist_rgb` with `mdp.object_position_in_robot_root_frame` in policy obs and drop the camera): should solve to ≥ 80 % success in ≤ 1500 PPO iterations (matches lift task's max_iterations). If not, the reward / DR / action are wrong — fix here before turning vision back on.
4. **Vision training run**: full DR, full vision policy. Train ~5000 iterations. Expect ≥ 60 % success in sim play.

Skipping step 3 is the single most common reason vision RL "doesn't work" — it almost always means the underlying MDP is broken and vision isn't the issue.

### 4.3 Headless training command

```bash
conda activate so_arm
export OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y

train --task Isaac-SO-ARM101-PickPlace-Bowl-v0 \
      --headless \
      --enable_cameras \
      --num_envs 2048 \
      --max_iterations 5000 \
      --logger tensorboard
```

Watch:

```bash
tensorboard --logdir isaac_so_arm101/logs/rsl_rl
```

Per-stage success curves (`reach`, `grasp`, `transport`, `place`, `release`) are essential to log — they tell you which stage is bottlenecking. Add them as scalar metrics inside the reward functions (write to `env.extras["log"]`).

### 4.4 Eval inside sim

```bash
play --task Isaac-SO-ARM101-PickPlace-Bowl-Play-v0 \
     --num_envs 16 \
     --enable_cameras \
     --checkpoint isaac_so_arm101/logs/rsl_rl/pickplace_bowl/<run>/model_*.pt
```

Add `--video --video_length 300` to dump rollouts as MP4s so you can visually inspect failure modes. The `play` script also auto-exports a JIT and ONNX policy under `<run>/exported/`, which we'll use for deployment.

---

## 5. Real-robot deployment plan

The actor only consumes things available on the real robot, so this is an interface problem, not an algorithmic one. The deploy script lives outside the gym env — it imports the trained policy weights (the JIT-exported `.pt` from §4.4) and runs an open-loop control loop talking to the wrist cam and the Feetech bus.

### 5.1 Camera calibration

1. Print a 7 × 5 chessboard (10 mm squares).
2. With the SO-ARM101 wrist cam, capture 30+ frames at varied angles.
3. Run `cv2.calibrateCamera` → `(fx, fy, cx, cy)` and distortion coefficients.
4. Plug those intrinsics into the sim `CameraCfg` (`focal_length` and `horizontal_aperture` are derived from `fx` and the image width via `focal_length = fx * horizontal_aperture / image_width`). Verify by rendering a simple checkerboard plane in sim and comparing.
5. Measure the wrist-cam mounting offset relative to the `gripper_link` body by hand (calipers or by holding a known target). Set the same offset on the sim camera prim's `OffsetCfg`.
6. Apply the same image preprocessing both sides: undistort → BGR→RGB → resize to 96 × 96 → `float / 255.0` (no further normalization unless the encoder does it internally).

### 5.2 Servo bus and joint mapping

SO-ARM101 uses Feetech STS-series servos (typically STS3215). The `feetech-servo-sdk` Python package gives `read_present_position`, `write_goal_position`, etc.

URDF facts to hard-pin in a small `so101_io.py`:

| Joint | URDF limit (rad) | Notes |
|---|---|---|
| `shoulder_pan` | ±1.91986 | full revolute |
| `shoulder_lift` | ±1.74533 | |
| `elbow_flex` | ±1.69 | |
| `wrist_flex` | ±1.65806 | home is `1.57` (gripper points down) |
| `wrist_roll` | [-2.74385, 2.84121] | asymmetric |
| `gripper` | [-0.17453, 1.74533] | **asymmetric**: 0 ≈ closed, larger = more open. Don't assume symmetry around zero! |

Things to verify once and pin down:

- **Servo ID order** vs URDF joint order. Build a `SERVO_TO_JOINT = [...]` list and unit-test on a known pose (e.g. arm fully extended).
- **Servo counts ↔ rad** scaling. Feetech default is 4096 counts per 360°. Each joint may have a sign flip relative to the URDF — record signs explicitly.
- **Zero offset per joint**: the home pose in sim is at `q = (0, 0, 0, 1.57, 0, 0)` but the servo home reads some non-zero counts. Capture this once and bake it into the IO layer.
- **Joint limits**: hardcode the soft limits above and clip every command before writing. This is your last line of defense against the policy commanding into the workspace floor.

### 5.3 Control loop

```python
policy = torch.jit.load("policy.pt").eval().cuda()
cam    = WristCam("/dev/video0", w=640, h=480)
bus    = FeetechBus("/dev/ttyUSB0", servo_ids=[1,2,3,4,5,6])

bowl_xy   = parse_args().bowl_xy           # e.g. [0.20, -0.05]
home_q    = torch.tensor([0.0, 0.0, 0.0, 1.57, 0.0, 0.0])  # matches SO_ARM101_CFG
ACTION_SCALE = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])     # arm only
last_action  = torch.zeros(6)
qpos_buf, qvel_buf, img_buf = deque(maxlen=3), deque(maxlen=1), deque(maxlen=3)

bus.go_home()                              # write home_q
rate = Rate(50)                            # 50 Hz control loop (matches sim)

while not user_stop():
    img  = preprocess(cam.read())                          # (3, 96, 96)
    qpos = bus.read_pos_rad()                              # (6,)
    qvel = bus.read_vel_rad()                              # (6,)

    img_buf.append(img); qpos_buf.append(qpos); qvel_buf.append(qvel)
    obs = pack(
        wrist_rgb     = stack(img_buf),
        joint_pos     = qpos - home_q,                     # joint_pos_rel
        joint_vel     = qvel,                              # joint_vel_rel (home_qvel = 0)
        bowl_xy       = bowl_xy,
        ee_proj_xy    = fk_proj(qpos),
        ee_to_bowl_xy = bowl_xy - fk_proj(qpos),
        last_action   = last_action,
    )

    with torch.inference_mode():
        action = policy(obs)                               # (6,), in [-1, 1]

    arm_cmd     = home_q[:5] + ACTION_SCALE * action[:5]   # absolute-around-home
    gripper_cmd = 1.5 if action[5] > 0 else 0.0            # binary thresholding
    target_q    = clamp(torch.cat([arm_cmd, gripper_cmd]), q_min, q_max)
    bus.write_pos_rad(target_q)
    last_action = action

    if success_check(qpos, bowl_xy):
        break
    rate.sleep()
```

Five things that must match the sim **exactly**:

1. **Control rate** (50 Hz). If the sim was 50 Hz and you run at 30 Hz on the robot, the policy gets stale obs and the action scale is effectively wrong.
2. **Action semantics** (absolute-around-home with `scale=0.5`, NOT delta; binary-thresholded gripper with `open=1.5, close=0.0`).
3. **Joint order and signs** (see §5.2). The joint_pos vector going into the policy must be in URDF order: `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`.
4. **Image preprocessing** (resize, color order, value range, frame stack).
5. **`fk_proj`**: a forward-kinematics function that returns `ee_xy` in the robot base frame, matching what `FrameTransformer(target=gripper_link, offset=[0.01,0,-0.09])` returns in sim. Use `pinocchio` or `urdfpy` on the host with the same URDF (`robots/trs_so101/urdf/so_arm101.urdf`) and apply the same offset on the gripper_link frame.

### 5.4 Safety

- A keyboard hook (e.g. `q` or Esc) sets `user_stop()` to True → loop exits → bus goes home.
- Workspace box check on every commanded `target_q`: project to ee_xyz via FK, and if it's outside `x ∈ [0, 0.35], y ∈ [-0.20, 0.20], z ∈ [0.00, 0.30]`, reject the command and hold last pose.
- Joint velocity limit: cap the per-step delta in IO layer regardless of what the policy outputs (the URDF lists `velocity=10` rad/s per joint, but Feetech STS3215 actually maxes at ~1.5 rad/s, which matches `velocity_limit_sim=1.5` in `SO_ARM101_CFG`).
- First runs **always** with the gripper open and the operator hand on the power switch.

### 5.5 Pre-flight diagnostic mode

Before running the policy, run a side-by-side check:

1. Move the real arm to a known pose, dump `qpos` from the bus.
2. Set the same `qpos` in sim (use a `zero_agent` modified to write fixed targets), dump the same.
3. Confirm they match within 1°.
4. Render a wrist-cam frame in sim with the same arm pose; compare to the real wrist-cam frame visually. Field of view, sense of motion, and approximate object size in frame should match. If not, fix camera intrinsics or extrinsics in sim.
5. Sweep the gripper through a small XY square in sim and on the real robot; record `ee_proj_xy` from both. They should overlay within ~5 mm.

This catches 90 % of "the policy worked in sim and didn't on the robot" surprises before they happen.

---

## 6. Risks, failure modes, and mitigations

| Risk | Why | Mitigation |
|---|---|---|
| Wrist-cam-only block search fails | When the gripper starts far from the block, the block isn't in the camera frame at all | Start the arm in a "scan" home pose where the wrist cam looks down on the workspace (the existing home `wrist_flex=1.57` already does this); widen the camera FOV in sim and real; add a brief hand-coded scan motion before policy hand-off |
| Monocular-RGB depth ambiguity at grasp | One camera, no stereo | 3-frame stack (parallax during descent); auxiliary `block_xy` regression head |
| 2 cm block not graspable by SO-ARM101 fingers | Gripper closure range may not reach 1.5 cm | Verify in URDF before training (gripper range `[-0.17, 1.75]` rad with finger geometry from `moving_jaw_so101_v1_link`); if needed, print finger pads. The lift task scales DexCube to 2.5 cm and grasps OK, so 2 cm should be feasible. |
| Sim block velocity / mass too unrealistic | Default density gives ~5 g — accurate, but if friction is wrong the block slides | Wide friction DR; test grasp with the actual wooden block once on the real arm and adjust |
| Action latency mismatch | Servo command-to-execution lag on real hw | Measure once with a step-response test, add the measured value (with margin) to sim DR |
| Policy releases too high | Reward only required xy and "block low", not gripper height | Add a soft penalty on gripper z at the moment of release |
| Bowl pose outside training distribution at eval | Graders pick `(x, y, z)` we never trained on | Sample bowl_xy widely (full reachable workspace), not just an "expected" region |
| Vision encoder collapses | PPO doesn't push enough gradient through the CNN | Auxiliary block-xy regression head; pretrained encoder fall-back |
| Real cam intrinsics drift after calibration | USB cam auto-focus / auto-exposure | Disable cam auto-* via v4l2-ctl before each run; recalibrate if the lens is bumped |
| Servo overheats during long sessions | Feetech STS series throttle when hot | Insert short cooldown sleeps between rollouts; monitor servo temp register |
| Contact sensors disabled in `SO_ARM101_CFG` | The cfg sets `activate_contact_sensors=False` ("waiting for capsule implementation"); critic loses contact flags | Approximate `is_grasped` from `gripper_cmd==CLOSE` AND `block_z > grasp_threshold`; flip flag back on once Isaac Lab supports it for capsule-replaced links |
| `BinaryJointPositionActionCfg` thresholding wastes policy entropy on the gripper dim | The gripper output is squashed to 0/1, so PPO's σ on that channel drifts | Optionally swap to `JointPositionActionCfg` for the gripper joint with `scale=0.875` (half the open range), but only after the binary version converges; binary is simpler for the first run |

---

## 7. Day-by-day execution plan

| Day | Goal | Deliverable |
|---|---|---|
| 1 | Scaffold the `tasks/pickplace/` module from `tasks/lift/`; register gym IDs; scene renders (gray table + 2 cm cube + wrist camera + ee_frame) | `zero_agent --task Isaac-SO-ARM101-PickPlace-Bowl-Play-v0 --enable_cameras` opens viewport with arm + table + block + wrist cam |
| 2 | Wire bowl `UniformPoseCommandCfg`, observations (incl. `wrist_rgb`, `ee_proj_xy`), action (lift-pattern), basic rewards (reach + transport from upstream + place stub), terminations | `random_agent` runs without crash; reward and obs values look sane in TB |
| 3 | **State-only training** (replace `wrist_rgb` with `object_position_in_robot_root_frame` in policy obs); confirm reward shaping works | ≥ 80 % success in sim play after ≤ 1500 PPO iterations |
| 4 | Add wrist `CameraCfg`, CNN actor, asymmetric critic; full DR enabled; **vision training** | First long run kicked off (overnight, 5000 iterations) |
| 5 | Iterate on DR ranges based on per-stage success; add aux block-xy regression head if encoder is collapsing | ≥ 60 % success in sim with full DR; checkpoint saved |
| 6 | Calibrate real wrist cam, write `so101_io.py`, run §5.5 diagnostic mode | Sim and real wrist frames look comparable; `ee_proj_xy` agrees within 5 mm |
| 7 | First real-robot rollouts; iterate on whichever DR axis was too narrow | Some non-zero successes on hardware |
| 8 | Tune (DR widening, retrain if needed); run 5 evaluation rollouts; record video | Submission-ready policy + video |

If day 3 doesn't produce a good state-only policy, **don't move to vision** — the underlying MDP needs fixing first. The most common cause is reward-stage gating; the second most common is action scale being too aggressive (arm flies past targets).

---

## 8. Open decisions

These are the things still TBD; pin them down once we hit them:

1. **Camera resolution** — start at 96 × 96, bump to 128² if grasp precision suffers.
2. **CNN vs frozen encoder** — start with end-to-end CNN; switch to R3M / DINOv2-small if we plateau.
3. **Auxiliary block-xy regression head** — recommended; turn on from the start of vision training.
4. **Gripper open/close action representation** — start with `BinaryJointPositionActionCfg` (matches the upstream lift pattern). Switch to a continuous `JointPositionActionCfg` on the gripper joint only if the binary thresholding becomes a bottleneck.
5. **Action `scale` for arm joints** — start with `0.5` (lift-task default). If the policy can't reach far block locations from home, bump to `1.0` before considering true delta semantics.
6. **Reachable workspace box** — measure once on the real SO-ARM101 (sweep manually, log ee_xyz extremes via the IO layer), then narrow the sim sampling to ~90 % of that.
7. **Whether to give the actor 1 frame or a 3-frame stack** — start with 3-frame stack at 50 Hz (cheap, helps depth).
8. **Episode length** — 6.0 s (300 steps @ 50 Hz) is the planned default; bump if the policy needs more time during search.
9. **Whether to use `is_grasped` as an explicit policy obs** — leave it privileged-only; the policy can infer it from gripper command + visual evidence.
10. **Camera type** — `CameraCfg` (per-env, simpler) vs `TiledCameraCfg` (batched render, faster with thousands of envs). Default to `TiledCameraCfg` if we hit render bottleneck at 2048 envs.

---

## 9. References

- Project doc: `Project 3_ Reinforcement Learning – Final Details.pdf`
- Daily Isaac Lab run instructions: `RUNNING.md`
- Upstream task we fork from: `isaac_so_arm101/src/isaac_so_arm101/tasks/lift/`
- Robot config: `isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/so_arm101.py`
- URDF (joint limits, link names): `isaac_so_arm101/src/isaac_so_arm101/robots/trs_so101/urdf/so_arm101.urdf`
- SO-ARM101 in Isaac Lab (upstream of `isaac_so_arm101`): <https://github.com/MuammerBay/isaac_so_arm101>
- SO-ARM101 in MuJoCo (alternative for quick-iteration sandbox): <https://github.com/RobotControlStack/robot-control-stack>
- SO-101 hardware tutorial: SO-101 (linked from project PDF)
- Asymmetric A-C in `rsl_rl` 3.0: see `rsl_rl.modules.ActorCritic` `obs_groups` argument.
