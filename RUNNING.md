# Running `isaac_so_arm101` (conda edition)

This guide is for the conda-based setup of [MuammerBay/isaac_so_arm101](https://github.com/MuammerBay/isaac_so_arm101) on this workstation. Upstream uses `uv`; we replaced it with `pip` inside a conda env. The repo lives at `./isaac_so_arm101/` and is installed editable, so changes under `isaac_so_arm101/src/` take effect without reinstalling.

## TL;DR

```bash
conda activate so_arm
export OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y

list_envs                                              # list 8 SO-ARM tasks
zero_agent   --task Isaac-SO-ARM100-Reach-Play-v0      # send zero actions
random_agent --task Isaac-SO-ARM100-Reach-Play-v0      # send random actions
train        --task Isaac-SO-ARM100-Reach-v0 --headless
play         --task Isaac-SO-ARM100-Reach-Play-v0
```

## Environment

| Component | Version |
|---|---|
| conda env | `so_arm` |
| Python | 3.11.15 |
| PyTorch | 2.7.0+cu128 |
| torchvision | 0.22.0+cu128 |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab | 2.3.0 |
| RSL-RL | 3.0.1 |
| skrl | 2.0.0 |
| stable-baselines3 | 2.8.0 |
| isaac-so-arm101 | 1.2.0 (editable, `./isaac_so_arm101`) |

GPU target: NVIDIA RTX 5090 (Blackwell, sm_120) — requires the `cu128` PyTorch wheel build. Driver 580+ is installed on this box.

## One-time setup (already done — kept here for reproducibility)

```bash
# 1. clone
git clone https://github.com/MuammerBay/isaac_so_arm101.git
cd isaac_so_arm101

# 2. conda env
conda create -y -n so_arm python=3.11 pip
conda activate so_arm

# 3. PyTorch (cu128 — required for RTX 5090)
pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 4. workaround for flatdict 4.0.1 (it imports pkg_resources at build,
#    which setuptools>=81 dropped)
pip install "setuptools<81" wheel
pip install "flatdict==4.0.1" --no-build-isolation

# 5. Isaac Lab + Isaac Sim from NVIDIA's index
pip install "isaaclab[all,isaacsim]==2.3.0" \
    --extra-index-url https://pypi.nvidia.com/ \
    --extra-index-url https://download.pytorch.org/whl/cu128

# 6. local package (editable). Build backend is uv_build, published on
#    PyPI as uv-build, so plain pip uses it during build isolation.
pip install -e .
```

## Daily use

### 1. Activate the env

```bash
conda activate so_arm
```

### 2. Accept the Omniverse EULA non-interactively

Isaac Sim 5.1 prompts for the EULA on first launch and dies with `Unable to bootstrap inner kit kernel: EOF when reading a line` if stdin is not a TTY. Two env vars silence both prompts:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y
```

To make this permanent for the env only (recommended), persist them with `conda env config vars`:

```bash
conda env config vars set OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y -n so_arm
conda activate so_arm   # reactivate to pick them up
```

### 3. List available tasks

```bash
list_envs
```

Expect 10 tasks — Reach, Lift, and PickPlace, with the ARM100/ARM101 and training/Play variants applicable to each:

| # | Task ID |
|---|---|
| 1 | `Isaac-SO-ARM100-Lift-Cube-v0` |
| 2 | `Isaac-SO-ARM100-Lift-Cube-Play-v0` |
| 3 | `Isaac-SO-ARM101-Lift-Cube-v0` |
| 4 | `Isaac-SO-ARM101-Lift-Cube-Play-v0` |
| 5 | `Isaac-SO-ARM101-PickPlace-Bowl-v0` |
| 6 | `Isaac-SO-ARM101-PickPlace-Bowl-Play-v0` |
| 7 | `Isaac-SO-ARM100-Reach-v0` |
| 8 | `Isaac-SO-ARM100-Reach-Play-v0` |
| 9 | `Isaac-SO-ARM101-Reach-v0` |
| 10 | `Isaac-SO-ARM101-Reach-Play-v0` |

`PickPlace-Bowl` is the SO-ARM101 single-object pick-and-place task added for Eval 1 of the project; see `EVAL1_PLAN.md` for design and `tasks/pickplace/` for the source. **It is vision-based** — a wrist-mounted `TiledCamera` is parented to `gripper_link` and ``wrist_rgb`` is a deployable obs group. Always pass `--enable_cameras` to `train`, `play`, `zero_agent`, `random_agent`, or any other entry point that instantiates this env; without the flag the camera fails to initialize and the env raises ``RuntimeError: A camera was spawned without the --enable_cameras flag`` on the first step.

> ⚠️ **The upstream README task names are stale.** It writes `SO-ARM100-Reach-Play-v0`; the registered gym IDs are prefixed with `Isaac-`. Use the table above (or `list_envs`) as the source of truth — `--task SO-ARM100-Reach-Play-v0` will fail with a gym registration error.

### 4. Sanity-check with dummy agents

`-Play-v0` variants spawn fewer envs (good for visual inspection); the non-Play variants are the heavy training versions.

```bash
zero_agent   --task Isaac-SO-ARM100-Reach-Play-v0     # arm sits still
random_agent --task Isaac-SO-ARM100-Reach-Play-v0     # arm flails
```

A native Isaac Sim viewport opens. Close it with the window's X or Ctrl-C in the terminal.

### 5. Train an RL policy (RSL-RL PPO)

```bash
train --task Isaac-SO-ARM100-Reach-v0 --headless
```

`--headless` disables the viewport, which is faster and works over SSH. Drop it if you want to watch training. Useful flags (forwarded to RSL-RL / Isaac Lab):

- `--num_envs N` — override parallel env count
- `--seed S` — seed
- `--max_iterations N` — stop after N PPO iterations
- `--video --video_length 200 --video_interval 2000` — record rollouts to MP4
- `--logger tensorboard|wandb|neptune` — defaults to TensorBoard

Logs and checkpoints land under `<CWD>/logs/rsl_rl/<experiment_name>/<run>/`. The training script uses a relative `logs/rsl_rl/...` path, so the destination is whatever directory you ran `train` from. Run from the project root and they end up at `project3/logs/rsl_rl/...`:

```bash
cd /home/rui/Projects/Course_Code/Robot_Learning/project3
tensorboard --logdir logs/rsl_rl
```

(Earlier versions of this doc pointed at `isaac_so_arm101/logs/rsl_rl` — that path is wrong unless you `cd isaac_so_arm101` before launching `train`.)

### 6. Evaluate a trained policy

```bash
play --task Isaac-SO-ARM100-Reach-Play-v0
```

By default `play` loads the latest checkpoint of the matching training task (`Isaac-SO-ARM100-Reach-v0`). Override with `--checkpoint /path/to/model_*.pt` or `--load_run <run_dir>`.

## Where things live

```
project3/
├── RUNNING.md                       ← this file
└── isaac_so_arm101/
    ├── pyproject.toml               (upstream, declares uv_build backend)
    ├── src/isaac_so_arm101/
    │   ├── robots/{trs_so100,trs_so101}/   # USD/URDF + Articulation cfgs
    │   ├── tasks/{reach,lift}/             # ManagerBasedRLEnvCfg subclasses
    │   └── scripts/
    │       ├── list_envs.py
    │       ├── zero_agent.py
    │       ├── random_agent.py
    │       └── rsl_rl/{train.py,play.py,cli_args.py}
    └── logs/rsl_rl/                  # created on first train run
```

Because the package is editable, edits to anything under `isaac_so_arm101/src/` take effect on the next run — no reinstall needed. New entry points in `[project.scripts]` do require `pip install -e .` again.

## Troubleshooting

**`Unable to bootstrap inner kit kernel: EOF when reading a line`** — you forgot the EULA env vars. Re-export `OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y` (see step 2).

**`gym.error.UnregisteredEnv: SO-ARM100-Reach-Play-v0`** — you used the README's task name. Add the `Isaac-` prefix.

**`TypeError: post_quit(): incompatible function arguments` at the very end of `list_envs`** — known cosmetic shutdown bug in Isaac Sim 5.1; it fires *after* the work completes, so output is fine. Ignore.

**`ModuleNotFoundError: No module named 'pkg_resources'` while pip-installing something** — same flatdict-style problem. Run `pip install "setuptools<81" wheel` and retry the install with `--no-build-isolation`.

**`Skipping unsupported non-NVIDIA GPU: AMD Ryzen 9 9950X ...`** — Isaac Sim's Vulkan probe lists the integrated AMD GPU in the Ryzen iGPU and skips it. Harmless; rendering uses the RTX 5090.

**`CPU performance profile is set to powersave`** — also harmless, but set `cpupower frequency-set -g performance` (needs sudo) for faster training if you care.

**`packaging 23.0` warning from pip's resolver** — Isaac Lab 2.3.0 pins `packaging==23.0`. Doesn't affect runtime; only matters if you build new wheels in the env.

**Training is slow / OOM** — drop `--num_envs` (default for the Reach task is 4096). 1024 still fits comfortably in 32 GB.

## Re-creating the env from scratch

```bash
conda env remove -n so_arm
# then redo the "One-time setup" section above
```

Total install time on this box was about 6–8 minutes, dominated by the ~5 GB Isaac Sim wheel set from `pypi.nvidia.com`.
