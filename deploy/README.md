# Real-Robot Deploy (Eval-1 / Eval-2 / Eval-3)

PPO checkpoint (trained in `isaac_so_arm101/`) → closed-loop on a real SO-ARM101. No `rsl_rl` / `isaaclab` needed on this PC — the actors are reimplemented in `deploy/ppo_actor.py` (Eval-1: `PPOActor`, Eval-2/3: `PPOActorClutter`).

Two entry points share the same hardware + image stack:

* `python -m deploy.deploy_real …` — Eval-1 (single dex_cube).
* `python -m deploy.deploy_real_clutter …` — Eval-2 (single target colour) and Eval-3 (3 sub-goals, shared bowl). See [§Eval-2/3](#eval-23--clutter-deploy) below.

## Step 1 — clone

```bash
git clone -b rui https://github.com/RuiZhou-cn/robot-learning.git project3
cd project3
```

## Step 2 — install

```bash
bash deploy/setup_inference_pc.sh
```

## Step 3 — checkpoint

Download a known-good vision-PPO checkpoint to `deploy/runs/model.pt`:

```bash
pip install --quiet gdown
mkdir -p deploy/runs
gdown "https://drive.google.com/uc?id=1ixXvEX-JmKj9aODik-4Mw9UqJ0yKi_cp" -O deploy/runs/model.pt
```

## Step 4 — real-arm rollout

```bash
conda activate so_arm
python -m deploy.deploy_real --bowl-xy 0.30,0.0
```

### Mask source (`--mask-source`)

The 4-channel wrist tensor the policy expects is `RGB + binary block mask`. Two ways to produce the mask channel on the real arm:

| Flag | Speed | Robustness | When to use |
|---|---|---|---|
| `--mask-source florence` *(default)* | ~2–5 s/frame on CPU | Open-vocabulary segmentation via Florence-2 (`microsoft/Florence-2-base`) with a text prompt fixed in code (Eval-1 is single-cube). Robust to clutter, lighting changes, shadows | Production deploy. First run downloads ~1 GB of weights to `~/.cache/huggingface/`. |
| `--mask-source hsv` | ~ms/frame, 50 Hz control loop | Brittle — `cv2.inRange` saturation gate + largest-CC pick latches onto the wrong blob whenever an in-FOV distractor (tool handle, cable, sticker) is more saturated than the cube | Clean scenes with only the cube on a near-white table; sim-vs-real visual gap A/B tests. Tune with `--hsv-low` / `--hsv-high`. |

The prompt for `florence` is hardcoded for Eval-1 (single cube, no target-color input). Color-aware prompts ("red cube", "blue cube", …) are an Eval-2 / Eval-3 concern where the task carries a target color. To swap in a different segmenter (GroundedSAM, SAM-3, CLIPSeg, …) implement the `Detector` protocol in `deploy/cube_detector.py` and register it in `build_detector`. The interface is one method: `mask(rgb_hwc_uint8) -> (H, W) float32 in {0, 1}`.

## Step 5 — sim↔real gap dump (optional)

To diagnose the sim-to-real gap, run the same Eval-1 PPO checkpoint in both stacks with `--debug-dump` and compare the resulting folders. Both sides emit `step_XXXX.png` (RGB | mask composite, 72×256), a `log.jsonl` row per step (`state`, `action`, `q_sim_rad`, `ee_xy`, `target_sim_rad`), and `meta.json`. Pick a fixed `--bowl-xy` so the rollouts are comparable.

```bash
# --- Real (inference PC, conda env so_arm) ---------------------------------
conda activate so_arm
python -m deploy.deploy_real --bowl-xy 0.20,-0.05 --debug-dump
# → deploy/runs/debug/<timestamp>/
```

```bash
# --- Sim (training PC, uv env inside isaac_so_arm101/) ---------------------
cd isaac_so_arm101
# Use --load_run alone; the runner cfg's `load_checkpoint` glob picks the
# highest-iter model_*.pt. If you need a specific iteration, pass --checkpoint
# with the FULL absolute path (bare filenames are not resolved).
uv run play --task Isaac-SO-ARM101-PickPlace-Bowl-Play-v0 \
    --load_run <run-timestamp> \
    --enable_cameras --debug-dump --dump-bowl-xy 0.20,-0.05
# → logs/rsl_rl/pickplace_bowl/<run-timestamp>/debug_dump/<timestamp>/
```

The sim dump also includes critic-only ground truth (`block_xyz_gt`, `block_to_bowl_xy_gt`, `is_grasped_gt`) per row so you can pin down whether a divergence is upstream of the policy (image / state distribution) or downstream (action execution).

The two folders share filenames and the jsonl schema (real dump simply omits the `*_gt` keys), so a row-by-row diff is direct.

## Eval-2/3 — clutter deploy

Same hardware loop, different policy class and observation contract. `deploy/deploy_real_clutter.py` consumes a Stage-3 vision PPO checkpoint from the `clutterpickplace` (Eval 2) or `eval3clutter` (Eval 3) task and routes the target colour through both the policy's FiLM head and Florence's text prompt.

Drop a Stage-3 checkpoint at one of the search paths (or pass `--ckpt`):

```bash
deploy/runs/clutter_model.pt   # Eval-2
deploy/runs/eval3_model.pt     # Eval-3
```

### Eval-2 — single target

```bash
conda activate so_arm
python -m deploy.deploy_real_clutter --mode eval2 \
    --target-color red --bowl-xy 0.22,0.0
```

What happens:

* Loads `PPOActorClutter` (frozen ResNet-18 layer1-2 + FiLM + spatial-softmax + MLP[256,128,64]).
* Builds the wrist tensor `(4, 72, 128)` with channel 3 from Florence-2 prompted `"red cube"`.
* State vector is 31-D: policy(25) ++ target_color_onehot(6). The trailing 6 dims are sliced inside the actor and fed to the FiLM head.
* One 5-s rollout, then exits.

### Eval-3 — 3 sub-goals, shared bowl

```bash
conda activate so_arm
python -m deploy.deploy_real_clutter --mode eval3 \
    --colors red,blue,yellow --bowl-xy 0.22,0.0 \
    --release-detect timed --subgoal-steps 250
```

What happens per sub-goal `k`:

* Calls `detector.set_prompt(f"{colors[k]} cube")` — cheap, no model reload.
* Overwrites the policy state's trailing 6-D one-hot with `onehot(colors[k])`.
* Runs `--subgoal-steps` ticks (default 5 s @ 50 Hz).
* If `--release-detect manual`, also accepts a non-blocking `<enter>` to advance early.

A `vision`-based release detector (HSV-match the target cube's colour against the bowl region) is the spec's third suggestion; not implemented — add a new class behind the flag if you want it.

### Mask source for Eval-2/3

`--mask-source` works the same as Eval-1's table above, with one difference: the Florence prompt is **derived from the current target colour** (not fixed in code). The HSV fallback (`--mask-source hsv`) does **not** discriminate by colour and will produce a single-blob mask of "the most-saturated thing in view" — useful only as an A/B for Eval-2 with one cube on the table; do not use it for the actual multi-cube eval.
