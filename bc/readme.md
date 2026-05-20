# Eval-1 — Real-Robot Deploy

## Step 0 — host pre-req

```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
bash /tmp/mc.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
exec bash
```

## Step 1 — clone

```bash
git clone -b rui https://github.com/RuiZhou-cn/robot-learning.git project3
cd project3
```

## Step 2 — install

```bash
bash bc/setup_inference_pc.sh
```

## Step 3 — checkpoint

Drive: <https://drive.google.com/file/d/1ixXvEX-JmKj9aODik-4Mw9UqJ0yKi_cp/view?usp=sharing>

```bash
mkdir -p bc/runs/deploy
pip install --quiet gdown
gdown "https://drive.google.com/uc?id=1ixXvEX-JmKj9aODik-4Mw9UqJ0yKi_cp" -O bc/runs/deploy/model.pt
```

## Step 4 — dry-run (no hardware)

```bash
conda activate so_arm
python -m bc.deploy_real --bowl-xy 0.30,0.0 --dry-run
```

## Step 5 — real-arm rollout

```bash
python -m bc.deploy_real --bowl-xy 0.30,0.0
```

## Step 6 — sim↔real gap dump (optional)

To diagnose the sim-to-real gap, run the same Eval-1 PPO checkpoint in both stacks with `--debug-dump` and compare the resulting folders. Both sides emit `step_XXXX.png` (RGB | mask composite, 72×256), a `log.jsonl` row per step (`state`, `action`, `q_sim_rad`, `ee_xy`, `target_sim_rad`), and `meta.json`.

Pick a fixed `--bowl-xy` so the rollouts are comparable.

```bash
# --- Real (inference PC, conda env so_arm) ---------------------------------
conda activate so_arm
python -m bc.deploy_real --bowl-xy 0.20,-0.05 --debug-dump
# → bc/runs/deploy/debug/<timestamp>/
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

# Other useful play-side flags:
#   --dump-steps 250          # default = 250 (5 s @ 50 Hz, matches real episode)
#   --dump-out <abs-path>     # override default location
#   --checkpoint /abs/path/to/model_1000.pt   # pin a specific iter (absolute path!)
```

The sim dump also includes critic-only ground truth (`block_xyz_gt`, `block_to_bowl_xy_gt`, `is_grasped_gt`) per row so you can pin down whether a divergence is upstream of the policy (image / state distribution) or downstream (action execution).

The two folders share filenames and the jsonl schema (real dump simply omits the `*_gt` keys), so a row-by-row diff is direct. To compare visually, e.g.:

```bash
for f in step_*.png; do
  montage -tile 1x2 -geometry +0+0 \
    sim/<sim-stamp>/$f real/<real-stamp>/$f compare_$f
done
```
