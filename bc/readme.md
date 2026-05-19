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
python -m bc.deploy_real --bowl-xy 0.20,-0.05 --dry-run
```

## Step 5 — real-arm rollout

```bash
python -m bc.deploy_real --bowl-xy 0.20,-0.05
python -m bc.deploy_real --bowl-xy 0.18,0.06
```
