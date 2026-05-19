# Eval-1 BC — Real-Robot Deploy

```bash
# 1. clone the repo (Eval-1 lives on the `rui` branch)
git clone -b rui https://github.com/RuiZhou-cn/robot-learning.git project3 && cd project3

# 2. one-time install (conda env + CPU PyTorch + numpy + opencv + lerobot + verify)
bash bc/setup_inference_pc.sh

# 3. download the trained BC run from Google Drive into bc/runs/
#    https://drive.google.com/file/d/1fnlNFXiMqMZsxhbM1DPW05M7xisu9a9Q/view?usp=sharing
mkdir -p bc/runs && cd bc/runs
pip install --quiet gdown
gdown "https://drive.google.com/uc?id=1fnlNFXiMqMZsxhbM1DPW05M7xisu9a9Q" -O bc_eval1_v2.zip
unzip -q bc_eval1_v2.zip && rm bc_eval1_v2.zip
cd ../..

# 4. run closed-loop BC on the real arm — only the bowl xy (metres, base frame).
conda activate so_arm
python -m bc.deploy_real --bowl-xy 0.20,-0.05
```

Host pre-req the setup script does **not** install: miniconda3 with `conda` on `PATH`.
