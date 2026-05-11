# Deployment Guide — Vision Student → Real SO-ARM101

How to set up the deploy host and run the trained student checkpoint on the physical SO-ARM101.

## 1. Environment setup

Same Ubuntu box as training is simplest — reuse the `so_arm` conda env. Isaac Lab is **not** needed at deploy time.

```bash
conda activate so_arm
pip install feetech-servo-sdk pyserial opencv-python pyyaml pin
# Depth Anything 3: clone + install per upstream (https://github.com/DepthAnything/Depth-Anything-V3).
# Pin a small variant (vits or vitb) so inference stays under ~30 ms at 128×72.
# SAM (cube mask): MobileSAM or SAM-HQ tiny variant for real-time use.
```

Linux serial + camera permissions (re-login after):

```bash
sudo usermod -a -G dialout,video $USER
```

## 2. Re-export the student checkpoint (one-off)

Isaac Lab's `export_policy_as_jit` only saves the student MLP head and silently drops the CNN encoder, so `logs/.../exported/policy.pt` is unusable on the real robot. Re-export bundles `student_cnn` + `student` MLP into a single TorchScript module that takes `(state, image)` and returns `action`:

```bash
python scripts/deploy/export_vision_student.py \
    --checkpoint logs/rsl_rl/pickplace_bowl_student/2026-05-10_23-38-19/model_1000.pt
# writes logs/.../exported/vision_policy.pt
```

Then write `deploy_meta.json` next to it (every value here must match sim — sources are in `joint_pos_env_cfg.py` and `pickplace_env_cfg.py`):

```json
{
  "joint_order":   ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"],
  "default_q":     [0.0, 0.0, 0.0, 1.57, 0.0, 0.5],
  "arm_action_scale":  0.5,
  "gripper_open_cmd":  0.5,
  "gripper_close_cmd": 0.0,
  "gripper_action_threshold": 0.0,
  "control_dt":    0.02,
  "image_size":    [72, 128],
  "image_channels": 5,
  "depth_clip_m":  0.5,
  "joint_lower":   [-1.91986,-1.74533,-1.69,-1.65806,-2.74385,-0.17453],
  "joint_upper":   [ 1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533],
  "ee_offset_in_gripper_link": [0.01, 0.0, -0.09]
}
```

## 3. Run a rollout

Plug the Feetech bus into `/dev/ttyUSB0`, the wrist USB cam into `cam_index 0`, place the bowl with its centre at the commanded `(x, y)`, drop the cube somewhere in the workspace, hand on the power switch:

```bash
python scripts/deploy/run_pickplace.py \
    --policy   logs/rsl_rl/pickplace_bowl_student/2026-05-10_23-38-19/exported/vision_policy.pt \
    --meta     logs/rsl_rl/pickplace_bowl_student/2026-05-10_23-38-19/exported/deploy_meta.json \
    --cam_intr camera_intrinsics.yaml \
    --bus_port /dev/ttyUSB0 --cam_index 0 \
    --bowl_xy 0.20 -0.05
```

`--bowl_xy` is in the robot base frame, in metres. Logs (qpos, action, raw wrist frames, processed 5-channel obs, success flag) land in `runs/<timestamp>/`.
