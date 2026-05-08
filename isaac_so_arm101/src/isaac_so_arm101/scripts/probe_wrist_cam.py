# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-shot probe: dump a wrist-cam frame at home pose to PNG.

Use this to visually verify the wrist-cam ``OffsetCfg`` on ``gripper_link``
matches the real WOWROBO mount (EVAL1_PLAN §3.5 / §5.1). Compare the
output PNG against a frame from the real wrist USB cam taken at the same
home pose — they should agree on FOV, table extent, and finger geometry
in the lower frame edge.

Run::

    conda activate so_arm
    cd /home/rui/Projects/Course_Code/Robot_Learning/project3
    python isaac_so_arm101/src/isaac_so_arm101/scripts/probe_wrist_cam.py \\
        --enable_cameras --headless

PNG lands at ``outputs/wrist_cam_probe.png``. Drop ``--headless`` to also
keep the viewport open (handy for picking the camera in the viewport's
camera dropdown and walking around the prim's frustum).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--out",
    default="outputs/wrist_cam_probe.png",
    help="PNG path (relative paths resolved against the project root).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v3 as iio  # noqa: E402
import torch  # noqa: E402

import isaac_so_arm101.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(
        "Isaac-SO-ARM101-PickPlace-Bowl-Play-v0", device=args_cli.device, num_envs=1
    )
    env = gym.make("Isaac-SO-ARM101-PickPlace-Bowl-Play-v0", cfg=env_cfg)
    env.reset()
    # Step a few times so the renderer warms up and the cam buffer is
    # populated. The first frame after reset is sometimes black on the
    # TiledCamera path.
    a = torch.zeros((1, 6), device=env.unwrapped.device)
    with torch.inference_mode():
        for _ in range(3):
            env.step(a)

    cam = env.unwrapped.scene.sensors["wrist_cam"]
    rgb = cam.data.output["rgb"][0].clone().cpu()  # (H, W, 3) uint8 or float
    if rgb.dtype.is_floating_point:
        rgb = (rgb.clamp(0.0, 1.0) * 255).to(torch.uint8)

    out = Path(args_cli.out)
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out, rgb.numpy())
    print(f"[probe] wrote {rgb.shape} -> {out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
