# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-shot probe: print the SO-ARM101 end-effector world pose at home.

Used by the grasped-init bootstrap (mdp.events.init_block_in_gripper) to
know *where* to teleport the block so it lands inside the jaws. The
``FrameTransformer`` doesn't have a current-frame world pose available at
the moment a reset event fires, so we hardcode the home-pose offset by
running this probe once.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaac_so_arm101.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(
        "Isaac-SO-ARM101-PickPlace-Bowl-Play-v0", device=args_cli.device, num_envs=1
    )
    env = gym.make("Isaac-SO-ARM101-PickPlace-Bowl-Play-v0", cfg=env_cfg)
    env.reset()
    # Step once with zero action to settle physics + populate body poses.
    a = torch.zeros((1, 6), device=env.unwrapped.device)
    with torch.inference_mode():
        env.step(a)

    robot = env.unwrapped.scene["robot"]
    ee_frame = env.unwrapped.scene["ee_frame"]
    root_w = robot.data.root_pos_w[0]
    ee_w = ee_frame.data.target_pos_w[0, 0]
    ee_b = ee_w - root_w
    print(f"\n[probe] robot root world : {root_w.tolist()}")
    print(f"[probe] ee target world : {ee_w.tolist()}")
    print(f"[probe] ee in robot frame: {ee_b.tolist()}")
    # Also print gripper-link world (the body, not the offset target frame).
    gl_idx = robot.find_bodies("gripper_link")[0][0]
    gl_w = robot.data.body_pos_w[0, gl_idx]
    gl_b = gl_w - root_w
    print(f"[probe] gripper_link world: {gl_w.tolist()}")
    print(f"[probe] gripper_link  in robot: {gl_b.tolist()}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
