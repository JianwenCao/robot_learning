# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic grasp-physics test for the SO-ARM101 PickPlace-Bowl env.

Drives a single env with a hand-crafted action sequence and prints the
block's z-coordinate at each step so we can verify whether the SO-ARM101
gripper can capture and lift a 2 cm cube under PhysX. If a hand-crafted
trajectory can't lift the cube, no amount of reward shaping will let PPO
discover lifting either.

Usage::

    conda activate so_arm
    export OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y
    python -m isaac_so_arm101.scripts.verify_grasp --headless

The script exits with code 0 if any grasp attempt lifted the cube above
2.5 cm, else 1.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

# CLI ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Deterministic grasp test for PickPlace-Bowl.")
parser.add_argument("--episodes", type=int, default=3, help="Number of grasp attempts.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim ---------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaac_so_arm101.tasks  # noqa: F401, E402  – registers gym IDs
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


# Action layout (6-D):
#   [0] shoulder_pan, [1] shoulder_lift, [2] elbow_flex,
#   [3] wrist_flex,   [4] wrist_roll,    [5] gripper_binary
#
# Action is normalized to [-1, 1] and the env applies the lift-task pattern
# ``target_q = default_q + scale * action`` for the arm joints.
# Gripper is binary: action[5] > 0 → open (0.5 rad), else close (0.0 rad).
def scripted_actions(t: int) -> list[float]:
    """Return action vector for global timestep t.

    Phase 0 (t < 30):  hold home pose, jaws open  (let physics settle).
    Phase 1 (30..60):  descend (lower shoulder_lift, push wrist down).
    Phase 2 (60..70):  close jaws (action[5] = -1).
    Phase 3 (70..150): lift back up (raise shoulder_lift, raise wrist).
    """
    if t < 30:
        return [0, 0, 0, 0, 0, +1.0]  # jaws open
    if t < 60:
        # Descend toward block: shoulder_lift down, elbow_flex down,
        # wrist_flex stays around 1.57 (gripper points down).
        return [0, -0.8, -0.6, 0, 0, +1.0]
    if t < 70:
        # Hold pose, command jaws to close.
        return [0, -0.8, -0.6, 0, 0, -1.0]
    if t < 150:
        # Ascend with jaws closed.
        return [0, +0.5, +0.3, 0, 0, -1.0]
    return [0, 0, 0, 0, 0, -1.0]  # default after sequence


def main() -> int:
    env_cfg = parse_env_cfg(
        "Isaac-SO-ARM101-PickPlace-Bowl-Play-v0",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    env = gym.make("Isaac-SO-ARM101-PickPlace-Bowl-Play-v0", cfg=env_cfg)
    print(f"[INFO] Action space: {env.action_space}")

    n_succ = 0
    for ep in range(args_cli.episodes):
        env.reset()
        # Read initial block z and bowl xy for context
        obj = env.unwrapped.scene["object"]
        block0 = obj.data.root_pos_w[0].clone()
        print(f"\n=== Episode {ep + 1}/{args_cli.episodes} ===")
        print(f"  block init  (world): {block0.tolist()}")

        max_z = float(block0[2])
        for t in range(160):
            a = scripted_actions(t)
            actions = torch.tensor([a], device=env.unwrapped.device, dtype=torch.float32)
            with torch.inference_mode():
                obs, rew, term, trunc, info = env.step(actions)

            block_z = float(obj.data.root_pos_w[0, 2])
            max_z = max(max_z, block_z)
            if t in (29, 59, 69, 100, 130, 159):
                print(f"  t={t:3d}  block_z={block_z:.4f}  max_so_far={max_z:.4f}")

            if term[0] or trunc[0]:
                # Episode ended — record outcome and break
                print(f"  episode ended at t={t}: term={term[0].item()} trunc={trunc[0].item()}")
                break

        ok = max_z > 0.025
        print(f"  -> max block_z = {max_z:.4f} m  ({'GRASP OK' if ok else 'NO LIFT'})")
        if ok:
            n_succ += 1

    env.close()
    print(f"\n=== Result: {n_succ}/{args_cli.episodes} episodes lifted the cube above 2.5 cm ===")
    return 0 if n_succ > 0 else 1


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    sys.exit(rc)
