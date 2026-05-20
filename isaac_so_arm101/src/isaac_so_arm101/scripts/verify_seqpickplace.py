# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless smoke test for the Eval-3 sequential pick-and-place env.

Loads ``Isaac-SO-ARM101-SeqPickPlace-Play-v0``, resets, inspects per-env
state buffers and the seq_goal command tensor, steps a few ticks with
zero actions to confirm no crashes, then resets a second time to verify
re-randomization.

Run::

    conda activate so_arm
    export OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y
    python -m isaac_so_arm101.scripts.verify_seqpickplace --headless --enable_cameras

Exits 0 on full PASS, 1 if any check fails.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test for Eval-3 SeqPickPlace env.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=20, help="Zero-action steps after reset.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-SO-ARM101-SeqPickPlace-Play-v0",
    help="Task id (Play variant by default for fast 16-env build).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaac_so_arm101.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from isaac_so_arm101.tasks.seqpickplace.mdp.events import (  # noqa: E402
    COLOR_NAMES,
    N_ACTIVE_BLOCKS,
    N_BOWLS,
    N_GOAL_STEPS,
    NUM_COLORS,
    BLOCK_COLORS,
    HIDDEN_PARK_XY,
)


def _ok(name: str, cond: bool, detail: str = "") -> bool:
    marker = "PASS" if cond else "FAIL"
    print(f"  [{marker}] {name}" + (f" — {detail}" if detail else ""))
    return cond


def main() -> int:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=cfg)
    inner = env.unwrapped
    N = args_cli.num_envs

    print(f"[info] task={args_cli.task}, num_envs={N}, device={inner.device}")
    print(f"[info] obs space: {env.observation_space}")
    print(f"[info] action space: {env.action_space}")

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        if not _ok(name, cond, detail):
            failures.append(name)

    # ---- reset ----------------------------------------------------------
    obs, _ = env.reset()

    # 1. Binding A: cube colors baked correctly
    print("\n[1] Scene-color binding (binding A)")
    for name in COLOR_NAMES:
        try:
            cube_cfg = inner.scene.cfg.__dict__[f"cube_{name}"]
            baked = tuple(float(c) for c in cube_cfg.spawn.visual_material.diffuse_color)
            expected = tuple(float(c) for c in BLOCK_COLORS[name])
            check(
                f"cube_{name}.diffuse_color matches BLOCK_COLORS",
                baked == expected,
                f"baked={baked}, expected={expected}",
            )
        except Exception as e:
            check(f"cube_{name} cfg lookup", False, f"err: {e}")

    # 2. Env-state buffer shapes (binding B)
    print("\n[2] Sequence-schedule buffers (binding B)")
    check(
        "_seq_active_indices shape",
        inner._seq_active_indices.shape == (N, N_ACTIVE_BLOCKS),
        f"got {tuple(inner._seq_active_indices.shape)}",
    )
    check(
        "_seq_goal_color_pos shape",
        inner._seq_goal_color_pos.shape == (N, N_GOAL_STEPS),
        f"got {tuple(inner._seq_goal_color_pos.shape)}",
    )
    check(
        "_seq_bowl_positions shape",
        inner._seq_bowl_positions.shape == (N, N_BOWLS, 2),
        f"got {tuple(inner._seq_bowl_positions.shape)}",
    )
    check(
        "_target_cube_idx_per_step shape",
        inner._target_cube_idx_per_step.shape == (N, N_GOAL_STEPS),
        f"got {tuple(inner._target_cube_idx_per_step.shape)}",
    )
    check(
        "_seq_step_idx starts at 0",
        bool((inner._seq_step_idx == 0).all().item()),
    )

    # active set: 4 distinct palette indices per env
    active = inner._seq_active_indices
    distinct = all(int(active[i].unique().numel()) == N_ACTIVE_BLOCKS for i in range(N))
    check("active_indices distinct per env", distinct)
    check(
        "active_indices in palette range",
        bool(((active >= 0) & (active < NUM_COLORS)).all().item()),
    )

    # _target_cube_idx_per_step[e, s] == active[e, goal_color_pos[e, s]]
    derived = active.gather(1, inner._seq_goal_color_pos)
    check(
        "_target_cube_idx_per_step matches active.gather(goal_color_pos)",
        bool((derived == inner._target_cube_idx_per_step).all().item()),
    )

    # 3. Bowl position: in workspace
    print("\n[3] Bowl placement constraints")
    bp = inner._seq_bowl_positions
    bx = bp[..., 0]
    by = bp[..., 1]
    in_x = ((bx >= 0.10) & (bx <= 0.30)).all().item()
    in_y = ((by >= -0.18) & (by <= 0.18)).all().item()
    check("bowl x in [0.10, 0.30]", bool(in_x))
    check("bowl y in [-0.18, 0.18]", bool(in_y))

    # 4. Cubes: 4 in workspace (low z), 2 parked (off-table x)
    print("\n[4] Cube placement (4 active / 2 parked)")
    robot = inner.scene["robot"]
    root_xy_w = robot.data.root_pos_w[:, :2]
    for env_i in range(N):
        active_set = set(active[env_i].tolist())
        for k, name in enumerate(COLOR_NAMES):
            cube = inner.scene[f"cube_{name}"]
            pos_w = cube.data.root_pos_w[env_i, :3]
            local_xy = pos_w[:2] - root_xy_w[env_i]
            on_table = (
                0.10 <= float(local_xy[0]) <= 0.25
                and -0.12 <= float(local_xy[1]) <= 0.12
                and float(pos_w[2]) < 0.05
            )
            parked = float(local_xy[0]) < -0.30
            if k in active_set:
                if env_i == 0:
                    detail = f"env0 cube_{name} local_xy=({float(local_xy[0]):.3f},{float(local_xy[1]):.3f}) z={float(pos_w[2]):.3f}"
                else:
                    detail = ""
                check(f"env{env_i} cube_{name} (active) on table", on_table, detail)
            else:
                check(f"env{env_i} cube_{name} (parked) off table", parked)

    # 5. Command tensor (binding B → policy obs)
    print("\n[5] seq_goal command tensor")
    cmd_term = inner.command_manager.get_term("seq_goal")
    cmd = cmd_term.command  # (N, 11)
    check("command shape (N, 8)", cmd.shape == (N, NUM_COLORS + 2))
    onehot = cmd[:, :NUM_COLORS]
    check(
        "color one-hot sums to 1",
        bool((onehot.sum(dim=1) - 1.0).abs().lt(1e-5).all().item()),
    )
    check(
        "color one-hot argmax == current_target_color_idx",
        bool((onehot.argmax(dim=1) == cmd_term.current_target_color_idx()).all().item()),
    )
    bowl_xy = cmd[:, NUM_COLORS:NUM_COLORS + 2]
    expected_bowl = cmd_term.current_target_bowl_xy()
    check(
        "command bowl_xy matches current_target_bowl_xy()",
        torch.allclose(bowl_xy, expected_bowl, atol=1e-5),
    )
    step_oh = cmd[:, NUM_COLORS + 2:]
    check(
        "step one-hot argmax == 0 (fresh reset)",
        bool((step_oh.argmax(dim=1) == 0).all().item()),
    )
    # color one-hot index matches palette of current target
    expected_color = inner._target_cube_idx_per_step[:, 0]
    check(
        "color one-hot points at active[goal_color_pos[0]]",
        bool((onehot.argmax(dim=1) == expected_color).all().item()),
    )

    # 6. Step a few ticks with zero action — no crashes
    print(f"\n[6] Stepping {args_cli.steps} zero-action ticks")
    try:
        for _ in range(args_cli.steps):
            actions = torch.zeros(env.action_space.shape, device=inner.device)
            env.step(actions)
        check("zero-action stepping survives", True)
    except Exception as e:
        check("zero-action stepping survives", False, f"crashed: {e}")

    # 7. Re-randomization on next reset
    print("\n[7] Re-randomization on reset")
    old_active = inner._seq_active_indices.clone()
    old_bowls = inner._seq_bowl_positions.clone()
    env.reset()
    diff_active = (inner._seq_active_indices != old_active).any(dim=1)
    diff_bowls = (inner._seq_bowl_positions - old_bowls).abs().sum(dim=(1, 2)) > 1e-3
    check(
        "active set re-sampled for >=1 env",
        bool(diff_active.any().item()),
        f"changed in {int(diff_active.sum())}/{N} envs",
    )
    check(
        "bowl positions re-sampled for >=1 env",
        bool(diff_bowls.any().item()),
        f"changed in {int(diff_bowls.sum())}/{N} envs",
    )

    # ---- summary --------------------------------------------------------
    print("\n" + "=" * 60)
    if failures:
        print(f"FAIL — {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        env.close()
        return 1
    print("PASS — all checks succeeded.")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
