"""Mirror a real SO-ARM101 into Isaac Sim from UDP joint packets."""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Mirror real SO-ARM101 joint positions in Isaac Sim.")
parser.add_argument("--task", type=str, default="Isaac-SO-ARM101-PickPlace-Bowl-Play-v0")
parser.add_argument("--listen-host", type=str, default="127.0.0.1")
parser.add_argument("--udp-port", type=int, default=5005)
parser.add_argument("--fps", type=int, default=50)
parser.add_argument("--timeout", type=float, default=2.0, help="Warn if no packet arrives for this many seconds.")
parser.add_argument("--out", type=str, default=None, help="Optional JSONL log path.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaac_so_arm101.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.driver import JOINT_NAMES  # noqa: E402


def _joint_indices(robot) -> list[int]:
    indices: list[int] = []
    for name in JOINT_NAMES:
        found = robot.find_joints(name)[0]
        if not found:
            raise RuntimeError(f"joint {name!r} not found; available joints: {robot.joint_names}")
        indices.append(int(found[0]))
    return indices


def _recv_latest(sock: socket.socket, current: np.ndarray | None) -> tuple[np.ndarray | None, dict | None]:
    latest_msg = None
    while True:
        try:
            payload, _ = sock.recvfrom(8192)
        except BlockingIOError:
            break
        latest_msg = json.loads(payload.decode("utf-8"))
    if latest_msg is None:
        return current, None
    q = np.asarray(latest_msg["q_sim_rad"], dtype=np.float32)
    if q.shape != (6,):
        raise RuntimeError(f"expected q_sim_rad shape (6,), got {q.shape}")
    return q, latest_msg


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    inner = env.unwrapped
    robot = inner.scene["robot"]
    joint_ids = _joint_indices(robot)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args_cli.listen_host, args_cli.udp_port))
    sock.setblocking(False)
    print(f"[teleop_sim_mirror] listening on {args_cli.listen_host}:{args_cli.udp_port}")

    log_f = open(args_cli.out, "w") if args_cli.out else None
    q_latest: np.ndarray | None = None
    last_packet_t = 0.0
    last_warn_t = 0.0
    dt = 1.0 / float(args_cli.fps)
    zero_action = torch.zeros((1, 6), dtype=torch.float32, device=inner.device)

    try:
        next_tick = time.perf_counter()
        step = 0
        while simulation_app.is_running():
            q_latest, msg = _recv_latest(sock, q_latest)
            now = time.perf_counter()
            if msg is not None:
                last_packet_t = now

            with torch.inference_mode():
                env.step(zero_action)

            if q_latest is not None:
                with torch.inference_mode():
                    q_full = robot.data.joint_pos.clone()
                    qd_full = torch.zeros_like(robot.data.joint_vel)
                    q_tensor = torch.as_tensor(q_latest, dtype=torch.float32, device=inner.device)
                    q_full[0, joint_ids] = q_tensor
                    robot.write_joint_state_to_sim(q_full, qd_full)
                    inner.scene.write_data_to_sim()

            if log_f is not None and q_latest is not None:
                q_sim = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy().astype(np.float32)
                row = {
                    "step": step,
                    "wall_time": time.time(),
                    "source_step": None if msg is None else msg.get("step"),
                    "q_received_sim_rad": q_latest.tolist(),
                    "q_sim_rad": q_sim.tolist(),
                    "joint_positions": dict(zip(JOINT_NAMES, q_sim.tolist())),
                }
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()

            if q_latest is None and now - last_warn_t > args_cli.timeout:
                print("[teleop_sim_mirror] waiting for real-arm UDP packets...")
                last_warn_t = now
            elif q_latest is not None and now - last_packet_t > args_cli.timeout and now - last_warn_t > args_cli.timeout:
                print(f"[teleop_sim_mirror] WARNING: no packet for {now - last_packet_t:.1f}s")
                last_warn_t = now

            step += 1
            next_tick += dt
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()
    finally:
        if log_f is not None:
            log_f.close()
        sock.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
