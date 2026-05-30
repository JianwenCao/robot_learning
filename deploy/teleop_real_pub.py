"""Publish real SO-ARM101 joint positions for Isaac Sim mirroring.

Run this in the real-robot LeRobot environment. The matching Isaac-side
receiver is ``isaac_so_arm101.scripts.teleop_sim_mirror``.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.driver import JOINT_NAMES, _gripper_sim_rad_from_pct


def _is_leader(robot_id: str) -> bool:
    return robot_id.endswith("-leader") or robot_id == "leader"


def _read_q_sim_rad(readings: dict[str, float]) -> np.ndarray:
    q = np.empty(6, dtype=np.float32)
    for i, name in enumerate(JOINT_NAMES[:5]):
        q[i] = float(readings[f"{name}.pos"]) * (np.pi / 180.0)
    q[5] = _gripper_sim_rad_from_pct(float(readings["gripper.pos"]))
    return q


def _connect_real_device(args: argparse.Namespace):
    if _is_leader(args.robot_id):
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        device = SO101Leader(SO101LeaderConfig(port=args.port, id=args.robot_id))
        read_fn = device.get_action
    else:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        device = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id))
        read_fn = device.get_observation

    device.connect(calibrate=args.calibrate)
    return device, read_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish real SO-ARM101 joints over UDP for Isaac mirroring.")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="LeRobot serial port.")
    parser.add_argument("--robot-id", type=str, default="eva-leader", help="LeRobot calibration id.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="UDP receiver host.")
    parser.add_argument("--udp-port", type=int, default=5005, help="UDP receiver port.")
    parser.add_argument("--fps", type=int, default=50, help="Publish frequency.")
    parser.add_argument("--calibrate", action="store_true", help="Allow LeRobot calibration if needed.")
    parser.add_argument(
        "--disable-torque",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable servo torque after connecting so the arm can be moved by hand.",
    )
    parser.add_argument("--out", type=str, default=None, help="Optional JSONL log path.")
    args = parser.parse_args()

    robot = None
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    log_f = open(args.out, "w") if args.out else None

    try:
        print(f"[teleop_real_pub] connecting {args.robot_id} on {args.port}")
        robot, read_real_joints = _connect_real_device(args)
        if args.disable_torque:
            robot.bus.disable_torque()
            print("[teleop_real_pub] torque disabled; move the arm by hand")
        else:
            print("[teleop_real_pub] torque left enabled; publishing encoder readings only")

        dt = 1.0 / float(args.fps)
        next_tick = time.perf_counter()
        step = 0
        dst = (args.host, args.udp_port)
        while True:
            q = _read_q_sim_rad(read_real_joints())
            msg = {
                "step": step,
                "wall_time": time.time(),
                "joint_names": JOINT_NAMES,
                "q_sim_rad": q.tolist(),
            }
            payload = json.dumps(msg, separators=(",", ":")).encode("utf-8")
            sock.sendto(payload, dst)
            if log_f is not None:
                log_f.write(json.dumps(msg) + "\n")
                log_f.flush()

            if step % max(args.fps, 1) == 0:
                print(f"[teleop_real_pub] step={step} q={np.round(q, 3).tolist()}")
            step += 1

            next_tick += dt
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("\n[teleop_real_pub] stopped")
    finally:
        if log_f is not None:
            log_f.close()
        sock.close()
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    main()
