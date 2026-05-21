"""Probe real SO-101 wrist_roll reachability without running a policy.

This script sends only ``wrist_roll.pos`` commands and prints raw readback
values. It intentionally does not use deploy.driver because that path wraps
the wrist_roll angle near the policy default and opens the wrist camera.
"""

from __future__ import annotations

import argparse
import math
import time


JOINT_NAME = "wrist_roll"
DEFAULT_TARGETS_RAD = [-2.8, -3.0, -3.2, -3.343, -3.5, -3.8, -4.0, -4.552000045776367]


def _parse_targets(text: str) -> list[float]:
    vals = []
    for item in text.split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    if not vals:
        raise argparse.ArgumentTypeError("target list is empty")
    return vals


def _read_wrist_rad(robot) -> float:
    obs = robot.get_observation()
    return float(obs[f"{JOINT_NAME}.pos"]) * math.pi / 180.0


def _send_wrist_rad(robot, target_rad: float) -> None:
    robot.send_action({f"{JOINT_NAME}.pos": float(target_rad) * 180.0 / math.pi})


def _move_to_target(
    robot,
    target_rad: float,
    *,
    fps: float,
    max_rad_per_step: float,
    settle_s: float,
    timeout_s: float,
    tol_rad: float,
) -> tuple[float, float, int]:
    dt = 1.0 / fps
    start = time.time()
    q = _read_wrist_rad(robot)
    n_cmd = 0
    while time.time() - start < timeout_s:
        err = target_rad - q
        if abs(err) <= tol_rad:
            break
        cmd = q + max(-max_rad_per_step, min(max_rad_per_step, err))
        _send_wrist_rad(robot, cmd)
        n_cmd += 1
        time.sleep(dt)
        q = _read_wrist_rad(robot)

    _send_wrist_rad(robot, target_rad)
    n_cmd += 1
    time.sleep(settle_s)
    q_final = _read_wrist_rad(robot)
    return q_final, q_final - target_rad, n_cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe wrist_roll raw rad/deg reachability on the real SO-101.")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Servo serial port.")
    parser.add_argument(
        "--targets",
        type=_parse_targets,
        default=DEFAULT_TARGETS_RAD,
        help="Comma-separated raw wrist_roll targets in radians.",
    )
    parser.add_argument("--fps", type=float, default=50.0, help="Command loop rate.")
    parser.add_argument("--max-rad-per-step", type=float, default=0.03, help="Slew cap for each command tick.")
    parser.add_argument("--settle-s", type=float, default=0.5, help="Readback wait after each target.")
    parser.add_argument("--timeout-s", type=float, default=3.0, help="Per-target timeout.")
    parser.add_argument("--tol-rad", type=float, default=0.03, help="Target tolerance.")
    parser.add_argument("--reverse", action="store_true", help="Run targets in reverse order.")
    args = parser.parse_args()

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    targets = list(args.targets)
    if args.reverse:
        targets.reverse()

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id="eva-follower"))
    robot.connect()
    try:
        q0 = _read_wrist_rad(robot)
        print(f"[probe-wrist] connected port={args.port}")
        print(f"[probe-wrist] initial {JOINT_NAME}: {q0:.6f} rad = {q0 * 180.0 / math.pi:.2f} deg")
        print(f"[probe-wrist] targets rad={targets}")
        print()

        prev = q0
        for i, target in enumerate(targets):
            q, err, n_cmd = _move_to_target(
                robot,
                float(target),
                fps=args.fps,
                max_rad_per_step=args.max_rad_per_step,
                settle_s=args.settle_s,
                timeout_s=args.timeout_s,
                tol_rad=args.tol_rad,
            )
            moved = q - prev
            status = "ok" if abs(err) <= args.tol_rad else "miss"
            print(
                f"[probe-wrist] {i:02d} target={target:+.6f} rad ({target * 180.0 / math.pi:+.2f} deg) "
                f"read={q:+.6f} rad ({q * 180.0 / math.pi:+.2f} deg) "
                f"err={err:+.6f} moved={moved:+.6f} cmds={n_cmd} {status}"
            )
            prev = q

        print()
        print("[probe-wrist] done")
    finally:
        try:
            robot.disconnect()
        except Exception as exc:
            print(f"[probe-wrist] WARNING: disconnect failed: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
