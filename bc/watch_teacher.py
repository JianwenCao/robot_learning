#!/usr/bin/env python3
"""Tail an RSL-RL teacher log, print one compact line per iter.

Highlights when ``Curriculum/log_success/success_rate`` crosses thresholds
so you don't have to babysit the raw log.

Usage:
    python -u bc/watch_teacher.py [--log /tmp/v4_teacher.log] [--target-sr 0.80]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import deque

# Lines from RSL-RL logs we care about. Maps regex → short label.
_PATTERNS = [
    (re.compile(r"Learning iteration\s+(\d+)/(\d+)"),                 "iter"),
    (re.compile(r"Curriculum/log_success/success_rate:\s+([\d.\-eE+]+)"), "sr"),
    (re.compile(r"Curriculum/log_success/over_bowl_high_rate:\s+([\d.\-eE+]+)"), "rim"),
    (re.compile(r"Episode_Reward/release_in_bowl:\s+([\d.\-eE+]+)"),  "rel"),
    (re.compile(r"Episode_Reward/lifting_object:\s+([\d.\-eE+]+)"),   "lift"),
    (re.compile(r"Curriculum/block_range_expand/x_radius:\s+([\d.\-eE+]+)"), "curr_x"),
    (re.compile(r"Curriculum/block_range_expand/y_radius:\s+([\d.\-eE+]+)"), "curr_y"),
    (re.compile(r"Mean action noise std:\s+([\d.\-eE+]+)"),           "sigma"),
    (re.compile(r"Mean reward:\s+([\d.\-eE+]+)"),                     "rew"),
    (re.compile(r"Iteration time:\s+([\d.]+)s"),                      "dt"),
    (re.compile(r"Time elapsed:\s+([\d:]+)"),                         "el"),
    (re.compile(r"ETA:\s+([\d:]+)"),                                  "eta"),
]

_END = re.compile(r"^\s*ETA:\s+[\d:]+")  # ETA line is the *last* one printed per iter


def _follow(path: str, from_start: bool = False):
    """Generator that yields lines from a growing file. Polls every 0.5 s."""
    while not os.path.isfile(path):
        time.sleep(1.0)
    f = open(path, "r")
    if not from_start:
        f.seek(0, os.SEEK_END)
    while True:
        line = f.readline()
        if not line:
            time.sleep(0.5)
            continue
        yield line


def _parse(stream, target_sr: float, threshold_hits: int = 3):
    """Accumulate one iter's metrics, emit when the iter block closes."""
    cur: dict[str, str] = {}
    iter_max = None
    sr_window = deque(maxlen=threshold_hits)
    above_target_emitted = False
    target_hi = 0.85   # secondary, brighter target

    for line in stream:
        for rx, key in _PATTERNS:
            m = rx.search(line)
            if m:
                if key == "iter":
                    # iter line opens a new block; reset accumulator.
                    cur = {"iter": m.group(1), "iter_total": m.group(2)}
                    iter_max = m.group(2)
                else:
                    cur[key] = m.group(1)
                break

        # End-of-iter — emit summary line after the ETA line (last per iter).
        if _END.match(line) and "iter" in cur:
            it = int(cur.get("iter", "0"))
            sr = float(cur.get("sr", "nan"))
            rim = float(cur.get("rim", "nan"))
            rel = float(cur.get("rel", "nan"))
            lift = float(cur.get("lift", "nan"))
            sig = float(cur.get("sigma", "nan"))
            rew = float(cur.get("rew", "nan"))
            dt = float(cur.get("dt", "nan"))
            curr_x = float(cur.get("curr_x", "nan"))
            curr_y = float(cur.get("curr_y", "nan"))
            el = cur.get("el", "?")
            eta = cur.get("eta", "?")

            # Flag SR threshold crossings (visible markers in stream).
            sr_window.append(sr)
            flag = ""
            if sr >= target_hi and not above_target_emitted:
                flag = "  ⭐ SR ≥ {:.2f}".format(target_hi)
                above_target_emitted = True
            elif sr >= target_sr:
                if len(sr_window) == sr_window.maxlen and all(
                    v >= target_sr for v in sr_window
                ):
                    flag = "  ✅ SR ≥ {:.2f} for {} iters — kill candidate".format(
                        target_sr, sr_window.maxlen
                    )

            print(
                f"iter {it:4d}/{iter_max} | "
                f"SR {sr:5.3f}  rim {rim:5.3f}  rel {rel:6.3f}  lift {lift:5.3f} | "
                f"σ {sig:4.2f}  rew {rew:6.2f}  curr ({curr_x:.3f},{curr_y:.3f}) | "
                f"dt {dt:4.2f}s  el {el}  eta {eta}"
                + flag,
                flush=True,
            )
            cur = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/tmp/v4_teacher.log",
                    help="Path to teacher training log.")
    ap.add_argument("--target-sr", type=float, default=0.80,
                    help="Flag when SR ≥ this and holds for --plateau iters.")
    ap.add_argument("--plateau", type=int, default=3,
                    help="Consecutive iters above --target-sr before flagging.")
    ap.add_argument("--from-start", action="store_true",
                    help="Replay the log from the beginning instead of tailing live.")
    args = ap.parse_args()

    print(f"watching {args.log}  target_sr={args.target_sr}  "
          f"plateau={args.plateau}\n" + "-" * 80, flush=True)
    try:
        _parse(_follow(args.log, from_start=args.from_start),
               target_sr=args.target_sr, threshold_hits=args.plateau)
    except KeyboardInterrupt:
        print("\n[watch] stopped.")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
