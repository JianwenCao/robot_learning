#!/usr/bin/env python3
"""Tail an RSL-RL DistillationRunner log, print one compact line per iter.

Distill emits ``Mean behavior loss`` (MSE between student and teacher actions)
in addition to the standard env/curriculum metrics. The Stage 2 stop signal
per ``EVAL1_PLAN.md`` §7 is ``release_from_scratch`` clearing ~30-50 % — not
the lowest behavior_loss. We flag SR + release crossings here, not loss.

Usage:
    python -u bc/watch_student.py [--log /tmp/v4_student.log] [--target-sr 0.40]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import deque

_PATTERNS = [
    (re.compile(r"Learning iteration\s+(\d+)/(\d+)"),                       "iter"),
    (re.compile(r"Curriculum/log_success/success_rate:\s+([\d.\-eE+]+)"),   "sr"),
    (re.compile(r"Curriculum/log_success/over_bowl_high_rate:\s+([\d.\-eE+]+)"), "rim"),
    (re.compile(r"Episode_Reward/release_in_bowl:\s+([\d.\-eE+]+)"),        "rel"),
    (re.compile(r"Episode_Reward/lifting_object:\s+([\d.\-eE+]+)"),         "lift"),
    (re.compile(r"Mean behavior loss:\s+([\d.\-eE+]+)"),                    "bloss"),
    (re.compile(r"Mean reward:\s+([\d.\-eE+]+)"),                           "rew"),
    (re.compile(r"Iteration time:\s+([\d.]+)s"),                            "dt"),
    (re.compile(r"Time elapsed:\s+([\d:]+)"),                               "el"),
    (re.compile(r"ETA:\s+([\d:]+)"),                                        "eta"),
]

_END = re.compile(r"^\s*ETA:\s+[\d:]+")  # last line per iter block


def _follow(path: str, from_start: bool = False):
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
    cur: dict[str, str] = {}
    iter_max = None
    sr_window = deque(maxlen=threshold_hits)
    above_target_emitted = False

    for line in stream:
        for rx, key in _PATTERNS:
            m = rx.search(line)
            if m:
                if key == "iter":
                    cur = {"iter": m.group(1), "iter_total": m.group(2)}
                    iter_max = m.group(2)
                else:
                    cur[key] = m.group(1)
                break

        if _END.match(line) and "iter" in cur:
            it = int(cur.get("iter", "0"))
            sr = float(cur.get("sr", "nan"))
            rim = float(cur.get("rim", "nan"))
            rel = float(cur.get("rel", "nan"))
            lift = float(cur.get("lift", "nan"))
            bloss = float(cur.get("bloss", "nan"))
            rew = float(cur.get("rew", "nan"))
            dt = float(cur.get("dt", "nan"))
            el = cur.get("el", "?")
            eta = cur.get("eta", "?")

            sr_window.append(sr)
            flag = ""
            if sr >= target_sr and not above_target_emitted:
                if len(sr_window) == sr_window.maxlen and all(
                    v >= target_sr for v in sr_window
                ):
                    flag = "  ✅ SR ≥ {:.2f} for {} iters — Stage 2 stop candidate".format(
                        target_sr, sr_window.maxlen
                    )
                    above_target_emitted = True

            print(
                f"iter {it:4d}/{iter_max} | "
                f"SR {sr:5.3f}  rim {rim:5.3f}  rel {rel:6.3f}  lift {lift:5.3f} | "
                f"bloss {bloss:6.3f}  rew {rew:6.2f} | "
                f"dt {dt:4.2f}s  el {el}  eta {eta}"
                + flag,
                flush=True,
            )
            cur = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/tmp/v4_student.log")
    ap.add_argument("--target-sr", type=float, default=0.40,
                    help="Stage 2 stop when SR ≥ this for --plateau iters.")
    ap.add_argument("--plateau", type=int, default=3)
    ap.add_argument("--from-start", action="store_true")
    args = ap.parse_args()

    print(f"watching {args.log}  target_sr={args.target_sr}  "
          f"plateau={args.plateau}\n" + "-" * 90, flush=True)
    try:
        _parse(_follow(args.log, from_start=args.from_start),
               target_sr=args.target_sr, threshold_hits=args.plateau)
    except KeyboardInterrupt:
        print("\n[watch] stopped.")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
