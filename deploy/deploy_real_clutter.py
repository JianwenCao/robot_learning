"""Closed-loop PPO deploy for Eval-2 / Eval-3 (clutter pick-and-place).

Single entry point with two modes:

* ``--mode eval2`` — one rollout per call. Target colour is fixed via
  ``--target-color`` and the bowl xy via ``--bowl-xy``. The Florence-2
  detector is prompted with ``"<target_color> cube"`` so its mask
  channel matches what the policy was trained against
  (``mdp.wrist_rgb_mask_dr`` in sim).

* ``--mode eval3`` — three sub-goals in one rollout, advancing on either
  a wall-clock timer or operator confirmation. Bowl xy is shared across
  the three sub-goals (per the spec); the target colour cycles through
  ``--colors`` (comma-separated, e.g. ``red,blue,yellow``). Between
  sub-goals we re-prompt Florence cheaply via :meth:`Detector.set_prompt`
  (no model reload) and overwrite the policy's target_color one-hot
  in the state vector.

Usage::

    # Eval-2 (single target):
    python -m deploy.deploy_real_clutter --mode eval2 \\
        --target-color red --bowl-xy 0.22,0.0

    # Eval-3 (3 sub-goals, timed advance, shared bowl):
    python -m deploy.deploy_real_clutter --mode eval3 \\
        --colors red,blue,yellow --bowl-xy 0.22,0.0

    # Dry-run (skip hardware; one synthetic forward):
    python -m deploy.deploy_real_clutter --mode eval2 \\
        --target-color red --bowl-xy 0.22,0.0 --dry-run

Observation pipeline (must match sim's
``ClutterPickPlaceEnvCfg.ObservationsCfg``):

* state (31,) — policy(25) + target_color_onehot(6).
    * policy(25) is byte-identical to Eval-1: joint_pos_rel(6) +
      joint_vel_rel(6) + gripper_state(1) + bowl_xy(2) + ee_proj_xy(2) +
      ee_to_bowl_xy(2) + last_action(6).
    * target_color_onehot(6) is the appended "goal" group — the trailing
      6 dims of state are sliced inside :class:`deploy.ppo_actor.PPOActorClutter`
      and passed to the CNN's FiLM head.
* image (4, 72, 128) — RGB + target-colour instance mask in ``[0, 1]``.
    * RGB: USB cam → undistort → resize → /255 (verbatim from Eval-1
      deploy).
    * Mask: Florence-2 prompted with ``"<color> cube"`` → polygon
      rasterize → largest-CC pick → nearest-neighbour resize to 128×72.
      Same `Detector` protocol as Eval-1; we just swap the prompt.

Action pipeline is identical to Eval-1
(:func:`deploy.deploy_real._decode_action`): 5 arm joints around home at
scale 0.5, binary gripper at the 0.5/0.0 sim rad endpoints.

Eval-3 release-detection modes (``--release-detect``):

* ``timed`` (default) — each sub-goal runs for ``--subgoal-steps``
  (default 250 = 5 s) and the scheduler advances unconditionally. Safe,
  unattended, but doesn't reward fast policies.
* ``manual`` — non-blocking stdin poll; press <enter> to advance.
  Useful for a human-in-the-loop demo where you watch the cube settle.

A ``vision``-based release detector (HSV-match the cube colour against
the bowl region) is the spec's third suggestion; not implemented here —
add a new class behind a flag if you want it.
"""
from __future__ import annotations

import argparse
import select
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

# Reuse Eval-1 helpers verbatim (FK chain, Lerobot driver, slew + homing,
# intrinsics, gripper conversion). The underscore prefix on a few of these
# is module-internal hinting; importing across deploy modules is fine.
from deploy.deploy_real import (
    CAM_HEIGHT,
    CAM_WIDTH,
    DEVICE,
    FPS,
    HSV_HIGH_DEFAULT,
    HSV_LOW_DEFAULT,
    IMG_H,
    IMG_W,
    JOINT_DEFAULTS_RAD,
    JOINT_NAMES,
    MAX_STEPS,
    PROJECT_ROOT,
    QDOT_EWMA_ALPHA,
    URDF_PATH,
    FK,
    LerobotSO101Driver,
    _build_image,
    _decode_action,
    _gripper_pct_from_sim_rad,
    _load_intrinsics,
    _parse_hsv,
    _rad_to_deg,
    _slew_limit,
    _slew_to_home,
)
from deploy.cube_detector import Detector, build_detector
from deploy.ppo_actor import CLUTTER_GOAL_DIM, PPOActorClutter

# Palette must match the sim's ``COLOR_NAMES`` in
# ``isaac_so_arm101.tasks.clutterpickplace.mdp.events``. Order maps directly
# to the one-hot bit (palette idx 0 = blue, …, 5 = red).
COLOR_NAMES: tuple[str, ...] = ("blue", "yellow", "purple", "orange", "green", "red")
COLOR_TO_IDX: dict[str, int] = {n: i for i, n in enumerate(COLOR_NAMES)}

CKPT_CANDIDATES = [
    PROJECT_ROOT / "deploy" / "runs" / "clutter_model.pt",
    PROJECT_ROOT / "deploy" / "runs" / "eval3_model.pt",
]


# ============================================================== state builder
def _onehot(color: str) -> np.ndarray:
    """6-D float32 one-hot of ``color``. Raises if not in the palette."""
    if color not in COLOR_TO_IDX:
        raise ValueError(
            f"unknown color {color!r}; expected one of {COLOR_NAMES}"
        )
    v = np.zeros(len(COLOR_NAMES), dtype=np.float32)
    v[COLOR_TO_IDX[color]] = 1.0
    return v


def _build_state_clutter(
    q_rad: np.ndarray,
    qdot_rad: np.ndarray,
    bowl_xy: np.ndarray,
    ee_xy: np.ndarray,
    last_action: np.ndarray,
    color_onehot: np.ndarray,
) -> np.ndarray:
    """31-D obs: policy(25) + target_color_onehot(6)."""
    joint_pos_rel = (q_rad - JOINT_DEFAULTS_RAD)[:6]
    joint_vel_rel = qdot_rad[:6]
    gripper_state = q_rad[5:6]
    ee_to_bowl = bowl_xy - ee_xy
    return np.concatenate([
        joint_pos_rel, joint_vel_rel, gripper_state,
        bowl_xy, ee_xy, ee_to_bowl, last_action,
        color_onehot,
    ]).astype(np.float32)


# ============================================================== sub-goal release detect
def _stdin_ready() -> bool:
    """Non-blocking poll for a newline on stdin (manual release-detect mode)."""
    rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
    if rlist:
        sys.stdin.readline()
        return True
    return False


# ============================================================== policy loop
def _run_subgoal(
    driver: LerobotSO101Driver,
    policy: PPOActorClutter,
    fk: FK,
    detector: Detector | None,
    K_mat,
    dist,
    args,
    bowl_xy: np.ndarray,
    color: str,
    q_rad_prev: np.ndarray,
    qdot_filt: np.ndarray,
    last_action: np.ndarray,
    max_steps: int,
    release_detect: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Run one sub-goal until ``max_steps`` or detected release.

    Returns the updated ``(q_rad_prev, qdot_filt, last_action, advanced)``
    so the caller can chain sub-goals without resetting state.
    """
    color_onehot = _onehot(color)
    if detector is not None and hasattr(detector, "set_prompt"):
        detector.set_prompt(f"{color} cube")

    dt = 1.0 / FPS
    next_tick = time.time()
    advanced = False

    for t in range(max_steps):
        q_rad = driver.read_proprio_sim_rad()
        qdot_raw = (q_rad - q_rad_prev) / dt
        qdot_filt = (QDOT_EWMA_ALPHA * qdot_raw
                     + (1.0 - QDOT_EWMA_ALPHA) * qdot_filt).astype(np.float32)
        q_rad_prev = q_rad

        ee_xy = fk.ee_xyz(q_rad)[:2]
        state = _build_state_clutter(
            q_rad, qdot_filt, bowl_xy[:2], ee_xy, last_action, color_onehot
        )

        rgb = driver.capture_wrist_rgb_hwc()
        image = _build_image(
            rgb, K_mat, dist,
            tuple(args.hsv_low), tuple(args.hsv_high),
            detector=detector,
        )

        with torch.no_grad():
            action = policy(
                torch.from_numpy(state).unsqueeze(0).to(DEVICE),
                torch.from_numpy(image).unsqueeze(0).to(DEVICE),
            )[0].cpu().numpy()
        last_action = action.astype(np.float32)

        target_rad = _decode_action(action)
        target_rad = _slew_limit(target_rad, q_rad)
        driver.send_joint_targets_sim_rad(target_rad)

        if (t + 1) % 30 == 0:
            print(
                f"  [{color:6s}] t={t+1:4d}  action={action.round(2)}  "
                f"ee_xy={ee_xy.round(3)}  bowl_xy={bowl_xy[:2].round(3)}"
            )

        # Manual release detect — non-blocking <enter> poll.
        if release_detect == "manual" and _stdin_ready():
            print(f"  [{color}] manual advance at t={t+1}")
            advanced = True
            break

        next_tick += dt
        sleep_for = next_tick - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.time()

    return q_rad_prev, qdot_filt, last_action, advanced


def run(args) -> None:
    ckpt_path = Path(args.ckpt) if args.ckpt else next(
        (p for p in CKPT_CANDIDATES if p.exists()), None
    )
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(
            "Clutter PPO checkpoint not found. Looked at:\n  "
            + "\n  ".join(str(p) for p in CKPT_CANDIDATES)
            + "\nPass --ckpt /path/to/model.pt or drop the file at one of the defaults."
        )
    policy = PPOActorClutter.from_checkpoint(ckpt_path, map_location=DEVICE).to(DEVICE)
    print(f"[clutter] loaded {ckpt_path}")

    fk = FK(URDF_PATH)
    K_mat, dist = _load_intrinsics()
    if K_mat is None:
        warnings.warn("camera_intrinsics.yaml missing — skipping undistort.")

    # Mode-dependent target list. Both modes share the same per-sub-goal
    # loop; eval2 is just the special case of len(targets) == 1.
    if args.mode == "eval2":
        if not args.target_color:
            raise ValueError("--target-color is required for --mode eval2")
        targets: list[str] = [args.target_color]
    else:  # eval3
        if not args.colors:
            raise ValueError("--colors red,blue,yellow is required for --mode eval3")
        targets = [c.strip() for c in args.colors.split(",") if c.strip()]
        if len(targets) == 0:
            raise ValueError("--colors must list at least one color")
        for c in targets:
            if c not in COLOR_TO_IDX:
                raise ValueError(f"unknown color {c!r}; expected one of {COLOR_NAMES}")

    x, y = (float(s) for s in args.bowl_xy.split(","))
    bowl_xy = np.array([x, y], dtype=np.float32)

    # Single detector instance, re-prompted between sub-goals. Florence is
    # ~1 GB on disk; build once and call set_prompt() per sub-goal so the
    # multi-sub-goal Eval-3 path doesn't pay the 30-s reload three times.
    initial_prompt = f"{targets[0]} cube"
    detector = build_detector(args.mask_source, prompt=initial_prompt)

    if args.dry_run:
        print(f"[clutter] --dry-run: single synthetic forward (mode={args.mode}, "
              f"targets={targets}, bowl_xy={bowl_xy.tolist()})")
        q_rad = JOINT_DEFAULTS_RAD.copy()
        qdot_rad = np.zeros(6, dtype=np.float32)
        ee_xy = fk.ee_xyz(q_rad)[:2]
        color_onehot = _onehot(targets[0])
        state = _build_state_clutter(
            q_rad, qdot_rad, bowl_xy[:2], ee_xy,
            np.zeros(6, dtype=np.float32), color_onehot,
        )
        synth_rgb = np.full((CAM_HEIGHT, CAM_WIDTH, 3), 128, dtype=np.uint8)
        image = _build_image(
            synth_rgb, K_mat, dist,
            tuple(args.hsv_low), tuple(args.hsv_high), detector=detector,
        )
        with torch.no_grad():
            a = policy(
                torch.from_numpy(state).unsqueeze(0).to(DEVICE),
                torch.from_numpy(image).unsqueeze(0).to(DEVICE),
            )[0].cpu().numpy()
        print(
            f"[clutter] dry-run state shape={state.shape}  image shape={image.shape}  "
            f"action mean={a.round(3)}"
        )
        return

    driver = LerobotSO101Driver()
    driver.connect()
    try:
        q0 = driver.read_proprio_sim_rad()
        print(f"[clutter] pre-home pose: q_sim_rad={q0.round(3)} "
              f"(arm_deg={_rad_to_deg(q0[:5]).round(2)}, "
              f"gripper_pct≈{_gripper_pct_from_sim_rad(float(q0[5])):.1f})")
        if not args.no_confirm:
            input("[clutter] arm will home to (shoulder=0, wrist=90°, gripper=open). "
                  "Clear the workspace. Press <enter> to home, ctrl-C to abort … ")
        q_rad_prev = _slew_to_home(driver)
        ee_xyz_now = fk.ee_xyz(q_rad_prev)
        print(
            f"[clutter] homed: q_sim_rad={q_rad_prev.round(3)}  "
            f"ee_xyz={ee_xyz_now.round(3)} "
            f"(sim home ≈ (0.247, 0.000, 0.063))"
        )
        if not args.no_confirm:
            input(
                f"[clutter] place cubes in workspace (x∈(0.13,0.25), y∈(-0.12,0.12)). "
                f"Targets (in order): {targets}. Bowl at {bowl_xy.tolist()}. "
                "Press <enter> to start rollout, ctrl-C to abort … "
            )

        qdot_filt = np.zeros(6, dtype=np.float32)
        last_action = np.zeros(6, dtype=np.float32)

        for k, color in enumerate(targets):
            print(f"[clutter] === sub-goal {k+1}/{len(targets)}: target={color} ===")
            q_rad_prev, qdot_filt, last_action, advanced = _run_subgoal(
                driver=driver,
                policy=policy,
                fk=fk,
                detector=detector,
                K_mat=K_mat,
                dist=dist,
                args=args,
                bowl_xy=bowl_xy,
                color=color,
                q_rad_prev=q_rad_prev,
                qdot_filt=qdot_filt,
                last_action=last_action,
                max_steps=args.subgoal_steps,
                release_detect=args.release_detect,
            )
            reason = "manual advance" if advanced else f"{args.subgoal_steps}-step timer"
            print(f"[clutter] sub-goal {k+1} ended ({reason})")
    finally:
        driver.disconnect()


# ============================================================== entry
def main() -> int:
    p = argparse.ArgumentParser(
        description="Eval-2/3 PPO closed-loop deploy on real SO-ARM101"
    )
    p.add_argument("--mode", choices=["eval2", "eval3"], required=True,
                   help="eval2 = single rollout with --target-color; "
                        "eval3 = 3 sub-goals from --colors with --release-detect.")
    p.add_argument("--bowl-xy", type=str, required=True,
                   help="Comma-separated 'x,y' metres, robot base frame.")
    p.add_argument("--target-color", type=str, default=None,
                   help=f"Eval-2 target color, one of {COLOR_NAMES}.")
    p.add_argument("--colors", type=str, default=None,
                   help="Eval-3 comma-separated color sequence, e.g. red,blue,yellow.")
    p.add_argument("--release-detect", choices=["timed", "manual"], default="timed",
                   help="How to advance between Eval-3 sub-goals. "
                        "timed = unconditional after --subgoal-steps; "
                        "manual = non-blocking <enter> poll.")
    p.add_argument("--subgoal-steps", type=int, default=MAX_STEPS,
                   help=f"Steps per sub-goal (default {MAX_STEPS} = 5 s @ {FPS} Hz).")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Path to a Stage-3 vision PPO checkpoint. "
                        f"If omitted, searches {[str(p) for p in CKPT_CANDIDATES]}.")
    p.add_argument("--mask-source", choices=["hsv", "florence"], default="florence",
                   help="Mask channel source. 'florence' (default) uses "
                        "Florence-2 with a color-prompt per sub-goal. "
                        "'hsv' falls back to the Eval-1 saturation gate — "
                        "DOES NOT discriminate by color, useful only for "
                        "Eval-2 with a single-cube scene.")
    p.add_argument("--hsv-low",  type=_parse_hsv, default=HSV_LOW_DEFAULT,
                   help="HSV lower bound (only used when --mask-source=hsv).")
    p.add_argument("--hsv-high", type=_parse_hsv, default=HSV_HIGH_DEFAULT,
                   help="HSV upper bound (only used when --mask-source=hsv).")
    p.add_argument("--no-confirm", action="store_true",
                   help="Skip pre-rollout <enter> prompts.")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip hardware; load model + run one synthetic forward.")
    args = p.parse_args()

    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
