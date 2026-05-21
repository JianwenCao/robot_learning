"""Closed-loop PPO deploy: state-only policy + AprilTag cube localisation.

Single entry point for Eval-1, Eval-2, and Eval-3. The wrist cam is used
only for AprilTag pose: every step the cam frame is undistorted,
``pupil-apriltags`` reads the target tag on top of the cube, the script
composes ``T_base_tag = T_base_ee · T_ee_cam · T_cam_tag``, and the
resulting ``(x, y)`` lands in the policy's 27-D state vector. There is
no image channel — the policy is a pure-state MLP (see
:mod:`deploy.ppo_actor` ``PPOActorState``).

Shared hardware / FK / action-decode helpers live in :mod:`deploy.driver`;
this file is the per-step closed loop plus the Eval-3 sub-goal
sequencer on top.

See ``docs/EVAL1_PLAN.md`` / ``docs/EVAL2_PLAN.md`` / ``docs/EVAL3_PLAN.md``
and ``deploy/README.md`` for the print/calibration prerequisites.
``deploy/calibrate_hand_eye.py`` must have been run once to populate
``deploy/hand_eye.yaml`` — without it this script errors out.

Per-step contract (mirroring the sim-side
:func:`mdp.observations.cube_pos_xy_noisy`:

* Detection valid this frame → publish ``(x, y)`` and store as
  ``last_cube_xy``.
* No detection this frame **and** not yet grasped → hold ``last_cube_xy``.
* Grasp latched (``is_grasped_now == True`` at any prior step) → skip
  detection entirely and hold ``last_cube_xy`` for the rest of the
  sub-goal rollout (matches sim's post-grasp deterministic freeze; the
  gripper occludes the tag).

The grasp latch fires when the gripper-close command has been issued
**and** the last detected cube was within ``GRASP_XY_TOL`` of the EE in
the table plane. Same shape as sim's kinematic ``is_grasped`` heuristic,
adapted to deploy where we don't have a privileged height signal.

Eval-1 / Eval-2 / Eval-3 selection: pass either ``--target-color`` (single
sub-goal, Eval-1/2) or ``--colors c1,c2,c3`` (3 sub-goals, Eval-3). For
multi-sub-goal runs the same policy weights drive every sub-goal —
only :meth:`AprilTagDetector.set_target_id` changes between them
(see ``docs/EVAL3_PLAN.md`` §10).
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from deploy.cube_detector import AprilTagDetector

# Tag-ID ↔ colour map. Source-of-truth: ``deploy/README.md`` §Step 3 print
# table. Keep this in lockstep with the physical labels stuck on the cubes;
# if you re-label a cube, update this dict (not the policy — the policy is
# colour-blind, the detector picks the right tag).
_COLOR_TO_ID: dict[str, int] = {
    "red": 0,
    "blue": 1,
    "yellow": 2,
    "green": 3,
    "purple": 4,
    "orange": 5,
}
from deploy.driver import (
    EE_LOCAL_OFFSET,
    FK,
    FPS,
    HOME_POSE_RAD,
    JOINT_DEFAULTS_RAD,
    JOINT_NAMES,
    MAX_STEPS,
    QDOT_EWMA_ALPHA,
    URDF_PATH,
    LerobotSO101Driver,
    _decode_action,
    _gripper_pct_from_sim_rad,
    _load_intrinsics,
    _rad_to_deg,
    _slew_limit,
    _slew_to_home,
)
from deploy.ppo_actor import PPOActorState, STATE_APRILTAG_STATE_DIM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = Path(__file__).resolve().parent

CKPT_CANDIDATES = [
    DEPLOY_DIR / "runs" / "state_apriltag_model.pt",
    DEPLOY_DIR / "runs" / "model.pt",
]

DEVICE = "cpu"

# Grasp-latch threshold: once we've commanded gripper-close AND the last
# detected cube was within this xy radius of the EE projection, latch
# ``_grasped`` to True for the rest of the rollout. Matches the sim-side
# post-grasp freeze (≈ 4 cm xy at the typical wrist-cam standoff).
GRASP_XY_TOL = 0.04

# Eval-3 ``--release-detect vision``: count target-tag-in-bowl frames before
# declaring release. Tag is occluded while grasped; after release the cube
# sits flat with the tag visible from above.
RELEASE_XY_TOL = 0.06   # m, matches sim release predicate
RELEASE_Z_TOL = 0.06    # m
RELEASE_DEBOUNCE = 5    # consecutive frames


def _resolve_color(name: str) -> int:
    """Look up tag-ID for a colour name. Raises on unknown colour."""
    key = name.strip().lower()
    if key not in _COLOR_TO_ID:
        raise ValueError(
            f"Unknown colour {name!r}. Known: {sorted(_COLOR_TO_ID)}."
        )
    return _COLOR_TO_ID[key]


def fk_T_base_ee(fk: FK, joint_pos_rad: np.ndarray) -> np.ndarray:
    """4×4 ``T_base_ee`` for the current joints. See calibrate_hand_eye.py."""
    arm_vals = {n: float(v) for n, v in zip(JOINT_NAMES[:5], joint_pos_rad[:5])}
    th = [arm_vals[n] for n in fk.chain.get_joint_parameter_names()]
    T = fk.chain.forward_kinematics(th)
    out = np.eye(4)
    out[:3, :3] = T.rot_mat
    out[:3, 3] = np.asarray(T.pos, dtype=np.float64) + T.rot_mat @ EE_LOCAL_OFFSET
    return out


def _build_state_27(
    q_rad: np.ndarray,
    qdot_rad: np.ndarray,
    bowl_xy: np.ndarray,
    ee_xy: np.ndarray,
    last_action: np.ndarray,
    cube_xy: np.ndarray,
) -> np.ndarray:
    """27-D state vector matching the sim policy obs (Eval-1/2/3 share schema).

    Field order matches the configclass declaration order — the
    ``cube_pos_xy_noisy`` field is appended last, so it lands at the
    trailing 2 dims here. If you reorder the sim PolicyCfg, reorder
    this too or the policy will get a scrambled obs vector.
    """
    joint_pos_rel = (q_rad - JOINT_DEFAULTS_RAD)[:6]
    joint_vel_rel = qdot_rad[:6]
    gripper_state = q_rad[5:6]
    ee_to_bowl = bowl_xy - ee_xy
    return np.concatenate(
        [
            joint_pos_rel, joint_vel_rel, gripper_state,
            bowl_xy, ee_xy, ee_to_bowl, last_action,
            cube_xy,
        ]
    ).astype(np.float32)  # 6+6+1+2+2+2+6+2 = 27


def _open_debug_dir(
    args, bowl_xy: np.ndarray, ckpt_path: Path, colors: list[str]
) -> Path | None:
    if not getattr(args, "debug_dump", False):
        return None
    import json
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = PROJECT_ROOT / "deploy" / "runs" / "debug_state" / stamp
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "ckpt": str(ckpt_path),
        "bowl_xy": [float(bowl_xy[0]), float(bowl_xy[1])],
        "joint_defaults_rad": JOINT_DEFAULTS_RAD.tolist(),
        "fps": FPS,
        "max_steps": MAX_STEPS,
        "colors": colors,
        "color_to_id": {c: _resolve_color(c) for c in colors},
        "tag_size_m": args.tag_size,
        "release_detect": args.release_detect,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[state] --debug-dump active: writing to {out}")
    return out


def _debug_dump_step(
    out: Path, sub_k: int, color: str, t: int,
    state: np.ndarray, action: np.ndarray,
    q_rad: np.ndarray, ee_xy: np.ndarray, target_rad: np.ndarray,
    cube_xy: np.ndarray, cube_valid: bool, grasped: bool,
) -> None:
    import json
    row = {
        "sub_k": int(sub_k),
        "color": color,
        "t": int(t),
        "state": state.tolist(),
        "action": action.tolist(),
        "q_sim_rad": q_rad.tolist(),
        "ee_xy": ee_xy.tolist(),
        "target_sim_rad": target_rad.tolist(),
        "cube_xy": cube_xy.tolist(),
        "cube_valid": bool(cube_valid),
        "grasped": bool(grasped),
    }
    with open(out / "log.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def _run_subgoal(
    *,
    sub_k: int,
    color: str,
    driver: LerobotSO101Driver,
    policy: PPOActorState,
    fk: FK,
    detector: AprilTagDetector,
    K_mat: np.ndarray | None,
    dist: np.ndarray | None,
    bowl_xy: np.ndarray,
    q_rad_prev: np.ndarray,
    dump_dir: Path | None,
    release_detect: str,
) -> tuple[np.ndarray, bool]:
    """One sub-goal rollout. Returns ``(q_rad_prev, released_early)``.

    All per-sub-goal latches (``grasped``, ``last_cube_xyz``, ``last_action``,
    ``qdot_filt``, detection counters) are local to this function and
    reset at the start of every call — mirroring the sim §10 contract
    where the env resets latches between sub-goals while world state
    persists. ``q_rad_prev`` is carried across sub-goals because joint
    velocity is a physical continuity, not a per-sub-goal latch.
    """
    detector.set_target_id(_resolve_color(color))
    print(f"\n[state] === sub-goal {sub_k}: color={color} "
          f"(tag_id={detector.target_id}) ===")

    qdot_filt = np.zeros(6, dtype=np.float32)
    last_action = np.zeros(6, dtype=np.float32)
    # Seed last_cube_xyz with the bowl xy and table-top cube center z as a
    # defensive default; the first
    # valid detection overrides it.
    last_cube_xyz = np.array([float(bowl_xy[0]), float(bowl_xy[1]), 0.01], dtype=np.float32)
    grasped = False
    n_dropout = 0
    n_detected = 0
    release_hits = 0
    released_early = False

    dt = 1.0 / FPS
    next_tick = time.time()

    for t in range(MAX_STEPS):
        # ---- proprio + FK -----------------------------------------------
        q_rad = driver.read_proprio_sim_rad()
        qdot_raw = (q_rad - q_rad_prev) / dt
        qdot_filt = (QDOT_EWMA_ALPHA * qdot_raw
                     + (1.0 - QDOT_EWMA_ALPHA) * qdot_filt).astype(np.float32)
        q_rad_prev = q_rad
        T_be = fk_T_base_ee(fk, q_rad)
        ee_xy = T_be[:2, 3].astype(np.float32)

        # ---- AprilTag detection -----------------------------------------
        # Once grasped, skip detection (tag occluded by gripper) and hold
        # last_cube_xyz — matches the sim post-grasp freeze.
        cube_valid = False
        cube_z = float("nan")
        if not grasped:
            rgb = driver.capture_wrist_rgb_hwc()
            import cv2  # lazy to allow --dry-run import on CPU-only boxes
            if K_mat is not None and dist is not None:
                rgb = cv2.undistort(rgb, K_mat, dist)
            cube_xy_now, cube_valid = detector.pose(rgb, T_be)
            if cube_valid:
                last_cube_xyz[:2] = cube_xy_now
                n_detected += 1
                # Cheap re-compose for the z component; ``detector.pose``
                # only returns xy.
                dets = detector.detect(rgb)
                tgt = next((d for d in dets if d["tag_id"] == detector.target_id), None)
                if tgt is not None:
                    T_base_tag = T_be @ detector.T_ee_cam @ tgt["T_cam_tag"]
                    cube_z = float(T_base_tag[2, 3])
                    last_cube_xyz[2] = cube_z
            else:
                n_dropout += 1
        cube_xyz_obs = last_cube_xyz

        # ---- Build state + policy ---------------------------------------
        state = _build_state_27(
            q_rad, qdot_filt, bowl_xy, ee_xy, last_action, cube_xyz_obs[:2],
        )
        with torch.no_grad():
            action = policy(
                torch.from_numpy(state).unsqueeze(0).to(DEVICE),
            )[0].cpu().numpy()
        last_action = action.astype(np.float32)

        # ---- Apply action, update grasp latch ---------------------------
        target_rad = _decode_action(action)
        target_rad = _slew_limit(target_rad, q_rad)
        driver.send_joint_targets_sim_rad(target_rad)

        if not grasped and action[5] < 0.0:
            xy_dist = float(np.linalg.norm(last_cube_xyz[:2] - ee_xy))
            if xy_dist < GRASP_XY_TOL:
                grasped = True
                print(f"[state] grasp latched at t={t} "
                      f"(xy_dist={xy_dist*100:.1f} cm; cube_pos frozen "
                      f"for remaining steps of sub-goal {sub_k})")

        # ---- Vision release-detect (Eval-3 early-advance) ---------------
        if release_detect == "vision" and grasped and cube_valid:
            # Once grasped we usually skip detection, but the operator may
            # want to detect re-acquisition after the gripper opens. We
            # only enter this branch when both flags are set, which is
            # rare — left here for symmetry; the common predicate fires
            # in the pre-grasp branch above and won't reach this point.
            pass
        if (
            release_detect == "vision"
            and not grasped
            and cube_valid
            and np.isfinite(cube_z)
        ):
            in_bowl = (
                np.linalg.norm(last_cube_xyz[:2] - bowl_xy) < RELEASE_XY_TOL
                and cube_z < RELEASE_Z_TOL
            )
            release_hits = release_hits + 1 if in_bowl else 0
            if release_hits >= RELEASE_DEBOUNCE:
                released_early = True
                print(f"[state] vision release detected at t={t} "
                      f"(sub-goal {sub_k}: {color})")

        if dump_dir is not None:
            _debug_dump_step(
                dump_dir, sub_k, color, t, state, action, q_rad, ee_xy,
                target_rad, cube_xyz_obs[:2], cube_valid, grasped,
            )

        if (t + 1) % 30 == 0:
            print(
                f"  k={sub_k}  t={t+1:4d}  action={action.round(2)}  "
                f"ee_xy={ee_xy.round(3)}  cube_xyz={cube_xyz_obs.round(3)}  "
                f"grasped={grasped}"
            )

        if released_early:
            break

        next_tick += dt
        sleep_for = next_tick - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.time()

    total = n_detected + n_dropout
    dropout_pct = (100.0 * n_dropout / total) if total else 0.0
    print(
        f"[state] sub-goal {sub_k} ({color}) done. "
        f"detections={n_detected}, dropouts={n_dropout} "
        f"({dropout_pct:.1f} %), grasped={grasped}, "
        f"released_early={released_early}."
    )
    return q_rad_prev, released_early


def run(bowl_xy: np.ndarray, colors: list[str], args) -> None:
    # --- Validate colours up-front (cheap, fails before hardware connect) -
    for c in colors:
        _resolve_color(c)
    multi = len(colors) > 1

    # --- Load policy ------------------------------------------------------
    ckpt_path = None
    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--ckpt: {ckpt_path} does not exist")
    else:
        ckpt_path = next((p for p in CKPT_CANDIDATES if p.exists()), None)
        if ckpt_path is None:
            raise FileNotFoundError(
                "state-only PPO checkpoint not found. Looked at:\n  "
                + "\n  ".join(str(p) for p in CKPT_CANDIDATES)
                + "\nPass --ckpt or place a state-only checkpoint at "
                  "deploy/runs/state_apriltag_model.pt."
            )
    policy = PPOActorState.from_checkpoint(ckpt_path, map_location=DEVICE).to(DEVICE)
    print(f"[state] loaded {ckpt_path}")
    print(f"[state] sub-goals: {colors}  (multi={multi})")

    fk = FK(URDF_PATH)

    # --- Dry run (short-circuit before detector / intrinsics) -------------
    # Dry-run is the offline smoke test: it must run on a freshly cloned
    # repo before the operator has done hand-eye calibration or camera
    # calibration. So we skip the detector and intrinsics here — they're
    # only needed by the live per-step loop.
    if args.dry_run:
        print("[state] --dry-run: synthetic forward, no hardware, no camera.")
        q_rad = JOINT_DEFAULTS_RAD.copy()
        qdot_rad = np.zeros(6, dtype=np.float32)
        ee_xy = fk.ee_xyz(q_rad)[:2]
        cube_xyz = np.array([0.20, 0.0, 0.01], dtype=np.float32)
        state = _build_state_27(
            q_rad, qdot_rad, bowl_xy, ee_xy,
            np.zeros(6, dtype=np.float32), cube_xyz[:2],
        )
        assert state.shape == (STATE_APRILTAG_STATE_DIM,), state.shape
        with torch.no_grad():
            a = policy(torch.from_numpy(state).unsqueeze(0).to(DEVICE))[0].cpu().numpy()
        print(f"[state] dry-run state shape = {state.shape}  action mean = {a.round(3)}")
        return

    K_mat, dist = _load_intrinsics()
    if K_mat is None:
        warnings.warn("camera_intrinsics.yaml missing — skipping undistort. Tag PnP will be wrong.")

    # --- Build AprilTag detector ------------------------------------------
    detector = AprilTagDetector(
        tag_size_m=args.tag_size,
        target_id=_resolve_color(colors[0]),
    )

    dump_dir = _open_debug_dir(args, bowl_xy, ckpt_path, colors)

    # --- Hardware loop ----------------------------------------------------
    driver = LerobotSO101Driver()
    driver.connect()
    try:
        q0 = driver.read_proprio_sim_rad()
        print(f"[state] pre-home pose: q_sim_rad={q0.round(3)} "
              f"(arm_deg={_rad_to_deg(q0[:5]).round(2)}, "
              f"gripper_pct≈{_gripper_pct_from_sim_rad(float(q0[5])):.1f})")
        if not args.no_confirm:
            input("[state] arm will home to (shoulder=0, wrist=90°, gripper=open). "
                  "Clear the workspace. Press <enter> to home, ctrl-C to abort … ")
        q_rad_prev = _slew_to_home(driver)
        ee_xyz_now = fk.ee_xyz(q_rad_prev)
        print(f"[state] homed: q_sim_rad={q_rad_prev.round(3)}  "
              f"ee_xyz={ee_xyz_now.round(3)}")
        if not args.no_confirm:
            input("[state] place block(s) within x∈(0.10,0.30), y∈(-0.15,0.15). "
                  "Press <enter> to start rollout, ctrl-C to abort … ")

        for sub_k, color in enumerate(colors):
            if sub_k > 0 and not args.no_home_between_subgoals:
                print(f"[state] homing between sub-goals "
                      f"({sub_k - 1} → {sub_k}) …")
                q_rad_prev = _slew_to_home(driver)

            q_rad_prev, _ = _run_subgoal(
                sub_k=sub_k,
                color=color,
                driver=driver,
                policy=policy,
                fk=fk,
                detector=detector,
                K_mat=K_mat,
                dist=dist,
                bowl_xy=bowl_xy,
                q_rad_prev=q_rad_prev,
                dump_dir=dump_dir,
                release_detect=args.release_detect,
            )

            if (
                multi
                and args.release_detect == "manual"
                and not args.no_confirm
                and sub_k < len(colors) - 1
            ):
                input(f"[state] sub-goal {sub_k} ({color}) complete. "
                      "Press <enter> for next sub-goal, ctrl-C to abort … ")

        print(f"\n[state] all {len(colors)} sub-goal(s) finished.")
    finally:
        driver.disconnect()
        if dump_dir is not None:
            print(f"[state] debug dump written to {dump_dir}")


def _parse_colors(args) -> list[str]:
    """Resolve the sub-goal colour sequence from CLI args.

    Precedence: ``--colors`` (Eval-3, comma-separated) → ``--target-color``
    (Eval-1/2, single). Exactly one must be passed. ``--colors red`` (single
    entry) is equivalent to ``--target-color red``.
    """
    if args.colors is not None:
        seq = [c.strip() for c in args.colors.split(",") if c.strip()]
        if not seq:
            raise ValueError("--colors parsed to an empty list")
        return seq
    if args.target_color is not None:
        return [args.target_color.strip()]
    # Default Eval-1 path: red cube.
    return ["red"]


def main() -> int:
    p = argparse.ArgumentParser(
        description="State-only + AprilTag PPO closed-loop deploy on real "
                    "SO-ARM101 (Eval-1 / Eval-2 / Eval-3).",
    )
    p.add_argument("--bowl-xy", type=str, required=True,
                   help="Comma-separated 'x,y' metres, robot base frame, "
                        "e.g. 0.20,-0.05. Shared across all sub-goals.")
    p.add_argument("--target-color", type=str, default=None,
                   help="Single target colour (Eval-1/2). One of "
                        f"{sorted(_COLOR_TO_ID)}. Default 'red' if neither "
                        "--target-color nor --colors is given.")
    p.add_argument("--colors", type=str, default=None,
                   help="Comma-separated colour sequence for Eval-3 "
                        "(e.g. 'red,blue,yellow'). Overrides --target-color.")
    p.add_argument("--tag-size", type=float, default=0.015,
                   help="Cube tag edge length in metres (default 15 mm).")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Override checkpoint path. Default search: "
                        "deploy/runs/state_apriltag_model.pt → "
                        "deploy/runs/model.pt.")
    p.add_argument("--release-detect", choices=["manual", "vision", "timed"],
                   default="manual",
                   help="Eval-3 sub-goal release-detect mode. 'manual' "
                        "(default): operator confirms between sub-goals. "
                        "'vision': early-advance when the target tag is in "
                        "the bowl region for several consecutive frames. "
                        "'timed': always run the full per-sub-goal budget.")
    p.add_argument("--no-home-between-subgoals", action="store_true",
                   help="Skip the homing step between Eval-3 sub-goals "
                        "(default: home on between sub-goals).")
    p.add_argument("--dry-run", action="store_true",
                   help="No hardware, no camera; run one forward with "
                        "synthetic obs.")
    p.add_argument("--no-confirm", action="store_true",
                   help="Skip the pre-rollout and inter-sub-goal <enter> "
                        "prompts and start immediately.")
    p.add_argument("--debug-dump", action="store_true",
                   help="Save per-step state/action JSONL under "
                        "deploy/runs/debug_state/<timestamp>/.")
    args = p.parse_args()

    x, y = (float(s) for s in args.bowl_xy.split(","))
    bowl_xy = np.array([x, y], dtype=np.float32)

    colors = _parse_colors(args)
    run(bowl_xy, colors, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
