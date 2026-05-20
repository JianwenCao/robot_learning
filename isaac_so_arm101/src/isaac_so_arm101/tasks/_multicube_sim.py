# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared simulation / PhysX param helpers for the multi-cube tasks.

Eval-1 (single cube, 1 rigid body / env) trains at num_envs=4096 with
the default PhysX GPU buffers. The Eval-2/3/Bonus tasks spawn six cubes
per env (one per palette color), which:

* Multiplies the number of dynamic rigid bodies by 6 — solver work
  scales roughly linearly.
* Multiplies the number of cube-cube contact pairs by up to ~6² (in
  practice, much less — only adjacent cubes touch). Singulation's
  stacked arrangement has up to 3 contact pairs per stack, plus
  cube-table contacts.
* Adds aggregate-pair work for each cube ↔ everything-else broadphase
  test, growing as ``num_envs × n_cubes``.

PhysX defaults pass for Eval-1 but **silently dropping contacts** when
the buffers overflow on multi-cube envs — visible only as a warning in
the kit log and degraded contact realism. The capacities below are
sized so 4096 envs × 6 cubes ≈ 24k bodies fit comfortably:

* ``gpu_total_aggregate_pairs_capacity = 256k`` (16× Eval-1's 16k) —
  Eval-1 was already on the edge.
* ``gpu_found_lost_aggregate_pairs_capacity = 16M`` (4× Eval-1's 4M) —
  scales with the broadphase update frequency × n_pairs.
* ``gpu_max_rigid_contact_count = 1M`` — Eval-1 default (524288) is
  plenty for a single cube but singulation stacks can blow through it
  during the brief moments when the policy slams cubes together.
* ``gpu_max_rigid_patch_count = 320k`` — same reasoning.

The recommended ``num_envs`` defaults are conservative starting points:

* ``training=2048`` — half of Eval-1's teacher (4096) because the per-
  step cost scales with cubes × envs. Good starting point on a 24 GB
  GPU. Scale up to 4096 if VRAM allows; scale down to 1024 on a 12 GB
  card.
* ``play=16`` — small, for visual inspection.

These knobs are kept on this helper so all three multi-cube tasks share
one source of truth — changing the buffer sizes here propagates to
every env that calls :func:`apply_multicube_sim_settings`.
"""

from __future__ import annotations


# Recommended num_envs — knobs the train script can override per run.
# Bumped 2048 → 4096 (2026-05-20) to match Eval-1's sample budget after
# Eval-2 v2 stalled at ~770 iters with no consistent lift. At 2048 envs,
# Eval-2's 6-cube physics used only ~4.5 GB / 32 GB VRAM; doubling to
# 4096 lands at ~9 GB and halves wall-clock per env-step of training
# signal. Matches what Eval-1's working state-AprilTag baseline used.
DEFAULT_TRAIN_NUM_ENVS = 4096
DEFAULT_PLAY_NUM_ENVS = 16

# Shared physics step parameters — match Eval-1 so action dynamics
# carry over to multi-cube tasks identically (50 Hz control / 100 Hz
# physics is the same rate the deploy loop hits on the real Feetech bus).
SIM_DT = 0.01            # 100 Hz physics
DECIMATION = 2            # → 50 Hz policy step
ENV_SPACING = 2.5         # m — matches Eval-1; comfortable for 6 cubes/env

# PhysX GPU buffer capacities — multi-cube safe sizes.
GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY = 256 * 1024            # 16× Eval-1's 16k
GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY = 16 * 1024 * 1024  # 4× Eval-1's 4M
GPU_MAX_RIGID_CONTACT_COUNT = 1024 * 1024                   # 2× PhysX default
GPU_MAX_RIGID_PATCH_COUNT = 320 * 1024                      # ~4× PhysX default

# Solver / stability knobs.
BOUNCE_THRESHOLD_VELOCITY = 0.01     # below this, contacts don't bounce — stable for cube stacks
FRICTION_CORRELATION_DISTANCE = 0.00625  # PhysX default; keeps friction patches accurate at cube scale


def apply_multicube_sim_settings(cfg) -> None:
    """Apply the shared timing + PhysX buffer settings to a ManagerBasedRLEnvCfg.

    Call from ``__post_init__`` after the parent's ``super().__post_init__()``.
    Does not change ``num_envs`` or ``episode_length_s`` (task-specific) or
    the viewer eye (env-specific). Only the timing + PhysX knobs that are
    safe to share across the multi-cube task family.
    """
    cfg.decimation = DECIMATION
    cfg.sim.dt = SIM_DT
    cfg.sim.render_interval = cfg.decimation

    cfg.sim.physx.bounce_threshold_velocity = BOUNCE_THRESHOLD_VELOCITY
    cfg.sim.physx.friction_correlation_distance = FRICTION_CORRELATION_DISTANCE

    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY
    cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY

    # Not all Isaac Lab builds expose every PhysX attribute — try/except
    # makes this forward-compatible. The two below are present on the
    # current pinned version (Isaac Lab 2.3.0); guard anyway.
    try:
        cfg.sim.physx.gpu_max_rigid_contact_count = GPU_MAX_RIGID_CONTACT_COUNT
    except AttributeError:
        pass
    try:
        cfg.sim.physx.gpu_max_rigid_patch_count = GPU_MAX_RIGID_PATCH_COUNT
    except AttributeError:
        pass
