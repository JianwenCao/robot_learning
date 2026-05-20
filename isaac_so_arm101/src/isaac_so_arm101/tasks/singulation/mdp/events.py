# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Placement and DR events for the Bonus-B singulation task.

Per reset, ``sample_active_set`` draws one of **11 arrangement families**
covering stacks, flat clusters, pyramids, and mixed stack+cluster
configurations (see ``ARRANGEMENT_SPECS`` below). For each family the
function:

1. Samples a per-env arrangement id from ``arrangement_weights``.
2. Looks up the n_active count from the arrangement spec.
3. Computes per-slot local xyz around (0, 0) — the layout relative to
   the cluster center, before yaw rotation and translation.
4. Rotates by a per-env yaw ``θ ∈ [0, 2π)`` and translates to the
   sampled cluster center ``(cx, cy)``.
5. Writes per-cube poses; parks the inactive palette cubes off-table.

Cached on the env for downstream consumers:

* ``env._singulation_n_active``         (N,)        long  — 3 or 4 per env
* ``env._singulation_active_mask``      (N, NUM_COLORS) bool — palette indices of active cubes
* ``env._singulation_arrangement_idx``  (N,)        long  — 0..10 (ARR_ID)
* ``env._singulation_cluster_center_xy``(N, 2)      float — for the bowl rejection sampler
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


COLOR_NAMES: tuple[str, ...] = ("blue", "yellow", "purple", "orange", "green", "red")
NUM_COLORS = len(COLOR_NAMES)

BLOCK_COLORS: dict[str, tuple[float, float, float]] = {
    "blue":   (0.17, 0.27, 0.62),
    "yellow": (0.95, 0.77, 0.06),
    "purple": (0.42, 0.22, 0.51),
    "orange": (0.90, 0.35, 0.16),
    "green":  (0.15, 0.49, 0.28),
    "red":    (0.78, 0.14, 0.17),
}

# Off-table parking slots for cubes not in the active set. Each cube has
# its own slot so parked cubes never collide. x = -0.60 keeps them out of
# the wrist-cam frame (workspace centre is x ∈ [0.16, 0.22]).
HIDDEN_PARK_XY: tuple[tuple[float, float], ...] = tuple(
    (-0.60, -0.25 + 0.10 * i) for i in range(NUM_COLORS)
)


# ---------------------------------------------------------------------------
# Arrangement registry
# ---------------------------------------------------------------------------


# Ordered tuple of (name, n_active) — index in this tuple == ARR_ID.
# Order is load-bearing: arrangement_onehot downstream reads this index.
ARRANGEMENT_SPECS: tuple[tuple[str, int], ...] = (
    ("STACK_3",              3),
    ("STACK_4",              4),
    ("CLUSTER_LINE_3",       3),
    ("CLUSTER_TRI_3",        3),
    ("CLUSTER_SQUARE_4",     4),
    ("CLUSTER_LINE_4",       4),
    ("CLUSTER_L_4",          4),
    ("PYRAMID_3",            3),
    ("PYRAMID_4",            4),
    ("MIXED_2STACK_PLUS_1",  3),
    ("MIXED_2STACK_PLUS_2",  4),
)
ARRANGEMENT_NAMES: tuple[str, ...] = tuple(name for name, _ in ARRANGEMENT_SPECS)
ARR_ID: dict[str, int] = {name: i for i, (name, _) in enumerate(ARRANGEMENT_SPECS)}
NUM_ARRANGEMENTS = len(ARRANGEMENT_SPECS)
MAX_N_ACTIVE = 4

# Default sampling weights per §3 of BONUS_B_PLAN.md.
# Stacks 0.25, flat clusters 0.40, pyramids 0.20, mixed 0.15.
DEFAULT_ARRANGEMENT_WEIGHTS: dict[str, float] = {
    "STACK_3":              0.125,
    "STACK_4":              0.125,
    "CLUSTER_LINE_3":       0.075,
    "CLUSTER_TRI_3":        0.075,
    "CLUSTER_SQUARE_4":     0.100,
    "CLUSTER_LINE_4":       0.075,
    "CLUSTER_L_4":          0.075,
    "PYRAMID_3":            0.100,
    "PYRAMID_4":            0.100,
    "MIXED_2STACK_PLUS_1":  0.075,
    "MIXED_2STACK_PLUS_2":  0.075,
}


# ---------------------------------------------------------------------------
# Per-family slot xyz computation
# ---------------------------------------------------------------------------
#
# Each helper returns ``slot_local_xyz`` of shape ``(n, MAX_N_ACTIVE, 3)`` —
# the local xyz of each slot relative to the cluster centre (cx, cy) and
# pre-yaw rotation. Slots beyond the family's n_active are placeholders
# (any value — they're masked off / parked by the caller).
#
# All helpers receive:
#   n: number of envs being placed for this family
#   cube_size: cube edge length (m)
#   table_z: top-of-table z in world frame (~ 0.01 for current setup)
#   device: torch device for the returned tensor
#   gen_kwargs: per-family knobs (jitter, spacing, etc.)
#
# Coordinate convention: x to the right, y forward (robot frame +x in the
# table plane). Yaw rotation is applied by the caller around (cx, cy).
# ---------------------------------------------------------------------------


def _stack_layout(
    n: int, n_active: int, cube_size: float, table_z: float, device,
    stack_lateral_jitter: float,
) -> torch.Tensor:
    """Vertical stack of ``n_active`` cubes with small lateral jitter."""
    slot_local = torch.zeros((n, MAX_N_ACTIVE, 3), device=device)
    jx = torch.empty((n, MAX_N_ACTIVE), device=device).uniform_(-stack_lateral_jitter, stack_lateral_jitter)
    jy = torch.empty((n, MAX_N_ACTIVE), device=device).uniform_(-stack_lateral_jitter, stack_lateral_jitter)
    k = torch.arange(MAX_N_ACTIVE, device=device).view(1, -1).float()
    slot_local[:, :, 0] = jx
    slot_local[:, :, 1] = jy
    slot_local[:, :, 2] = table_z + cube_size * (k + 0.5)
    return slot_local


def _line_layout(
    n: int, n_active: int, cube_size: float, table_z: float, device,
    inter_spacing: float, position_jitter: float,
) -> torch.Tensor:
    """``n_active`` cubes in a line along local +x, centred on (0, 0)."""
    slot_local = torch.zeros((n, MAX_N_ACTIVE, 3), device=device)
    # Slot k at x = (k - (n_active-1)/2) * inter_spacing
    k = torch.arange(MAX_N_ACTIVE, device=device).float()
    xs = (k - (n_active - 1) * 0.5) * inter_spacing      # (MAX_N_ACTIVE,)
    slot_local[:, :, 0] = xs.view(1, -1).expand(n, -1)
    slot_local[:, :, 0] += torch.empty((n, MAX_N_ACTIVE), device=device).uniform_(-position_jitter, position_jitter)
    slot_local[:, :, 1] = torch.empty((n, MAX_N_ACTIVE), device=device).uniform_(-position_jitter, position_jitter)
    slot_local[:, :, 2] = table_z + cube_size * 0.5
    return slot_local


def _triangle_layout(
    n: int, cube_size: float, table_z: float, device,
    inter_spacing: float, position_jitter: float,
) -> torch.Tensor:
    """3 cubes in an equilateral triangle, side = inter_spacing, centred on (0, 0)."""
    slot_local = torch.zeros((n, MAX_N_ACTIVE, 3), device=device)
    # Circumradius of an equilateral triangle with side s: R = s / sqrt(3)
    r = inter_spacing / math.sqrt(3.0)
    # Vertices at angles 90°, 210°, 330° (cube 0 above centre, then CCW)
    angles = torch.tensor(
        [math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 - 2 * math.pi / 3],
        device=device,
    )
    corners = torch.stack([r * angles.cos(), r * angles.sin()], dim=-1)  # (3, 2)
    slot_local[:, :3, :2] = corners.unsqueeze(0).expand(n, -1, -1)
    slot_local[:, :, :2] += torch.empty((n, MAX_N_ACTIVE, 2), device=device).uniform_(-position_jitter, position_jitter)
    slot_local[:, :, 2] = table_z + cube_size * 0.5
    return slot_local


def _square_layout(
    n: int, cube_size: float, table_z: float, device,
    inter_spacing: float, position_jitter: float,
) -> torch.Tensor:
    """2×2 attached cluster — Eval-3's placement."""
    slot_local = torch.zeros((n, MAX_N_ACTIVE, 3), device=device)
    s = inter_spacing * 0.5
    corners = torch.tensor(
        [[+s, +s], [-s, +s], [-s, -s], [+s, -s]], device=device,
    )  # (4, 2)
    slot_local[:, :, :2] = corners.unsqueeze(0).expand(n, -1, -1)
    slot_local[:, :, :2] += torch.empty((n, MAX_N_ACTIVE, 2), device=device).uniform_(-position_jitter, position_jitter)
    slot_local[:, :, 2] = table_z + cube_size * 0.5
    return slot_local


def _l_layout(
    n: int, cube_size: float, table_z: float, device,
    inter_spacing: float, position_jitter: float,
) -> torch.Tensor:
    """L-shape: 3 in a row along +x, 1 perpendicular at the +x end (+y)."""
    slot_local = torch.zeros((n, MAX_N_ACTIVE, 3), device=device)
    s = inter_spacing
    # 3 in a line, centred on (-s/2, 0): (-1.5s, 0), (-0.5s, 0), (+0.5s, 0)
    # 4th cube at (+0.5s, +s) — perpendicular at the +x end.
    corners = torch.tensor(
        [[-1.5 * s, 0.0], [-0.5 * s, 0.0], [+0.5 * s, 0.0], [+0.5 * s, +s]],
        device=device,
    )
    # Recenter so the L's centroid is at origin (better behaviour under yaw rotation).
    corners = corners - corners.mean(dim=0, keepdim=True)
    slot_local[:, :, :2] = corners.unsqueeze(0).expand(n, -1, -1)
    slot_local[:, :, :2] += torch.empty((n, MAX_N_ACTIVE, 2), device=device).uniform_(-position_jitter, position_jitter)
    slot_local[:, :, 2] = table_z + cube_size * 0.5
    return slot_local


def _pyramid_3_layout(
    n: int, cube_size: float, table_z: float, device,
    position_jitter: float,
) -> torch.Tensor:
    """2-1 pyramid: 2 cubes touching face-to-face on the table + 1 on top.

    Bottom cubes share a face along local x; top cube straddles the seam,
    resting in the saddle. Top cube z spawn is slightly above the saddle
    to let it settle (avoids initial-penetration artefacts).
    """
    slot_local = torch.zeros((n, MAX_N_ACTIVE, 3), device=device)
    # Bottom pair touching face-to-face along x (centres ±cube_size/2 apart).
    half = cube_size * 0.5
    slot_local[:, 0, 0] = -half  # left bottom
    slot_local[:, 1, 0] = +half  # right bottom
    slot_local[:, 0:2, 2] = table_z + half
    # Top cube above the seam — sits at z ≈ table_z + 1.5*cube + small eps so
    # it falls into the saddle without initial penetration. Saddle equilibrium
    # is z ≈ table_z + cube_size * (1 + sin(45°)/2) ≈ table_z + 0.027 for 2 cm cubes.
    slot_local[:, 2, 0] = 0.0
    slot_local[:, 2, 2] = table_z + cube_size * 1.5  # spawn a bit high, gravity settles
    slot_local[:, :3, :2] += torch.empty((n, 3, 2), device=device).uniform_(-position_jitter, position_jitter)
    return slot_local


def _pyramid_4_layout(
    n: int, cube_size: float, table_z: float, device,
    position_jitter: float,
) -> torch.Tensor:
    """3-1 pyramid: 3 bottom in equilateral triangle (touching) + 1 on top centroid."""
    slot_local = torch.zeros((n, MAX_N_ACTIVE, 3), device=device)
    # Bottom triangle with side = cube_size (touching faces).
    side = cube_size
    r = side / math.sqrt(3.0)
    angles = torch.tensor(
        [math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 - 2 * math.pi / 3],
        device=device,
    )
    corners = torch.stack([r * angles.cos(), r * angles.sin()], dim=-1)  # (3, 2)
    slot_local[:, :3, :2] = corners.unsqueeze(0).expand(n, -1, -1)
    slot_local[:, :3, 2] = table_z + cube_size * 0.5
    # Top cube at (0, 0), spawn z above saddle so it settles cleanly.
    slot_local[:, 3, 0] = 0.0
    slot_local[:, 3, 1] = 0.0
    slot_local[:, 3, 2] = table_z + cube_size * 1.7
    slot_local[:, :, :2] += torch.empty((n, MAX_N_ACTIVE, 2), device=device).uniform_(-position_jitter, position_jitter)
    return slot_local


def _mixed_2stack_plus_1_layout(
    n: int, cube_size: float, table_z: float, device,
    stack_lateral_jitter: float, gap: float, position_jitter: float,
) -> torch.Tensor:
    """2-stack + 1 standalone cube ~gap away on the table.

    Stack at local (-gap/2, 0); standalone at (+gap/2, 0).
    """
    slot_local = torch.zeros((n, MAX_N_ACTIVE, 3), device=device)
    half_g = gap * 0.5
    jx = torch.empty((n, 2), device=device).uniform_(-stack_lateral_jitter, stack_lateral_jitter)
    jy = torch.empty((n, 2), device=device).uniform_(-stack_lateral_jitter, stack_lateral_jitter)
    # Slot 0, 1: stack bottom + top, at (-half_g, 0)
    slot_local[:, 0, 0] = -half_g + jx[:, 0]
    slot_local[:, 0, 1] = jy[:, 0]
    slot_local[:, 0, 2] = table_z + cube_size * 0.5
    slot_local[:, 1, 0] = -half_g + jx[:, 1]
    slot_local[:, 1, 1] = jy[:, 1]
    slot_local[:, 1, 2] = table_z + cube_size * 1.5
    # Slot 2: standalone at (+half_g, 0)
    slot_local[:, 2, 0] = +half_g + torch.empty(n, device=device).uniform_(-position_jitter, position_jitter)
    slot_local[:, 2, 1] = torch.empty(n, device=device).uniform_(-position_jitter, position_jitter)
    slot_local[:, 2, 2] = table_z + cube_size * 0.5
    return slot_local


def _mixed_2stack_plus_2_layout(
    n: int, cube_size: float, table_z: float, device,
    stack_lateral_jitter: float, gap: float, inter_spacing: float, position_jitter: float,
) -> torch.Tensor:
    """2-stack + 2 in flat contact ~gap away on the table."""
    slot_local = torch.zeros((n, MAX_N_ACTIVE, 3), device=device)
    half_g = gap * 0.5
    jx = torch.empty((n, 2), device=device).uniform_(-stack_lateral_jitter, stack_lateral_jitter)
    jy = torch.empty((n, 2), device=device).uniform_(-stack_lateral_jitter, stack_lateral_jitter)
    # Slot 0, 1: stack at (-half_g, 0)
    slot_local[:, 0, 0] = -half_g + jx[:, 0]
    slot_local[:, 0, 1] = jy[:, 0]
    slot_local[:, 0, 2] = table_z + cube_size * 0.5
    slot_local[:, 1, 0] = -half_g + jx[:, 1]
    slot_local[:, 1, 1] = jy[:, 1]
    slot_local[:, 1, 2] = table_z + cube_size * 1.5
    # Slots 2, 3: pair at (+half_g, ±inter_spacing/2)
    s = inter_spacing * 0.5
    slot_local[:, 2, 0] = +half_g
    slot_local[:, 2, 1] = +s
    slot_local[:, 2, 2] = table_z + cube_size * 0.5
    slot_local[:, 3, 0] = +half_g
    slot_local[:, 3, 1] = -s
    slot_local[:, 3, 2] = table_z + cube_size * 0.5
    slot_local[:, 2:4, :2] += torch.empty((n, 2, 2), device=device).uniform_(-position_jitter, position_jitter)
    return slot_local


def _compute_family_xyz(
    name: str, n: int, n_active: int, cube_size: float, table_z: float, device,
    stack_lateral_jitter: float, cluster_inter_spacing: float, cluster_position_jitter: float,
    mixed_gap: float,
) -> torch.Tensor:
    """Dispatch to the per-family helper. Returns (n, MAX_N_ACTIVE, 3) local xyz."""
    if name in ("STACK_3", "STACK_4"):
        return _stack_layout(n, n_active, cube_size, table_z, device, stack_lateral_jitter)
    if name in ("CLUSTER_LINE_3", "CLUSTER_LINE_4"):
        return _line_layout(n, n_active, cube_size, table_z, device, cluster_inter_spacing, cluster_position_jitter)
    if name == "CLUSTER_TRI_3":
        return _triangle_layout(n, cube_size, table_z, device, cluster_inter_spacing, cluster_position_jitter)
    if name == "CLUSTER_SQUARE_4":
        return _square_layout(n, cube_size, table_z, device, cluster_inter_spacing, cluster_position_jitter)
    if name == "CLUSTER_L_4":
        return _l_layout(n, cube_size, table_z, device, cluster_inter_spacing, cluster_position_jitter)
    if name == "PYRAMID_3":
        return _pyramid_3_layout(n, cube_size, table_z, device, cluster_position_jitter)
    if name == "PYRAMID_4":
        return _pyramid_4_layout(n, cube_size, table_z, device, cluster_position_jitter)
    if name == "MIXED_2STACK_PLUS_1":
        return _mixed_2stack_plus_1_layout(
            n, cube_size, table_z, device, stack_lateral_jitter, mixed_gap, cluster_position_jitter,
        )
    if name == "MIXED_2STACK_PLUS_2":
        return _mixed_2stack_plus_2_layout(
            n, cube_size, table_z, device, stack_lateral_jitter, mixed_gap, cluster_inter_spacing, cluster_position_jitter,
        )
    raise ValueError(f"Unknown arrangement name: {name!r}")


# ---------------------------------------------------------------------------
# Main reset event — sample active set + arrangement + cube poses
# ---------------------------------------------------------------------------


def sample_active_set(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    arrangement_weights: dict[str, float] | None = None,
    cube_size: float = 0.02,
    table_z: float = 0.01,
    stack_lateral_jitter: float = 0.005,
    cluster_inter_spacing: float = 0.021,
    cluster_position_jitter: float = 0.003,
    mixed_gap: float = 0.07,
    center_x: tuple[float, float] = (0.16, 0.22),
    center_y: tuple[float, float] = (-0.08, 0.08),
    cube_prefix: str = "cube_",
) -> None:
    """Sample active cubes + initial arrangement and write per-cube poses.

    See module docstring for the list of side effects on the env.
    """
    if isinstance(env_ids, torch.Tensor):
        if env_ids.numel() == 0:
            return
        ids = env_ids.long()
    else:
        if len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    n = ids.numel()
    device = env.device

    # ---- Sample arrangement per env --------------------------------------
    weights_dict = arrangement_weights if arrangement_weights is not None else DEFAULT_ARRANGEMENT_WEIGHTS
    weights = torch.tensor(
        [weights_dict.get(name, 0.0) for name in ARRANGEMENT_NAMES],
        device=device, dtype=torch.float32,
    )
    if weights.sum().item() <= 0.0:
        raise ValueError(f"All arrangement weights are zero: {weights_dict}")
    weights = weights / weights.sum()
    arrangement_idx = torch.multinomial(weights, n, replacement=True)  # (n,) long ∈ [0, NUM_ARRANGEMENTS)

    # Per-env n_active derived from arrangement spec.
    n_active_per_arr = torch.tensor([na for _, na in ARRANGEMENT_SPECS], device=device, dtype=torch.long)
    n_active = n_active_per_arr[arrangement_idx]  # (n,)

    # ---- Sample which palette cubes are active --------------------------
    perms = torch.argsort(torch.rand((n, NUM_COLORS), device=device), dim=1)  # (n, 6)
    slot_indices = torch.arange(NUM_COLORS, device=device).view(1, -1).expand(n, -1)
    active_in_slot = slot_indices < n_active.view(-1, 1)
    active_palette_mask = torch.zeros((n, NUM_COLORS), dtype=torch.bool, device=device)
    active_palette_mask.scatter_(1, perms, active_in_slot)
    slot_order = perms[:, :MAX_N_ACTIVE]  # (n, MAX_N_ACTIVE) — palette idx per slot

    # ---- Workspace centre + yaw per env ---------------------------------
    cx = torch.empty(n, device=device).uniform_(*center_x)
    cy = torch.empty(n, device=device).uniform_(*center_y)
    yaw = torch.empty(n, device=device).uniform_(0.0, 2.0 * math.pi)

    # ---- Per-family local slot xyz (pre-yaw, pre-translation) -----------
    slot_local_xyz = torch.zeros((n, MAX_N_ACTIVE, 3), device=device)
    for arr_name, arr_id in ARR_ID.items():
        mask = arrangement_idx == arr_id
        if not mask.any():
            continue
        idx_in_family = mask.nonzero(as_tuple=True)[0]
        n_fam = idx_in_family.numel()
        n_active_fam = n_active_per_arr[arr_id].item()
        family_xyz = _compute_family_xyz(
            arr_name, n_fam, n_active_fam, cube_size, table_z, device,
            stack_lateral_jitter, cluster_inter_spacing, cluster_position_jitter, mixed_gap,
        )
        slot_local_xyz[idx_in_family] = family_xyz

    # ---- Apply yaw rotation around (0, 0) then translate to (cx, cy) -----
    cos_y = yaw.cos().view(-1, 1)  # (n, 1)
    sin_y = yaw.sin().view(-1, 1)
    x_local = slot_local_xyz[:, :, 0]
    y_local = slot_local_xyz[:, :, 1]
    x_rot = cos_y * x_local - sin_y * y_local
    y_rot = sin_y * x_local + cos_y * y_local
    slot_local_xyz[:, :, 0] = x_rot + cx.view(-1, 1)
    slot_local_xyz[:, :, 1] = y_rot + cy.view(-1, 1)

    # ---- Write poses per cube -------------------------------------------
    robot = env.scene["robot"]
    root_xy_w = robot.data.root_pos_w[ids, :2]  # (n, 2)

    quat = torch.zeros((n, 4), device=device)
    quat[:, 0] = 1.0
    zero_vel = torch.zeros((n, 6), device=device)

    for k, name in enumerate(COLOR_NAMES):
        cube = env.scene[f"{cube_prefix}{name}"]
        in_slot = (slot_order == k)  # (n, MAX_N_ACTIVE) bool
        is_active = active_palette_mask[:, k]
        slot_idx = in_slot.float().argmax(dim=1).long()  # (n,)
        pos_local_active = slot_local_xyz.gather(
            1, slot_idx.view(-1, 1, 1).expand(-1, 1, 3)
        ).squeeze(1)  # (n, 3)

        park_x, park_y = HIDDEN_PARK_XY[k]
        pos_local_parked = torch.stack(
            [
                torch.full((n,), park_x, device=device),
                torch.full((n,), park_y, device=device),
                torch.full((n,), 0.05,   device=device),
            ],
            dim=1,
        )
        valid_slot = slot_idx < n_active
        is_truly_active = is_active & valid_slot

        pos_local = torch.where(
            is_truly_active.view(-1, 1), pos_local_active, pos_local_parked
        )
        pos_w = pos_local.clone()
        pos_w[:, :2] += root_xy_w
        pose = torch.cat([pos_w, quat], dim=1)
        cube.write_root_pose_to_sim(pose, env_ids=ids)
        cube.write_root_velocity_to_sim(zero_vel, env_ids=ids)

    # ---- Cache buffers ---------------------------------------------------
    if not hasattr(env, "_singulation_n_active"):
        env._singulation_n_active = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        env._singulation_active_mask = torch.zeros((env.num_envs, NUM_COLORS), dtype=torch.bool, device=device)
        env._singulation_arrangement_idx = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        env._singulation_cluster_center_xy = torch.zeros((env.num_envs, 2), device=device)
    env._singulation_n_active[ids] = n_active
    env._singulation_active_mask[ids] = active_palette_mask
    env._singulation_arrangement_idx[ids] = arrangement_idx
    env._singulation_cluster_center_xy[ids, 0] = cx
    env._singulation_cluster_center_xy[ids, 1] = cy


def reset_singulation_latches(env: "ManagerBasedEnv", env_ids: torch.Tensor) -> None:
    """Clear per-episode success latches."""
    if isinstance(env_ids, torch.Tensor):
        if env_ids.numel() == 0:
            return
        ids = env_ids.long()
    else:
        if len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    for name in ("_singulation_success_latch",):
        flag = getattr(env, name, None)
        if flag is not None:
            flag[ids] = False


# ---------------------------------------------------------------------------
# Per-episode physics DR — cube mass + friction
# ---------------------------------------------------------------------------


def randomize_cube_physics(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    mass_range: tuple[float, float] = (0.016, 0.024),
    friction_range: tuple[float, float] = (0.7, 1.3),
    cube_prefix: str = "cube_",
) -> None:
    """Per-episode DR on cube mass and cube↔cube / cube↔table friction.

    Width sized to bracket the most likely real-cube parameters:

    * Mass ±20% — wooden cubes typically 15-25 g for a 2 cm edge.
    * Friction ±30% — real paint finish + table varies more than cube
      mass; static = dynamic (we don't bother distinguishing).

    Writes via the PhysX view APIs directly (no curriculum wrappers) so
    the new values take effect from the next physics step.
    """
    if isinstance(env_ids, torch.Tensor):
        if env_ids.numel() == 0:
            return
        ids_t = env_ids.long()
    else:
        if len(env_ids) == 0:
            return
        ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    n = ids_t.numel()
    device = env.device
    ids_cpu = ids_t.detach().cpu()

    new_masses = torch.empty(n, device=device).uniform_(*mass_range)
    # Friction is sampled per-env per-cube — keeps each cube's contact
    # behaviour distinct (mimics real cubes having slightly different
    # paint / wear).
    new_friction = torch.empty((n, NUM_COLORS), device=device).uniform_(*friction_range)

    for k, name in enumerate(COLOR_NAMES):
        cube: RigidObject = env.scene[f"{cube_prefix}{name}"]

        # Mass — RigidObject exposes root_physx_view.
        try:
            view = cube.root_physx_view
            masses = view.get_masses()  # (num_envs, num_bodies) on CPU per Isaac Lab
            masses[ids_cpu, 0] = new_masses.detach().cpu()
            view.set_masses(masses, indices=ids_cpu)
        except (AttributeError, RuntimeError):
            pass

        # Friction — sized per (num_envs, num_shapes, 3) [static, dynamic, restitution].
        try:
            view = cube.root_physx_view
            mats = view.get_material_properties()  # CPU tensor
            fr_k = new_friction[:, k].detach().cpu()
            mats[ids_cpu, 0, 0] = fr_k        # static friction
            mats[ids_cpu, 0, 1] = fr_k        # dynamic friction (= static)
            view.set_material_properties(mats, indices=ids_cpu)
        except (AttributeError, RuntimeError):
            pass
