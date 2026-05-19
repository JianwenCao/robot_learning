# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Placement events for the Bonus-B singulation task.

Each episode samples:

1. ``n_active`` ∈ {3, 4} — number of cubes to place in the workspace.
2. ``arrangement`` ∈ {"stacked", "clustered"} — initial config. With
   ``stacked_prob`` we draw a vertical stack; otherwise a flat cluster.

Both arrangement types satisfy the spec's "stack or clustered
arrangement of three or four blocks". Their initial pairwise distances
are well below the singulation threshold, so the policy starts every
episode in a configuration that requires separation.

The selected cubes (palette indices) are also randomized per episode so
the policy can't memorize per-color placements.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


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

HIDDEN_PARK_XY: tuple[tuple[float, float], ...] = tuple(
    (-0.60, -0.25 + 0.10 * i) for i in range(NUM_COLORS)
)


# ---------------------------------------------------------------------------
# Per-episode active set + arrangement sampling
# ---------------------------------------------------------------------------


def sample_active_set(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    n_active_choices: tuple[int, ...] = (3, 4),
    stacked_prob: float = 0.5,
    cube_size: float = 0.02,
    table_z: float = 0.01,
    # Stack arrangement parameters.
    stack_lateral_jitter: float = 0.003,
    # Cluster arrangement parameters.
    cluster_inter_spacing: float = 0.023,  # touching for 2cm cubes (≈ size + 0.3 cm gap)
    cluster_position_jitter: float = 0.002,
    # Workspace center where the arrangement is placed.
    center_x: tuple[float, float] = (0.16, 0.22),
    center_y: tuple[float, float] = (-0.08, 0.08),
    cube_prefix: str = "cube_",
) -> None:
    """Sample active cubes + initial arrangement and write per-cube poses.

    Side effects (allocated lazily):

    * ``env._singulation_n_active``    (N,)  long — 3 or 4 per env.
    * ``env._singulation_active_mask`` (N, NUM_COLORS) bool — palette
      indices of active cubes per env.
    * ``env._singulation_arrangement`` (N,)  long — 0 = stacked, 1 = clustered.
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
    max_n_active = max(n_active_choices)

    # Per-env counts.
    choice_t = torch.tensor(n_active_choices, device=device, dtype=torch.long)
    n_active = choice_t[torch.randint(0, len(n_active_choices), (n,), device=device)]

    # Per-env arrangement (0=stacked, 1=clustered).
    arrangement = (torch.rand(n, device=device) >= stacked_prob).long()

    # Sample which `n_active` of the 6 palette cubes are active (top-k of
    # a random permutation per env, masked by n_active).
    perms = torch.argsort(torch.rand((n, NUM_COLORS), device=device), dim=1)  # (n, 6)
    slot_indices = torch.arange(NUM_COLORS, device=device).view(1, -1).expand(n, -1)
    active_in_slot = slot_indices < n_active.view(-1, 1)        # (n, 6) bool over slot
    # Build (n, NUM_COLORS) bool mask: True if palette idx k is in active set.
    active_palette_mask = torch.zeros((n, NUM_COLORS), dtype=torch.bool, device=device)
    active_palette_mask.scatter_(1, perms, active_in_slot)
    # Slot order (n, max_n_active) gives the *palette index* of cube at
    # slot 0, slot 1, ... — we'll use this to assign positions to slots.
    slot_order = perms[:, :max_n_active]  # (n, 4)

    # Workspace center per env.
    cx = torch.empty(n, device=device).uniform_(*center_x)
    cy = torch.empty(n, device=device).uniform_(*center_y)

    # ----- Compute target local xyz for each slot ---------------------------
    # ``slot_local_xyz[i, slot]`` = (x, y, z) for the cube currently
    # holding slot ``slot`` in env ``i``. For slot >= n_active[i], the
    # entry is ignored (we'll park that cube).
    slot_local_xyz = torch.zeros((n, max_n_active, 3), device=device)

    # ----- Stacked arrangement -----
    # All slots at same xy (with small jitter), z = table_z + cube_size *
    # slot_idx. Small lateral jitter is sampled per slot to avoid a
    # perfectly aligned tower (more realistic, plus PhysX is happier
    # with small misalignments than perfect ones).
    is_stack = arrangement == 0
    if is_stack.any():
        stack_jx = torch.empty((n, max_n_active), device=device).uniform_(
            -stack_lateral_jitter, stack_lateral_jitter
        )
        stack_jy = torch.empty((n, max_n_active), device=device).uniform_(
            -stack_lateral_jitter, stack_lateral_jitter
        )
        slot_idx_t = torch.arange(max_n_active, device=device).view(1, -1).expand(n, -1)
        # bottom face of slot k is at table_z + k*cube_size; center at + 0.5*cube_size
        stack_z = table_z + cube_size * (slot_idx_t.float() + 0.5)
        stack_x = cx.view(-1, 1).expand(-1, max_n_active) + stack_jx
        stack_y = cy.view(-1, 1).expand(-1, max_n_active) + stack_jy
        stack_xyz = torch.stack([stack_x, stack_y, stack_z], dim=-1)  # (n, 4, 3)
        slot_local_xyz = torch.where(is_stack.view(-1, 1, 1), stack_xyz, slot_local_xyz)

    # ----- Clustered arrangement -----
    # Place slots at corners of a small square (slots 0..3) or triangle
    # (slots 0..2 when n_active=3). All at z = table_z (resting on table
    # surface center).
    is_cluster = arrangement == 1
    if is_cluster.any():
        # Offsets for up to 4 slots — square layout with side
        # ``cluster_inter_spacing``. Slot 0 at +x+y, 1 at -x+y, 2 at -x-y,
        # 3 at +x-y. For n_active=3 we drop slot 3 (parked).
        s = cluster_inter_spacing * 0.5
        offsets = torch.tensor(
            [
                [+s, +s],
                [-s, +s],
                [-s, -s],
                [+s, -s],
            ],
            device=device,
        )  # (4, 2)
        cluster_xy = offsets.view(1, max_n_active, 2).expand(n, -1, -1).clone()
        # Add small jitter
        cluster_xy += torch.empty_like(cluster_xy).uniform_(-cluster_position_jitter, cluster_position_jitter)
        cluster_xy[:, :, 0] += cx.view(-1, 1)
        cluster_xy[:, :, 1] += cy.view(-1, 1)
        cluster_z = torch.full((n, max_n_active), table_z + cube_size * 0.5, device=device)
        cluster_xyz = torch.cat([cluster_xy, cluster_z.unsqueeze(-1)], dim=-1)
        slot_local_xyz = torch.where(is_cluster.view(-1, 1, 1), cluster_xyz, slot_local_xyz)

    # ----- Write poses per cube ---------------------------------------------
    robot = env.scene["robot"]
    root_xy_w = robot.data.root_pos_w[ids, :2]  # (n, 2)

    quat = torch.zeros((n, 4), device=device)
    quat[:, 0] = 1.0
    zero_vel = torch.zeros((n, 6), device=device)

    for k, name in enumerate(COLOR_NAMES):
        cube = env.scene[f"{cube_prefix}{name}"]
        # Is cube k active? If so, in which slot?
        in_slot = (slot_order == k)  # (n, max_n_active) bool
        is_active = active_palette_mask[:, k]  # (n,) bool

        # Slot index of cube k per env (default 0 — only read when active).
        slot_idx = in_slot.float().argmax(dim=1).long()  # (n,)

        # Gather slot_local_xyz at slot_idx.
        pos_local_active = slot_local_xyz.gather(
            1, slot_idx.view(-1, 1, 1).expand(-1, 1, 3)
        ).squeeze(1)  # (n, 3)

        # Parked default
        park_x, park_y = HIDDEN_PARK_XY[k]
        pos_local_parked = torch.stack(
            [
                torch.full((n,), park_x, device=device),
                torch.full((n,), park_y, device=device),
                torch.full((n,), 0.05, device=device),
            ],
            dim=1,
        )
        # Slots beyond n_active[i] (e.g. slot 3 when n_active=3) → parked.
        # Because n_active is variable, the slot_order might point cube k
        # to a slot >= n_active for some env — those envs should park k.
        valid_slot = slot_idx < n_active  # (n,)
        is_truly_active = is_active & valid_slot

        pos_local = torch.where(
            is_truly_active.view(-1, 1), pos_local_active, pos_local_parked
        )
        pos_w = pos_local.clone()
        pos_w[:, :2] += root_xy_w
        pose = torch.cat([pos_w, quat], dim=1)
        cube.write_root_pose_to_sim(pose, env_ids=ids)
        cube.write_root_velocity_to_sim(zero_vel, env_ids=ids)

    # ----- Cache buffers ----------------------------------------------------
    if not hasattr(env, "_singulation_n_active"):
        env._singulation_n_active = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        env._singulation_active_mask = torch.zeros((env.num_envs, NUM_COLORS), dtype=torch.bool, device=device)
        env._singulation_arrangement = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    env._singulation_n_active[ids] = n_active
    env._singulation_active_mask[ids] = active_palette_mask
    env._singulation_arrangement[ids] = arrangement


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
