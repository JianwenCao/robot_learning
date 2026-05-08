# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bowl-goal command for the pick-and-place task.

This is a thin subclass of Isaac Lab's :class:`UniformPoseCommand` that
**rejects samples too close to the block** so the policy never gets a free
"block already in bowl" episode. EVAL1_PLAN §3.1 calls for the constraint
``‖block_xy − bowl_xy‖ ≥ min_distance``; we enforce it here instead of in
a generic event term because:

* The reset order in :class:`ManagerBasedRLEnv` is:
  ``event_manager(mode="reset")`` → ``command_manager.reset``. So when the
  command resamples, the *block* is already in its new pose — we can read
  it cheaply.
* If we tried to enforce the constraint in an event term, the command
  manager would later overwrite our work because it always resamples on
  reset (see ``CommandManager.reset`` → ``CommandTerm._resample``).

The motivating observation: a 100-iter state-only run with overlap-allowed
sampling produced ``Episode_Reward/place ≈ 1.24`` per episode while
``grasp ≈ 0`` — i.e., the policy was getting paid for blocks that
*spawned* near the bowl, not blocks it placed there.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class BowlPoseCommand(UniformPoseCommand):
    """Uniform pose command with rejection sampling against a target asset.

    Re-samples the (x, y) component up to ``max_attempts`` times until
    ``‖command_xy − target_xy_b‖ ≥ min_distance`` (in the robot root frame).
    If the budget is exhausted we keep the last sample — this fails
    gracefully and shows up in the per-episode metrics rather than as a
    silent infinite loop.
    """

    cfg: BowlPoseCommandCfg

    def __init__(self, cfg: BowlPoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._target_asset: RigidObject = env.scene[cfg.target_asset_name]

    def _resample_command(self, env_ids: Sequence[int]):
        # Take the upstream uniform sample first (sets pose_command_b in
        # robot frame for env_ids).
        super()._resample_command(env_ids)

        # Compute the target asset's xy in the robot root frame for the
        # envs being resampled. The asset is already at its post-reset
        # pose because event_manager(mode="reset") ran before us.
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        robot_root_xy_w = self.robot.data.root_pos_w[env_ids_t, :2]
        target_xy_w = self._target_asset.data.root_pos_w[env_ids_t, :2]
        target_xy_b = target_xy_w - robot_root_xy_w  # robot-frame xy

        min_d = float(self.cfg.min_distance)
        for _ in range(int(self.cfg.max_attempts)):
            cur_xy = self.pose_command_b[env_ids_t, :2]
            dist = torch.norm(cur_xy - target_xy_b, dim=1)
            bad = dist < min_d
            if not torch.any(bad):
                break
            # Resample only the bad envs, in the same xy ranges as the
            # parent UniformPoseCommandCfg.
            bad_global_ids = env_ids_t[bad]
            n = bad_global_ids.numel()
            r = torch.empty(n, device=self.device)
            self.pose_command_b[bad_global_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
            self.pose_command_b[bad_global_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)


@configclass
class BowlPoseCommandCfg(UniformPoseCommandCfg):
    """Config for :class:`BowlPoseCommand`."""

    class_type: type = BowlPoseCommand

    target_asset_name: str = MISSING
    """Name of the asset (in :attr:`InteractiveScene`) that the bowl xy
    must stay at least :attr:`min_distance` away from. Typically ``"object"``."""

    min_distance: float = 0.10
    """Minimum xy-plane distance between the sampled bowl and the target
    asset, in meters (robot frame). Defaults to 10 cm per EVAL1_PLAN §3.1."""

    max_attempts: int = 8
    """Maximum number of rejection-sampling attempts before giving up.
    8 attempts × ~80 % accept rate per draw → < 0.001 % stuck, which is
    fine: stuck envs simply train on a slightly close bowl that episode."""
