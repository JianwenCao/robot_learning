# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bowl-goal command for the singulation task.

The bowl is **not a scene prim** — only a 2-D xy target, exposed in the
policy observation so the singulation policy can avoid dropping cubes
into it and so the chained P2 (Eval-3 pick-and-place policy) has the
same ``bowl_xy`` state slot it expects after handoff.

We rejection-sample the bowl xy ≥ ``min_distance`` from the cluster
centre cached on the env by
:func:`~isaac_so_arm101.tasks.singulation.mdp.events.sample_active_set`.
Compare to:

* :class:`pickplace.mdp.commands.BowlPoseCommand` — rejects against a
  single asset's xy. Doesn't apply (no single "target" cube here).
* :class:`clutterpickplace.mdp.commands.ClusterBowlPoseCommand` — rejects
  against ``env._active_cube_indices`` (2 cubes). Singulation has 3–4
  active cubes per env but they're tightly clustered, so it's cheaper
  to reject vs the cached cluster-centre xy than to gather all active
  cube positions on every resample.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SingulationBowlPoseCommand(UniformPoseCommand):
    """Uniform pose command with rejection sampling vs the cluster centre.

    Reads ``env._singulation_cluster_center_xy`` (written by
    :func:`mdp.events.sample_active_set`, which runs BEFORE
    ``command_manager.reset`` per ``ManagerBasedRLEnv._reset_idx``).

    If the ``max_attempts`` budget is exhausted, we keep the last sample —
    a slightly-close bowl that episode rather than a silent infinite loop.
    """

    cfg: "SingulationBowlPoseCommandCfg"

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return

        center_xy_b = getattr(self._env, "_singulation_cluster_center_xy", None)
        if center_xy_b is None:
            return  # event hasn't run yet (very first reset) — accept the uniform sample
        center_xy = center_xy_b[env_ids_t]  # (n, 2) — already in robot frame (xy local to root)

        min_d = float(self.cfg.min_distance)
        for _ in range(int(self.cfg.max_attempts)):
            cur_xy = self.pose_command_b[env_ids_t, :2]
            d = torch.norm(cur_xy - center_xy, dim=1)
            bad = d < min_d
            if not torch.any(bad):
                break
            bad_global_ids = env_ids_t[bad]
            n_bad = bad_global_ids.numel()
            r = torch.empty(n_bad, device=self.device)
            self.pose_command_b[bad_global_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
            self.pose_command_b[bad_global_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)


@configclass
class SingulationBowlPoseCommandCfg(UniformPoseCommandCfg):
    """Config for :class:`SingulationBowlPoseCommand`."""

    class_type: type = SingulationBowlPoseCommand

    min_distance: float = 0.15
    """Minimum xy distance between sampled bowl and cluster centre, in
    robot frame. 15 cm gives ~5 cm clearance from the outermost cube of
    a 4-stack / pyramid / spread cluster."""

    max_attempts: int = 8
    """Rejection-sampling budget. 8 × ~80 % accept rate per draw → <
    0.001 % stuck-envs."""
