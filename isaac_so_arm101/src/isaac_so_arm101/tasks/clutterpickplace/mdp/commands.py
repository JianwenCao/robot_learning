# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Target-color command for Eval-2 (targeted pick-and-place in clutter).

This command is **passive** — it doesn't sample anything in
``_resample_command``. Instead, the per-episode active pair and target
index are sampled inside the event term :func:`mdp.events.place_clutter_blocks`,
which writes them onto the env buffers (``env._active_cube_indices``,
``env._target_cube_idx``, ``env._target_color_onehot``).

Why split it this way: Isaac Lab's ``ManagerBasedRLEnv._reset_idx`` runs
the event manager *before* the command manager at reset time, so any
sampling done inside ``_resample_command`` would be too late to inform
the cube placement. Doing all sampling inside the event (which runs
first) keeps the event ↔ command state consistent on every episode.

The command's own buffers (``active_indices`` etc.) are aliased to the
env-level buffers in ``__init__`` so callers querying via
``env.command_manager.get_term("target_color").active_indices`` see the
same tensor the event wrote into.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .events import NUM_COLORS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TargetColorCommand(CommandTerm):
    """Per-episode (active pair, target-in-pair) sampler.

    Samples are drawn once per :meth:`_resample_command` call (i.e. once per
    episode reset, because we set ``resampling_time_range = (T, T)`` with
    ``T >= episode_length_s``). Same pattern as :class:`BowlPoseCommand`
    in Eval-1.

    Buffers (all ``(num_envs, ...)``, allocated lazily):

    * ``active_indices``    — long, ``(N, 2)``, palette indices of active cubes.
    * ``target_idx_in_pair`` — long, ``(N,)``, 0 or 1.
    * ``target_color_idx``  — long, ``(N,)``, palette index of the target.
    * ``onehot``            — float, ``(N, NUM_COLORS)``, target color one-hot.
    """

    cfg: "TargetColorCommandCfg"

    def __init__(self, cfg: "TargetColorCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        N = self.num_envs
        device = self.device
        # Allocate the env-level buffers eagerly so the place_clutter_blocks
        # event term can write to them without lazy-init checks. The
        # command exposes these via property aliases below — there's only
        # one copy of the state, owned by the env.
        if not hasattr(env, "_active_cube_indices"):
            env._active_cube_indices = torch.zeros((N, 2), dtype=torch.long, device=device)
        if not hasattr(env, "_target_idx_in_pair"):
            env._target_idx_in_pair = torch.zeros(N, dtype=torch.long, device=device)
        if not hasattr(env, "_target_cube_idx"):
            env._target_cube_idx = torch.zeros(N, dtype=torch.long, device=device)
        # Output buffer for the command tensor (one-hot of target color).
        self._onehot = torch.zeros((N, NUM_COLORS), dtype=torch.float32, device=device)

    # ----------------------------------------------------------------- API

    # Property aliases — the source of truth is the env buffer, populated
    # by :func:`mdp.events.place_clutter_blocks`.

    @property
    def active_indices(self) -> torch.Tensor:
        return self._env._active_cube_indices

    @property
    def target_idx_in_pair(self) -> torch.Tensor:
        return self._env._target_idx_in_pair

    @property
    def target_color_idx(self) -> torch.Tensor:
        return self._env._target_cube_idx

    @property
    def command(self) -> torch.Tensor:  # noqa: D401 — Isaac Lab convention
        """The target-color one-hot, ``(N, NUM_COLORS)``. Re-computed from env state each call."""
        self._onehot.zero_()
        self._onehot.scatter_(1, self.target_color_idx.view(-1, 1), 1.0)
        return self._onehot

    # -------------------------------------------------------------- resample

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        # No-op: sampling happens in the event term (which runs BEFORE
        # command_manager.reset in ManagerBasedRLEnv._reset_idx). The env
        # buffers are already populated by the time we'd be called.
        pass

    # ------------------------------------------------ no-op visualization

    def _update_command(self) -> None:
        # No per-step update — the command is constant within an episode.
        pass

    def _update_metrics(self) -> None:
        # No per-step metrics tracked here; reward/termination handle that.
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:  # noqa: D401
        # No visualization marker for this command (the target color is
        # already visible as the cube material color). The Eval-1
        # bowl_pose command still draws its red sphere.
        pass

    def _debug_vis_callback(self, event) -> None:  # noqa: D401
        pass


@configclass
class TargetColorCommandCfg(CommandTermCfg):
    """Config for :class:`TargetColorCommand`."""

    class_type: type = TargetColorCommand

    asset_name: str = MISSING
    """Name of the robot articulation (used only for ``num_envs``/device
    introspection — there's no continuous goal in robot frame to track).
    Defaults filled in by ``joint_pos_env_cfg``."""
