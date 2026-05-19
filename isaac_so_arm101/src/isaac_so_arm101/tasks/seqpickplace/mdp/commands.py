# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SequentialGoalCommand — Eval-3's all-in-one goal command.

Encapsulates everything the 3-step sequential task needs:

* Per-episode sampling of:
  - ``active_indices`` ``(N, 4)`` long — which 4 of the 6 palette cubes
    are placed in the workspace.
  - ``goal_color_pos`` ``(N, 3)`` long ∈ [0, 4) — for each of the 3
    steps, which position inside ``active_indices`` is the target.
  - ``goal_bowl_idx`` ``(N, 3)`` long ∈ [0, 3) — for each step, which of
    the 3 bowls is the target. Independent of color sampling.
  - ``bowl_positions`` ``(N, 3, 2)`` float — the three bowl xy positions
    in the robot frame, sampled with rejection so they're ≥
    ``min_bowl_separation`` apart and the *first step's* bowl is
    ``min_block_distance`` from the first step's target cube.
* Per-step "advance step on success" book-keeping:
  ``_update_command`` is called every env step by the CommandManager; it
  reads ``env._seq_step_release_indicator`` (an OR-latch maintained by
  the reward function) and increments ``env._seq_step_idx`` accordingly.

The command's "command" tensor is a 6 + 2 + 3 = 11-D vector per env:
``[current_target_color_onehot(6), current_target_bowl_xy(2), current_step_idx_onehot(3)]``.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .events import N_ACTIVE_BLOCKS, N_GOAL_STEPS, NUM_COLORS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class SequentialGoalCommand(CommandTerm):
    """Goal command for Eval-3 (sequential 3-step pick-and-place)."""

    cfg: "SequentialGoalCommandCfg"

    def __init__(self, cfg: "SequentialGoalCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        N = self.num_envs
        device = self.device

        # The buffers live on the env (so the place_seq_blocks event,
        # which runs BEFORE command_manager.reset, can write them and
        # this command can read them). Eagerly allocated so the event
        # term doesn't need to lazy-init.
        if not hasattr(env, "_seq_active_indices"):
            env._seq_active_indices = torch.zeros((N, N_ACTIVE_BLOCKS), dtype=torch.long, device=device)
        if not hasattr(env, "_seq_goal_color_pos"):
            env._seq_goal_color_pos = torch.zeros((N, N_GOAL_STEPS), dtype=torch.long, device=device)
        if not hasattr(env, "_seq_goal_bowl_idx"):
            env._seq_goal_bowl_idx = torch.zeros((N, N_GOAL_STEPS), dtype=torch.long, device=device)
        if not hasattr(env, "_seq_bowl_positions"):
            env._seq_bowl_positions = torch.zeros((N, N_GOAL_STEPS, 2), dtype=torch.float32, device=device)
        if not hasattr(env, "_seq_step_idx"):
            env._seq_step_idx = torch.zeros(N, dtype=torch.long, device=device)

        # Output buffer for self.command (re-computed each call).
        self._cmd = torch.zeros((N, NUM_COLORS + 2 + N_GOAL_STEPS), dtype=torch.float32, device=device)

    # Aliases — the source of truth is env state.

    @property
    def active_indices(self) -> torch.Tensor:
        return self._env._seq_active_indices

    @property
    def goal_color_pos(self) -> torch.Tensor:
        return self._env._seq_goal_color_pos

    @property
    def goal_bowl_idx(self) -> torch.Tensor:
        return self._env._seq_goal_bowl_idx

    @property
    def bowl_positions(self) -> torch.Tensor:
        return self._env._seq_bowl_positions

    # ----------------------------------------------------------- properties

    @property
    def command(self) -> torch.Tensor:
        """``(N, 11)`` — current target color one-hot + current bowl xy + step one-hot."""
        return self._compose_command()

    def current_target_color_idx(self) -> torch.Tensor:
        """Palette index ``[0, NUM_COLORS)`` of the current step's target color, ``(N,)``."""
        step = self._env._seq_step_idx.clamp(max=N_GOAL_STEPS - 1)
        # goal_color_pos[:, step] → position in active; then active[:, that pos] → palette idx
        color_pos = self.goal_color_pos.gather(1, step.view(-1, 1)).squeeze(1)
        return self.active_indices.gather(1, color_pos.view(-1, 1)).squeeze(1)

    def current_target_bowl_xy(self) -> torch.Tensor:
        """``(N, 2)`` current step's bowl xy in robot frame."""
        step = self._env._seq_step_idx.clamp(max=N_GOAL_STEPS - 1)
        bowl_idx = self.goal_bowl_idx.gather(1, step.view(-1, 1)).squeeze(1)
        return self.bowl_positions.gather(
            1, bowl_idx.view(-1, 1, 1).expand(-1, 1, 2)
        ).squeeze(1)

    def all_steps_done(self) -> torch.Tensor:
        """Bool mask ``(N,)`` — current step idx ≥ ``N_GOAL_STEPS`` (final
        step completed and incremented past)."""
        return self._env._seq_step_idx >= N_GOAL_STEPS

    # ---------------------------------------------------------- compose cmd

    def _compose_command(self) -> torch.Tensor:
        step_clamped = self._env._seq_step_idx.clamp(max=N_GOAL_STEPS - 1)
        color_idx = self.current_target_color_idx()
        bowl_xy = self.current_target_bowl_xy()

        # Color one-hot
        self._cmd[:, :NUM_COLORS] = 0.0
        self._cmd.scatter_(1, color_idx.view(-1, 1), 1.0)
        # bowl xy
        self._cmd[:, NUM_COLORS:NUM_COLORS + 2] = bowl_xy
        # step one-hot — clamp the (already-incremented-past-last) case to
        # last step so the obs stays well-defined after termination.
        self._cmd[:, NUM_COLORS + 2:] = 0.0
        idx = NUM_COLORS + 2 + step_clamped
        self._cmd.scatter_(1, idx.view(-1, 1), 1.0)
        # Zero the color/bowl one-hots for envs that have finished all
        # steps so the policy gets a clear "done" signal (caller's
        # responsibility to also terminate / mask reward).
        done = self.all_steps_done()
        if done.any():
            self._cmd[done, :NUM_COLORS] = 0.0
            self._cmd[done, NUM_COLORS:NUM_COLORS + 2] = 0.0
            # leave step one-hot at the last step
        return self._cmd

    # ---------------------------------------------------------- resampling

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        # No-op: all per-episode sampling happens in the event term
        # :func:`mdp.events.place_seq_blocks` (which runs BEFORE
        # command_manager.reset in ManagerBasedRLEnv._reset_idx).
        pass

    # ---------------------------------------------------------- per-step update

    def _update_command(self) -> None:
        """Advance step idx whenever the reward indicator latches True.

        The reward term :func:`mdp.rewards.release_current_target_in_bowl`
        sets ``env._seq_step_release_indicator`` to True for envs that
        satisfied the release predicate this step. We advance those envs'
        step counters and clear the per-step latches so the next sub-goal
        starts fresh.
        """
        env = self._env
        ind = getattr(env, "_seq_step_release_indicator", None)
        if ind is None:
            return
        # Only advance envs that haven't finished yet.
        not_done = env._seq_step_idx < N_GOAL_STEPS
        advance = ind & not_done
        if not advance.any():
            return
        env._seq_step_idx[advance] += 1
        # Clear per-step gating latches for the envs advancing — the
        # next step's target is a different cube + bowl, so the
        # was-lifted / was-over-rim semantics don't carry over.
        if hasattr(env, "_seq_was_grasped"):
            env._seq_was_grasped[advance] = False
        if hasattr(env, "_seq_was_over_bowl_above_rim"):
            env._seq_was_over_bowl_above_rim[advance] = False
        # Consume the indicator so we don't double-advance next step.
        env._seq_step_release_indicator[advance] = False

    def _update_metrics(self) -> None:
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        pass

    def _debug_vis_callback(self, event) -> None:
        pass


@configclass
class SequentialGoalCommandCfg(CommandTermCfg):
    """Config for :class:`SequentialGoalCommand`.

    The command itself is passive — per-episode sampling (active set,
    color sequence, bowl positions, distinct-bowl constraint) lives in
    :func:`mdp.events.place_seq_blocks`. This cfg is intentionally
    minimal as a result.
    """

    class_type: type = SequentialGoalCommand

    asset_name: str = MISSING
