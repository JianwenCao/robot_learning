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
  - ``bowl_positions`` ``(N, 1, 2)`` float — the single bowl xy position
    in the robot frame, sampled with rejection so it's ≥
    ``min_bowl_block_separation`` from every cube. All 3 sequential
    placements target this bowl.
* Per-step "advance step on success" book-keeping:
  ``_update_command`` is called every env step by the CommandManager; it
  reads ``env._seq_step_release_indicator`` (an OR-latch maintained by
  the reward function) and increments ``env._seq_step_idx`` accordingly.

The command's "command" tensor is a 6 + 2 = 8-D vector per env:
``[current_target_color_onehot(6), current_target_bowl_xy(2)]`` — the
deployable input is just what the spec lists: target color + target
bowl position. The current step counter is tracked internally on the
env (``env._seq_step_idx``) for reward gating but NOT exposed to the
policy.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .events import N_ACTIVE_BLOCKS, N_BOWLS, N_GOAL_STEPS, NUM_COLORS

# Per-sub-goal step budget. 250 steps = 5 s @ 50 Hz, matching the
# Eval-1 / Eval-2 single-shot episode length the zero-shot actor was
# trained for. If a sub-goal isn't released within this budget, the
# command term auto-advances to the next sub-goal (after arm return-
# to-home) so the rollout can still attempt the remaining sub-goals.
# Total rollout budget should be ≥ N_GOAL_STEPS × MAX_STEPS_PER_SUBGOAL
# (= 3 × 250 = 750 = the env's 15 s ``episode_length_s``).
MAX_STEPS_PER_SUBGOAL = 250

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
        if not hasattr(env, "_seq_bowl_positions"):
            env._seq_bowl_positions = torch.zeros((N, N_BOWLS, 2), dtype=torch.float32, device=device)
        if not hasattr(env, "_seq_step_idx"):
            env._seq_step_idx = torch.zeros(N, dtype=torch.long, device=device)
        # Per-sub-goal step counter — incremented every env step in
        # _update_command, reset on advance. Used to enforce a per-sub-
        # goal timeout so a failed sub-goal doesn't consume the whole
        # 15 s rollout and starve later sub-goals.
        if not hasattr(env, "_seq_sub_step_count"):
            env._seq_sub_step_count = torch.zeros(N, dtype=torch.long, device=device)
        # Hot-path lookup buffers used by observations and rewards. Must be
        # allocated here (before the first reset) because the obs manager
        # probes obs shapes at env construction, which calls obs functions
        # that read these buffers before place_seq_blocks ever runs.
        if not hasattr(env, "_active_cube_indices"):
            env._active_cube_indices = torch.zeros((N, N_ACTIVE_BLOCKS), dtype=torch.long, device=device)
        if not hasattr(env, "_target_cube_idx_per_step"):
            env._target_cube_idx_per_step = torch.zeros((N, N_GOAL_STEPS), dtype=torch.long, device=device)

        # Output buffer for self.command (re-computed each call).
        # 6 (color one-hot) + 2 (bowl xy) = 8-D. Step idx is internal only.
        self._cmd = torch.zeros((N, NUM_COLORS + 2), dtype=torch.float32, device=device)

    # Aliases — the source of truth is env state.

    @property
    def active_indices(self) -> torch.Tensor:
        return self._env._seq_active_indices

    @property
    def goal_color_pos(self) -> torch.Tensor:
        return self._env._seq_goal_color_pos

    @property
    def bowl_positions(self) -> torch.Tensor:
        """``(N, 1, 2)`` float — xy position of the single bowl in robot frame."""
        return self._env._seq_bowl_positions

    # ----------------------------------------------------------- properties

    @property
    def command(self) -> torch.Tensor:
        """``(N, 8)`` — current target color one-hot (6) + current bowl xy (2).

        This matches the spec's per-goal info: target color + target bowl
        position. The step counter is internal (used by reward gating)
        and intentionally NOT exposed to the policy.
        """
        return self._compose_command()

    def current_target_color_idx(self) -> torch.Tensor:
        """Palette index ``[0, NUM_COLORS)`` of the current step's target color, ``(N,)``."""
        step = self._env._seq_step_idx.clamp(max=N_GOAL_STEPS - 1)
        # goal_color_pos[:, step] → position in active; then active[:, that pos] → palette idx
        color_pos = self.goal_color_pos.gather(1, step.view(-1, 1)).squeeze(1)
        return self.active_indices.gather(1, color_pos.view(-1, 1)).squeeze(1)

    def current_target_bowl_xy(self) -> torch.Tensor:
        """``(N, 2)`` bowl xy in robot frame — constant across the 3
        sequential steps within a rollout (single bowl per rollout)."""
        return self.bowl_positions[:, 0]

    def all_steps_done(self) -> torch.Tensor:
        """Bool mask ``(N,)`` — current step idx ≥ ``N_GOAL_STEPS`` (final
        step completed and incremented past)."""
        return self._env._seq_step_idx >= N_GOAL_STEPS

    # ---------------------------------------------------------- compose cmd

    def _compose_command(self) -> torch.Tensor:
        color_idx = self.current_target_color_idx()
        bowl_xy = self.current_target_bowl_xy()

        # Color one-hot
        self._cmd[:, :NUM_COLORS] = 0.0
        self._cmd.scatter_(1, color_idx.view(-1, 1), 1.0)
        # bowl xy (target bowl is per-rollout fixed, but exposed each
        # call for consistency)
        self._cmd[:, NUM_COLORS:NUM_COLORS + 2] = bowl_xy
        # Zero the color/bowl entries for envs that have finished all
        # steps so the policy gets a clear "done" signal (caller's
        # responsibility to also terminate / mask reward).
        done = self.all_steps_done()
        if done.any():
            self._cmd[done, :NUM_COLORS] = 0.0
            self._cmd[done, NUM_COLORS:NUM_COLORS + 2] = 0.0
        return self._cmd

    # ---------------------------------------------------------- resampling

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        # No-op: all per-episode sampling happens in the event term
        # :func:`mdp.events.place_seq_blocks` (which runs BEFORE
        # command_manager.reset in ManagerBasedRLEnv._reset_idx).
        pass

    # ---------------------------------------------------------- per-step update

    def _update_command(self) -> None:
        """Advance step idx on release-success OR per-sub-goal timeout.

        Two advance triggers:

        * **Success** — the reward term
          :func:`mdp.rewards.release_current_target_in_bowl` sets
          ``env._seq_step_release_indicator`` True for envs that
          satisfied the release predicate this step.
        * **Timeout** — ``env._seq_sub_step_count`` reached
          ``MAX_STEPS_PER_SUBGOAL`` without a release. Without this,
          a failed first sub-goal consumes the whole 15 s rollout and
          sub-goals 2/3 never get attempted (the Eval-1 zero-shot
          actor is a single-shot "from home, grasp + place" policy —
          if its first attempt misses, no recovery in-place is
          possible without explicit retraction).

        On advance, we also **teleport the arm back to its home pose**
        (zero velocity, default joint positions including gripper
        open at 0.5). This matches what the deploy script must do
        between sub-goals — re-key the AprilTag target ID and start
        the next reach from a known clean state. Without it, the next
        sub-goal would start from "above the (previous) bowl with
        gripper open" which is out-of-distribution for the policy
        and tanks subsequent SR.
        """
        env = self._env
        not_done = env._seq_step_idx < N_GOAL_STEPS

        # Tick per-sub-goal counter for envs still working.
        env._seq_sub_step_count[not_done] += 1

        ind = getattr(env, "_seq_step_release_indicator", None)
        if ind is None:
            ind = torch.zeros_like(not_done)
        timeout = env._seq_sub_step_count >= MAX_STEPS_PER_SUBGOAL
        advance = (ind | timeout) & not_done
        if not advance.any():
            return

        env._seq_step_idx[advance] += 1
        env._seq_sub_step_count[advance] = 0

        # Clear per-step gating latches for the envs advancing — the
        # next step's target is a different cube + bowl, so the
        # was-lifted / was-over-rim semantics don't carry over.
        if hasattr(env, "_seq_was_grasped"):
            env._seq_was_grasped[advance] = False
        if hasattr(env, "_seq_was_over_bowl_above_rim"):
            env._seq_was_over_bowl_above_rim[advance] = False
        # Consume the indicator (if it was the trigger) so we don't
        # double-advance next step.
        if ind is not None:
            env._seq_step_release_indicator[advance] = False

        # Arm return-to-home for envs that just advanced — only those
        # still inside the rollout (skip envs that finished all sub-
        # goals; their idx is now ≥ N_GOAL_STEPS and they'll be reset
        # by the episode-level pipeline anyway).
        still_running = env._seq_step_idx < N_GOAL_STEPS
        retract = advance & still_running
        if retract.any():
            retract_ids = torch.nonzero(retract, as_tuple=True)[0]
            home_q = self.robot.data.default_joint_pos[retract_ids].clone()
            home_qvel = torch.zeros_like(home_q)
            self.robot.write_joint_state_to_sim(home_q, home_qvel, env_ids=retract_ids)

    def _update_metrics(self) -> None:
        pass

    # ---------------------------------------------------------- debug viz
    #
    # SequentialGoalCommand carries its OWN bowl position (vs Eval-1/2's
    # UniformPoseCommand which auto-draws via Isaac Lab's command-marker
    # path). We implement the marker manually so ``debug_vis=True`` on
    # the cfg renders the single per-rollout bowl as a red sphere.

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "_bowl_markers") or self._bowl_markers is None:
                import isaaclab.sim as sim_utils
                from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
                marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/seq_bowls",
                    markers={
                        # Single per-rollout target bowl (red).
                        "target": sim_utils.SphereCfg(
                            radius=0.03,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.0, 0.0)
                            ),
                        ),
                    },
                )
                self._bowl_markers = VisualizationMarkers(marker_cfg)
            self._bowl_markers.set_visibility(True)
        else:
            if hasattr(self, "_bowl_markers") and self._bowl_markers is not None:
                self._bowl_markers.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        if not getattr(self, "_bowl_markers", None):
            return
        # Bowl xy in robot frame → world. Single bowl per env.
        bowl_pos_b = self._env._seq_bowl_positions  # (N, 1, 2)
        robot_root_xy_w = self.robot.data.root_pos_w[:, :2]
        bowl_pos_w = torch.zeros(
            (self.num_envs, N_BOWLS, 3),
            device=self.device,
            dtype=torch.float32,
        )
        bowl_pos_w[..., :2] = bowl_pos_b + robot_root_xy_w.unsqueeze(1)
        bowl_pos_w[..., 2] = 0.01  # at table top
        translations = bowl_pos_w.reshape(-1, 3)
        marker_ids = torch.zeros(translations.shape[0], device=self.device, dtype=torch.long)
        self._bowl_markers.visualize(
            translations=translations,
            marker_indices=marker_ids,
        )


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
