# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Eval-2 vision actor-critic — extends Eval-1 with FiLM goal conditioning.

Subclasses :class:`pickplace.agents.vision_actor_critic.PickPlaceVisionActorCritic`
to route the ``goal`` obs group (target-color one-hot) into the CNN's
FiLM head, so each of the spatial-softmax keypoint channels becomes
color-conditional. See ``docs/EVAL2_PLAN.md`` §3.1.

The base class already handles:

* ``cnn_class="resnet"`` dispatch → frozen ImageNet ResNet-18 trunk +
  trainable 1×1 conv + spatial-softmax head.
* ``cnn_kwargs={"truncate_at": "layer2", "film_cond_dim": 6}`` to
  truncate the trunk early (9×16 spatial map for 2 cm cubes) and
  allocate a FiLM head conditioned on the 6-D color one-hot.

What this subclass adds:

* Reads ``obs["goal"]`` and passes it as ``film_cond=`` to the encoder.
* The MLP also receives ``goal`` because the runner cfg includes it
  in ``obs_groups`` — belt-and-suspenders (the MLP can use the
  one-hot for non-visual decisions like bowl approach angle).
"""

from __future__ import annotations

import torch

from isaac_so_arm101.tasks.pickplace.agents.vision_actor_critic import (
    DRQ_PAD_PIXELS,
    PickPlaceVisionActorCritic,
    _random_shift_pad,
)

DEFAULT_GOAL_GROUP = "goal"


class ClutterPickPlaceVisionActorCritic(PickPlaceVisionActorCritic):
    """Vision A-C with FiLM goal conditioning on the target-color one-hot."""

    def __init__(self, *args, goal_group_name: str = DEFAULT_GOAL_GROUP, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal_group_name = goal_group_name

    # ---- encode overrides — pass goal one-hot to CNN's FiLM head -----------

    def _gather_film_cond(self, obs) -> torch.Tensor | None:
        """Read the goal-group one-hot from obs if present, else None."""
        if self.goal_group_name in obs:
            return self._safe(obs[self.goal_group_name])
        return None

    def _encode_actor(self, obs) -> torch.Tensor:
        state = self._gather_state(obs, self.obs_groups["policy"])
        state = self.actor_obs_normalizer(state)
        if self.actor_cnn is None:
            return state
        img = self._safe(obs[self.image_group_name])
        if self.training:
            img = _random_shift_pad(img, DRQ_PAD_PIXELS)
        film_cond = self._gather_film_cond(obs)
        feat = self.actor_cnn(img, film_cond=film_cond)
        return torch.cat([state, feat], dim=-1)

    def _encode_critic(self, obs) -> torch.Tensor:
        state = self._gather_state(obs, self.obs_groups["critic"])
        state = self.critic_obs_normalizer(state)
        if self.critic_cnn is None:
            return state
        # If the critic ever uses the image (not by default — see plan
        # §6 obs_groups), pass the same FiLM conditioning. In the default
        # asymmetric setup the critic_cnn is None so this branch never
        # fires.
        film_cond = self._gather_film_cond(obs)
        feat = self.critic_cnn(self._safe(obs[self.image_group_name]), film_cond=film_cond)
        return torch.cat([state, feat], dim=-1)
