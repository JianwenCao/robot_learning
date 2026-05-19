# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pretrained-backbone Actor-Critic for the SO-ARM101 pick-and-place task.

EVAL1_PLAN §9 alternative path — **cold-start, no teacher**. This module is
deliberately parallel to :mod:`vision_actor_critic` (the §7 production path)
so that the verified Stage 1–3 pipeline stays untouched.

Architecture (§9.3): dual-stream encoder + fused spatial-softmax head.

* **RGB stream** — ImageNet-pretrained ResNet-18, with two surgical mods:
  (a) ``conv1`` inflated 3 → 5 input channels (only ch 0–2 receive pretrained
  weights; ch 3, 4 are zero-init slots we don't actually feed) so the same
  module slot can host the 5-ch obs path used by Stage 3. In practice this
  class slices the 5-ch obs into RGB (ch 0–2) and Depth+Mask (ch 3–4) before
  the encoders, so ``conv1`` only sees 3 channels at runtime — channel
  inflation is kept as a forward-compat option, but the default RGB-only
  conv1 path is what runs;
  (b) all ``BatchNorm2d`` → ``GroupNorm(32, …)`` so the encoder is robust to
  PPO's non-stationary rollout statistics (BN running stats drift
  catastrophically under on-policy RL with the policy distribution shifting
  iter-by-iter).

  **The trunk is FROZEN by default** (``freeze_backbone=True``,
  ``requires_grad=False`` on all ResNet params, eval-mode locked). This
  diverges from §9.3's original fine-tuning spec on supervisor
  recommendation. Frozen-backbone reasoning: PPO's distribution shift
  destabilizes a fine-tuned ImageNet encoder before the head learns to
  read it; freezing keeps the feature extractor reliable and concentrates
  all gradient on the head + MLPs (~5× fewer trainable params, faster,
  more stable). The forward pass for the trunk runs under ``torch.no_grad()``
  to free activation memory. Fallback path if frozen-ImageNet plateaus:
  swap to R3M or MVP (manipulation-pretrained) — *not* unfreezing ImageNet,
  which the literature shows underperforms manipulation-pretrained
  alternatives at the cost of distribution-shift risk.

  We stop at ``layer3`` (output shape ``(256, 5, 8)`` at 128×72 input)
  rather than ``layer4`` (``(512, 3, 4)``) — a 5×8 = 40-cell spatial-softmax
  grid keeps enough positional resolution to localize the 2 cm cube, where
  3×4 = 12 cells discards most of it. This is the only deliberate divergence
  from the §9.3 spec; documented inline below.

* **Depth+Mask stream** — small 3-layer ELU CNN, from scratch, designed so
  its output spatial dims match the RGB stream at ``layer3`` ((5, 8)) for
  channel-wise concat.

* **Fusion head** — concat → 1×1 Conv → spatial softmax → LayerNorm. The
  Levine-style soft-argmax keypoint inductive bias is preserved on top of
  the pretrained features (same head shape as :mod:`vision_actor_critic`).

The DrQ pad-and-crop augmentation is reused from :mod:`vision_actor_critic`
unchanged (training-only, ±4 px replicate-pad-and-crop). This module imports
``_random_shift_pad`` directly; no copy.

Registered into RSL-RL's runner namespace at import time
(see :mod:`agents.pretrained_ppo_cfg`) so ``class_name="PickPlaceResNetActorCritic"``
resolves in ``OnPolicyRunner.__init__``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP, EmpiricalNormalization
from torch.distributions import Normal
from torchvision.models import resnet18, ResNet18_Weights

# DrQ shift helper and image-group name are reused unchanged from the §7 path.
from .vision_actor_critic import (
    DEFAULT_IMAGE_GROUP,
    DRQ_PAD_PIXELS,
    _random_shift_pad,
)


# ---------------------------------------------------------------------------
# Helpers — BN→GN, channel inflation, depth/mask CNN, fused softmax head.
# ---------------------------------------------------------------------------


def _bn_to_gn(module: nn.Module, num_groups: int = 32) -> nn.Module:
    """Recursively replace every ``BatchNorm2d`` in ``module`` with ``GroupNorm``.

    ImageNet weights of the parent ``BatchNorm2d`` (affine ``γ`` and ``β``)
    are copied into the new ``GroupNorm`` — both have the same per-channel
    affine parameters, so the encoder output distribution is unchanged at
    the moment of swap. Running statistics (``running_mean``, ``running_var``)
    are dropped: ``GroupNorm`` normalizes per-sample within groups and has
    no running stats, which is exactly why we want it for PPO (BN's running
    stats drift with the non-stationary rollout distribution and corrupt
    the encoder output once the policy starts updating).

    ``num_groups=32`` is the original GroupNorm paper's default; it divides
    cleanly into ResNet-18's channel counts (64, 128, 256, 512) so every
    group has 2/4/8/16 channels respectively. Reference: Wu & He, 2018.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(
                num_groups=min(num_groups, child.num_features),
                num_channels=child.num_features,
                eps=child.eps,
                affine=True,
            )
            # Copy affine params from BN — same shape, same semantics.
            if child.affine:
                with torch.no_grad():
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)
            setattr(module, name, gn)
        else:
            _bn_to_gn(child, num_groups=num_groups)
    return module


def _build_resnet18_rgb_encoder(pretrained: bool = True) -> nn.Sequential:
    """Build a ResNet-18 truncated at ``layer3``, with BN→GN.

    Returns a ``Sequential`` that maps ``(N, 3, 72, 128)`` to ``(N, 256, 5, 8)``.
    Pretrained weights are ImageNet-1k V1; we keep them on by default — the
    whole point of §9 is to compare against the §7 from-scratch CNN baseline,
    so dropping pretraining would defeat the experiment.

    Why truncate at ``layer3``: at 128×72 input, ``layer4`` outputs
    ``(512, 3, 4)`` — a spatial-softmax over 12 cells discards most of the
    positional information the head was chosen for. ``layer3`` gives
    ``(256, 5, 8)`` = 40 cells, preserving enough resolution to localize a
    2 cm cube while still benefiting from C3 semantic features (the
    middle-block features that the IL/manipulation literature consistently
    finds most useful for keypoint-style heads).
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = resnet18(weights=weights)
    # Truncate after layer3. nn.Sequential preserves submodule order; the
    # listed pieces match ResNet-18's forward up to (and including) layer3.
    encoder = nn.Sequential(
        backbone.conv1,    # 3 → 64,  stride 2  → (64, 36, 64)
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,  # stride 2            → (64, 18, 32)
        backbone.layer1,   # 64 → 64             → (64, 18, 32)
        backbone.layer2,   # 64 → 128, stride 2  → (128, 9, 16)
        backbone.layer3,   # 128 → 256, stride 2 → (256, 5, 8)
    )
    _bn_to_gn(encoder, num_groups=32)
    return encoder


class _DepthMaskCNN(nn.Module):
    """Small 3-layer ELU CNN for the 2-channel depth+mask stream.

    Designed to match the RGB stream's output spatial dims ``(5, 8)`` at
    128×72 input, so the two streams can concat channel-wise before the
    fused 1×1 + spatial softmax. From-scratch — depth and the binary block
    mask have no analogue in ImageNet pretraining, so pretrained init would
    be misleading.

    Shape contract: ``(N, 2, 72, 128) → (N, 64, 5, 8)``.
    """

    def __init__(self) -> None:
        super().__init__()
        # k7 s4 p3   : (72, 128) → (18, 32)
        # k3 s2 p1   : (18, 32)  → (9,  16)
        # k3 s2 p1   : (9,  16)  → (5,  8)
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=7, stride=4, padding=3),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _FusedSpatialSoftmax(nn.Module):
    """Spatial-softmax head over fused (RGB + depth/mask) feature map.

    Input shape: ``(N, C_in, H, W)`` — here ``C_in = 256 + 64 = 320`` and
    ``(H, W) = (5, 8)``. The 1×1 conv collapses to ``num_keypoints`` channels;
    spatial softmax converts each channel into a soft-argmax expected
    ``(x, y)`` coordinate in ``[-1, 1]``. Output dim is ``2 * num_keypoints``.

    Same head pattern as :class:`vision_actor_critic._SpatialSoftmaxCNN`
    minus the conv stack (which the pretrained backbone replaces). LayerNorm
    at the output keeps keypoint magnitudes bounded under random init when
    the conv heatmaps may be one-hot peaky.
    """

    def __init__(self, in_channels: int, feat_h: int, feat_w: int, out_dim: int = 128):
        super().__init__()
        if out_dim % 2 != 0:
            raise ValueError(f"out_dim={out_dim} must be even (2 coords per keypoint)")
        num_keypoints = out_dim // 2
        self.reduce = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, feat_h),
            torch.linspace(-1.0, 1.0, feat_w),
            indexing="ij",
        )
        self.register_buffer("x_grid", xs.reshape(-1))
        self.register_buffer("y_grid", ys.reshape(-1))
        self.norm = nn.LayerNorm(out_dim)
        self.num_keypoints = num_keypoints
        self.out_dim = out_dim

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        feat = self.reduce(feat)  # (N, K, H, W)
        n, k, h, w = feat.shape
        attn = torch.softmax(feat.view(n, k, h * w), dim=-1)
        ex = (attn * self.x_grid).sum(dim=-1)
        ey = (attn * self.y_grid).sum(dim=-1)
        kp = torch.stack([ex, ey], dim=-1).reshape(n, 2 * k)
        return self.norm(kp)


class _PretrainedActorEncoder(nn.Module):
    """Full actor-side image encoder: RGB-ResNet + Depth/Mask-CNN + fused head.

    Splits the 5-channel ``wrist_image`` into RGB (ch 0–2) and depth+mask
    (ch 3–4), runs each through its dedicated stream, concatenates the
    feature maps along the channel dim, then passes through the fused
    spatial-softmax head.

    ResNet-18 expects ImageNet-normalized RGB (mean/std), so we apply the
    standard normalization on the fly. The wrist obs is already in ``[0, 1]``
    (see ``mdp.wrist_image``), so the normalization is just an affine remap.

    **Frozen backbone (default, per supervisor):** with ``freeze_backbone=True``
    the ResNet trunk's parameters get ``requires_grad=False`` and the trunk
    is forced into ``eval()`` mode permanently — the parent ``train()`` call
    is intercepted to keep it eval-mode even when RSL-RL toggles the rest of
    the network. The RGB stream then acts as a fixed feature extractor; only
    the depth/mask CNN, the fused 1×1 conv, the spatial-softmax head, and the
    actor/value MLPs train. Three reasons this is preferred over fine-tuning:

    * **Stability** — PPO's on-policy distribution shift can wreck a
      fine-tuned ImageNet encoder before the head learns to read it.
    * **Sample efficiency** — fewer trainable parameters → smaller effective
      hypothesis space → faster convergence on the bits that matter (the
      task-specific spatial mapping).
    * **Memory** — gradients don't propagate through layer1–3, freeing
      activation memory; the forward pass for the frozen trunk runs under
      ``torch.no_grad()`` to recover that memory.

    **Caveat / fallback path.** ImageNet features were optimized for
    classification, not spatial localization of a 2 cm wooden cube in a
    wrist-cam view. If frozen-ImageNet plateaus, the next step is swapping
    to **R3M** or **MVP** (pretrained on robotic manipulation video — same
    interface, different ``weights=...``). Don't fine-tune ImageNet as the
    fallback; the literature consistently shows R3M/MVP > fine-tuned
    ImageNet for manipulation.
    """

    # ImageNet-1k normalization constants used by torchvision's resnet18.
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        out_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.rgb_encoder = _build_resnet18_rgb_encoder(pretrained=pretrained)
        self.depth_mask_encoder = _DepthMaskCNN()
        # Freeze the ResNet trunk if requested. We do this BEFORE the dummy
        # forward below — no_grad context covers the probe naturally because
        # the params have requires_grad=False.
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.rgb_encoder.parameters():
                p.requires_grad_(False)
            self.rgb_encoder.eval()
        # Probe spatial dims from a dummy forward — assert both streams
        # produce the same (H, W) so the channel-wise concat is well-defined.
        with torch.no_grad():
            dummy_rgb = torch.zeros(1, 3, 72, 128)
            dummy_dm = torch.zeros(1, 2, 72, 128)
            rgb_out = self.rgb_encoder(dummy_rgb)
            dm_out = self.depth_mask_encoder(dummy_dm)
        assert rgb_out.shape[2:] == dm_out.shape[2:], (
            f"RGB stream output {tuple(rgb_out.shape)} and Depth/Mask stream "
            f"output {tuple(dm_out.shape)} must have matching (H, W). "
            f"Adjust _DepthMaskCNN strides/kernels."
        )
        in_channels = int(rgb_out.shape[1] + dm_out.shape[1])
        feat_h, feat_w = int(rgb_out.shape[2]), int(rgb_out.shape[3])
        self.head = _FusedSpatialSoftmax(in_channels, feat_h, feat_w, out_dim=out_dim)
        # Register normalization buffers (move with .to(device), not trained).
        self.register_buffer(
            "_imnet_mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "_imnet_std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1)
        )
        self.out_dim = out_dim
        self.feat_h = feat_h
        self.feat_w = feat_w

    def train(self, mode: bool = True):
        """Override ``train()`` to keep a frozen trunk in eval() mode.

        RSL-RL calls ``policy.train()`` between rollout and update; without
        this override, the recursive ``train()`` would flip the ResNet trunk
        back to train mode. Even though GroupNorm has no running stats (the
        practical reason train/eval differ), we keep the trunk in eval mode
        for consistency — any future change to the trunk (e.g. swapping in
        an encoder that *does* have train-mode-only behavior) won't silently
        regress.
        """
        super().train(mode)
        if self.freeze_backbone:
            self.rgb_encoder.eval()
        return self

    def forward(self, img5: torch.Tensor) -> torch.Tensor:
        # Slice 5-ch input → (RGB, Depth+Mask). Order is fixed by
        # ``mdp.wrist_image``: ch 0-2 RGB, ch 3 depth, ch 4 mask.
        rgb = img5[:, :3]
        dm = img5[:, 3:]
        # ImageNet-normalize the RGB stream so the pretrained ResNet sees the
        # input distribution it was trained on. Depth/mask stream is from
        # scratch and stays in ``[0, 1]``.
        rgb = (rgb - self._imnet_mean) / self._imnet_std
        # Frozen trunk → no_grad to save activation memory. The
        # requires_grad=False on params already prevents grad computation;
        # the no_grad context is what releases the saved-for-backward
        # tensors of each conv layer. Big memory win at 1024+ envs.
        if self.freeze_backbone:
            with torch.no_grad():
                rgb_feat = self.rgb_encoder(rgb)
        else:
            rgb_feat = self.rgb_encoder(rgb)
        dm_feat = self.depth_mask_encoder(dm)
        fused = torch.cat([rgb_feat, dm_feat], dim=1)
        return self.head(fused)


# ---------------------------------------------------------------------------
# Main module — Actor-Critic with pretrained backbone.
# ---------------------------------------------------------------------------


class PickPlaceResNetActorCritic(ActorCritic):
    """Asymmetric A-C with an ImageNet-pretrained ResNet-18 actor encoder.

    Constructor signature mirrors :class:`rsl_rl.modules.ActorCritic` — RSL-RL
    instantiates it via ``cls(obs, obs_groups, num_actions, **policy_cfg_kwargs)``.

    The actor consumes ``policy + wrist_image``; the critic consumes
    ``policy + critic`` (no image, same as the §7 path) so the privileged
    critic still has ground-truth block pose and the encoder remains
    actor-only. Setting the policy cfg ``class_name="PickPlaceResNetActorCritic"``
    makes ``OnPolicyRunner`` pick this class up via name lookup.

    EVAL1_PLAN §9 is a **cold-start** path — there is no distillation
    warm-start to unpack and no teacher critic overlay. We deliberately do
    not override ``load_state_dict``: a normal resume (``--resume``) works
    via the parent class's vanilla load.
    """

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims=(256, 128, 64),
        critic_hidden_dims=(256, 128, 64),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        image_group_name: str = DEFAULT_IMAGE_GROUP,
        image_feat_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        **kwargs,
    ):
        # As in PickPlaceVisionActorCritic: skip the parent ``ActorCritic.__init__``
        # because it asserts every obs group is 1-D, which fails for the image
        # group. Replicate the parts we need (MLPs, normalizers, distribution
        # params).
        nn.Module.__init__(self)
        if kwargs:
            print(
                "PickPlaceResNetActorCritic.__init__ got unexpected kwargs (ignored): "
                + str(list(kwargs.keys()))
            )

        self.obs_groups = obs_groups
        self.image_group_name = image_group_name

        # Dim accounting for the state MLP inputs.
        actor_state_dim = self._sum_state_dims(obs, obs_groups["policy"], image_group_name)
        critic_state_dim = self._sum_state_dims(obs, obs_groups["critic"], image_group_name)
        actor_uses_image = image_group_name in obs_groups["policy"]
        critic_uses_image = image_group_name in obs_groups["critic"]

        # Verify the image obs shape matches what the encoder is built for.
        if actor_uses_image or critic_uses_image:
            img_sample = obs[image_group_name]
            assert img_sample.dim() == 4, (
                f"Expected image obs (N, C, H, W); got {tuple(img_sample.shape)}"
            )
            c, h, w = int(img_sample.shape[1]), int(img_sample.shape[2]), int(img_sample.shape[3])
            # §9 design assumes the 5-channel wrist_image at 128×72 (the same
            # tensor the §7 path consumes). Refuse to silently degrade if the
            # env cfg ever changes the channel count or resolution out from
            # under us — a misconfigured encoder would just bake in slow loss.
            assert c == 5, (
                f"PickPlaceResNetActorCritic expects a 5-channel image (R, G, B, depth, mask); "
                f"got {c} channels. Check mdp.wrist_image channel construction."
            )
            assert (h, w) == (72, 128), (
                f"PickPlaceResNetActorCritic was designed for 72×128 wrist images "
                f"(matches ResNet/Depth-CNN strides → 5×8 feature map). Got {h}×{w}. "
                f"Either change WRIST_RGB_{{HEIGHT,WIDTH}} back or adjust the encoder strides."
            )

        # Actor image encoder. The critic does not see images (§9.3, §2) —
        # same asymmetric A-C principle as the §7 path. Keeping critic_cnn as
        # None matches the vision_actor_critic interface.
        self.actor_cnn: nn.Module | None = (
            _PretrainedActorEncoder(
                out_dim=image_feat_dim,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
            )
            if actor_uses_image
            else None
        )
        # We do not support image-to-critic in §9 (no use case and it would
        # need a second pretrained backbone — double the params for no
        # asymmetric-AC benefit). Fail loud if someone wires it up.
        if critic_uses_image:
            raise ValueError(
                "PickPlaceResNetActorCritic does not support image obs in the "
                "critic group — the asymmetric design (§2, §9.3) keeps the "
                "encoder actor-only. Remove 'wrist_image' from obs_groups['critic']."
            )
        self.critic_cnn = None

        actor_in = actor_state_dim + (image_feat_dim if actor_uses_image else 0)
        critic_in = critic_state_dim

        self.actor = MLP(actor_in, num_actions, list(actor_hidden_dims), activation)
        self.critic = MLP(critic_in, 1, list(critic_hidden_dims), activation)
        print(f"Actor CNN: {self.actor_cnn}")
        print(f"Critic CNN: {self.critic_cnn}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # State-only normalizer (image stream has its own ImageNet
        # normalization inside the encoder and a LayerNorm at the head).
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(actor_state_dim)
        else:
            self.actor_obs_normalizer = nn.Identity()
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(critic_state_dim)
        else:
            self.critic_obs_normalizer = nn.Identity()

        # Action noise — single scalar σ across all dims, stock RSL-RL
        # semantics (the same choice the §7 production path settled on).
        # ``init_noise_std=1.0`` is the §9 cold-start default — see
        # :class:`pretrained_ppo_cfg.PickPlaceBowlPretrainedPPORunnerCfg`.
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type {noise_std_type!r}")

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    # ----------------------------------------------------------------------
    # Helpers (mirror :class:`PickPlaceVisionActorCritic`)
    # ----------------------------------------------------------------------

    @staticmethod
    def _sum_state_dims(obs, group_names, image_group_name) -> int:
        total = 0
        for name in group_names:
            if name == image_group_name:
                continue
            t = obs[name]
            assert t.dim() == 2, (
                f"State obs group {name!r} must be 1-D per env; got shape {tuple(t.shape)}"
            )
            total += t.shape[-1]
        return total

    @staticmethod
    def _safe(t: torch.Tensor) -> torch.Tensor:
        # See PickPlaceVisionActorCritic._safe for the inference-tensor rationale.
        out = torch.empty_like(t)
        out.copy_(t)
        return out

    def _gather_state(self, obs, group_names) -> torch.Tensor:
        parts = [self._safe(obs[g]) for g in group_names if g != self.image_group_name]
        return (
            torch.cat(parts, dim=-1)
            if parts
            else torch.empty(
                obs[self.image_group_name].shape[0], 0, device=obs[self.image_group_name].device
            )
        )

    def _encode_actor(self, obs) -> torch.Tensor:
        state = self._gather_state(obs, self.obs_groups["policy"])
        state = self.actor_obs_normalizer(state)
        if self.actor_cnn is None:
            return state
        img = self._safe(obs[self.image_group_name])
        if self.training:
            img = _random_shift_pad(img, DRQ_PAD_PIXELS)
        feat = self.actor_cnn(img)
        return torch.cat([state, feat], dim=-1)

    def _encode_critic(self, obs) -> torch.Tensor:
        state = self._gather_state(obs, self.obs_groups["critic"])
        return self.critic_obs_normalizer(state)

    # ----------------------------------------------------------------------
    # ActorCritic interface
    # ----------------------------------------------------------------------

    # v8 (2026-05-15): σ clamp to kill the KL-controller runaway.
    # v7 saw σ pumped 1.0 → 1.70 over 4000 iters, which destroyed a working
    # SR≈0.62 policy at iter 1500. Clamping log_std at [log(0.1), log(1.0)]
    # caps σ ∈ [0.1, 1.0]: prevents both the runaway-up and the collapse-down
    # failure modes. Gradient flows normally inside the band; at either
    # boundary the gradient through clamp is zero, so the parameter cannot
    # be pushed past the cap by entropy bonus or surrogate loss.
    _LOG_STD_MIN = math.log(0.1)
    _LOG_STD_MAX = math.log(1.0)

    def update_distribution(self, x):
        mean = self.actor(x)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            log_std_clamped = self.log_std.clamp(
                min=self._LOG_STD_MIN, max=self._LOG_STD_MAX
            )
            std = torch.exp(log_std_clamped).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        x = self._encode_actor(obs)
        self.update_distribution(x)
        return self.distribution.sample()

    def act_inference(self, obs):
        x = self._encode_actor(obs)
        return self.actor(x)

    def evaluate(self, obs, **kwargs):
        x = self._encode_critic(obs)
        return self.critic(x)

    def get_actor_obs(self, obs):
        return self._encode_actor(obs)

    def get_critic_obs(self, obs):
        return self._encode_critic(obs)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(
                self._gather_state(obs, self.obs_groups["policy"])
            )
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(
                self._gather_state(obs, self.obs_groups["critic"])
            )
