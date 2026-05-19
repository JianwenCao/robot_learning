# Copyright (c) 2024-2025, Rui Zhou
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision Actor-Critic for the SO-ARM101 pick-and-place task.

The stock ``rsl_rl.modules.ActorCritic`` requires every observation group to
be 1-D — it asserts ``len(obs[group].shape) == 2`` and concatenates groups
along the last dim. That doesn't work for image obs, which arrive as
``(N, 3, H, W)``. This module subclasses ``ActorCritic`` to:

* route the ``wrist_rgb`` group (one of the obs groups) through a small CNN
  encoder before flattening,
* concatenate the encoded image features with the 1-D state groups,
* feed the result to the standard ``MLP`` actor / critic stack.

The asymmetric structure is preserved: actor reads ``policy + wrist_rgb``,
critic reads ``policy + critic + wrist_rgb`` (the privileged group adds
block ground-truth and distances).

The CNN is small on purpose — the wrist image is 128×72 and we only need
to localize a 2 cm cube on a uniform table. Bigger encoders would just
overfit DR knobs. Sized for plan §3.9 Option A (end-to-end training); if
the encoder collapses, swap to a frozen R3M / DINOv2-small variant.

This module is **registered into RSL-RL's runner namespace** at import time
(see :mod:`agents.rsl_rl_ppo_cfg`) so that
``eval(self.policy_cfg.pop("class_name"))`` inside
``rsl_rl.runners.on_policy_runner.OnPolicyRunner.__init__`` can resolve
``"PickPlaceVisionActorCritic"``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP, EmpiricalNormalization
from torch.distributions import Normal

# Default name of the image obs group. Kept in sync with
# ``ObservationsCfg.WristImageCfg`` in :mod:`pickplace_env_cfg`. As of v4
# the image is 4-channel ``(R, G, B, mask)`` — see :func:`mdp.wrist_image`
# for channel construction. The CNN's first conv reads ``in_channels``
# from the input shape so the same class adapts to 3-/4-/5-channel
# inputs without code changes; if you load a 5-ch checkpoint into this
# 4-ch model the conv1 keys won't match (expected — v4 retrains from scratch).
DEFAULT_IMAGE_GROUP = "wrist_image"

# Default pad amount for the DrQ-style random-shift augmentation. 4 pixels
# is ~3% of width / ~6% of height for our 128×72 wrist image and matches
# the original DrQ paper's setting for cheap-resolution control inputs.
DRQ_PAD_PIXELS = 4


def _random_shift_pad(images: torch.Tensor, pad: int = DRQ_PAD_PIXELS) -> torch.Tensor:
    """DrQ-style random shift via pad-and-crop (Kostrikov et al., ICLR 2021).

    Pads the input by ``pad`` pixels on every side using *replicate* mode
    (mirroring the table edge / gripper edge so the policy doesn't have to
    learn that black borders are part of the scene), then crops back to
    the original H × W at a random offset in ``[0, 2*pad]`` per env. Each
    env in the batch gets its own offset, so this acts as a per-sample
    regularizer on the conv features.

    Net effect: the encoder learns to be invariant to small translations
    of the cube in image space, which is essentially-free domain
    randomization for the wrist-cam pose. The augmentation is the second
    half of the "spatial softmax + DrQ" stack the literature reports as
    sufficient for from-scratch visual RL without a pretrained backbone.
    """
    if pad <= 0:
        return images
    n, c, h, w = images.shape
    # Replicate-pad the borders so we don't introduce hard edges.
    padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
    # Per-env random offsets. ``torch.randint`` is fast on GPU.
    offsets = torch.randint(0, 2 * pad + 1, (n, 2), device=images.device)
    # Build per-env crop windows. We use ``torch.gather`` via index tensors
    # that select [oy : oy+h, ox : ox+w] for each env. The vectorized
    # variant uses ``F.grid_sample`` with shifted affine grids.
    base_y = torch.arange(h, device=images.device).view(1, h, 1).expand(n, h, w)
    base_x = torch.arange(w, device=images.device).view(1, 1, w).expand(n, h, w)
    iy = base_y + offsets[:, 0:1].unsqueeze(-1)  # (n, h, w)
    ix = base_x + offsets[:, 1:2].unsqueeze(-1)
    # Flat-index into padded.view(n, c, (h+2pad)*(w+2pad)).
    flat = padded.reshape(n, c, (h + 2 * pad) * (w + 2 * pad))
    flat_idx = (iy * (w + 2 * pad) + ix).unsqueeze(1).expand(-1, c, -1, -1)
    return torch.gather(flat, 2, flat_idx.reshape(n, c, h * w)).reshape(n, c, h, w)


class _SpatialSoftmaxCNN(nn.Module):
    """Small CNN with a Levine-style spatial softmax head.

    Shape contract: input ``(N, C, H, W)`` in ``[0, 1]``, output
    ``(N, 2 * num_keypoints)``.

    Why spatial softmax instead of flatten+Linear: for wrist-camera
    manipulation where the policy needs to localize a small object on a
    uniform table, an MLP projection averages out spatial information; PPO
    has to discover from the dense grasp gradient that "the cube position
    in the image matters", which (per run 5 diagnostic) it never did. The
    spatial softmax converts each channel of the final conv into a soft
    expected (x, y) image-plane coordinate via:

        attn[k]   = softmax over spatial dims of channel k
        kp_x[k]   = sum(attn[k] * x_grid)
        kp_y[k]   = sum(attn[k] * y_grid)

    so each output coordinate pair is geometrically meaningful and the
    encoder is biased toward "treat each channel as detecting a feature at
    a location", which is exactly the inductive bias for finding a cube
    on a flat workspace. This is the architecture from Levine et al. 2016
    (`End-to-End Training of Deep Visuomotor Policies`, JMLR).

    Output dim is ``2 * num_keypoints`` (kept at 128 to match the prior
    Impala projection so downstream MLP shapes are unchanged). The final
    conv produces ``num_keypoints`` channels — fewer than the original
    64, so the model is slightly smaller too. The output goes through
    LayerNorm to keep keypoint magnitudes bounded; image coords are
    naturally in [-1, 1] but LayerNorm stabilizes early training.
    """

    def __init__(self, in_shape: tuple[int, int, int], out_dim: int = 128):
        super().__init__()
        if out_dim % 2 != 0:
            raise ValueError(f"out_dim={out_dim} must be even (2 coords per keypoint)")
        num_keypoints = out_dim // 2
        in_channels, in_h, in_w = in_shape
        # Same first two convs as the prior Impala-style stack; the final
        # conv outputs ``num_keypoints`` channels (not 64) so each channel
        # corresponds to one keypoint after the spatial softmax.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(64, num_keypoints, kernel_size=3, stride=1),
        )
        # Probe the conv output spatial dims so we can register the
        # coordinate grid as a buffer of the right size.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, in_h, in_w)
            out = self.conv(dummy)
        feat_h, feat_w = int(out.shape[2]), int(out.shape[3])
        # x_grid / y_grid are flattened (H*W,) tensors in [-1, 1]. They
        # are buffers so they move with .to(device) but aren't trained.
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, feat_h),
            torch.linspace(-1.0, 1.0, feat_w),
            indexing="ij",
        )
        self.register_buffer("x_grid", xs.reshape(-1))
        self.register_buffer("y_grid", ys.reshape(-1))
        # LayerNorm over the 2K output keeps init activations stable even
        # if the conv heatmaps start very peaky (one-hot soft-argmax).
        self.norm = nn.LayerNorm(out_dim)
        self.num_keypoints = num_keypoints
        self.out_dim = out_dim
        self.feat_h = feat_h
        self.feat_w = feat_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)  # (N, K, H, W)
        n, k, h, w = feat.shape
        # Softmax over spatial dims gives a (N, K, H*W) attention map.
        attn = torch.softmax(feat.view(n, k, h * w), dim=-1)
        # Expected x / y coordinate per channel.
        ex = (attn * self.x_grid).sum(dim=-1)  # (N, K)
        ey = (attn * self.y_grid).sum(dim=-1)  # (N, K)
        # Interleave as (kp0_x, kp0_y, kp1_x, kp1_y, ...) — slightly nicer
        # for downstream interpretation (the aux head can reshape to
        # (N, K, 2) and pick a subset). Total dim = 2 * K = out_dim.
        kp = torch.stack([ex, ey], dim=-1).reshape(n, 2 * k)
        return self.norm(kp)


# Legacy alias so any downstream module that still imports the old name
# continues to work. The class name swap is intentional — anyone reading
# ``self.actor_cnn = _SpatialSoftmaxCNN(...)`` sees the architecture.
_ImpalaSmallCNN = _SpatialSoftmaxCNN


class _ResNetSpatialSoftmaxCNN(nn.Module):
    """Frozen ResNet-18 (ImageNet-pretrained) trunk + trainable 1×1 conv +
    spatial-softmax head. Drop-in replacement for :class:`_SpatialSoftmaxCNN`
    when sim2real robustness matters more than from-scratch fitting speed.

    Shape contract: input ``(N, C, H, W)`` in ``[0, 1]``, output
    ``(N, 2 * num_keypoints) = (N, out_dim)`` — identical to
    :class:`_SpatialSoftmaxCNN` so the surrounding actor/critic MLPs are
    unchanged.

    Why this class exists (v4 design notes, see EVAL1_PLAN §7 fallback):

    1. **ImageNet features are domain-general.** Real-world lighting and
       color variation that the sim domain randomization can't perfectly
       cover are inside ImageNet's pretraining distribution.
    2. **Frozen trunk = stable PPO.** The most common visual-PPO failure
       mode in this repo is encoder gradients fighting RL gradients. With
       ``requires_grad=False`` on the trunk, only the 1×1 conv + spatial
       softmax + downstream MLPs see PPO gradients.
    3. **Spatial-softmax inductive bias kept.** ResNet features by
       themselves are not localization-friendly; the 1×1 conv re-projects
       to per-keypoint heatmaps and the soft-argmax extracts (x, y)
       coords. Same Levine-2016 head as the from-scratch CNN.
    4. **Channel inflation, not RGB-only.** v4 keeps the binary block mask
       as channel 3. We inflate ResNet's ``conv1`` (3 → C input channels)
       and init the mask-channel weights from the RGB-channel mean. This
       avoids a separate mask CNN at the cost of a slight init mismatch
       (binary input through a kernel trained on float RGB) — empirically
       fine because the mask conv1 weights immediately adapt during the
       brief distillation phase that comes before frozen-trunk PPO.

    Note: when ``freeze=True`` we also force the trunk into ``.eval()``
    mode so BatchNorm running stats don't drift under PPO's non-stationary
    rollouts — see Wu & He, GroupNorm, ECCV 2018 for why BN under
    distribution shift is a known landmine. (Frozen trunk side-steps the
    issue without needing a BN→GN conversion.)
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int],
        out_dim: int = 128,
        freeze: bool = True,
        bc_v1_weights_path: str | None = None,
    ):
        super().__init__()
        if out_dim % 2 != 0:
            raise ValueError(f"out_dim={out_dim} must be even (2 coords per keypoint)")
        num_keypoints = out_dim // 2
        in_channels, in_h, in_w = in_shape

        # Lazy import — torchvision is a heavy import and not all callers
        # need it. The class is gated on a cfg flag so the existing
        # from-scratch path doesn't pull torchvision in.
        from torchvision import models as tvm

        weights = tvm.ResNet18_Weights.IMAGENET1K_V1
        backbone = tvm.resnet18(weights=weights)

        # ---- conv1 inflation if in_channels != 3 ---------------------------
        if in_channels != 3:
            old_conv1 = backbone.conv1
            new_conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False,
            )
            with torch.no_grad():
                # Copy first 3 channels (RGB) verbatim from ImageNet.
                new_conv1.weight[:, :3] = old_conv1.weight
                if in_channels > 3:
                    # Init extra channels (e.g. mask) from RGB-mean — keeps
                    # activation statistics roughly the same scale as the
                    # ImageNet-trained channels would produce.
                    rgb_mean = old_conv1.weight.mean(dim=1, keepdim=True)
                    new_conv1.weight[:, 3:] = rgb_mean.expand(-1, in_channels - 3, -1, -1)
            backbone.conv1 = new_conv1

        # Truncate at layer3 — gives a richer spatial map than layer4
        # (which is 3×4 at 72×128) without losing too much resolution.
        self.trunk = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3,
        )

        # Optionally overlay BC v1's encoder weights — torchvision's
        # ResNet-18 children layout matches what bc/model.py saves under
        # ``img_enc.backbone.*`` (also a ``nn.Sequential`` of the same
        # children, minus avgpool/fc). Loaded with strict=False so missing
        # keys (avgpool/fc) and extra keys (BC v1's ``proj``) are tolerated.
        if bc_v1_weights_path is not None:
            self._maybe_load_bc_v1(bc_v1_weights_path, expected_in_channels=in_channels)

        # ---- ImageNet input normalization (for RGB channels only) ----------
        # Channel 3+ (mask) is binary {0, 1}; we leave it untouched.
        _imagenet_mean = torch.tensor([0.485, 0.456, 0.406, 0.0] + [0.0] * max(0, in_channels - 4))
        _imagenet_std = torch.tensor([0.229, 0.224, 0.225, 1.0] + [1.0] * max(0, in_channels - 4))
        self.register_buffer("_in_mean", _imagenet_mean[:in_channels].view(1, in_channels, 1, 1))
        self.register_buffer("_in_std", _imagenet_std[:in_channels].view(1, in_channels, 1, 1))

        # Freeze trunk (so PPO grads can't destabilize the encoder).
        self.freeze = bool(freeze)
        if self.freeze:
            for p in self.trunk.parameters():
                p.requires_grad = False
            self.trunk.eval()

        # ---- Probe spatial dims after trunk ---------------------------------
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, in_h, in_w)
            feat = self.trunk(self._normalize_input(dummy))
        trunk_c, feat_h, feat_w = int(feat.shape[1]), int(feat.shape[2]), int(feat.shape[3])

        # 1×1 conv to num_keypoints channels — small, trainable.
        self.head = nn.Conv2d(trunk_c, num_keypoints, kernel_size=1)

        # Soft-argmax grid (same as _SpatialSoftmaxCNN).
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
        self.feat_h = feat_h
        self.feat_w = feat_w

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._in_mean) / self._in_std

    def _maybe_load_bc_v1(self, path: str, expected_in_channels: int) -> None:
        """Best-effort overlay of BC v1's ResNet-18 weights onto this trunk."""
        import os
        if not os.path.isfile(path):
            print(f"[ResNetEncoder] BC v1 ckpt not found at {path} — keeping ImageNet init.")
            return
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[ResNetEncoder] failed to load BC v1 ckpt: {e!r} — ImageNet init.")
            return
        sd = ck.get("model", ck)
        # Extract keys matching ``img_enc.backbone.<idx>.*`` and rename to
        # ``<idx>.*`` so they line up with our ``trunk`` Sequential indices.
        backbone_prefix = "img_enc.backbone."
        overlay = {}
        for k, v in sd.items():
            if k.startswith(backbone_prefix):
                overlay[k[len(backbone_prefix):]] = v
        if not overlay:
            print("[ResNetEncoder] BC v1 ckpt has no img_enc.backbone.* keys — skipping.")
            return
        # If conv1 was inflated (channels > 3), the saved BC v1 conv1 is
        # (64, 3, 7, 7) — strict load would fail. Pad it to current channels.
        conv1_key = "0.weight"  # trunk[0] = conv1
        if conv1_key in overlay:
            w = overlay[conv1_key]
            if w.shape[1] != expected_in_channels:
                with torch.no_grad():
                    new_w = torch.zeros(w.shape[0], expected_in_channels, w.shape[2], w.shape[3])
                    new_w[:, :min(w.shape[1], expected_in_channels)] = w[:, :min(w.shape[1], expected_in_channels)]
                    if expected_in_channels > w.shape[1]:
                        new_w[:, w.shape[1]:] = w.mean(dim=1, keepdim=True)
                    overlay[conv1_key] = new_w
        missing, unexpected = self.trunk.load_state_dict(overlay, strict=False)
        print(f"[ResNetEncoder] loaded BC v1 weights from {path}: "
              f"{len(overlay)} keys overlaid, missing={len(missing)} unexpected={len(unexpected)}")

    def train(self, mode: bool = True):  # noqa: D401
        """Override so the frozen trunk stays in eval mode regardless."""
        super().train(mode)
        if self.freeze:
            self.trunk.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize_input(x)
        if self.freeze:
            with torch.no_grad():
                feat = self.trunk(x)
        else:
            feat = self.trunk(x)
        heat = self.head(feat)                                   # (N, K, H, W)
        n, k, h, w = heat.shape
        attn = torch.softmax(heat.view(n, k, h * w), dim=-1)
        ex = (attn * self.x_grid).sum(dim=-1)                    # (N, K)
        ey = (attn * self.y_grid).sum(dim=-1)
        kp = torch.stack([ex, ey], dim=-1).reshape(n, 2 * k)
        return self.norm(kp)


class PickPlaceVisionActorCritic(ActorCritic):
    """Asymmetric A-C with a CNN encoder for the wrist image.

    Constructor signature mirrors :class:`rsl_rl.modules.ActorCritic` — RSL-RL
    instantiates it via
    ``cls(obs, obs_groups, num_actions, **policy_cfg_kwargs)``. We keep that
    contract intact so the runner doesn't need any patching beyond the
    ``class_name`` lookup (see :func:`agents.rsl_rl_ppo_cfg._register_class`).

    Args:
        image_group_name: name of the obs group carrying ``(N, 3, H, W)``
            image tensors. Default ``"wrist_rgb"``.
        image_feat_dim: latent dimension of the CNN output. Default 128.
        Other args are forwarded to :class:`ActorCritic`.
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
        **kwargs,
    ):
        # We deliberately do NOT call super().__init__ — the parent's init
        # asserts every obs group is 1-D, which fails for the image group.
        # We replicate the parts of the parent we still need (actor/critic
        # MLPs, action distribution params, normalizers).
        nn.Module.__init__(self)
        if kwargs:
            print(
                "PickPlaceVisionActorCritic.__init__ got unexpected kwargs (ignored): "
                + str(list(kwargs.keys()))
            )

        self.obs_groups = obs_groups
        self.image_group_name = image_group_name

        # ------------------------------------------------------------------
        # Inspect obs to compute input dims for actor / critic MLPs.
        # ------------------------------------------------------------------
        # The image group is routed through a CNN; every other group is
        # treated as a 1-D state vector (assert kept for those).
        actor_state_dim = self._sum_state_dims(obs, obs_groups["policy"], image_group_name)
        critic_state_dim = self._sum_state_dims(obs, obs_groups["critic"], image_group_name)
        actor_uses_image = image_group_name in obs_groups["policy"]
        critic_uses_image = image_group_name in obs_groups["critic"]

        # Image shape (C, H, W) — captured from a sample tensor so the CNN's
        # ``proj`` layer is sized correctly at construction (not lazily during
        # the first forward, which would skip optimizer registration).
        if actor_uses_image or critic_uses_image:
            img_sample = obs[image_group_name]
            assert (
                img_sample.dim() == 4
            ), f"Expected image obs of shape (N, C, H, W); got {tuple(img_sample.shape)}"
            img_in_shape = (
                int(img_sample.shape[1]),
                int(img_sample.shape[2]),
                int(img_sample.shape[3]),
            )
        else:
            img_in_shape = None

        # Two encoders — keep actor/critic decoupled so the privileged
        # critic info doesn't leak into the actor's encoder gradients.
        if actor_uses_image:
            self.actor_cnn: nn.Module = _ImpalaSmallCNN(img_in_shape, image_feat_dim)
        else:
            self.actor_cnn = None
        if critic_uses_image:
            self.critic_cnn: nn.Module = _ImpalaSmallCNN(img_in_shape, image_feat_dim)
        else:
            self.critic_cnn = None

        actor_in = actor_state_dim + (image_feat_dim if actor_uses_image else 0)
        critic_in = critic_state_dim + (image_feat_dim if critic_uses_image else 0)

        # Standard MLPs — same as parent class.
        self.actor = MLP(actor_in, num_actions, list(actor_hidden_dims), activation)
        self.critic = MLP(critic_in, 1, list(critic_hidden_dims), activation)
        print(f"Actor CNN: {self.actor_cnn}")
        print(f"Critic CNN: {self.critic_cnn}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Observation normalization is only over the *state* portion. The
        # CNN already has a LayerNorm at its head, so image features are
        # well-conditioned without an extra normalizer.
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

        # Action noise — per-dim init: arm dims at the cfg-supplied
        # ``init_noise_std``, gripper dim hardcoded to 0.1.
        #
        # Why a smaller gripper σ:
        #     The gripper action goes through ``BinaryJointPositionAction``
        #     which thresholds ``action[5] > 0`` → open / ≤ 0 → close.
        #     With scalar σ=0.5 across all dims, the gripper command flips
        #     ~once every 2-3 steps even when μ_gripper sits at e.g. -0.3
        #     (P(action[5] > 0) ≈ 28%). The cube cannot survive 30 frames
        #     in a closed grasp under that flip rate, so PPO never observes
        #     a successful close-and-hold trajectory and never learns the
        #     credit assignment "close jaws at cube → +15 grasp reward".
        #     Run-12 diagnostic (2026-05-09 iter 100): grasp_bootstrap
        #     decayed 0.41 → 0.010 within ~30 sim steps; grasp_from_scratch
        #     stayed flat at 0.0 across 100 iters even after the
        #     ``pre_grasp_pose`` latch fix removed the open-jaws attractor.
        #     Bootstrap can't deliver useful gradient because the cube
        #     drops out before any value-function tail can form.
        # Lowering gripper-dim σ to 0.1 makes the binary command "decisive
        # open" or "decisive close" for the first ~50-100 iters of training,
        # giving sustained-grasp trajectories a fighting chance to appear in
        # rollouts. ``self.std`` is still a learnable nn.Parameter, so PPO
        # adapts the noise level from there as usual.
        # Action noise — stock RSL-RL semantics (uniform σ across all dims).
        # Earlier we had a per-dim override hardcoding σ_gripper to 0.1 / 0.2
        # on the theory that the binary gripper action needed quieter
        # exploration than the arm joints. Stock Isaac Lab Franka Lift PPO
        # uses ``init_noise_std=1.0`` across all dims with the same
        # ``BinaryJointPositionActionCfg`` gripper and converges in ~1500
        # iters — proving the override was unnecessary (and possibly
        # harmful, since it locked early exploration too narrow to
        # discover sustained-close trajectories). Removing the override
        # restores stock semantics; ``self.std`` is still a learnable
        # Parameter so PPO adapts σ as usual.
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
    # Helpers
    # ----------------------------------------------------------------------

    @staticmethod
    def _sum_state_dims(obs, group_names, image_group_name) -> int:
        """Sum last-dim sizes of all *non-image* groups in ``group_names``."""
        total = 0
        for name in group_names:
            if name == image_group_name:
                continue
            t = obs[name]
            assert (
                t.dim() == 2
            ), f"State obs group {name!r} must be 1-D per env; got shape {tuple(t.shape)}"
            total += t.shape[-1]
        return total

    @staticmethod
    def _safe(t: torch.Tensor) -> torch.Tensor:
        """Return a tensor that is safe to back-prop through.

        Obs tensors collected under Isaac Lab's ``torch.inference_mode()``
        rollout end up in the rollout buffer (non-inference) — but the
        intermediate ``act()`` calls during the *rollout* itself receive
        inference tensors, and during the PPO *update* the buffer slices
        sometimes still carry the inference flag depending on the TensorDict
        view path. ``torch.empty_like`` allocates a fresh non-inference
        storage in any context; ``.copy_()`` then transfers the values. This
        is the cheapest unconditional way to break the inference link.
        """
        out = torch.empty_like(t)
        out.copy_(t)
        return out

    def _gather_state(self, obs, group_names) -> torch.Tensor:
        parts = [self._safe(obs[g]) for g in group_names if g != self.image_group_name]
        return (
            torch.cat(parts, dim=-1)
            if parts
            else torch.empty(obs[self.image_group_name].shape[0], 0, device=obs[self.image_group_name].device)
        )

    def _encode_actor(self, obs) -> torch.Tensor:
        state = self._gather_state(obs, self.obs_groups["policy"])
        state = self.actor_obs_normalizer(state)
        if self.actor_cnn is None:
            return state
        # Apply DrQ-style random-shift augmentation when training. We
        # gate on ``self.training`` so deployed policies (set to .eval()
        # by the play script) see clean, non-augmented frames — the
        # encoder still consumes the same image distribution because the
        # augmentation is small (4-px shift) and the network is trained
        # to be invariant to it.
        img = self._safe(obs[self.image_group_name])
        if self.training:
            img = _random_shift_pad(img, DRQ_PAD_PIXELS)
        feat = self.actor_cnn(img)
        return torch.cat([state, feat], dim=-1)

    def _encode_critic(self, obs) -> torch.Tensor:
        state = self._gather_state(obs, self.obs_groups["critic"])
        state = self.critic_obs_normalizer(state)
        if self.critic_cnn is None:
            return state
        feat = self.critic_cnn(self._safe(obs[self.image_group_name]))
        return torch.cat([state, feat], dim=-1)

    # ----------------------------------------------------------------------
    # ActorCritic interface
    # ----------------------------------------------------------------------

    def update_distribution(self, x):
        mean = self.actor(x)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
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

    # The parent ``get_actor_obs`` / ``get_critic_obs`` are 1-D-only and would
    # crash on the image group; we override them so callers (e.g. PPO storage
    # bookkeeping) get a sensible flattened vector. Image features are
    # pre-encoded so the rest of PPO sees a 1-D tensor downstream.
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

    # ----------------------------------------------------------------------
    # Warm-start from distillation checkpoint (EVAL1_PLAN §7 stage 3)
    # ----------------------------------------------------------------------

    def load_state_dict(self, state_dict, strict=True):
        """Load a normal PPO checkpoint **or** warm-start from a distillation one.

        Distillation (stage 2) saves a :class:`PickPlaceVisionStudentTeacher`,
        which has keys ``student_cnn.*`` / ``student.*`` / ``std`` /
        ``teacher.*``. The vision A-C (this class) expects ``actor_cnn.*`` /
        ``actor.*`` / ``std`` / ``critic.*``. Architectures match exactly:

        * ``student_cnn`` ↔ ``actor_cnn`` — same ``_SpatialSoftmaxCNN``, same
          5-channel input, same 128-D output.
        * ``student`` ↔ ``actor`` — same MLP[256,128,64], same 153-D input
          (state + image features), same 6-D output.
        * ``std`` is **DROPPED** — the distill checkpoint stores ``std=0.1``
          (BC regression init), too narrow for PPO exploration. We keep the
          cfg's ``init_noise_std`` (currently 0.5) instead. The binary gripper
          action at σ=0.1 is effectively deterministic, killing exploration
          on the open/close decision; restoring σ=0.5 lets PPO probe out of
          the imitation basin. See EVAL1_PLAN §7.2 intervention #2.
        * ``teacher.*`` is dropped: it's the *policy* head of the state-only
          teacher, not a value function, so it cannot meaningfully initialize
          ``critic`` (different objective). The critic warm-start happens
          OUT-OF-BAND in train.py via the ``--teacher_ckpt`` flag (§7.2
          intervention #5), which loads ``critic.*`` keys directly from the
          stage-1 teacher PPO checkpoint.

        Returns ``False`` for distill warm-starts so
        :meth:`OnPolicyRunner.load` skips optimizer-state loading and resets
        the iteration counter — this is a fresh PPO run that happens to
        start from warm weights, *not* a resume.
        """
        is_distill = any(
            k.startswith("student_cnn.") or k.startswith("student.")
            for k in state_dict
        )
        if not is_distill:
            return super().load_state_dict(state_dict, strict=strict)

        remapped = {}
        for k, v in state_dict.items():
            if k.startswith("student_cnn."):
                remapped["actor_cnn." + k[len("student_cnn.") :]] = v
            elif k.startswith("student."):
                remapped["actor." + k[len("student.") :]] = v
            elif k == "std":
                # DROP — keep cfg's init_noise_std (0.5). The distill ckpt's
                # std=0.1 is too narrow for PPO exploration; loading it here
                # kills the binary gripper's open/close exploration and
                # leaves PPO unable to probe out of the imitation basin.
                # EVAL1_PLAN §7.2 intervention #2.
                continue
            elif k.startswith("teacher."):
                # See docstring — teacher is a policy, not a value head.
                continue
            else:
                # Pass through anything else (e.g. optional normalizers).
                remapped[k] = v

        # strict=False because the fresh ``critic.*`` keys and the deliberately-
        # skipped ``std`` are expected to be missing from the remapped dict.
        # Any *other* missing/unexpected key is a real mismatch and should
        # fail loud.
        missing, unexpected = nn.Module.load_state_dict(self, remapped, strict=False)
        expected_missing = {
            k for k in missing
            if k == "std"  # intentionally kept at cfg's init_noise_std
            or k.startswith("critic.")
            or k.startswith("critic_cnn.")
            or k.startswith("critic_obs_normalizer.")
        }
        real_missing = [k for k in missing if k not in expected_missing]
        if real_missing or unexpected:
            raise RuntimeError(
                "Distill warm-start: unexpected key mismatch.\n"
                f"  unexpected: {list(unexpected)}\n"
                f"  missing (non-critic, non-std): {real_missing}"
            )
        std_val = float(self.std.detach().mean().item()) if hasattr(self, "std") else None
        print(
            f"[PickPlaceVisionActorCritic] Warm-started from distillation checkpoint: "
            f"loaded {len(remapped)} keys into actor_cnn/actor; "
            f"std kept at cfg init_noise_std={std_val}; "
            f"critic initializes fresh ({sum(1 for k in expected_missing if k != 'std')} keys — "
            f"load teacher critic via --teacher_ckpt to avoid the random-init advantage shock)."
        )
        return False  # not a resume — runner should skip optimizer load
