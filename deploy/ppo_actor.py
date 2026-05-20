"""Minimal PPO actors for real-robot deploy.

Forward-only mirrors of the training-time actor classes, without any
``rsl_rl`` / ``isaaclab`` runtime dependency so the inference PC doesn't
need the simulator installed.

Two classes:

* :class:`PPOActor` — mirrors
  ``isaac_so_arm101.tasks.pickplace.agents.vision_actor_critic.PickPlaceVisionActorCritic``
  (Eval-1, single-cube). Small-CNN encoder + MLP[256, 128, 64].

  State (B, 25): joint_pos_rel(6) + joint_vel_rel(6) + gripper_state(1) +
  bowl_xy(2) + ee_proj_xy(2) + ee_to_bowl_xy(2) + last_action(6).
  Image (B, 4, 72, 128): RGB + binary block mask in ``[0, 1]``.

* :class:`PPOActorClutter` — mirrors
  ``isaac_so_arm101.tasks.clutterpickplace.agents.vision_actor_critic.ClutterPickPlaceVisionActorCritic``
  (Eval-2/3). Frozen ImageNet ResNet-18 (truncated at layer2) + 1×1 conv
  with FiLM goal-conditioning + spatial-softmax + MLP[256, 128, 64].

  State (B, 31): policy(25) + target_color_onehot(6). Trailing 6 are
  routed to the FiLM head; the full 31 also feeds the MLP (matches the
  training-side ``obs_groups["policy"] = ["policy", "goal", "wrist_image"]``).
  Image (B, 4, 72, 128): RGB + target-color instance mask (from Florence-2
  with a color-prompt at deploy).

In both cases the actor outputs the Gaussian's deterministic mean — no
noise at deploy.
"""
from __future__ import annotations

import torch
import torch.nn as nn

STATE_DIM = 25
IMAGE_SHAPE = (4, 72, 128)
ACTION_DIM = 6
IMAGE_FEAT_DIM = 128


class _SpatialSoftmaxCNN(nn.Module):
    """Levine-style spatial-softmax CNN. Bit-for-bit copy of training-time class."""

    def __init__(self, in_shape: tuple[int, int, int] = IMAGE_SHAPE, out_dim: int = IMAGE_FEAT_DIM):
        super().__init__()
        if out_dim % 2 != 0:
            raise ValueError(f"out_dim={out_dim} must be even")
        num_keypoints = out_dim // 2
        c, h, w = in_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(64, num_keypoints, kernel_size=3, stride=1),
        )
        with torch.no_grad():
            probe = self.conv(torch.zeros(1, c, h, w))
        fh, fw = int(probe.shape[2]), int(probe.shape[3])
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, fh),
            torch.linspace(-1.0, 1.0, fw),
            indexing="ij",
        )
        self.register_buffer("x_grid", xs.reshape(-1))
        self.register_buffer("y_grid", ys.reshape(-1))
        self.norm = nn.LayerNorm(out_dim)
        self.num_keypoints = num_keypoints

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)                                  # (N, K, h, w)
        n, k, h, w = feat.shape
        attn = torch.softmax(feat.view(n, k, h * w), dim=-1)
        ex = (attn * self.x_grid).sum(dim=-1)                # (N, K)
        ey = (attn * self.y_grid).sum(dim=-1)
        kp = torch.stack([ex, ey], dim=-1).reshape(n, 2 * k)
        return self.norm(kp)


def _build_actor_mlp(in_dim: int, hidden=(256, 128, 64), out_dim: int = ACTION_DIM) -> nn.Sequential:
    """MLP layout that matches RSL-RL's ``MLP`` keying (linear at even indices)."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ELU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class PPOActor(nn.Module):
    """Deterministic actor: encode image → concat with state → MLP → action mean."""

    def __init__(self):
        super().__init__()
        self.actor_cnn = _SpatialSoftmaxCNN(IMAGE_SHAPE, IMAGE_FEAT_DIM)
        self.actor = _build_actor_mlp(STATE_DIM + IMAGE_FEAT_DIM)

    def forward(self, state: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        feat = self.actor_cnn(image)
        return self.actor(torch.cat([state, feat], dim=-1))

    @classmethod
    def from_checkpoint(cls, ckpt_path, map_location="cpu") -> "PPOActor":
        ck = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
        # Keep only actor-side keys; drop critic/std/optimizer/etc.
        actor_sd = {
            k: v for k, v in sd.items()
            if k.startswith("actor.") or k.startswith("actor_cnn.")
        }
        model = cls()
        missing, unexpected = model.load_state_dict(actor_sd, strict=False)
        # ``x_grid`` / ``y_grid`` are registered buffers we recompute in __init__
        # at the same resolution, so the checkpoint copies are interchangeable
        # and any "missing" entry for them is fine. Anything else is a mismatch.
        unexpected = [k for k in unexpected if not k.endswith(("x_grid", "y_grid"))]
        missing = [k for k in missing if not k.endswith(("x_grid", "y_grid"))]
        if missing or unexpected:
            raise RuntimeError(
                f"PPOActor checkpoint key mismatch.\n"
                f"  missing:    {missing}\n  unexpected: {unexpected}"
            )
        model.eval()
        return model


# ============================================================== Eval-2/3 actor
# Bit-for-bit mirror of
# ``isaac_so_arm101.tasks.pickplace.agents.vision_actor_critic._ResNetSpatialSoftmaxCNN``
# with ``truncate_at="layer2"`` and ``film_cond_dim=6`` (matches the Eval-2
# rsl_rl_ppo_cfg). conv1 is inflated to 4-channel input: RGB channels keep
# the ImageNet weights; channel 3 (mask) is initialised to the RGB-channel
# mean. Trunk is frozen at construction so the checkpoint's trunk weights
# overwrite the inflated init verbatim — no ImageNet download needed at
# deploy (`torchvision.models.resnet18(weights=None)` works for the
# scaffold; the checkpoint provides the actual weights).
CLUTTER_STATE_DIM = 31           # policy(25) + target_color_onehot(6)
CLUTTER_IMAGE_SHAPE = (4, 72, 128)
CLUTTER_GOAL_DIM = 6
CLUTTER_IMAGE_FEAT_DIM = 128
CLUTTER_FILM_HIDDEN = 64
CLUTTER_KEYPOINTS = CLUTTER_IMAGE_FEAT_DIM // 2  # 64


class _ResNetSpatialSoftmaxCNN(nn.Module):
    """Deploy mirror of the training-time ResNet+FiLM CNN.

    Args:
        in_shape: ``(C, H, W)``. ``C=4`` for clutter (RGB + target_mask).
        out_dim: keypoint feature dim. ``128`` to match training (64 kpts × 2).
        film_cond_dim: target_color one-hot dim. ``6`` to match training.
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int] = CLUTTER_IMAGE_SHAPE,
        out_dim: int = CLUTTER_IMAGE_FEAT_DIM,
        film_cond_dim: int = CLUTTER_GOAL_DIM,
    ):
        super().__init__()
        if out_dim % 2 != 0:
            raise ValueError(f"out_dim={out_dim} must be even")
        num_keypoints = out_dim // 2
        in_channels, in_h, in_w = in_shape

        from torchvision import models as tvm

        # No weights download at deploy — checkpoint provides them. The
        # bare resnet18(weights=None) scaffold has the same module layout
        # as the trained one, so state_dict keys match.
        backbone = tvm.resnet18(weights=None)

        if in_channels != 3:
            new_conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False,
            )
            with torch.no_grad():
                # Same initialisation as training-side, but the values get
                # overwritten by the checkpoint anyway.
                new_conv1.weight[:, :3] = backbone.conv1.weight
                if in_channels > 3:
                    rgb_mean = backbone.conv1.weight.mean(dim=1, keepdim=True)
                    new_conv1.weight[:, 3:] = rgb_mean.expand(-1, in_channels - 3, -1, -1)
            backbone.conv1 = new_conv1

        # Trunk = conv1+bn1+relu+maxpool+layer1+layer2 (truncate at layer2).
        self.trunk = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2,
        )

        # Input normalization buffers — matches training side.
        _mean = torch.tensor([0.485, 0.456, 0.406, 0.0] + [0.0] * max(0, in_channels - 4))
        _std = torch.tensor([0.229, 0.224, 0.225, 1.0] + [1.0] * max(0, in_channels - 4))
        self.register_buffer("_in_mean", _mean[:in_channels].view(1, in_channels, 1, 1))
        self.register_buffer("_in_std", _std[:in_channels].view(1, in_channels, 1, 1))

        # Freeze trunk (matters for eval mode determinism of BN running stats).
        for p in self.trunk.parameters():
            p.requires_grad = False
        self.trunk.eval()

        # Probe spatial dims with a dummy forward.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, in_h, in_w)
            feat = self.trunk((dummy - self._in_mean) / self._in_std)
        trunk_c, feat_h, feat_w = int(feat.shape[1]), int(feat.shape[2]), int(feat.shape[3])

        self.head = nn.Conv2d(trunk_c, num_keypoints, kernel_size=1)
        self.trunk_c = trunk_c

        if film_cond_dim > 0:
            self.film_mlp = nn.Sequential(
                nn.Linear(film_cond_dim, CLUTTER_FILM_HIDDEN),
                nn.ELU(),
                nn.Linear(CLUTTER_FILM_HIDDEN, 2 * trunk_c),
            )
        else:
            self.film_mlp = None
        self.film_cond_dim = int(film_cond_dim)

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

    def train(self, mode: bool = True):
        """Frozen trunk stays in eval mode regardless."""
        super().train(mode)
        self.trunk.eval()
        return self

    def forward(self, x: torch.Tensor, film_cond: torch.Tensor | None = None) -> torch.Tensor:
        x = (x - self._in_mean) / self._in_std
        with torch.no_grad():
            feat = self.trunk(x)
        if self.film_mlp is not None and film_cond is not None:
            gamma_beta = self.film_mlp(film_cond)
            gamma, beta = gamma_beta[:, :self.trunk_c], gamma_beta[:, self.trunk_c:]
            feat = gamma.unsqueeze(-1).unsqueeze(-1) * feat + beta.unsqueeze(-1).unsqueeze(-1)
        heat = self.head(feat)
        n, k, h, w = heat.shape
        attn = torch.softmax(heat.view(n, k, h * w), dim=-1)
        ex = (attn * self.x_grid).sum(dim=-1)
        ey = (attn * self.y_grid).sum(dim=-1)
        kp = torch.stack([ex, ey], dim=-1).reshape(n, 2 * k)
        return self.norm(kp)


class PPOActorClutter(nn.Module):
    """Eval-2/3 deterministic actor: ResNet+FiLM CNN + MLP.

    Forward signature: ``(state_31, image_4chw) → action_6``. The trailing
    6 dims of ``state_31`` are the target-color one-hot, sliced off and
    passed to the CNN's FiLM head; the full 31-D state also flows into
    the MLP (mirrors training where ``obs_groups["policy"] = ["policy",
    "goal", "wrist_image"]`` concatenates policy + goal before the MLP).
    """

    def __init__(self):
        super().__init__()
        self.actor_cnn = _ResNetSpatialSoftmaxCNN(
            in_shape=CLUTTER_IMAGE_SHAPE,
            out_dim=CLUTTER_IMAGE_FEAT_DIM,
            film_cond_dim=CLUTTER_GOAL_DIM,
        )
        self.actor = _build_actor_mlp(CLUTTER_STATE_DIM + CLUTTER_IMAGE_FEAT_DIM)

    def forward(self, state: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        # Trailing 6 dims of state == target_color_onehot (the "goal" group).
        goal_onehot = state[:, CLUTTER_STATE_DIM - CLUTTER_GOAL_DIM:CLUTTER_STATE_DIM]
        feat = self.actor_cnn(image, film_cond=goal_onehot)
        return self.actor(torch.cat([state, feat], dim=-1))

    @classmethod
    def from_checkpoint(cls, ckpt_path, map_location="cpu") -> "PPOActorClutter":
        ck = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
        actor_sd = {
            k: v for k, v in sd.items()
            if k.startswith("actor.") or k.startswith("actor_cnn.")
        }
        model = cls()
        missing, unexpected = model.load_state_dict(actor_sd, strict=False)
        # Recomputed buffers — interchangeable across checkpoints at same
        # spatial resolution. _in_mean/_in_std are also buffers we
        # re-register in __init__; safe to skip.
        skip_suffixes = ("x_grid", "y_grid", "_in_mean", "_in_std")
        unexpected = [k for k in unexpected if not k.endswith(skip_suffixes)]
        missing = [k for k in missing if not k.endswith(skip_suffixes)]
        if missing or unexpected:
            raise RuntimeError(
                f"PPOActorClutter checkpoint key mismatch.\n"
                f"  missing:    {missing}\n  unexpected: {unexpected}"
            )
        model.eval()
        return model
