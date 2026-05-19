"""Minimal PPO actor for real-robot deploy.

Re-implements ``PickPlaceVisionActorCritic`` (actor side only) without any
``rsl_rl`` / ``isaaclab`` runtime dependency, so the inference PC doesn't need
the simulator installed. The architecture exactly matches
``isaac_so_arm101.tasks.pickplace.agents.vision_actor_critic`` — same conv
stack, same spatial softmax head, same MLP[256, 128, 64] — so a checkpoint
saved by training loads with no key remapping.

Shape contract:

* state input  : ``(B, 25)``  — joint_pos_rel(6) + joint_vel_rel(6) +
                                 gripper_state(1) + bowl_xy(2) +
                                 ee_proj_xy(2) + ee_to_bowl_xy(2) +
                                 last_action(6)
* image input  : ``(B, 4, 72, 128)`` — RGB + binary block mask in ``[0, 1]``
* action output: ``(B, 6)``   — deterministic mean of the Gaussian
                                 (no noise at deploy)
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
