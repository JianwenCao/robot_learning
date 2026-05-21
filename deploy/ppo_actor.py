"""Minimal PPO actor for real-robot deploy (state-only + AprilTag).

Forward-only mirror of the training-time
:class:`PickPlaceVisionActorCritic` *when the wrist_image obs group is
absent from obs_groups* — i.e. the state-only + AprilTag deploy path.
With no image input the training-side class collapses to a pure
MLP[256,128,64] on the deployable state vector. Checkpoint keys are the
same MLP keys (``actor.0.weight``, ...) so we just slice those.

State (B, 27): policy(25) + cube_pos_xy_noisy(2). The trailing 2 dims
come from AprilTag pose at deploy (see :mod:`deploy.cube_detector`) or
from :func:`mdp.observations.cube_pos_xy_noisy` in sim.

The actor outputs the Gaussian's deterministic mean — no noise at deploy.

No ``rsl_rl`` / ``isaaclab`` runtime dependency so the inference PC
doesn't need the simulator installed.
"""
from __future__ import annotations

import torch
import torch.nn as nn

STATE_DIM = 25
ACTION_DIM = 6
STATE_APRILTAG_STATE_DIM = STATE_DIM + 2  # 27
VISION_IMAGE_SHAPE = (3, 240, 320)
VISION_FEATURE_DIM = 128


def _build_actor_mlp(in_dim: int, hidden=(256, 128, 64), out_dim: int = ACTION_DIM) -> nn.Sequential:
    """MLP layout that matches RSL-RL's ``MLP`` keying (linear at even indices)."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ELU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class PPOActorState(nn.Module):
    """Deterministic state-only actor — MLP-only, no image branch.

    Forward signature: ``(state_27,) -> action_6``. The trailing 2 dims
    of ``state_27`` are absolute ``cube_xy`` as produced by the AprilTag
    detector on real (or by :func:`mdp.observations.cube_pos_xy_noisy`
    in sim).
    """

    def __init__(self):
        super().__init__()
        self.actor = _build_actor_mlp(STATE_APRILTAG_STATE_DIM)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state)

    @classmethod
    def from_checkpoint(cls, ckpt_path, map_location="cpu") -> "PPOActorState":
        ck = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
        cnn_keys = [k for k in sd if k.startswith("actor_cnn.")]
        if cnn_keys:
            raise RuntimeError(
                "PPOActorState got a checkpoint with actor_cnn.* keys — "
                "this looks like a vision checkpoint, not a state-only one."
            )
        actor_sd = {k: v for k, v in sd.items() if k.startswith("actor.")}
        model = cls()
        missing, unexpected = model.load_state_dict(actor_sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"PPOActorState checkpoint key mismatch.\n"
                f"  missing:    {missing}\n  unexpected: {unexpected}"
            )
        model.eval()
        return model


class _SpatialSoftmaxCNN(nn.Module):
    """Deploy mirror of the training-side RGB spatial-softmax CNN."""

    def __init__(self, in_shape: tuple[int, int, int] = VISION_IMAGE_SHAPE, out_dim: int = VISION_FEATURE_DIM):
        super().__init__()
        if out_dim % 2 != 0:
            raise ValueError("out_dim must be even")
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
            out = self.conv(torch.zeros(1, c, h, w))
        feat_h, feat_w = int(out.shape[2]), int(out.shape[3])
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, feat_h),
            torch.linspace(-1.0, 1.0, feat_w),
            indexing="ij",
        )
        self.register_buffer("x_grid", xs.reshape(-1))
        self.register_buffer("y_grid", ys.reshape(-1))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        n, k, h, w = feat.shape
        attn = torch.softmax(feat.view(n, k, h * w), dim=-1)
        ex = (attn * self.x_grid).sum(dim=-1)
        ey = (attn * self.y_grid).sum(dim=-1)
        return self.norm(torch.stack([ex, ey], dim=-1).reshape(n, 2 * k))


class PPOActorVision(nn.Module):
    """Deterministic vision student actor: state_25 + RGB_240x320 -> action_6."""

    def __init__(self):
        super().__init__()
        self.student_cnn = _SpatialSoftmaxCNN()
        self.student = _build_actor_mlp(STATE_DIM + VISION_FEATURE_DIM)

    def forward(self, state: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        feat = self.student_cnn(image)
        return self.student(torch.cat([state, feat], dim=-1))

    @classmethod
    def from_checkpoint(cls, ckpt_path, map_location="cpu") -> "PPOActorVision":
        ck = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
        model = cls()
        if any(k.startswith("student_cnn.") or k.startswith("student.") for k in sd):
            model_sd = {
                k: v
                for k, v in sd.items()
                if k.startswith("student_cnn.") or k.startswith("student.")
            }
        elif any(k.startswith("actor_cnn.") or k.startswith("actor.") for k in sd):
            model_sd = {}
            for k, v in sd.items():
                if k.startswith("actor_cnn."):
                    model_sd["student_cnn." + k[len("actor_cnn."):]] = v
                elif k.startswith("actor."):
                    model_sd["student." + k[len("actor."):]] = v
        else:
            raise RuntimeError(f"No student/actor vision keys found in checkpoint: {ckpt_path}")
        missing, unexpected = model.load_state_dict(model_sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"PPOActorVision checkpoint key mismatch.\n"
                f"  missing:    {missing}\n  unexpected: {unexpected}"
            )
        model.eval()
        return model
