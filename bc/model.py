"""Goal-conditioned BC policy.

Architecture:
  wrist image (B, 3, 72, 128) uint8 → ImageNet-normalized float → ResNet-18 →
    GAP → Linear(512 → 256)                                                (img feat)
  proprio (B, 6) normalized → MLP → 128                                    (proprio feat)
  bowl    (B, 3) normalized → MLP → 64                                     (goal feat)
  concat (448) → trunk MLP (512 → 512) → head Linear(512 → k*6)
  reshape → (B, k, 6)  — normalized actions

Outputs are in the normalized action space. Callers (eval/deploy) denormalize
through `Stats.denormalize("action", ...)`.

Image augmentation lives here (train-time only, applied inside forward when
`self.training`), so the dataset can keep returning raw uint8 CHW images and
augmentation is consistent regardless of dataloader workers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm
from torchvision.transforms import v2 as T

from .config import ACTION_DIM, BOWL_DIM, CHUNK_K, IMG_H, IMG_W, PROPRIO_DIM

# ImageNet stats — input to ResNet must be normalized this way.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.resnet18(weights=weights)
        # drop final fc; keep conv trunk + avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 512, 1, 1)
        self.proj = nn.Linear(512, out_dim)
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, img_u8: torch.Tensor) -> torch.Tensor:
        # img_u8: (B, 3, H, W) uint8
        x = img_u8.float() / 255.0
        x = (x - self._mean) / self._std
        x = self.backbone(x).flatten(1)         # (B, 512)
        return self.proj(x)                     # (B, out_dim)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def _make_train_augmentations(strength: str = "v1") -> nn.Module:
    """Image augmentations applied at train time. Operate on uint8 (B,3,H,W).

    ``strength="v1"`` matches the original baseline.
    ``strength="v3"`` is heavier color/lighting DR — broader ColorJitter,
    higher grayscale prob, random gamma, additive pixel noise, larger crop.
    Used to combat the sim2real gap in background lighting / block tint.
    """
    if strength == "v1":
        return T.Compose([
            T.RandomCrop((IMG_H, IMG_W), padding=6, padding_mode="reflect"),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomGrayscale(p=0.1),
        ])
    if strength == "v3":
        return T.Compose([
            T.RandomCrop((IMG_H, IMG_W), padding=10, padding_mode="reflect"),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.10),
            T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
            T.RandomAutocontrast(p=0.3),
            T.RandomEqualize(p=0.1),
            T.RandomGrayscale(p=0.2),
        ])
    raise ValueError(f"unknown aug strength {strength!r}")


class GoalCondBCPolicy(nn.Module):
    def __init__(
        self,
        k: int = CHUNK_K,
        proprio_dim: int = PROPRIO_DIM,
        bowl_dim: int = BOWL_DIM,
        action_dim: int = ACTION_DIM,
        img_feat: int = 256,
        proprio_feat: int = 128,
        goal_feat: int = 64,
        trunk_hidden: int = 512,
        pretrained_backbone: bool = True,
        aug_strength: str = "v1",
    ):
        super().__init__()
        self.k = k
        self.action_dim = action_dim

        self.img_enc = _ImageEncoder(out_dim=img_feat, pretrained=pretrained_backbone)
        self.proprio_enc = _MLP(proprio_dim, 128, proprio_feat)
        self.goal_enc = _MLP(bowl_dim, 64, goal_feat)

        fused = img_feat + proprio_feat + goal_feat
        self.trunk = nn.Sequential(
            nn.Linear(fused, trunk_hidden), nn.GELU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.GELU(),
        )
        self.head = nn.Linear(trunk_hidden, k * action_dim)

        self._train_aug = _make_train_augmentations(strength=aug_strength)

    # -------- per-parameter-group LR helper (for AdamW config in train.py) --
    def param_groups(self, base_lr: float, backbone_lr_mult: float = 0.1):
        backbone_params = list(self.img_enc.backbone.parameters())
        other_params = [p for n, p in self.named_parameters()
                        if not n.startswith("img_enc.backbone.")]
        return [
            {"params": backbone_params, "lr": base_lr * backbone_lr_mult},
            {"params": other_params, "lr": base_lr},
        ]

    # -------- forward ------------------------------------------------------
    def forward(
        self,
        img: torch.Tensor,        # (B, 3, H, W) uint8
        proprio: torch.Tensor,    # (B, 6) normalized f32
        bowl: torch.Tensor,       # (B, 3) normalized f32
    ) -> torch.Tensor:
        if self.training:
            img = self._train_aug(img)
        f_img = self.img_enc(img)
        f_prop = self.proprio_enc(proprio)
        f_goal = self.goal_enc(bowl)
        z = torch.cat([f_img, f_prop, f_goal], dim=-1)
        z = self.trunk(z)
        a = self.head(z).view(-1, self.k, self.action_dim)
        return a                 # normalized actions


# --------------------------------------------------------------- smoke test ---

def _smoke_test() -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GoalCondBCPolicy().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device={device}, params={n_params/1e6:.2f}M (trainable {n_trainable/1e6:.2f}M)")

    B = 4
    img = torch.randint(0, 256, (B, 3, IMG_H, IMG_W), dtype=torch.uint8, device=device)
    proprio = torch.randn(B, PROPRIO_DIM, device=device)
    bowl = torch.randn(B, BOWL_DIM, device=device)

    # train-mode forward (uses augmentation)
    model.train()
    a_train = model(img, proprio, bowl)
    assert a_train.shape == (B, CHUNK_K, ACTION_DIM), a_train.shape
    assert torch.isfinite(a_train).all()
    print(f"train-mode out: shape={tuple(a_train.shape)}, "
          f"mean={a_train.mean().item():+.3f}, std={a_train.std().item():.3f}")

    # eval-mode forward (no augmentation)
    model.eval()
    with torch.no_grad():
        a_eval = model(img, proprio, bowl)
    assert a_eval.shape == (B, CHUNK_K, ACTION_DIM)
    assert torch.isfinite(a_eval).all()
    print(f"eval-mode  out: shape={tuple(a_eval.shape)}, "
          f"mean={a_eval.mean().item():+.3f}, std={a_eval.std().item():.3f}")

    # Determinism in eval mode — same input twice → same output
    with torch.no_grad():
        a_eval2 = model(img, proprio, bowl)
    assert torch.allclose(a_eval, a_eval2)
    print("eval-mode determinism: OK")

    # Param groups for LR multiplier
    groups = model.param_groups(base_lr=3e-4, backbone_lr_mult=0.1)
    print(f"backbone params: {sum(p.numel() for p in groups[0]['params'])/1e6:.2f}M @ lr={groups[0]['lr']}")
    print(f"other    params: {sum(p.numel() for p in groups[1]['params'])/1e6:.2f}M @ lr={groups[1]['lr']}")

    # Backprop smoke
    model.train()
    target = torch.randn_like(a_train)
    loss = F.l1_loss(model(img, proprio, bowl), target)
    loss.backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    print(f"loss={loss.item():.4f}, grad-abs-sum={grad_norm:.2f}")
    assert grad_norm > 0

    print("\nOK: model smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
