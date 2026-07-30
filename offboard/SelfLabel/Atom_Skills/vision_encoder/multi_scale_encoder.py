from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MultiScaleEncoderConfig:
    image_size: tuple[int, int] = (224, 224)
    patch_sizes: list[int] = field(default_factory=lambda: [8, 16, 32])
    embed_dim: int = 1024
    num_heads: int = 16
    num_layers: int = 24
    mlp_ratio: float = 4.0
    use_flash_attn: bool = True
    output_modalities: list[str] = field(default_factory=lambda: ["object", "material", "global"])
    drop_path_rate: float = 0.1
    use_checkpointing: bool = True

    def __post_init__(self):
        assert all(p % 2 == 0 for p in self.patch_sizes), "patch_sizes must be even"


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class ScaleSpecificBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ScaleBranch(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int, num_heads: int, num_layers: int, mlp_ratio: float):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(224, patch_size, 3, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([
            ScaleSpecificBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        return cls_token, patch_tokens


class MultiScaleFusion(nn.Module):
    def __init__(self, embed_dim: int, num_scales: int):
        super().__init__()
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        self.fusion_proj = nn.Linear(embed_dim * num_scales, embed_dim)

    def forward(self, cls_tokens: list[torch.Tensor]) -> torch.Tensor:
        weights = F.softmax(self.scale_weights, dim=0)
        weighted = sum(w * t for w, t in zip(weights, cls_tokens))
        return self.fusion_proj(torch.cat(cls_tokens, dim=-1)) + weighted


class MultiScaleVisionEncoder(nn.Module):
    def __init__(self, config: MultiScaleEncoderConfig):
        super().__init__()
        self.config = config
        self.scales = nn.ModuleList([
            ScaleBranch(
                patch_size=ps,
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                num_layers=config.num_layers // len(config.patch_sizes),
                mlp_ratio=config.mlp_ratio,
            )
            for ps in config.patch_sizes
        ])
        self.fusion = MultiScaleFusion(config.embed_dim, len(config.patch_sizes))

        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B = x.shape[0]
        outputs = {}

        cls_tokens = []
        patch_tokens_list = []
        for branch in self.scales:
            cls_t, patch_t = branch(x)
            cls_tokens.append(cls_t)
            patch_tokens_list.append(patch_t)

        fused = self.fusion(cls_tokens)
        outputs["global"] = fused

        best_patches = patch_tokens_list[1]
        outputs["patch_features"] = best_patches

        scale_aware = []
        for i, pt in enumerate(patch_tokens_list):
            pooled = self.global_pool(pt.transpose(1, 2)).squeeze(-1)
            scale_aware.append(pooled.unsqueeze(1))
        outputs["multi_scale"] = torch.cat(scale_aware, dim=1)

        return outputs
