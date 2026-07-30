from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


MATERIAL_CATEGORIES = [
    "metal",
    "plastic",
    "glass",
    "wood",
    "fabric",
    "paper_cardboard",
    "ceramic",
    "rubber",
    "leather",
    "organic_food",
    "electronic",
    "stone_concrete",
    "unknown",
]

OBJECT_MATERIAL_MAP: dict[str, str] = {
    "can": "metal",
    "aluminum_foil": "metal",
    "metal_tool": "metal",
    "coin": "metal",
    "bottle": "glass",
    "jar": "glass",
    "window": "glass",
    "plastic_bottle": "plastic",
    "plastic_bag": "plastic",
    "container": "plastic",
    "toy": "plastic",
    "paper": "paper_cardboard",
    "cardboard_box": "paper_cardboard",
    "book": "paper_cardboard",
    "newspaper": "paper_cardboard",
    "wooden_furniture": "wood",
    "branch": "organic_food",
    "apple": "organic_food",
    "banana": "organic_food",
    "leftover": "organic_food",
    "t_shirt": "fabric",
    "towel": "fabric",
    "curtain": "fabric",
    "phone": "electronic",
    "laptop": "electronic",
    "battery": "electronic",
    "ceramic_plate": "ceramic",
    "cup": "ceramic",
    "tire": "rubber",
    "shoe": "leather",
    "brick": "stone_concrete",
}


@dataclass
class MaterialTaxonomy:
    categories: list[str] = field(default_factory=lambda: MATERIAL_CATEGORIES)
    num_classes: int = len(MATERIAL_CATEGORIES)

    def get_material_for_object(self, object_name: str) -> str:
        return OBJECT_MATERIAL_MAP.get(object_name.lower(), "unknown")

    def encode(self, material_name: str) -> int:
        return self.categories.index(material_name) if material_name in self.categories else self.categories.index("unknown")

    def decode(self, idx: int) -> str:
        return self.categories[idx] if 0 <= idx < len(self.categories) else "unknown"


class TextureFeatureExtractor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.input_proj = nn.Conv2d(in_dim, 32, kernel_size=1)
        self.lbp_conv = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gabor_conv = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.freq_proj = nn.Linear(128, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        lbp_feat = F.relu(self.lbp_conv(x)).view(x.size(0), 64, -1).mean(dim=-1)
        gabor_feat = F.relu(self.gabor_conv(x)).view(x.size(0), 64, -1).mean(dim=-1)
        texture = torch.cat([lbp_feat, gabor_feat], dim=-1)
        return self.freq_proj(texture)


class PixelTextureEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).flatten(1)
        return self.proj(x)


class MaterialClassifier(nn.Module):
    def __init__(self, embed_dim: int, taxonomy: MaterialTaxonomy | None = None):
        super().__init__()
        self.taxonomy = taxonomy or MaterialTaxonomy()
        self.texture_extractor = TextureFeatureExtractor(embed_dim)
        self.pixel_encoder = PixelTextureEncoder(embed_dim)

        self.material_head = nn.Sequential(
            nn.Linear(embed_dim + embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, self.taxonomy.num_classes),
        )

        self.reflectance_head = nn.Sequential(
            nn.Linear(embed_dim + embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

        self.embedding_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, global_feat: torch.Tensor, patch_feats: torch.Tensor, pixel_feats: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if pixel_feats is not None and pixel_feats.size(1) == 3:
            texture_feat = self.pixel_encoder(pixel_feats)
        elif pixel_feats is not None:
            texture_feat = self.texture_extractor(pixel_feats)
        else:
            texture_feat = torch.zeros_like(global_feat)
        combined = torch.cat([global_feat, texture_feat], dim=-1)

        material_logits = self.material_head(combined)
        reflectance = torch.sigmoid(self.reflectance_head(combined))

        return {
            "material_logits": material_logits,
            "material_probs": F.softmax(material_logits, dim=-1),
            "reflectance": reflectance,
            "predicted_material": material_logits.argmax(dim=-1),
            "material_embedding": self.embedding_proj(combined),
        }
