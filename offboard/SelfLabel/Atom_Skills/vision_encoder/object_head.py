from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


OBJECT_CATEGORIES = [
    "bottle",
    "can",
    "plastic_bottle",
    "cup",
    "plate",
    "bag",
    "box",
    "paper",
    "cardboard_box",
    "fruit",
    "vegetable",
    "leftover_food",
    "electronic_device",
    "phone",
    "tool",
    "toy",
    "furniture",
    "clothing",
    "book",
    "battery",
    "jar",
    "tire",
    "unknown",
]


@dataclass
class ObjectTaxonomy:
    categories: list[str] = field(default_factory=lambda: OBJECT_CATEGORIES)
    num_classes: int = len(OBJECT_CATEGORIES)

    def encode(self, object_name: str) -> int:
        return self.categories.index(object_name) if object_name in self.categories else self.categories.index("unknown")

    def decode(self, idx: int) -> str:
        return self.categories[idx] if 0 <= idx < len(self.categories) else "unknown"


class ObjectIdentifier(nn.Module):
    def __init__(self, embed_dim: int, taxonomy: ObjectTaxonomy | None = None):
        super().__init__()
        self.taxonomy = taxonomy or ObjectTaxonomy()

        self.object_head = nn.Sequential(
            nn.Linear(embed_dim, 768),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(384, self.taxonomy.num_classes),
        )

        self.bbox_regressor = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 4),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, global_feat: torch.Tensor, multi_scale: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        feat = global_feat
        if multi_scale is not None:
            ms_pooled = multi_scale.mean(dim=1)
            feat = feat + ms_pooled

        object_logits = self.object_head(feat)
        bbox = torch.sigmoid(self.bbox_regressor(feat))
        confidence = torch.sigmoid(self.confidence_head(feat))

        return {
            "object_logits": object_logits,
            "object_probs": F.softmax(object_logits, dim=-1),
            "bbox": bbox,
            "confidence": confidence,
            "predicted_object": object_logits.argmax(dim=-1),
            "object_embedding": feat,
        }
