from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, object_feat: torch.Tensor, material_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        obj = object_feat.unsqueeze(1)
        mat = material_feat.unsqueeze(1)

        fused, attn_weights = self.cross_attn(obj, mat, mat)
        fused = self.norm(fused + obj)
        fused = fused.squeeze(1)

        gate_val = self.gate(torch.cat([object_feat, material_feat], dim=-1))
        weighted = gate_val * object_feat + (1 - gate_val) * material_feat

        logit = self.classifier(weighted)

        return {
            "fused_embedding": fused,
            "weighted_embedding": weighted,
            "classification_logit": logit,
            "gate_value": gate_val,
            "attention_weights": attn_weights,
        }


class JointOutputHead(nn.Module):
    def __init__(self, embed_dim: int, num_objects: int, num_materials: int):
        super().__init__()
        self.fusion = CrossModalFusion(embed_dim)

        self.object_out = nn.Linear(embed_dim, num_objects)
        self.material_out = nn.Linear(embed_dim, num_materials)
        self.joint_out = nn.Linear(embed_dim, num_objects + num_materials)

    def forward(self, object_feat: torch.Tensor, material_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        fused_dict = self.fusion(object_feat, material_feat)
        fused = fused_dict["fused_embedding"]

        obj_logits = self.object_out(object_feat + fused * 0.3)
        mat_logits = self.material_out(material_feat + fused * 0.3)
        joint_logits = self.joint_out(fused)

        return {
            "object_logits": obj_logits,
            "material_logits": mat_logits,
            "joint_logits": joint_logits,
            "object_pred": obj_logits.argmax(dim=-1),
            "material_pred": mat_logits.argmax(dim=-1),
            **fused_dict,
        }
