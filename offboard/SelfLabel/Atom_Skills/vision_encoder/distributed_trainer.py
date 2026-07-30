from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class DistributedConfig:
    num_machines: int = 1
    machine_rank: int = 0
    sync_interval_steps: int = 100
    checkpoint_dir: str = "checkpoints/vision_encoder"
    sync_method: str = "manual"
    master_addr: str = ""
    master_port: int = 29500
    backend: str = "gloo"
    use_fsdp: bool = False


class DistributedTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        material_taxonomy: Any = None,
        object_taxonomy: Any = None,
    ):
        self.model = model
        self.config = config
        self.material_taxonomy = material_taxonomy
        self.object_taxonomy = object_taxonomy
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.step = 0
        self._checkpoint_dir = Path(config.checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_step(self, images: torch.Tensor, object_labels: torch.Tensor, material_labels: torch.Tensor) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        vision_out = self.model(images)

        obj_logits = vision_out.get("joint_object_logits")
        if obj_logits is None:
            obj_logits = vision_out.get("object_object_logits")

        mat_logits = vision_out.get("joint_material_logits")
        if mat_logits is None:
            mat_logits = vision_out.get("material_material_logits")

        obj_loss = F.cross_entropy(obj_logits, object_labels) if obj_logits is not None else torch.tensor(0.0)
        mat_loss = F.cross_entropy(mat_logits, material_labels) if mat_logits is not None else torch.tensor(0.0)

        total_loss = obj_loss + mat_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.step += 1

        return {
            "loss": total_loss.item(),
            "obj_loss": obj_loss.item() if isinstance(obj_loss, torch.Tensor) else 0.0,
            "mat_loss": mat_loss.item() if isinstance(mat_loss, torch.Tensor) else 0.0,
        }

    def save_checkpoint(self, tag: str = "latest", metrics: dict | None = None) -> str:
        path = self._checkpoint_dir / f"vision_encoder_{tag}_step{self.step}.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step": self.step,
                "config": self.config,
                "metrics": metrics or {},
                "timestamp": datetime.now().isoformat(),
            },
            path,
        )
        return str(path)

    def load_checkpoint(self, path: str) -> int:
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(data["model_state_dict"])
        self.optimizer.load_state_dict(data["optimizer_state_dict"])
        self.step = data["step"]
        print(f"Loaded checkpoint from {path} (step {self.step})")
        return self.step

    def export_weights(self, output_path: str):
        torch.save(self.model.state_dict(), output_path)
        print(f"Exported weights to {output_path}")

    def sync_with_peer(self, peer_checkpoint_path: str) -> int:
        if not os.path.exists(peer_checkpoint_path):
            print(f"Peer checkpoint {peer_checkpoint_path} not found, skipping sync")
            return self.step

        peer_ckpt = torch.load(peer_checkpoint_path, map_location="cpu", weights_only=False)
        peer_step = peer_ckpt["step"]

        if peer_step > self.step:
            self.model.load_state_dict(peer_ckpt["model_state_dict"])
            self.optimizer.load_state_dict(peer_ckpt["optimizer_state_dict"])
            self.step = peer_step
            print(f"Synced from peer (step {peer_step})")
        else:
            ours_path = self.save_checkpoint("for_peer")
            print(f"Our checkpoint is newer, exported for peer at {ours_path}")

        return self.step
