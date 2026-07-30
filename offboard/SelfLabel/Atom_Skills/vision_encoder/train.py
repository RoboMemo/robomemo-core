from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from .full_pipeline import build_model, VisionEncoderVLM
from .distributed_trainer import DistributedTrainer, DistributedConfig
from .material_head import MaterialTaxonomy, MATERIAL_CATEGORIES
from .object_head import ObjectTaxonomy, OBJECT_CATEGORIES


@dataclass
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 50
    steps_per_epoch: int = 200
    warmup_steps: int = 50
    image_size: tuple[int, int] = (224, 224)
    checkpoint_dir: str = "checkpoints/vision_encoder"
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    sync_interval: int = 100
    use_optimized_eval: bool = True


class VisionLabelDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        label_file: str | Path | None = None,
        object_taxonomy: ObjectTaxonomy | None = None,
        material_taxonomy: MaterialTaxonomy | None = None,
        image_size: tuple[int, int] = (224, 224),
        augment: bool = True,
        synthetic: bool = False,
        num_synthetic: int = 10000,
    ):
        self.image_dir = Path(image_dir)
        self.object_taxonomy = object_taxonomy or ObjectTaxonomy()
        self.material_taxonomy = material_taxonomy or MaterialTaxonomy()
        self.synthetic = synthetic

        if synthetic:
            self.samples = self._make_synthetic(num_synthetic)
        elif label_file and Path(label_file).exists():
            self.samples = self._load_labels(label_file)
        else:
            self.samples = self._scan_images()

        t_list = [transforms.Resize(image_size), transforms.ToTensor()]
        if augment:
            t_list = [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(0.1, 0.1, 0.05, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        else:
            t_list = [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        self.transform = transforms.Compose(t_list)

    def _make_synthetic(self, n: int) -> list[dict]:
        samples = []
        for i in range(n):
            obj_idx = random.randint(0, self.object_taxonomy.num_classes - 1)
            obj_name = self.object_taxonomy.decode(obj_idx)
            mat_name = self.material_taxonomy.get_material_for_object(obj_name)
            mat_idx = self.material_taxonomy.encode(mat_name)
            samples.append({
                "img_path": None,
                "object_id": obj_idx,
                "material_id": mat_idx,
            })
        return samples

    def _load_labels(self, label_file: Path) -> list[dict]:
        samples = []
        data = json.loads(Path(label_file).read_text())
        for entry in data:
            img_path = self.image_dir / entry["image"]
            obj_name = entry.get("object", "unknown")
            mat_name = entry.get("material", "unknown")
            if mat_name == "infer":
                mat_name = self.material_taxonomy.get_material_for_object(obj_name)
            obj_id = self.object_taxonomy.encode(obj_name)
            mat_id = self.material_taxonomy.encode(mat_name)
            samples.append({
                "img_path": img_path,
                "object_id": obj_id,
                "material_id": mat_id,
            })
        return samples

    def _scan_images(self) -> list[dict]:
        samples = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            for f in sorted(self.image_dir.glob(ext)):
                obj_name = self.image_dir.name
                obj_id = self.object_taxonomy.encode(obj_name)
                mat_name = self.material_taxonomy.get_material_for_object(obj_name)
                mat_id = self.material_taxonomy.encode(mat_name)
                samples.append({
                    "img_path": f,
                    "object_id": obj_id,
                    "material_id": mat_id,
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        if self.synthetic or s["img_path"] is None:
            img = torch.randn(3, *self.transform.transforms[-1].size if hasattr(self.transform.transforms[-1], "size") else (224, 224))
        else:
            img = Image.open(s["img_path"]).convert("RGB")
            img = self.transform(img)
        return {
            "images": img,
            "object_labels": torch.tensor(s["object_id"], dtype=torch.long),
            "material_labels": torch.tensor(s["material_id"], dtype=torch.long),
        }


class LabeledImageWriter:
    @staticmethod
    def create_label_template(output_path: str = "labels_template.json"):
        template = []
        for obj in OBJECT_CATEGORIES:
            mat = MaterialTaxonomy().get_material_for_object(obj)
            template.append({"image": "path/to/image.jpg", "object": obj, "material": mat if mat != "unknown" else "infer"})
        Path(output_path).write_text(json.dumps(template, indent=2))
        print(f"Template written to {output_path}")
        return output_path


class Trainer:
    def __init__(
        self,
        model: VisionEncoderVLM,
        config: TrainingConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_epochs * config.steps_per_epoch,
        )
        self.step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []

        self.dist_trainer = DistributedTrainer(
            model,
            DistributedConfig(
                checkpoint_dir=str(self.checkpoint_dir),
                sync_interval_steps=config.sync_interval,
            ),
            material_taxonomy=MaterialTaxonomy(),
            object_taxonomy=ObjectTaxonomy(),
        )

        self.optimized_engine = None
        if config.use_optimized_eval and torch.cuda.is_available():
            try:
                from .optimized_runner import OptimizedVisionEngine, VisionEncoderOptimizedConfig
                eng = OptimizedVisionEngine(
                    VisionEncoderOptimizedConfig(
                        precision="bf16",
                        use_cuda_graph=True,
                        use_torch_compile=False,
                        use_flash_attn=True,
                        batch_size=config.batch_size,
                    )
                )
                eng.build()
                model_state = {k: v for k, v in model.state_dict().items()}
                eng.model.model.load_state_dict(model_state)
                eng.warmup()
                self.optimized_engine = eng
                print(f"  Optimized eval engine ready")
            except Exception as e:
                print(f"  Optimized eval engine skipped: {e}")
                self.optimized_engine = None

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        images = batch["images"].to(self.device)
        obj_labels = batch["object_labels"].to(self.device)
        mat_labels = batch["material_labels"].to(self.device)

        out = self.model(images)

        obj_logits = out.get("joint_object_logits")
        if obj_logits is None:
            obj_logits = out.get("object_object_logits")
        mat_logits = out.get("joint_material_logits")
        if mat_logits is None:
            mat_logits = out.get("material_material_logits")

        obj_loss = F.cross_entropy(obj_logits, obj_labels)
        mat_loss = F.cross_entropy(mat_logits, mat_labels)

        consistency = out.get("joint_classification_logit", torch.zeros(images.size(0), 1, device=self.device)).squeeze(-1)
        joint_labels = (obj_labels == self._map_obj_to_mat_idx(mat_labels)).float()
        cons_loss = F.binary_cross_entropy_with_logits(consistency, joint_labels)

        total_loss = obj_loss + mat_loss + 0.1 * cons_loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        obj_acc = (obj_logits.argmax(dim=-1) == obj_labels).float().mean().item()
        mat_acc = (mat_logits.argmax(dim=-1) == mat_labels).float().mean().item()

        self.step += 1
        return {
            "loss": total_loss.item(),
            "obj_loss": obj_loss.item(),
            "mat_loss": mat_loss.item(),
            "cons_loss": cons_loss.item(),
            "obj_acc": obj_acc,
            "mat_acc": mat_acc,
            "lr": self.scheduler.get_last_lr()[0],
        }

    def _map_obj_to_mat_idx(self, mat_labels: torch.Tensor) -> torch.Tensor:
        return mat_labels

    @torch.no_grad()
    def sync_eval_weights(self):
        if self.optimized_engine is not None:
            sd = {k: v for k, v in self.model.state_dict().items()}
            self.optimized_engine.model.model.load_state_dict(sd)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_obj_acc = 0.0
        total_mat_acc = 0.0
        total_loss = 0.0
        count = 0

        for batch in loader:
            images = batch["images"].to(self.device)
            obj_labels = batch["object_labels"].to(self.device)
            mat_labels = batch["material_labels"].to(self.device)

            if self.optimized_engine is not None:
                out = self.optimized_engine.infer(images)
                obj_logits = out["object_logits"]
                mat_logits = out["material_logits"]
            else:
                out = self.model(images)
                obj_logits = out.get("joint_object_logits")
                if obj_logits is None:
                    obj_logits = out.get("object_object_logits")
                mat_logits = out.get("joint_material_logits")
                if mat_logits is None:
                    mat_logits = out.get("material_material_logits")

            obj_loss = F.cross_entropy(obj_logits, obj_labels)
            mat_loss = F.cross_entropy(mat_logits, mat_labels)

            total_obj_acc += (obj_logits.argmax(dim=-1) == obj_labels).float().sum().item()
            total_mat_acc += (mat_logits.argmax(dim=-1) == mat_labels).float().sum().item()
            total_loss += (obj_loss + mat_loss).item() * images.size(0)
            count += images.size(0)

        return {
            "eval_loss": total_loss / count,
            "eval_obj_acc": total_obj_acc / count,
            "eval_mat_acc": total_mat_acc / count,
        }

    def save(self, tag: str = "latest", metrics: dict | None = None):
        path = self.checkpoint_dir / f"vision_encoder_{tag}_step{self.step}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "config": self.config,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat(),
        }, path)
        return str(path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(data["model_state_dict"])
        self.optimizer.load_state_dict(data["optimizer_state_dict"])
        self.scheduler.load_state_dict(data["scheduler_state_dict"])
        self.step = data["step"]
        self.epoch = data.get("epoch", 0)
        self.best_loss = data.get("best_loss", float("inf"))
        print(f"Resumed from {path} (epoch {self.epoch}, step {self.step})")
        return self

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None):
        steps_per_epoch = len(train_loader)
        print(f"Training: {steps_per_epoch} steps/epoch, {self.config.max_epochs} epochs")

        for epoch in range(self.epoch, self.config.max_epochs):
            self.epoch = epoch
            for batch in train_loader:
                metrics = self.train_step(batch)

                if self.step % self.config.log_interval == 0:
                    lr = metrics["lr"]
                    print(f"  E{epoch:03d} S{self.step:06d} | "
                          f"loss={metrics['loss']:.4f} obj={metrics['obj_acc']:.3f} mat={metrics['mat_acc']:.3f} "
                          f"lr={lr:.2e}")
                    self.history.append({"step": self.step, **metrics})

                if self.step % self.config.save_interval == 0:
                    path = self.save(f"step{self.step}")
                    print(f"  Saved: {path}")

                if val_loader and self.step % self.config.eval_interval == 0:
                    self.sync_eval_weights()
                    eval_metrics = self.evaluate(val_loader)
                    print(f"  Eval: loss={eval_metrics['eval_loss']:.4f} "
                          f"obj_acc={eval_metrics['eval_obj_acc']:.3f} "
                          f"mat_acc={eval_metrics['eval_mat_acc']:.3f}")
                    if eval_metrics["eval_loss"] < self.best_loss:
                        self.best_loss = eval_metrics["eval_loss"]
                        best_path = self.save("best", eval_metrics)
                        print(f"  New best: {best_path}")

            path = self.save(f"epoch{epoch}")
            if self.step % self.config.sync_interval == 0 and epoch % 5 == 0:
                self.dist_trainer.save_checkpoint(f"latest_epoch{epoch}")
                print(f"  Checkpoint synced for distributed training")
            print(f"  Epoch {epoch} done: {path}")

        self.save("final")
        print("Training complete")


def make_dataloaders(config: TrainingConfig) -> tuple[DataLoader, DataLoader | None]:
    train_ds = VisionLabelDataset(
        image_dir=Path("data/train"),
        label_file=Path("data/labels.json") if Path("data/labels.json").exists() else None,
        image_size=config.image_size,
        augment=True,
        synthetic=not Path("data").exists(),
        num_synthetic=config.steps_per_epoch * 2,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = None
    if Path("data/val").exists():
        val_ds = VisionLabelDataset(
            image_dir=Path("data/val"),
            label_file=Path("data/val_labels.json") if Path("data/val_labels.json").exists() else None,
            image_size=config.image_size,
            augment=False,
        )
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    else:
        val_ds = VisionLabelDataset(
            image_dir=Path("."),
            image_size=config.image_size,
            augment=False,
            synthetic=True,
            num_synthetic=config.steps_per_epoch // 4,
        )
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train Vision Encoder VLM")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--no-optimized-eval", action="store_true", help="Disable optimized eval engine")
    parser.add_argument("--template", action="store_true", help="Generate label template")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark optimized engine")
    args = parser.parse_args()

    if args.template:
        LabeledImageWriter.create_label_template()
        return

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = build_model(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total:,} params ({trainable:,} trainable)")

    config = TrainingConfig(
        batch_size=args.batch,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        use_optimized_eval=not args.no_optimized_eval,
    )

    trainer = Trainer(model, config, device)
    if args.resume:
        trainer.load(args.resume)

    if args.benchmark:
        from .optimized_runner import run_benchmark
        run_benchmark()
        return

    train_loader, val_loader = make_dataloaders(config)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
