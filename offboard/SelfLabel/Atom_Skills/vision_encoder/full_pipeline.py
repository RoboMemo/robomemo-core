from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .multi_scale_encoder import MultiScaleVisionEncoder, MultiScaleEncoderConfig
from .material_head import MaterialClassifier, MaterialTaxonomy, MATERIAL_CATEGORIES
from .object_head import ObjectIdentifier, ObjectTaxonomy, OBJECT_CATEGORIES
from .fusion import JointOutputHead
from .distributed_trainer import DistributedTrainer, DistributedConfig


class VisionEncoderVLM(nn.Module):
    def __init__(
        self,
        encoder_cfg: MultiScaleEncoderConfig | None = None,
        material_taxonomy: MaterialTaxonomy | None = None,
        object_taxonomy: ObjectTaxonomy | None = None,
    ):
        super().__init__()
        self.encoder_cfg = encoder_cfg or MultiScaleEncoderConfig()
        self.material_taxonomy = material_taxonomy or MaterialTaxonomy()
        self.object_taxonomy = object_taxonomy or ObjectTaxonomy()

        self.encoder = MultiScaleVisionEncoder(self.encoder_cfg)
        self.object_head = ObjectIdentifier(self.encoder_cfg.embed_dim, self.object_taxonomy)
        self.material_head = MaterialClassifier(self.encoder_cfg.embed_dim, self.material_taxonomy)
        self.joint_head = JointOutputHead(
            self.encoder_cfg.embed_dim,
            self.object_taxonomy.num_classes,
            self.material_taxonomy.num_classes,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(images)

        obj_out = self.object_head(features["global"], features.get("multi_scale"))
        mat_out = self.material_head(
            features["global"],
            features["patch_features"],
            pixel_feats=images.detach() if images.dim() == 4 else None,
        )
        joint_out = self.joint_head(obj_out["object_embedding"], mat_out["material_embedding"])

        return {
            **features,
            **{f"object_{k}": v for k, v in obj_out.items()},
            **{f"material_{k}": v for k, v in mat_out.items()},
            **{f"joint_{k}": v for k, v in joint_out.items()},
        }

    def predict(self, images: torch.Tensor) -> dict[str, Any]:
        self.eval()
        with torch.no_grad():
            out = self.forward(images)

        object_ids = out.get("object_predicted_object")
        if object_ids is None:
            object_ids = out.get("joint_object_pred")
        if object_ids is None:
            object_ids = torch.zeros(images.size(0), dtype=torch.long, device=images.device)

        material_ids = out.get("material_predicted_material")
        if material_ids is None:
            material_ids = out.get("joint_material_pred")
        if material_ids is None:
            material_ids = torch.zeros(images.size(0), dtype=torch.long, device=images.device)

        object_names = [self.object_taxonomy.decode(int(i)) for i in object_ids]
        material_names = [self.material_taxonomy.decode(int(i)) for i in material_ids]

        conf_vals = out.get("object_confidence")
        if conf_vals is None:
            conf_vals = out.get("joint_classification_logit")
        if conf_vals is None:
            conf_vals = torch.ones(len(object_ids), device=images.device)
        if conf_vals.dim() > 1:
            conf_vals = conf_vals.squeeze(-1)

        results = []
        for obj_name, mat_name, conf in zip(object_names, material_names, conf_vals):
            results.append({
                "object": obj_name,
                "material": mat_name,
                "confidence": float(conf),
                "expected_material": self.material_taxonomy.get_material_for_object(obj_name),
            })

        return {
            "results": results,
            "object_probs": out.get("object_object_probs", out.get("joint_object_logits")),
            "material_probs": out.get("material_material_probs", out.get("joint_material_logits")),
            "timing_ms": 0.0,
        }


def build_model(device: str = "cuda") -> VisionEncoderVLM:
    cfg = MultiScaleEncoderConfig(
        image_size=(224, 224),
        patch_sizes=[8, 16, 32],
        embed_dim=1024,
        num_heads=16,
        num_layers=24,
        use_flash_attn=True,
        output_modalities=["object", "material", "global"],
    )
    model = VisionEncoderVLM(encoder_cfg=cfg)
    return model.to(device)


def run_demo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device)

    print(f"VisionEncoderVLM on {device}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable:,} trainable")

    dummy = torch.randn(4, 3, 224, 224, device=device)
    out = model.predict(dummy)

    print(f"  Output: {json.dumps(out['results'], indent=2)}")

    t0 = time.perf_counter()
    for _ in range(100):
        _ = model.predict(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_ms = (time.perf_counter() - t0) / 100 * 1000
    print(f"  Inference: {t_ms:.2f} ms per batch of 4")


def main():
    parser = argparse.ArgumentParser(description="Vision Encoder VLM for Object → Material Classification")
    parser.add_argument("--demo", action="store_true", help="Run demo inference")
    parser.add_argument("--train", action="store_true", help="Run training loop (mock data)")
    parser.add_argument("--export", type=str, default="", help="Export weights to path")
    parser.add_argument("--checkpoint", type=str, default="", help="Load checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    device = args.device if torch.cuda.is_available() else "cpu"
    model = build_model(device)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False))
        print(f"Loaded checkpoint: {args.checkpoint}")

    dist_cfg = DistributedConfig(checkpoint_dir="checkpoints/vision_encoder")
    trainer = DistributedTrainer(
        model,
        dist_cfg,
        material_taxonomy=MaterialTaxonomy(),
        object_taxonomy=ObjectTaxonomy(),
    )

    if args.train:
        print("Training on mock data (replace with real dataset)...")
        for step in range(100):
            images = torch.randn(8, 3, 224, 224, device=device)
            obj_labels = torch.randint(0, len(OBJECT_CATEGORIES), (8,), device=device)
            mat_labels = torch.randint(0, len(MATERIAL_CATEGORIES), (8,), device=device)
            metrics = trainer.train_step(images, obj_labels, mat_labels)
            if (step + 1) % 20 == 0:
                print(f"  step {step+1}: loss={metrics['loss']:.4f}")
        trainer.save_checkpoint("final")

    if args.export:
        trainer.export_weights(args.export)
        print(f"Exported to {args.export}")


if __name__ == "__main__":
    main()
