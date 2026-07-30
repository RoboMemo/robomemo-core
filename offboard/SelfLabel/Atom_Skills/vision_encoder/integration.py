from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from .full_pipeline import VisionEncoderVLM, build_model
from .multi_scale_encoder import MultiScaleEncoderConfig


class OptimizedVisionAdapter:
    def __init__(self, model: VisionEncoderVLM, device: str = "cuda"):
        self.model = model.to(device).eval()
        self.device = device

    def warmup(self, batch_size: int = 1, n_iters: int = 5):
        dummy = torch.randn(batch_size, 3, 224, 224, device=self.device)
        print(f"[OptimizedVisionAdapter] Warming up (bs={batch_size})")
        for _ in range(n_iters):
            _ = self.model(dummy)
        if self.device == "cuda":
            torch.cuda.synchronize()
        print(f"  warmup done")

    @torch.no_grad()
    def infer(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(images)

    @torch.no_grad()
    def classify(self, images: torch.Tensor) -> list[dict[str, Any]]:
        out = self.model.predict(images)
        return out["results"]

    def timed_infer(self, images: torch.Tensor, n_iters: int = 100) -> tuple[dict[str, torch.Tensor], float]:
        if self.device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            out = self.infer(images)
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end)
        else:
            import time
            t0 = time.perf_counter()
            out = self.infer(images)
            ms = (time.perf_counter() - t0) * 1000
        return out, ms


def create_optimized_model(
    image_size: tuple[int, int] = (224, 224),
    embed_dim: int = 1024,
    device: str = "cuda",
    use_torch_compile: bool = True,
) -> tuple[VisionEncoderVLM, OptimizedVisionAdapter]:
    cfg = MultiScaleEncoderConfig(
        image_size=image_size,
        patch_sizes=[8, 16, 32],
        embed_dim=embed_dim,
        num_heads=16,
        num_layers=24,
        use_flash_attn=True,
        use_checkpointing=True,
    )
    model = VisionEncoderVLM(encoder_cfg=cfg)

    if use_torch_compile and sys.platform != "win32":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[create_optimized_model] torch.compile enabled")
        except Exception as e:
            print(f"[create_optimized_model] torch.compile skipped ({e})")

    model = model.to(device)
    adapter = OptimizedVisionAdapter(model, device)
    return model, adapter
