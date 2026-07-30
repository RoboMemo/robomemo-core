from __future__ import annotations

import enum
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from ._cuda_graph import CUDAGraphRunner
from .full_pipeline import VisionEncoderVLM, build_model as _build_model


class PrecisionMode(enum.Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

    def torch_dtype(self) -> torch.dtype:
        return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[self.value]

    def label(self) -> str:
        return self.value.upper()


@dataclass
class VisionEncoderOptimizedConfig:
    precision: str = "bf16"
    use_cuda_graph: bool = True
    use_torch_compile: bool = True
    use_flash_attn: bool = True
    warmup_iters: int = 10
    batch_size: int = 1
    image_size: tuple[int, int] = (224, 224)


class VisionEncoderFlatWrapper(nn.Module):
    def __init__(self, model: VisionEncoderVLM):
        super().__init__()
        self.model = model
        self.num_objects = model.object_taxonomy.num_classes
        self.num_materials = model.material_taxonomy.num_classes

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        out = self.model(images)
        obj = out["joint_object_logits"]
        mat = out["joint_material_logits"]
        return torch.cat([obj, mat], dim=-1)

    def decode(self, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        n = self.num_objects
        obj_logits = flat[:, :n]
        mat_logits = flat[:, n : n + self.num_materials]
        return {
            "object_logits": obj_logits,
            "material_logits": mat_logits,
            "object_pred": obj_logits.argmax(dim=-1),
            "material_pred": mat_logits.argmax(dim=-1),
            "object_probs": torch.softmax(obj_logits, dim=-1),
            "material_probs": torch.softmax(mat_logits, dim=-1),
        }


class _SimpleInferenceEngine:
    def __init__(self, model: nn.Module, config: VisionEncoderOptimizedConfig, device: str = "cuda"):
        self.model = model.eval()
        self.config = config
        self.device = torch.device(device)
        self._graph_runner: CUDAGraphRunner | None = None
        self._compile_log: list[str] = []

    def warmup(self, sample_inputs: dict[str, torch.Tensor]):
        target_dtype = PrecisionMode(self.config.precision).torch_dtype()
        self.model = self.model.to(device=self.device, dtype=target_dtype)
        self.model.eval()

        sample_inputs = {k: v.to(device=self.device, dtype=target_dtype) for k, v in sample_inputs.items()}

        if self.config.use_flash_attn:
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            except Exception:
                pass

        if self.config.use_torch_compile and sys.platform != "win32":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
                self._compile_log.append("torch.compile applied")
            except Exception as e:
                self._compile_log.append(f"torch.compile skipped: {e}")

        if self.config.use_cuda_graph and sys.platform != "win32":
            self._graph_runner = CUDAGraphRunner(
                model=self.model,
                device=self.device,
                warmup_iters=self.config.warmup_iters,
            )
            self._graph_runner.capture(sample_inputs)

        torch.cuda.synchronize()

    def infer(self, inputs: dict[str, torch.Tensor]) -> Any:
        target_dtype = PrecisionMode(self.config.precision).torch_dtype()
        inputs = {k: v.to(dtype=target_dtype) if v.dtype != target_dtype else v for k, v in inputs.items()}
        if self._graph_runner is not None:
            return self._graph_runner.replay(inputs)
        with torch.no_grad():
            return self.model(**inputs)

    def timed_infer(self, inputs: dict[str, torch.Tensor]) -> tuple[Any, float]:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        out = self.infer(inputs)
        end.record()
        torch.cuda.synchronize()
        return out, start.elapsed_time(end)

    def cleanup(self):
        if self._graph_runner is not None:
            self._graph_runner.reset()
        self.model = self.model.cpu()
        gc.collect()
        torch.cuda.empty_cache()


class OptimizedVisionEngine:
    def __init__(self, cfg: VisionEncoderOptimizedConfig | None = None):
        self.cfg = cfg or VisionEncoderOptimizedConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: VisionEncoderFlatWrapper | None = None
        self._engine: _SimpleInferenceEngine | None = None

    def build(self):
        model = _build_model(self.device)
        self.model = VisionEncoderFlatWrapper(model).to(self.device).eval()
        return self

    def warmup(self):
        if self.model is None:
            self.build()
        self._engine = _SimpleInferenceEngine(self.model, self.cfg, self.device)
        dtype = PrecisionMode(self.cfg.precision).torch_dtype()
        sample = {"images": torch.randn(self.cfg.batch_size, 3, *self.cfg.image_size, device=self.device, dtype=dtype)}
        self._engine.warmup(sample)
        return self

    @torch.no_grad()
    def infer(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if self._engine is not None:
            out = self._engine.infer({"images": images})
        else:
            out = self.model(images)
        return self.model.decode(out) if isinstance(out, torch.Tensor) else out

    def benchmark(self, batch_size: int | None = None, n_iters: int = 500) -> dict[str, float]:
        bs = batch_size if batch_size is not None else self.cfg.batch_size
        dtype = PrecisionMode(self.cfg.precision).torch_dtype()
        images = torch.randn(bs, 3, *self.cfg.image_size, device=self.device, dtype=dtype)
        latencies = []
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = self.infer(images)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
        import numpy as np
        arr = np.array(latencies)
        return {
            "mean_ms": float(np.mean(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "fps": 1000.0 / float(np.mean(arr)) * bs,
            "batch_size": bs,
        }


def run_benchmark():
    print("=== Optimized Vision Engine ===")
    base_bs = 1
    engine = OptimizedVisionEngine(
        VisionEncoderOptimizedConfig(
            precision="bf16",
            use_cuda_graph=True,
            use_torch_compile=sys.platform != "win32",
            use_flash_attn=True,
            batch_size=base_bs,
        )
    )
    engine.build()
    print("Warming up (bs={})...".format(base_bs))
    engine.warmup()
    r = engine.benchmark(n_iters=400)
    print("  bs={}: {:.2f} ms, {:.0f} FPS".format(r["batch_size"], r["mean_ms"], r["fps"]))

    print("\n  (static cuda_graph — re-capturing for bs=4)...")
    engine2 = OptimizedVisionEngine(
        VisionEncoderOptimizedConfig(
            precision="bf16",
            use_cuda_graph=True,
            use_torch_compile=False,
            use_flash_attn=True,
            batch_size=4,
        )
    )
    engine2.build()
    engine2.warmup()
    r4 = engine2.benchmark(n_iters=200)
    print("  bs={}: {:.2f} ms, {:.0f} FPS".format(r4["batch_size"], r4["mean_ms"], r4["fps"]))

    if torch.cuda.is_available():
        print("  VRAM: {:.0f} MB".format(torch.cuda.memory_allocated() / 1024 / 1024))


if __name__ == "__main__":
    run_benchmark()
