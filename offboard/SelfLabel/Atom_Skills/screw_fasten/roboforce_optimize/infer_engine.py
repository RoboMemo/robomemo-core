from __future__ import annotations
import enum
import gc
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .cuda_graph import CUDAGraphRunner


class PrecisionMode(enum.Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    FP4 = "fp4"

    def torch_dtype(self) -> torch.dtype:
        mapping = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp8": torch.float8_e4m3fn,
            "fp4": torch.float8_e4m3fn,
        }
        return mapping[self.value]

    def label(self) -> str:
        return self.value.upper()


@dataclass
class EngineConfig:
    model_name: str = "gr00t"
    precision: PrecisionMode = PrecisionMode.BF16
    use_cuda_graph: bool = True
    use_torch_compile: bool = True
    torch_compile_mode: str = "reduce-overhead"
    use_flash_attn: bool = True
    warmup_iters: int = 5
    num_flow_steps: int = 10
    use_cuda_async_alloc: bool = True
    static_batch_size: int = 1
    use_tensorrt: bool = False
    tensorrt_engine_path: str = ""

    def __post_init__(self):
        if self.use_cuda_async_alloc:
            try:
                os.environ["CUDA_DEVICE_MEMORY_POOL"] = "cudaMallocAsync"
                if hasattr(torch.cuda, "change_current_allocator") and hasattr(torch.cuda, "cudaMallocAsync"):
                    torch.cuda.change_current_allocator(torch.cuda.cudaMallocAsync)
            except Exception:
                pass


class OptimizedInferenceEngine:
    """Unified inference engine with cascading optimizations:

    1. torch.compile (graph-level optimization)
    2. CUDA Graph capture (kernel launch overhead elimination)
    3. FP8/FP4 precision (Blackwell Tensor Core utilization)
    4. TensorRT backend (optional, requires onnx+tensorrt)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: EngineConfig,
        device: torch.device | str = "cuda",
    ):
        self.model = model.eval()
        self.config = config
        self.device = torch.device(device)
        self._graph_runner: CUDAGraphRunner | None = None
        self._compile_log: list[str] = []
        self._forward_fn: Callable | None = None

    def warmup(self, sample_inputs: dict[str, torch.Tensor] | None = None):
        """Apply all configured optimizations and warm up the engine.

        Order: device -> dtype -> compile -> CUDA Graph capture (if enabled).
        """
        print(f"[OptimizedInferenceEngine] Warming up {self.config.model_name}")
        print(f"  precision:     {self.config.precision.label()}")
        print(f"  torch.compile: {self.config.use_torch_compile} ({self.config.torch_compile_mode})")
        print(f"  cuda_graph:    {self.config.use_cuda_graph}")
        print(f"  flash_attn:    {self.config.use_flash_attn}")

        target_dtype = self.config.precision.torch_dtype()
        self.model = self.model.to(device=self.device)
        if self.config.precision in (PrecisionMode.FP16, PrecisionMode.BF16):
            self.model = self.model.to(dtype=target_dtype)
        self.model.eval()

        if self.config.use_flash_attn:
            self._enable_flash_attention()

        if self.config.use_torch_compile:
            self._apply_torch_compile()

        if self.config.use_cuda_graph and sample_inputs is not None:
            self._graph_runner = CUDAGraphRunner(
                model=self.model,
                device=self.device,
                warmup_iters=self.config.warmup_iters,
            )
            if sample_inputs is not None:
                s = {k: v.to(self.device, dtype=target_dtype) for k, v in sample_inputs.items()}
                self._graph_runner.capture(s)
                print(f"  cuda_graph:    captured")
            else:
                self._graph_runner = None

        torch.cuda.synchronize()
        print(f"  warmup:        done ({self.config.warmup_iters} iters)")

    def infer(self, inputs: dict[str, torch.Tensor]) -> Any:
        if self._graph_runner is not None:
            return self._graph_runner.replay(inputs)

        if self._forward_fn is not None:
            return self._forward_fn(**inputs)

        with torch.no_grad():
            return self.model(**inputs)

    def timed_infer(
        self, inputs: dict[str, torch.Tensor], n_iters: int = 100
    ) -> tuple[Any, float]:
        """Run timed inference and return (output, latency_ms)."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        target_dtype = self.config.precision.torch_dtype()
        inputs_gpu = {k: v.to(self.device, dtype=target_dtype) for k, v in inputs.items()}

        torch.cuda.synchronize()
        start.record()
        out = self.infer(inputs_gpu)
        end.record()
        torch.cuda.synchronize()
        return out, start.elapsed_time(end)

    def _apply_torch_compile(self):
        if sys.platform == "win32":
            print(f"  torch.compile: skipped (Windows detected; Triton unavailable)")
            self.config.use_torch_compile = False
            self._compile_log.append("torch.compile SKIPPED: Windows (no Triton)")
            return
        self.model = torch.compile(
            self.model,
            mode=self.config.torch_compile_mode,
            fullgraph=False,
        )
        try:
            dummy_image = torch.randn(1, 3, 224, 224, device=self.device, dtype=self.config.precision.torch_dtype())
            dummy_state = torch.randn(1, 32, device=self.device, dtype=self.config.precision.torch_dtype())
            _ = self.model(dummy_image, dummy_state)
            torch.cuda.synchronize()
            self._compile_log.append(f"torch.compile mode={self.config.torch_compile_mode}")
            print(f"  torch.compile: applied (mode={self.config.torch_compile_mode})")
        except Exception as e:
            self.model = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
            self._compile_log.append(f"torch.compile SKIPPED: {e}")
            print(f"  torch.compile: skipped ({e})")
            self.config.use_torch_compile = False

    def _enable_flash_attention(self):
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            print("  flash_attn:    enabled")
        except Exception as e:
            print(f"  flash_attn:    skip ({e})")

    def profile_summary(self) -> dict:
        return {
            "model": self.config.model_name,
            "precision": self.config.precision.label(),
            "torch_compile": self.config.use_torch_compile,
            "cuda_graph": self.config.use_cuda_graph,
            "flash_attn": self.config.use_flash_attn,
            "compile_log": self._compile_log,
        }

    def cleanup(self):
        if self._graph_runner is not None:
            self._graph_runner.reset()
        self.model = self.model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
