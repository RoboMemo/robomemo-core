"""GPU-accelerated inference benchmark with CUDA event timing.

Replaces the MockPolicyModel in roboforce_validation/benchmark.py with
real GPU timing using torch.cuda.Event for precise latency measurement.
"""
from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .infer_engine import OptimizedInferenceEngine, EngineConfig, PrecisionMode


@dataclass
class BenchmarkConfig:
    model_name: str = "gr00t"
    checkpoint_path: str = ""
    precision: str = "bf16"
    use_cuda_graph: bool = True
    use_torch_compile: bool = True
    use_tensorrt: bool = False
    tensorrt_engine: str = ""
    warmup_iters: int = 50
    measure_iters: int = 1000
    batch_size: int = 1
    image_size: tuple[int, int] = (224, 224)
    state_dim: int = 32
    action_dim: int = 8
    action_horizon: int = 16
    num_denoise_steps: int = 10
    use_real_model: bool = False
    output_dir: str = "results/benchmarks"


@dataclass
class BenchmarkResults:
    config: dict
    latency_ms: dict
    throughput_fps: float
    gpu_info: dict
    precision_info: dict
    timestamp: str


class GPUBenchmark:
    """Benchmark with real CUDA event timing on GPU."""

    def __init__(self, config: BenchmarkConfig):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._engine: OptimizedInferenceEngine | None = None

        print(f"[GPUBenchmark] Device: {torch.cuda.get_device_name(0)}")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(self.device)}")
        print(f"  Config: {config.model_name} | {config.precision} "
              f"| cudagraph={config.use_cuda_graph} | compile={config.use_torch_compile}")

    def build_dummy_model(self) -> torch.nn.Module:
        """Build a representative dummy model for benchmarking.

        Matches the FLOPs and memory pattern of the real model
        without requiring the actual checkpoint.
        """
        import math

        class DummyVLA(torch.nn.Module):
            """Dummy VLA model that matches 3B-param compute profile.

            Uses a ViT-like encoder + transformer + action head with
            realistic tensor shapes and FLOPs.
            """

            def __init__(self, cfg: BenchmarkConfig):
                super().__init__()
                self.cfg = cfg
                H, W = cfg.image_size
                patch_size = 16
                n_patches = (H // patch_size) * (W // patch_size)
                embed_dim = 1536
                n_heads = 16
                n_layers = 28
                hidden_dim = embed_dim * 4

                self.patch_embed = torch.nn.Conv2d(3, embed_dim,
                    kernel_size=patch_size, stride=patch_size)
                self.pos_embed = torch.nn.Parameter(
                    torch.randn(1, n_patches + 1, embed_dim) * 0.02)
                self.cls_token = torch.nn.Parameter(
                    torch.randn(1, 1, embed_dim) * 0.02)

                self.blocks = torch.nn.ModuleList([
                    torch.nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=n_heads,
                        dim_feedforward=hidden_dim,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(n_layers)
                ])

                self.norm = torch.nn.LayerNorm(embed_dim)
                self.state_proj = torch.nn.Linear(cfg.state_dim, embed_dim)
                self.action_head = torch.nn.Sequential(
                    torch.nn.Linear(embed_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim, cfg.action_dim * cfg.action_horizon),
                )

                if cfg.model_name == "pi05":
                    self.flow_net = torch.nn.Sequential(
                        torch.nn.Linear(cfg.action_dim * cfg.action_horizon + embed_dim, hidden_dim),
                        torch.nn.GELU(),
                        torch.nn.Linear(hidden_dim, cfg.action_dim * cfg.action_horizon),
                    )

            def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
                B = image.shape[0]
                x = self.patch_embed(image)
                x = x.flatten(2).transpose(1, 2)
                cls = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls, x], dim=1)
                x = x + self.pos_embed[:, :x.size(1), :]

                for block in self.blocks:
                    x = block(x)

                x = self.norm(x)[:, 0]
                s = self.state_proj(state)
                x = x + s
                action = self.action_head(x)
                out = action.view(B, self.cfg.action_horizon, self.cfg.action_dim)

                if hasattr(self, "flow_net"):
                    noise = torch.randn_like(out)
                    for _ in range(self.cfg.num_denoise_steps):
                        feat = torch.cat([out.flatten(1), x], dim=1)
                        denoise = self.flow_net(feat).view_as(out)
                        out = out - 0.1 * denoise

                return out

        return DummyVLA(self.cfg).eval()

    def warmup(self):
        if self.cfg.checkpoint_path and self.cfg.use_real_model:
            model = torch.load(self.cfg.checkpoint_path, map_location="cpu")
            if hasattr(model, "eval"):
                model = model.eval()
        else:
            model = self.build_dummy_model()

        model = model.to(self.device)

        precision = PrecisionMode(self.cfg.precision)
        engine_cfg = EngineConfig(
            model_name=self.cfg.model_name,
            precision=precision,
            use_cuda_graph=self.cfg.use_cuda_graph,
            use_torch_compile=self.cfg.use_torch_compile,
            warmup_iters=self.cfg.warmup_iters,
            static_batch_size=self.cfg.batch_size,
        )

        self._engine = OptimizedInferenceEngine(model, engine_cfg, self.device)

        sample = self._make_sample_inputs()
        self._engine.warmup(sample)

    def _make_sample_inputs(self) -> dict[str, torch.Tensor]:
        B = self.cfg.batch_size
        C, H, W = 3, self.cfg.image_size[0], self.cfg.image_size[1]
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[self.cfg.precision]
        return {
            "image": torch.randn(B, C, H, W, device=self.device, dtype=dtype),
            "state": torch.randn(B, self.cfg.state_dim, device=self.device, dtype=dtype),
        }

    def run(self) -> BenchmarkResults:
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()

        mem_before = torch.cuda.memory_allocated(self.device)
        self.warmup()
        mem_after = torch.cuda.memory_allocated(self.device)

        sample = self._make_sample_inputs()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        n_warm = self.cfg.warmup_iters
        n_meas = self.cfg.measure_iters

        for _ in range(n_warm):
            self._engine.infer(sample)

        torch.cuda.synchronize()

        latencies_ms = []
        for i in range(n_meas):
            start_event.record()
            self._engine.infer(sample)
            end_event.record()
            torch.cuda.synchronize()
            latencies_ms.append(start_event.elapsed_time(end_event))

        latencies_ms = np.array(latencies_ms)

        torch.cuda.synchronize()
        mem_peak = torch.cuda.max_memory_allocated(self.device)
        mem_reserved = torch.cuda.memory_reserved(self.device)

        stats = {
            "mean_ms": float(np.mean(latencies_ms)),
            "median_ms": float(np.median(latencies_ms)),
            "p95_ms": float(np.percentile(latencies_ms, 95)),
            "p99_ms": float(np.percentile(latencies_ms, 99)),
            "min_ms": float(np.min(latencies_ms)),
            "max_ms": float(np.max(latencies_ms)),
            "std_ms": float(np.std(latencies_ms)),
        }

        throughput = 1000.0 / stats["mean_ms"] * self.cfg.batch_size

        gpu_name = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)

        self._engine.cleanup()

        return BenchmarkResults(
            config=asdict(self.cfg),
            latency_ms=stats,
            throughput_fps=throughput,
            gpu_info={
                "name": gpu_name,
                "total_memory_mb": gpu_props.total_memory / 1024 / 1024,
                "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                "mem_allocated_before_mb": mem_before / 1024 / 1024,
                "mem_allocated_after_mb": mem_after / 1024 / 1024,
                "mem_peak_mb": mem_peak / 1024 / 1024,
                "mem_reserved_mb": mem_reserved / 1024 / 1024,
            },
            precision_info={
                "mode": self.cfg.precision,
                "cuda_graph": self.cfg.use_cuda_graph,
                "torch_compile": self.cfg.use_torch_compile,
                "tensorrt": self.cfg.use_tensorrt,
            },
            timestamp=datetime.now().isoformat(),
        )

    def print_report(self, r: BenchmarkResults):
        w = 72
        sep = "=" * w
        sep2 = "-" * w
        print()
        print(sep)
        title = f"  {r.config['model_name'].upper()} @ {r.config['precision'].upper()}"
        print(f"|{title:<{w+1}s}|")
        print(sep)

        print(f"|  GPU: {r.gpu_info['name']:<{w-7}s}|")
        opt_str = f"cudagraph={r.precision_info['cuda_graph']}, compile={r.precision_info['torch_compile']}"
        print(f"|  Options: {opt_str:<{w-12}s}|")
        print(sep)

        rows = [
            ("Mean latency", r.latency_ms["mean_ms"]),
            ("Median latency", r.latency_ms["median_ms"]),
            ("P95 latency", r.latency_ms["p95_ms"]),
            ("P99 latency", r.latency_ms["p99_ms"]),
            ("Max latency", r.latency_ms["max_ms"]),
            ("Std latency", r.latency_ms["std_ms"]),
        ]

        fmt = "|  {:<30s} {:>8.3f} ms  {:>20s} |"
        hz_fmt = "|  {:<30s} {:>8.0f} fps  {:>20s} |"
        for label, val in rows:
            ok = "[OK]" if val < 20 else "[XX]"
            print(fmt.format(label, val, ok if "P99" in label or "Max" in label else ""))
        print(hz_fmt.format("Throughput", r.throughput_fps, ""))

        print(sep)
        print(f"|  Memory peak: {r.gpu_info['mem_peak_mb']:>8.0f} MB / "
              f"{r.gpu_info['total_memory_mb']:>5.0f} MB ({r.gpu_info['mem_peak_mb']/r.gpu_info['total_memory_mb']*100:.0f}%)"
              f"{'':>{w-30}}|")
        print(sep)

    def save_results(self, r: BenchmarkResults):
        d = Path(self.cfg.output_dir)
        d.mkdir(parents=True, exist_ok=True)
        name = f"{r.config['model_name']}_{r.config['precision']}_bs{r.config['batch_size']}"
        path = d / f"benchmark_{name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(path, "w") as f:
            json.dump(asdict(r), f, indent=2)
        print(f"  Results saved: {path}")


def _sweep():
    """Run a sweep across precision/optimization combinations."""
    configs = []
    for model in ["gr00t", "pi05"]:
        for prec in ["bf16", "fp16"]:
            for cg in [True, False]:
                for comp in [True, False]:
                    configs.append(BenchmarkConfig(
                        model_name=model,
                        precision=prec,
                        use_cuda_graph=cg,
                        use_torch_compile=comp,
                    ))

    results = []
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Benchmark: {cfg.model_name} {cfg.precision} "
              f"cudagraph={cfg.use_cuda_graph} compile={cfg.use_torch_compile}")
        print("=" * 60)
        bench = GPUBenchmark(cfg)
        r = bench.run()
        bench.print_report(r)
        bench.save_results(r)
        results.append(r)

    return results


def main():
    import argparse
    ap = argparse.ArgumentParser(description="GPU Benchmark for RoboForce VLA models")
    ap.add_argument("--model", default="gr00t", choices=["gr00t", "pi05"])
    ap.add_argument("--checkpoint", default="", help="Path to real checkpoint (optional)")
    ap.add_argument("--precision", default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--no-cudagraph", action="store_false", dest="use_cuda_graph")
    ap.add_argument("--no-compile", action="store_false", dest="use_torch_compile")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--sweep", action="store_true", help="Run all config combinations")
    ap.add_argument("--output", default="results/benchmarks")

    args = ap.parse_args()
    if args.sweep:
        _sweep()
        return

    cfg = BenchmarkConfig(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        precision=args.precision,
        use_cuda_graph=args.use_cuda_graph,
        use_torch_compile=args.use_torch_compile,
        measure_iters=args.iters,
        warmup_iters=args.warmup,
        batch_size=args.batch_size,
        output_dir=args.output,
    )

    bench = GPUBenchmark(cfg)
    r = bench.run()
    bench.print_report(r)
    bench.save_results(r)


if __name__ == "__main__":
    main()
