# RoboForce Optimization (RTX 5080 Blackwell)

GPU inference optimization suite for GR00T N1.6 and pi0.5 on RTX 5080/5070Ti.

## Quick Start

```bash
# 1. See what GPU optimizations are available
python -m roboforce_optimize info

# 2. Run benchmark sweep (all precision/optimization combos)
python -m roboforce_optimize benchmark --sweep

# 3. Quick test on a single config
python -m roboforce_optimize benchmark --model gr00t --precision fp16

# 4. Export TensorRT FP8 engine
python -m roboforce_optimize tensorrt --model gr00t --checkpoint model.pt --precision fp8

# 5. Export Blackwell FP4 engine
python -m roboforce_optimize tensorrt --model gr00t --checkpoint model.pt --fp4
```

## Optimization Stack (ordered by ROI)

| Optimization | Speedup | Complexity | Files |
|---|---|---|---|
| **torch.compile** | 1.2-1.4x | Zero (add flag) | `infer_engine.py` |
| **CUDA Graph** | 1.3-1.5x | Low | `cuda_graph.py`, `infer_engine.py` |
| **Flash Attention** | 1.1-1.3x | Zero (config) | `infer_engine.py` |
| **Async pipeline** | 1.2-1.3x | Medium | `async_pipeline.py` |
| **FP8 TensorRT** | 2-3x | High | `tensorrt_export.py` |
| **FP4 TensorRT (Blackwell)** | 4-6x | High | `tensorrt_export.py` |
| **Flow distillation (pi0.5)** | 2-2.5x | High | `distill_flow.py` |

## Files

```
roboforce_optimize/
  __init__.py              # Package exports
  __main__.py              # CLI entry point (python -m roboforce_optimize)
  infer_engine.py          # Unified engine: compile + CUDA Graph + precision
  cuda_graph.py            # CUDA Graph capture/replay wrapper
  tensorrt_export.py       # TensorRT FP8/FP4 engine export
  async_pipeline.py        # CUDA stream overlap + NVDEC decoder
  distill_flow.py          # Flow step distillation for pi0.5
  benchmark_prof.py        # Real CUDA event benchmark (not mock)
  requirements_optimize.txt
  configs/
    optimize_gr00t.json    # GR00T optimization config
    optimize_pi05.json     # pi0.5 optimization config
  scripts/
    deploy_to_5080.sh      # rsync to production machine
    run_benchmark.sh       # benchmark runner script
```

## Target: 10^14 calculations/sec

RTX 5080 theoretical performance by precision (Blackwell, sm_120):

| Precision | Tensor Core Throughput | Reaches 100 TFLOPS? |
|-----------|----------------------|-------------------|
| FP32 | ~56 TFLOPS | No |
| FP16 | ~224 TFLOPS | **Yes** (2.2x headroom) |
| FP8 | ~448 TOPS | **Yes** (4.5x headroom) |
| FP4 | ~896 TOPS | **Yes** (9x headroom) |

Real-world utilization is ~40-60% of theoretical, so FP16 Tensor Cores
are the minimum viable path. FP8 via TensorRT is recommended.
