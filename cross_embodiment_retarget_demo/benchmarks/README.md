# SONIC Inference Benchmark Suite

Comprehensive benchmarks for the SONIC retarget pipeline, designed to validate
real-time deployment readiness on humanoid robots (Unitree G1, H1, Fourier GR1T2).

## Quick Start

```bash
cd /home/siyu/Projects/Retarget

# Full benchmark (default: 10,000 iterations, 500 warmup)
python3 -m cross_embodiment_retarget_demo.benchmarks.sonic_benchmark

# Or run directly
python3 cross_embodiment_retarget_demo/benchmarks/sonic_benchmark.py

# Quick benchmark (fewer iterations)
python3 -m cross_embodiment_retarget_demo.benchmarks.sonic_benchmark -n 2000 -w 200

# Custom config
python3 -m cross_embodiment_retarget_demo.benchmarks.sonic_benchmark \
    --config cross_embodiment_retarget_demo/configs/demo_config.yaml \
    --iterations 10000 --warmup 500 --sustained 10
```

## What It Measures

### 1. Inference Latency (most critical)
- **Proprioception pack**: `np.concatenate([joint_pos, joint_vel, gravity])`
- **Backend.infer**: Raw analytical IK (MockSonicBackend) or ONNX encoder+decoder
- **Post-processing**: Velocity limiting + joint clamping
- **End-to-end**: Full `SonicRetarget.infer()` call

Statistics: mean, median, P95, P99, min, max, std

### 2. Throughput
- **Sequential FPS**: Maximum single-inference throughput
- **Sustained FPS**: Actual achieved FPS over a 10-second run
- **Batch throughput**: Batch sizes 1, 4, 8, 16, 32

### 3. Full Pipeline
Per-step breakdown of the complete control loop:
```
Input capture → SONIC infer → Sim step → Proprioception update
```
Uses `MockPhysicsEnv` and `MockMotionSource`.

### 4. Per-Robot Comparison
Runs end-to-end latency for all three robots:
| Robot | DOF | Description |
|-------|-----|-------------|
| Unitree G1 | 29 | Primary deployment target |
| Unitree H1 | 19 | Simpler humanoid |
| Fourier GR1T2 | 32 | Most DOFs, includes head |

### 5. Stability / Jitter Analysis
- Latency spikes (>2x median)
- Maximum spike magnitude
- Jitter: standard deviation of consecutive latency differences
- Helps identify GC pauses, context switches, cache misses

### 6. Memory Profiling
- CPU RSS before/after model load
- GPU memory usage (via `nvidia-smi`)
- Peak GPU memory during inference

## Target Metrics for Unitree G1 Deployment

**Control loop: 50 Hz → 20 ms budget per step**

| Metric | Target | Notes |
|--------|--------|-------|
| End-to-end latency | < 5 ms | Leaves margin for comm + safety |
| SONIC infer | < 2 ms | Core inference |
| Sim step | < 10 ms | Physics + collision |
| Total pipeline | < 20 ms | Must not exceed 50 Hz budget |
| Latency spikes | < 1% | At 2x median threshold |
| Jitter | < 1 ms | Smooth control requires consistency |
| FPS headroom | > 10x | Safety margin for worst case |

## Output Files

Results are saved to `cross_embodiment_retarget_demo/benchmarks/results/`:

```
results/
├── sonic_benchmark_YYYYMMDD_HHMMSS.json   # Machine-readable, full data
├── sonic_benchmark_YYYYMMDD_HHMMSS.md     # Human-readable summary
└── sonic_latency_dist_YYYYMMDD_HHMMSS.png # Latency distribution plot
```

### JSON Schema
```json
{
  "timestamp": "2026-03-13 10:00:00",
  "system": { "gpu": "...", "python": "...", "numpy": "...", "onnxruntime": "...", "backend": "Mock|ONNX" },
  "config": { "iterations": 10000, "warmup": 500, "target_hz": 50 },
  "latency": [ { "stage": "...", "mean_ms": 0.1, "median_ms": 0.09, ... } ],
  "throughput": [ { "label": "Sequential FPS", "fps": 100000, "batch_size": 1 } ],
  "pipeline": [ { "stage": "Per-step total", "mean_ms": 0.5 } ],
  "per_robot": [ { "robot": "unitree_g1", "dof": 29, "mean_ms": 0.1, "fps": 10000 } ],
  "stability": { "spikes": 12, "spike_pct": 0.12, "jitter_ms": 0.02 },
  "memory": { "rss_before_mb": 45, "rss_after_mb": 120, "gpu_peak_mb": 95 }
}
```

## Backends

### Mock Backend (default)
Uses `MockSonicBackend` — an analytical inverse kinematics mapping. No ONNX
models required. Good for pipeline validation and establishing baseline timing.

### ONNX Backend
When SONIC models are downloaded (`model_encoder.onnx` + `model_decoder.onnx`),
the benchmark automatically uses `OnnxSonicBackend` with CUDA or CPU providers.

To download models:
```bash
python3 cross_embodiment_retarget_demo/setup/download_checkpoints.py \
    --output-dir cross_embodiment_retarget_demo/checkpoints/sonic
```

## Implementation Notes

- Uses `time.perf_counter_ns()` for nanosecond-resolution timing
- Process pinned to CPU core 0 (`os.sched_setaffinity`) for reduced jitter
- Warmup phase (default 500 iters) excluded from measurements
- Handles missing deps gracefully: onnxruntime, matplotlib, psutil
- Self-contained script — works as both module and direct execution

## Dependencies

Required:
- Python 3.10+
- numpy
- PyYAML

Optional (for full features):
- `onnxruntime-gpu` — ONNX model inference with CUDA
- `psutil` — accurate memory profiling
- `matplotlib` — latency distribution plots

```bash
pip3 install onnxruntime-gpu psutil matplotlib
```
