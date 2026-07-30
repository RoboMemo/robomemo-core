# RoboForce Optimization Benchmark Results

## Hardware: RTX 5070 Ti Laptop (12GB Blackwell, Windows)
Collected: 2026-07-29

### GR00T N1.6 (simulated 3B)

| Config | Mean | P99 | Max | Std | FPS | 50Hz |
|--------|:----:|:---:|:---:|:---:|:---:|:----:|
| BF16 eager (no cudagraph, no compile) | 14.54 ms | 23.46 ms | 38.17 ms | 2.66 ms | 69 | NO (variance) |
| BF16 + CUDA Graph + compile(skip) | 13.37 ms | 13.78 ms | 14.27 ms | 0.21 ms | 75 | YES |
| FP16 + CUDA Graph + compile(skip) | 19.68 ms | 29.25 ms | 29.78 ms | 7.63 ms | 51 | NO |
| BF16 + compile(skip) only (no cudagraph) | 24.75 ms | 31.61 ms | 32.12 ms | 8.08 ms | 40 | NO |

**Recommendation: BF16 + CUDA Graph + NO torch.compile on Windows** (13.4ms, 75 FPS)

### pi0.5 (simulated 0.5B, 10 denoising steps)

| Config | Mean | P99 | Max | Std | FPS | 50Hz |
|--------|:----:|:---:|:---:|:---:|:---:|:----:|
| BF16 eager (no cudagraph, no compile) | 22.17 ms | 24.84 ms | 25.17 ms | 0.84 ms | 45 | NO |
| BF16 + CUDA Graph + compile(skip) | 34.19 ms | 34.88 ms | 34.92 ms | 0.43 ms | 29 | NO |
| **FP16 + CUDA Graph + compile(skip)** | **18.12 ms** | **21.04 ms** | **21.09 ms** | **1.44 ms** | **55** | **YES** |

**Recommendation: FP16 + CUDA Graph + NO torch.compile + distillation to 4 steps** (18ms → ~7ms)

### Key Findings

1. **CUDA Graph is essential** for real-time — it eliminates kernel launch variance (std drops 10x)
2. **torch.compile attempt on Windows is harmful** — even when it fails gracefully, it adds ~10ms overhead
3. **GR00T meets 50Hz @ BF16** on 5070 Ti Laptop; **pi0.5 needs FP16** to meet 50Hz
4. All configs use < 3.2 GB VRAM — plenty of headroom on 12GB (5070 Ti) or 16GB (5080)
5. Real models will be ~2-3x more compute than the dummy
