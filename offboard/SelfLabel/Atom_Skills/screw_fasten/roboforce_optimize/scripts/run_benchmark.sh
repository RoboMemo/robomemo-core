#!/usr/bin/env bash
# Run the full optimization benchmark sweep on the RTX 5080 / 5070Ti.
#
# This runs all precision/optimization combinations and saves results.
#
# Usage:
#   bash scripts/run_benchmark.sh                    # full sweep (all combos)
#   bash scripts/run_benchmark.sh gr00t              # GR00T only
#   bash scripts/run_benchmark.sh pi05               # pi0.5 only
#   bash scripts/run_benchmark.sh quick              # quick test (100 iters)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-sweep}"
ITERS="${2:-1000}"
WARMUP="${3:-50}"

echo "============================================"
echo "  RoboForce Optimization Benchmark"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null)"
echo "  Mode: $MODE"
echo "============================================"
echo ""

PY="python -m roboforce_optimize"

case "$MODE" in
    sweep)
        echo "==== Full sweep ===="
        # GR00T: best config = BF16 + CUDA Graph + compile (on Linux)
        $PY benchmark --model gr00t --precision bf16 --iters "$ITERS" --warmup "$WARMUP"
        $PY benchmark --model gr00t --precision fp16 --iters "$ITERS" --warmup "$WARMUP"
        # pi0.5: best config = FP16 + CUDA Graph + compile (on Linux)
        $PY benchmark --model pi05 --precision bf16 --iters "$((ITERS / 2))" --warmup "$WARMUP"
        $PY benchmark --model pi05 --precision fp16 --iters "$((ITERS / 2))" --warmup "$WARMUP"
        # Eager baselines (no CUDA Graph, no compile)
        $PY benchmark --model gr00t --precision bf16 --no-cudagraph --no-compile --iters "$ITERS" --warmup "$WARMUP"
        $PY benchmark --model pi05 --precision bf16 --no-cudagraph --no-compile --iters "$((ITERS / 2))" --warmup "$WARMUP"
        ;;
    gr00t)
        $PY benchmark --model gr00t --precision bf16 --iters "$ITERS" --warmup "$WARMUP"
        $PY benchmark --model gr00t --precision fp16 --iters "$ITERS" --warmup "$WARMUP"
        ;;
    pi05)
        $PY benchmark --model pi05 --precision fp16 --iters "$((ITERS / 2))" --warmup "$WARMUP"
        $PY benchmark --model pi05 --precision bf16 --iters "$((ITERS / 2))" --warmup "$WARMUP"
        ;;
    quick|smoke)
        echo "Quick smoke test..."
        $PY benchmark --model gr00t --precision bf16 --iters 100 --warmup 10
        $PY benchmark --model pi05 --precision fp16 --iters 50 --warmup 10
        ;;
    tensorrt)
        echo "Building TensorRT engines..."
        for model in gr00t pi05; do
            for prec in fp8 fp16; do
                echo "  Building $model $prec engine..."
                $PY tensorrt --model "$model" --precision "$prec" \
                    --checkpoint "checkpoints/${model}_screw_driving/best.pt" \
                    --output "engines/${model}_${prec}.engine" \
                    --workspace 12 || echo "  [SKIP] TensorRT not available"
            done
        done
        ;;
    fp4)
        echo "Building FP4 (Blackwell) engine for GR00T..."
        $PY tensorrt --model gr00t --precision fp8 --fp4 \
            --checkpoint "checkpoints/gr00t_screw_driving/best.pt" \
            --output "engines/gr00t_fp4.engine" \
            --workspace 12 || echo "  [SKIP] FP4 requires Blackwell + TensorRT >= 10.6"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [sweep|gr00t|pi05|quick|tensorrt|fp4] [iters] [warmup]"
        exit 1
        ;;
esac

echo ""
echo "=== Benchmark complete ==="
echo "Results in: results/benchmarks/"
ls -la results/benchmarks/ 2>/dev/null || echo "  (no results yet)"
