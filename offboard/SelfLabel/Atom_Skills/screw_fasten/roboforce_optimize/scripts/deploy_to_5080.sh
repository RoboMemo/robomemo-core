#!/usr/bin/env bash
# Deploy optimization code to the 5080 (or 5070Ti) production machine.
#
# Usage:
#   bash scripts/deploy_to_5080.sh              # rsync to 5080 (default)
#   bash scripts/deploy_to_5080.sh 5070ti       # rsync to 5070Ti instead
#   bash scripts/deploy_to_5080.sh dry          # dry run (show what would copy)
#
# The 5080 box: siyu@100.70.68.31 (Tailscale)
# Remote path: ~/robomemo-core/offboard/SelfLabel/Atom_Skills/screw_fasten/

set -euo pipefail

MACHINE="${1:-5080}"
DRY=""

case "$MACHINE" in
    dry|--dry|-n)
        DRY="--dry-run"
        REMOTE="siyu@100.70.68.31"
        ;;
    5080)
        REMOTE="siyu@100.70.68.31"
        ;;
    5070ti|5070)
        REMOTE="siyu@100.79.88.6"
        ;;
    *)
        echo "Usage: $0 [5080|5070ti|dry]"
        exit 1
        ;;
esac

REMOTE_PATH="~/robomemo-core/offboard/SelfLabel/Atom_Skills/screw_fasten/"
LOCAL="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== Deploying optimizations to $REMOTE:$REMOTE_PATH ==="
echo "  local:  $LOCAL"
echo "  remote: $REMOTE:$REMOTE_PATH"
echo ""

rsync -avz $DRY \
    --progress \
    --include="roboforce_optimize/***" \
    --include="roboforce_optimize/configs/***" \
    --include="roboforce_optimize/scripts/***" \
    --include="roboforce_skills/gr00t_finetune_config.py" \
    --include="roboforce_skills/openpi_finetune_config.py" \
    --include="results/benchmarks/***" \
    --exclude="roboforce_optimize/__pycache__/***" \
    --exclude="**/*.pyc" \
    --exclude="**/__pycache__/" \
    -e "ssh -o StrictHostKeyChecking=no" \
    "$LOCAL/" \
    "$REMOTE:$REMOTE_PATH"

echo ""
echo "=== Deploy complete ==="
echo ""
echo "Next steps on $REMOTE (5080 has Linux + Triton):"
echo "  1. ssh $REMOTE"
echo "  2. cd $REMOTE_PATH"
echo "  3. conda activate pose3d"
echo "  4. pip install -r requirements_optimize.txt"
echo "  5. Run benchmark:    bash scripts/run_benchmark.sh gr00t"
echo "  6. Full sweep:       bash scripts/run_benchmark.sh sweep"
echo ""
echo "Recommended configs (proven on 5070 Ti Linux):"
echo "  GR00T: BF16 + CUDA Graph + torch.compile (target: <10ms)"
echo "  pi0.5: FP16 + CUDA Graph + torch.compile + distill 4 steps (target: <10ms)"
echo ""
echo "TensorRT export (requires tensorrt>=10.6):"
echo "  python -m roboforce_optimize tensorrt --model gr00t --checkpoint <ckpt> --precision fp8"
echo "  python -m roboforce_optimize tensorrt --model pi05  --checkpoint <ckpt> --fp4"
