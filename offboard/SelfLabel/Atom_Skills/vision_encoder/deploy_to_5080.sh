#!/usr/bin/env bash
set -euo pipefail

REMOTE="siyu@100.70.68.31"
REMOTE_PATH="~/robomemo-core/offboard/SelfLabel/Atom_Skills/vision_encoder/"
LOCAL="$(cd "$(dirname "$0")" && pwd)"

echo "=== Deploying Vision Encoder to RTX 5080 ==="
echo "  local:  $LOCAL"
echo "  remote: $REMOTE:$REMOTE_PATH"

rsync -avz --progress \
    --include="*.py" \
    --exclude="__pycache__/" \
    --exclude="*.pyc" \
    -e "ssh -o StrictHostKeyChecking=no" \
    "$LOCAL/" \
    "$REMOTE:$REMOTE_PATH"

echo ""
echo "=== Deploy complete ==="
echo ""
echo "Next steps on 5080:"
echo "  ssh $REMOTE"
echo "  source ~/miniforge3/etc/profile.d/conda.sh && conda activate pose3d"
echo "  cd $REMOTE_PATH"
echo "  python -m vision_encoder.full_pipeline --demo"
echo ""
echo "Distributed sync between peers:"
echo "  python -m vision_encoder.distributed_trainer --sync <peer_checkpoint.pt>"
