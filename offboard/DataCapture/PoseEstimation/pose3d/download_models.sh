#!/usr/bin/env bash
# =====================================================================
# download_models.sh — fetch model code + weights for the pose3d pipeline.
# Run from PoseEstimation/pose3d/, on your Linux+CUDA machine, INSIDE the
# conda env:  conda activate pose3d && bash download_models.sh
#
# Fetches:
#   1. SMPLer-X source (body+hands main model)  -> third_party/SMPLer-X
#   2. SMPLer-X-H32* checkpoint (camera-fix)    -> pretrained_models/
#   3. mmdet faster_rcnn detector (person crop) -> pretrained_models/mmdet/
#   4. VERIFIES the SMPL-X body model (already provided; no download)
#   5. SMPLer-X bundled mmpose fork (transformer_utils) pip-installed
#   6. Prints OPTIONAL HaMeR / MANO instructions (disabled by default)
# =====================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATACAP_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
THIRD_PARTY="$SCRIPT_DIR/third_party"
mkdir -p "$THIRD_PARTY"

# Pin a commit for reproducibility (unset -> latest main). Set to a known SHA.
SMPLERX_COMMIT="${SMPLERX_COMMIT:-}"
SMPLERX_DIR="$THIRD_PARTY/SMPLer-X"

echo "============================================================"
echo " pose3d — download_models.sh"
echo " DataCapture root: $DATACAP_ROOT"
echo "============================================================"

require() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: '$1' not found." >&2; exit 1; }; }
dl() { # dl URL OUTPATH
  if [ -f "$2" ]; then echo "      exists: $2"; return; fi
  echo "      downloading -> $2"; curl -fSL -o "$2" "$1" || { echo "      ! failed: $1"; return 1; }
}

# ---------------------------------------------------------------- 1. source
if [ ! -d "$SMPLERX_DIR/.git" ]; then
  echo "[1/6] Cloning SMPLer-X (caizhongang/SMPLer-X) ..."
  git clone https://github.com/caizhongang/SMPLer-X "$SMPLERX_DIR"
  if [ -n "$SMPLERX_COMMIT" ]; then
    git -C "$SMPLERX_DIR" checkout "$SMPLERX_COMMIT"
  else
    echo "      (no SMPLERX_COMMIT set; using latest main. Set it to pin.)"
  fi
else
  echo "[1/6] SMPLer-X present ($SMPLERX_DIR)."
fi
PRE="$SMPLERX_DIR/pretrained_models"
mkdir -p "$PRE"

# ---------------------------------------------------------------- 2. H32* ckpt
echo "[2/6] SMPLer-X-H32* checkpoint (camera-fix) ..."
H32_URL="https://huggingface.co/caizhongang/SMPLer-X/resolve/main/smpler_x_h32_correct.pth.tar"
dl "$H32_URL" "$PRE/smpler_x_h32_correct.pth.tar"

# ---------------------------------------------------------------- 3. YOLO detector
echo "[3/6] ultralytics YOLO person detector (yolov8n.pt) ..."
# ultralytics auto-downloads yolov8n.pt on first inference; we also pre-fetch it.
YOLO_PT="$SCRIPT_DIR/yolov8n.pt"
if [ ! -f "$YOLO_PT" ]; then
  python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>/dev/null \
    && cp -f yolov8n.pt "$YOLO_PT" 2>/dev/null \
    || dl "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt" "$YOLO_PT"
fi
[ -f "$YOLO_PT" ] && echo "      OK yolov8n.pt" || echo "      ! yolov8n.pt missing (ultralytics will fetch on first run)"

# ---------------------------------------------------------------- 4. SMPL-X (provided)
echo "[4/6] Verifying SMPL-X body model (already provided) ..."
SMPLX_DIR="$DATACAP_ROOT/dataset/models/smplx"
ok=1
for f in SMPLX_NEUTRAL.npz SMPLX_MALE.npz SMPLX_FEMALE.npz; do
  [ -f "$SMPLX_DIR/$f" ] && echo "      OK  $f" || { echo "      MISSING  $f"; ok=0; }
done
if [ $ok -eq 1 ]; then
  echo "      -> provided, no download needed."
else
  echo "  ! Some SMPLX_*.npz missing. Get them from https://smpl.is.tue.mpg.de/ -> $SMPLX_DIR"
fi
# SMPLer-X reads body models from common/utils/human_model_files/
HMF="$SMPLERX_DIR/common/utils/human_model_files"; mkdir -p "$HMF"
[ -d "$SMPLX_DIR" ] && { ln -sfn "$SMPLX_DIR" "$HMF/smplx" 2>/dev/null || true; }

# ---------------------------------------------------------------- 5. mmpose fork
echo "[5/6] Installing SMPLer-X bundled mmpose (transformer_utils) ..."
echo "      NOTE: the SMPLer-X REGRESSOR still imports mmcv (common/nets/smpler_x.py:"
echo "      'from mmcv.ops.roi_align import roi_align' + main/SMPLer_X.py 'build_posenet')."
echo "      mmdet is GONE (YOLO detects), but mmcv + transformer_utils must stay."
echo "      On RTX 5080/Blackwell there is no mmcv-full cu124/torch2.6 wheel — see README §9."
TU="$SMPLERX_DIR/main/transformer_utils"
if [ -d "$TU" ]; then
  pip install -v -e "$TU" -q || echo "  ! transformer_utils install failed (see README §2)."
else
  echo "  ! $TU not found in this commit; SMPLer-X may use a stock mmpose instead."
fi

# ---------------------------------------------------------------- 6. HaMeR (optional)
echo "[6/6] HaMeR / MANO -> OPTIONAL, NOT enabled by default."
echo "      Enable only if SMPLer-X hand precision is insufficient:"
echo "        git clone --depth 1 https://github.com/geopavlakos/hamer $THIRD_PARTY/hamer"
echo "        # MANO_RIGHT.pkl from https://mano.is.tue.mpg.de/ -> dataset/models/mano/"
echo "        # set config.yaml: hand.backend=hamer; wire hamer_wrapper.infer_hands()"

echo "============================================================"
echo " Done. Next: python run_pipeline.py --recording 0721-1"
echo "============================================================"
