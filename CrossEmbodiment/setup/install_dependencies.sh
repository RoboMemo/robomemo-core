#!/usr/bin/env bash
# install_dependencies.sh — One-click dependency installer
# Run from project root: bash setup/install_dependencies.sh
set -euo pipefail

echo "=============================================="
echo "  Cross-Embodiment Retarget Demo — Installer"
echo "=============================================="

# Detect conda
if command -v conda &>/dev/null; then
    echo "[✓] Conda detected: $(conda --version)"
else
    echo "[!] Conda not found — using system pip"
fi

# Create/activate conda env
ENV_NAME="retarget"
if conda env list 2>/dev/null | grep -q "$ENV_NAME"; then
    echo "[✓] Conda env '$ENV_NAME' already exists"
else
    echo "[→] Creating conda env '$ENV_NAME' with Python 3.10…"
    conda create -n "$ENV_NAME" python=3.10 -y
fi

echo "[→] Activating env '$ENV_NAME'…"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "[→] Installing core Python dependencies…"
pip install --upgrade pip
pip install \
    numpy \
    pyyaml \
    pyzmq \
    websockets \
    matplotlib

echo "[→] Installing optional dependencies (may skip if unavailable)…"
# MediaPipe for webcam mode
pip install mediapipe opencv-python || echo "[!] MediaPipe/OpenCV install failed (webcam mode won't work)"

# ONNX Runtime with CUDA for real SONIC models
pip install onnxruntime-gpu || pip install onnxruntime || echo "[!] ONNX Runtime install failed (will use mock backend)"

echo ""
echo "=============================================="
echo "  Installation complete!"
echo ""
echo "  Activate env:  conda activate retarget"
echo "  Run tests:     python -m pytest tests/"
echo "  Run demo:      python -m src.demo_runner"
echo "=============================================="
