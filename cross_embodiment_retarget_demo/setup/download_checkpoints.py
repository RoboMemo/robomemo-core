"""
Download SONIC model checkpoints from HuggingFace.

Downloads:
  - nvidia/GEAR-SONIC → policy/release/model_encoder.onnx, model_decoder.onnx
  - nvidia/GEAR-SONIC → planner/target_vel/V2/planner_sonic.onnx
  - nvidia/GEAR-SONIC → policy/release/observation_config.yaml

Usage:
  python setup/download_checkpoints.py [--output-dir checkpoints/sonic]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def download_sonic_checkpoints(output_dir: str = "checkpoints/sonic"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub…")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import snapshot_download

    print(f"Downloading SONIC checkpoints to {out.resolve()}…")
    print("Repository: nvidia/GEAR-SONIC")

    try:
        snapshot_download(
            repo_id="nvidia/GEAR-SONIC",
            local_dir=str(out),
            allow_patterns=[
                "policy/release/model_encoder.onnx",
                "policy/release/model_decoder.onnx",
                "policy/release/observation_config.yaml",
                "planner/target_vel/V2/planner_sonic.onnx",
            ],
        )
        print(f"\n[✓] Checkpoints downloaded to: {out.resolve()}")
        print("Files:")
        for f in sorted(out.rglob("*")):
            if f.is_file():
                size = f.stat().st_size / (1024 * 1024)
                print(f"  {f.relative_to(out)} ({size:.1f} MB)")
    except Exception as e:
        print(f"\n[!] Download failed: {e}")
        print("    You may need to accept the model license at:")
        print("    https://huggingface.co/nvidia/GEAR-SONIC")
        print("    Then run: huggingface-cli login")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SONIC model checkpoints")
    parser.add_argument("--output-dir", default="checkpoints/sonic",
                        help="Output directory for model files")
    args = parser.parse_args()
    download_sonic_checkpoints(args.output_dir)
