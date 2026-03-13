"""
Download SONIC model checkpoints from HuggingFace (pure urllib, no pip required).

Downloads:
  - nvidia/GEAR-SONIC → policy/release/model_encoder.onnx, model_decoder.onnx
  - nvidia/GEAR-SONIC → planner/target_vel/V2/planner_sonic.onnx
  - nvidia/GEAR-SONIC → policy/release/observation_config.yaml

Usage:
  python setup/download_checkpoints.py [--output-dir checkpoints/sonic]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import urllib.error
from pathlib import Path


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress indicator."""
    try:
        print(f"  Downloading: {output_path.name}…", end=" ", flush=True)
        
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            
            with open(output_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(f"\r  Downloading: {output_path.name}… {pct:.0f}%", end="", flush=True)
        
        print(f" ✓ ({output_path.stat().st_size / (1024*1024):.1f} MB)")
        return True
    except Exception as e:
        print(f" ✗ ({e})")
        return False


def download_sonic_checkpoints(output_dir: str = "checkpoints/sonic"):
    """Download SONIC models from HuggingFace CDN."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # HuggingFace CDN URLs for SONIC models
    files_to_download = {
        "policy/release/model_encoder.onnx": 
            "https://huggingface.co/nvidia/GEAR-SONIC/resolve/main/policy/release/model_encoder.onnx",
        "policy/release/model_decoder.onnx":
            "https://huggingface.co/nvidia/GEAR-SONIC/resolve/main/policy/release/model_decoder.onnx",
        "policy/release/observation_config.yaml":
            "https://huggingface.co/nvidia/GEAR-SONIC/resolve/main/policy/release/observation_config.yaml",
        "planner/target_vel/V2/planner_sonic.onnx":
            "https://huggingface.co/nvidia/GEAR-SONIC/resolve/main/planner/target_vel/V2/planner_sonic.onnx",
    }

    print(f"Downloading SONIC checkpoints to {out.resolve()}…")
    print("Repository: nvidia/GEAR-SONIC\n")

    success_count = 0
    for rel_path, url in files_to_download.items():
        file_out = out / rel_path
        file_out.parent.mkdir(parents=True, exist_ok=True)
        
        if file_out.exists():
            print(f"  Skipping: {rel_path} (already exists)")
            success_count += 1
            continue
        
        if download_file(url, file_out):
            success_count += 1

    print(f"\n[{'✓' if success_count == len(files_to_download) else '⚠'}] {success_count}/{len(files_to_download)} files downloaded")
    
    if success_count > 0:
        print(f"\nCheckpoints location: {out.resolve()}")
        print("Update demo_config.yaml:")
        print(f"  sonic.model_dir: {out.resolve()}")
    else:
        print("\n[!] Download failed. You may need to:")
        print("    1. Accept the model license at: https://huggingface.co/nvidia/GEAR-SONIC")
        print("    2. Install huggingface_hub: pip install huggingface_hub")
        print("    3. Login: huggingface-cli login")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download SONIC model checkpoints")
    parser.add_argument("--output-dir", default="checkpoints/sonic",
                        help="Output directory for model files")
    args = parser.parse_args()
    download_sonic_checkpoints(args.output_dir)
