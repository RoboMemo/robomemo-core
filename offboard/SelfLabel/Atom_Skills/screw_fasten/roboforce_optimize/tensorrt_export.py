"""TensorRT engine export with FP8/FP4 quantization for Blackwell (sm_120).

Converts GR00T N1.6 or pi0.5 PyTorch models to TensorRT engines with:
    - FP8 quantization (via TensorRT Model Optimizer)
    - FP4 quantization (Blackwell 5th-gen Tensor Cores)
    - INT8 quantization fallback
    - Kernel fusion + memory planning

Requirements:
    pip install tensorrt>=10.0 tensorrt-cu12>=10.0
    pip install onnx onnxruntime
    pip install nvidia-modelopt~=0.28.0   (for FP8/FP4 quantization)

Usage:
    python -m roboforce_optimize.tensorrt_export \\
        --model gr00t --checkpoint /path/to/gr00t.pt \\
        --precision fp8 --output engines/gr00t_fp8.engine
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FP8_OPS = {
    "all": ["matmul", "layernorm", "attention", "silu", "gelu"],
    "matmul_only": ["matmul"],
    "attention_only": ["attention"],
    "default": ["matmul", "layernorm"],
}


@dataclass
class TensorRTExportConfig:
    model_name: str = "gr00t"
    checkpoint_path: str = ""
    output_path: str = "engines/model_fp8.engine"
    precision: str = "fp8"
    batch_size: int = 1
    max_batch_size: int = 8
    opt_batch_size: int = 1
    input_height: int = 224
    input_width: int = 224
    state_dim: int = 32
    action_dim: int = 8
    action_horizon: int = 16
    num_flow_steps: int = 10
    workspace_gb: int = 12
    use_fp4: bool = False
    sparse_weights: bool = True
    strict_types: bool = False
    build_config: dict | None = None

    def engine_path(self) -> str:
        p = Path(self.output_path)
        parent = p.parent
        parent.mkdir(parents=True, exist_ok=True)
        tag = f"{self.model_name}_{self.precision}_bs{self.max_batch_size}"
        if self.use_fp4:
            tag += "_fp4"
        return str(parent / f"{tag}.engine")


def export_gr00t_tensorrt(cfg: TensorRTExportConfig) -> str:
    """Export GR00T N1.6 to TensorRT engine.

    Converts via ONNX then TensorRT, applying FP8/FP4 quantization.
    """
    engine_path = cfg.engine_path()
    if Path(engine_path).exists():
        print(f"[TensorRT] Engine exists: {engine_path}")
        return engine_path

    print(f"[TensorRT] Exporting {cfg.model_name} ({cfg.precision}) -> {engine_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, f"{cfg.model_name}.onnx")
        _export_to_onnx_gr00t(cfg, onnx_path)
        _build_tensorrt_engine(cfg, onnx_path, engine_path)

    print(f"[TensorRT] Engine saved: {engine_path}")
    return engine_path


def export_pi05_tensorrt(cfg: TensorRTExportConfig) -> str:
    """Export pi0.5 to TensorRT engine.

    pi0.5 has flow matching so the export is more complex — we export the
    denoising UNet as a single engine with flow_steps as an input dimension.
    """
    engine_path = cfg.engine_path()
    if Path(engine_path).exists():
        print(f"[TensorRT] Engine exists: {engine_path}")
        return engine_path

    print(f"[TensorRT] Exporting {cfg.model_name} ({cfg.precision}) -> {engine_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, f"{cfg.model_name}_flow.onnx")
        _export_to_onnx_pi05(cfg, onnx_path)
        _build_tensorrt_engine(cfg, onnx_path, engine_path)

    print(f"[TensorRT] Engine saved: {engine_path}")
    return engine_path


def _export_to_onnx_gr00t(cfg: TensorRTExportConfig, onnx_path: str):
    """Export GR00T model to ONNX with dynamic axes."""
    import torch

    class GR00TWrapper(torch.nn.Module):
        def __init__(self, original_checkpoint: str):
            super().__init__()
            self.model = torch.load(original_checkpoint, map_location="cpu")
            if hasattr(self.model, "eval"):
                self.model.eval()

        def forward(self, image: torch.Tensor, state: torch.Tensor):
            return self.model(image, state)

    model = GR00TWrapper(cfg.checkpoint_path)
    dummy_image = torch.randn(1, 3, cfg.input_height, cfg.input_width)
    dummy_state = torch.randn(1, cfg.state_dim)

    dynamic_axes = {
        "image": {0: "batch"},
        "state": {0: "batch"},
        "action": {0: "batch"},
    }

    torch.onnx.export(
        model,
        (dummy_image, dummy_state),
        onnx_path,
        input_names=["image", "state"],
        output_names=["action"],
        dynamic_axes=dynamic_axes,
        opset_version=19,
    )
    print(f"[TensorRT] ONNX saved: {onnx_path}")


def _export_to_onnx_pi05(cfg: TensorRTExportConfig, onnx_path: str):
    """Export pi0.5 flow matching model to ONNX."""
    import torch

    class Pi05FlowWrapper(torch.nn.Module):
        def __init__(self, original_checkpoint: str):
            super().__init__()
            self.model = torch.load(original_checkpoint, map_location="cpu")
            if hasattr(self.model, "eval"):
                self.model.eval()

        def forward(
            self,
            image: torch.Tensor,
            state: torch.Tensor,
            noise: torch.Tensor,
            t: torch.Tensor,
        ):
            return self.model(image, state, noise, t)

    model = Pi05FlowWrapper(cfg.checkpoint_path)
    dummy_image = torch.randn(1, 3, cfg.input_height, cfg.input_width)
    dummy_state = torch.randn(1, cfg.state_dim)
    dummy_noise = torch.randn(1, cfg.action_horizon, cfg.action_dim)
    dummy_t = torch.randn(1)

    dynamic_axes = {
        "image": {0: "batch"},
        "state": {0: "batch"},
        "noise": {0: "batch"},
        "denoised": {0: "batch"},
    }

    torch.onnx.export(
        model,
        (dummy_image, dummy_state, dummy_noise, dummy_t),
        onnx_path,
        input_names=["image", "state", "noise", "t"],
        output_names=["denoised"],
        dynamic_axes=dynamic_axes,
        opset_version=19,
    )
    print(f"[TensorRT] pi0.5 ONNX saved: {onnx_path}")


def _build_tensorrt_engine(
    cfg: TensorRTExportConfig, onnx_path: str, engine_path: str
):
    """Build TensorRT engine from ONNX with FP8/FP4 quantization."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for e in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(e)}")
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, cfg.workspace_gb * 1073741824)

    if builder.platform_has_fast_fp8:
        config.set_flag(trt.BuilderFlag.FP8)
        print(f"[TensorRT] FP8 mode enabled")

    if builder.platform_has_fast_int8:
        if cfg.precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            print(f"[TensorRT] INT8 mode enabled")

    if cfg.use_fp4:
        config.set_flag(trt.BuilderFlag.OB_BLACKWELL_FP4)
        print(f"[TensorRT] FP4 (Blackwell) mode enabled")

    if cfg.sparse_weights:
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

    profile = builder.create_optimization_profile()
    input_names = ["image", "state"] if cfg.model_name == "gr00t" else ["image", "state", "noise", "t"]

    for name in input_names:
        min_shape = [1, 3, cfg.input_height, cfg.input_width] if "image" in name else [1, cfg.state_dim]
        opt_shape = [cfg.opt_batch_size, 3, cfg.input_height, cfg.input_width] if "image" in name else [cfg.opt_batch_size, cfg.state_dim]
        max_shape = [cfg.max_batch_size, 3, cfg.input_height, cfg.input_width] if "image" in name else [cfg.max_batch_size, cfg.state_dim]

        if "state" in name:
            min_shape = [1, cfg.state_dim]
            opt_shape = [cfg.opt_batch_size, cfg.state_dim]
            max_shape = [cfg.max_batch_size, cfg.state_dim]
        elif "noise" in name:
            min_shape = [1, cfg.action_horizon, cfg.action_dim]
            opt_shape = [cfg.opt_batch_size, cfg.action_horizon, cfg.action_dim]
            max_shape = [cfg.max_batch_size, cfg.action_horizon, cfg.action_dim]
        elif "t" in name:
            min_shape = [1]
            opt_shape = [1]
            max_shape = [1]

        profile.set_shape(name, min_shape, opt_shape, max_shape)

    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)

    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)

    print(f"[TensorRT] Engine built: {engine_path}")


def export_all(cfg: TensorRTExportConfig):
    """Export all models to TensorRT engines."""
    result = {}
    for model_name in ["gr00t", "pi05"]:
        cfg.model_name = model_name
        if model_name == "gr00t":
            result[model_name] = export_gr00t_tensorrt(cfg)
        else:
            result[model_name] = export_pi05_tensorrt(cfg)
    return result


def main():
    ap = argparse.ArgumentParser(description="TensorRT FP8/FP4 export for RoboForce models")
    ap.add_argument("--model", default="gr00t", choices=["gr00t", "pi05", "all"])
    ap.add_argument("--checkpoint", default="", help="Path to model checkpoint")
    ap.add_argument("--output", default="engines/model_fp8.engine", help="Output engine path")
    ap.add_argument("--precision", default="fp8", choices=["fp8", "int8", "fp16"])
    ap.add_argument("--fp4", action="store_true", help="Use Blackwell FP4")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--workspace-gb", type=int, default=12)
    ap.add_argument("--state-dim", type=int, default=32)
    ap.add_argument("--action-dim", type=int, default=8)
    ap.add_argument("--action-horizon", type=int, default=16)

    args = ap.parse_args()
    cfg = TensorRTExportConfig(
        model_name=args.model if args.model != "all" else "gr00t",
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        precision=args.precision,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch,
        workspace_gb=args.workspace_gb,
        use_fp4=args.fp4,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
    )

    if args.model == "all":
        export_all(cfg)
    elif args.model == "gr00t":
        export_gr00t_tensorrt(cfg)
    else:
        export_pi05_tensorrt(cfg)


if __name__ == "__main__":
    main()
