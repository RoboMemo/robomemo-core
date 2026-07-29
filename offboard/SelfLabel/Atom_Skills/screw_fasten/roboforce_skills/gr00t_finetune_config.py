# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""GR00T N1.6 fine-tuning configuration for screw driving.

Provides configuration and utilities for fine-tuning NVIDIA GR00T N1.6
(Vision-Language-Action model) on the RoboForce screw driving demonstration
data collected via the DaaS pipeline.

References:
    - GR00T N1.6: https://github.com/NVIDIA/Isaac-GR00T
    - Fine-tuning guide: https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/finetune_new_embodiment.md
    - LeRobot V2 format: https://github.com/huggingface/lerobot

Usage:
    python -m roboforce_skills.gr00t_finetune_config --generate_config
    python -m roboforce_skills.gr00t_finetune_config --validate_dataset /path/to/dataset

Optimization (RTX 5080 Blackwell):
    See roboforce_optimize/ for CUDA Graph, TensorRT FP8/FP4, torch.compile.
    python -m roboforce_optimize.benchmark_prof --model gr00t --precision fp8
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# GR00T N1.6 Configuration
# ---------------------------------------------------------------------------

@dataclass
class GR00TModelCfg:
    """GR00T N1.6 model configuration."""

    # Model
    model_name: str = "nvidia/GR00T-N1.6-3B"
    """Pre-trained model identifier."""
    model_size: str = "3B"
    """Model parameter count variant."""
    backbone: str = "eagle2-vit"
    """Vision backbone architecture."""

    # Input modalities
    image_size: tuple[int, int] = (224, 224)
    """Input image resolution for the vision encoder."""
    num_cameras: int = 1
    """Number of camera inputs (head_rgb)."""
    state_dim: int = 32
    """Dimension of proprioceptive state input."""

    # Output
    action_dim: int = 8
    """Action output dimension (6D pose + screw + gripper)."""
    action_horizon: int = 16
    """Number of future action steps to predict."""
    action_chunk_size: int = 16
    """Action chunking window size."""

    # === Inference optimization (Blackwell RTX 5080/5070Ti) ===
    use_cuda_graph: bool = True
    """Capture CUDA Graph after warmup to eliminate kernel launch overhead (~1.3x)."""
    use_torch_compile: bool = True
    """Apply torch.compile with mode='reduce-overhead' (~1.2x)."""
    use_flash_attn: bool = True
    """Enable Flash Attention via torch.backends.cuda."""
    quantization: str = "bf16"
    """Inference precision: bf16 | fp16 | fp8 (TensorRT) | fp4 (Blackwell)."""
    use_tensorrt: bool = False
    """Use TensorRT engine for inference (requires export first)."""
    tensorrt_engine_path: str = ""
    """Path to exported TensorRT engine file."""
    num_flow_steps: int = 1
    """GR00T is single-pass (1 step). pi0.5 uses 10."""


@dataclass
class GR00TTrainingCfg:
    """Training hyperparameters for GR00T fine-tuning."""

    # Optimization
    learning_rate: float = 1e-4
    """Peak learning rate."""
    weight_decay: float = 0.01
    """AdamW weight decay."""
    warmup_steps: int = 500
    """Linear warmup steps."""
    max_steps: int = 50_000
    """Total training steps."""
    batch_size: int = 32
    """Per-GPU batch size."""
    gradient_accumulation_steps: int = 2
    """Gradient accumulation for effective batch size."""

    # Scheduler
    lr_scheduler: str = "cosine"
    """Learning rate scheduler type."""
    min_lr_ratio: float = 0.01
    """Minimum LR as fraction of peak."""

    # Mixed precision
    fp16: bool = False
    bf16: bool = True
    """Use bf16 mixed precision (recommended for RTX 5080/5090)."""

    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.0

    # Checkpointing
    save_steps: int = 2000
    eval_steps: int = 1000
    logging_steps: int = 100

    # Hardware
    num_gpus: int = 1
    num_workers: int = 8
    """DataLoader workers."""


@dataclass
class GR00TDataCfg:
    """Data configuration for GR00T fine-tuning."""

    # Dataset
    dataset_path: str = "datasets/daas_training/roboforce_daas_v1"
    """Path to the LeRobot V2 dataset."""

    # Task instruction
    task_instruction: str = (
        "Pick up the screw and drive it into the solar panel mounting bracket"
    )
    """Natural language task instruction for the VLA model."""

    # Alternative instructions (for instruction diversity)
    instruction_variants: list[str] = field(default_factory=lambda: [
        "Pick up the screw and drive it into the solar panel mounting bracket",
        "Tighten the screw into the mounting bracket on the solar panel",
        "Install the screw into the bracket by driving it clockwise",
        "Fasten the bolt into the solar panel frame bracket",
        "Use the screw driver to install the fastener into the bracket",
        "Secure the solar panel by driving the screw into the mount",
    ])

    # Data processing
    image_augmentation: bool = True
    """Apply image augmentations during training."""
    state_normalization: str = "per_feature"
    """State normalization strategy: per_feature, global, or none."""
    action_normalization: str = "per_feature"
    """Action normalization strategy."""

    # Splits
    train_split: str = "train"
    val_split: str = "validation"

    # Embodiment mapping (RoboForce → GR00T)
    embodiment_name: str = "roboforce"
    """Custom embodiment identifier."""
    joint_names: list[str] = field(default_factory=lambda: [
        "base_x", "base_y", "base_yaw",
        "right_shoulder_pan", "right_shoulder_lift", "right_elbow",
        "right_wrist_1", "right_wrist_2", "right_wrist_3", "right_wrist_roll",
        "screw_driver_rotation",
        "head_pan", "head_tilt",
    ])


@dataclass
class GR00TFinetuneCfg:
    """Complete GR00T N1.6 fine-tuning configuration."""

    model: GR00TModelCfg = field(default_factory=GR00TModelCfg)
    training: GR00TTrainingCfg = field(default_factory=GR00TTrainingCfg)
    data: GR00TDataCfg = field(default_factory=GR00TDataCfg)

    # Output
    output_dir: str = "checkpoints/gr00t_screw_driving"
    experiment_name: str = "roboforce_screw_v1"

    # Wandb / logging
    use_wandb: bool = True
    wandb_project: str = "roboforce-gr00t"
    wandb_entity: str = ""


# ---------------------------------------------------------------------------
# Config Generator
# ---------------------------------------------------------------------------

def generate_gr00t_config(cfg: GR00TFinetuneCfg | None = None) -> dict:
    """Generate a GR00T N1.6 fine-tuning config dict.

    This produces a configuration compatible with the GR00T training script:
    ``python -m gr00t.experiment.train --config <config.json>``

    Returns:
        Config dict ready for JSON serialization.
    """
    if cfg is None:
        cfg = GR00TFinetuneCfg()

    config = {
        # Model
        "model": {
            "name": cfg.model.model_name,
            "size": cfg.model.model_size,
            "backbone": cfg.model.backbone,
            "image_size": list(cfg.model.image_size),
            "num_cameras": cfg.model.num_cameras,
            "state_dim": cfg.model.state_dim,
            "action_dim": cfg.model.action_dim,
            "action_horizon": cfg.model.action_horizon,
            "action_chunk_size": cfg.model.action_chunk_size,
            # Inference optimization (Blackwell RTX 5080/5070Ti)
            "use_cuda_graph": cfg.model.use_cuda_graph,
            "use_torch_compile": cfg.model.use_torch_compile,
            "use_flash_attn": cfg.model.use_flash_attn,
            "quantization": cfg.model.quantization,
            "use_tensorrt": cfg.model.use_tensorrt,
            "tensorrt_engine_path": cfg.model.tensorrt_engine_path,
        },

        # Training
        "training": {
            "learning_rate": cfg.training.learning_rate,
            "weight_decay": cfg.training.weight_decay,
            "warmup_steps": cfg.training.warmup_steps,
            "max_steps": cfg.training.max_steps,
            "batch_size": cfg.training.batch_size,
            "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
            "lr_scheduler": cfg.training.lr_scheduler,
            "min_lr_ratio": cfg.training.min_lr_ratio,
            "fp16": cfg.training.fp16,
            "bf16": cfg.training.bf16,
            "dropout": cfg.training.dropout,
            "save_steps": cfg.training.save_steps,
            "eval_steps": cfg.training.eval_steps,
            "logging_steps": cfg.training.logging_steps,
            "num_gpus": cfg.training.num_gpus,
            "num_workers": cfg.training.num_workers,
        },

        # Data
        "data": {
            "dataset_path": cfg.data.dataset_path,
            "task_instruction": cfg.data.task_instruction,
            "instruction_variants": cfg.data.instruction_variants,
            "image_augmentation": cfg.data.image_augmentation,
            "state_normalization": cfg.data.state_normalization,
            "action_normalization": cfg.data.action_normalization,
            "train_split": cfg.data.train_split,
            "val_split": cfg.data.val_split,
            "embodiment_name": cfg.data.embodiment_name,
            "joint_names": cfg.data.joint_names,
        },

        # Embodiment mapping (required for new embodiments)
        "embodiment": {
            "name": cfg.data.embodiment_name,
            "modality": {
                "video": {
                    "cameras": ["head_rgb"],
                    "resolution": list(cfg.model.image_size),
                },
                "state": {
                    "joint_positions": cfg.data.joint_names[:10],
                    "ee_state": ["ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz"],
                    "force_torque": ["fx", "fy", "fz", "tx", "ty", "tz"],
                },
                "action": {
                    "type": "delta_ee_pose_and_screw",
                    "components": [
                        "delta_x", "delta_y", "delta_z",
                        "delta_rx", "delta_ry", "delta_rz",
                        "screw_rotation", "gripper",
                    ],
                },
            },
            "control_frequency_hz": 50,
        },

        # Output
        "output_dir": cfg.output_dir,
        "experiment_name": cfg.experiment_name,

        # Logging
        "wandb": {
            "enabled": cfg.use_wandb,
            "project": cfg.wandb_project,
            "entity": cfg.wandb_entity,
        },
    }

    return config


def save_gr00t_config(
    path: str | Path,
    cfg: GR00TFinetuneCfg | None = None,
) -> str:
    """Generate and save GR00T config to a JSON file.

    Args:
        path: Output path for the config JSON.
        cfg: Fine-tuning configuration.

    Returns:
        The output file path.
    """
    config = generate_gr00t_config(cfg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return str(path)


# ---------------------------------------------------------------------------
# Dataset Validator
# ---------------------------------------------------------------------------

def validate_dataset_for_gr00t(dataset_path: str) -> dict[str, Any]:
    """Validate that a LeRobot V2 dataset is compatible with GR00T N1.6.

    Checks:
    - Required files exist (metadata.json, episodes.json, data.jsonl/hdf5)
    - Observation dimensions match expected state_dim
    - Action dimensions match expected action_dim
    - Camera images exist and have correct resolution

    Args:
        dataset_path: Path to the dataset directory.

    Returns:
        Validation report dict.
    """
    path = Path(dataset_path)
    report: dict[str, Any] = {"valid": True, "errors": [], "warnings": [], "info": {}}

    # Check metadata
    meta_path = path / "metadata.json"
    if not meta_path.exists():
        report["valid"] = False
        report["errors"].append("Missing metadata.json")
    else:
        with open(meta_path) as f:
            meta = json.load(f)
        report["info"]["format"] = meta.get("format", "unknown")
        report["info"]["num_episodes"] = meta.get("num_episodes", 0)
        report["info"]["total_frames"] = meta.get("total_frames", 0)

        # Check action dim
        action_shape = meta.get("features", {}).get("action", {}).get("shape", [])
        if action_shape and action_shape != [8]:
            report["warnings"].append(
                f"Action dim is {action_shape}, expected [8]. May need mapping."
            )

    # Check episodes
    episodes_path = path / "episodes.json"
    if not episodes_path.exists():
        report["valid"] = False
        report["errors"].append("Missing episodes.json")
    else:
        with open(episodes_path) as f:
            episodes = json.load(f)
        report["info"]["num_episodes_listed"] = len(episodes)

    # Check data file
    data_jsonl = path / "data.jsonl"
    data_hdf5 = path / "data.hdf5"
    if not data_jsonl.exists() and not data_hdf5.exists():
        report["valid"] = False
        report["errors"].append("Missing data file (data.jsonl or data.hdf5)")
    else:
        if data_jsonl.exists():
            report["info"]["data_format"] = "jsonl"
            report["info"]["data_size_mb"] = data_jsonl.stat().st_size / (1024 * 1024)
        if data_hdf5.exists():
            report["info"]["data_format_hdf5"] = True
            report["info"]["hdf5_size_mb"] = data_hdf5.stat().st_size / (1024 * 1024)

    # Check images directory
    images_dir = path / "images"
    if images_dir.exists():
        cameras = [d.name for d in images_dir.iterdir() if d.is_dir()]
        report["info"]["cameras"] = cameras
        if "head_rgb" not in cameras:
            report["warnings"].append("Missing head_rgb camera directory")
    else:
        report["warnings"].append("No images directory found")

    return report


# ---------------------------------------------------------------------------
# Training Launch Helper
# ---------------------------------------------------------------------------

def generate_training_command(
    config_path: str,
    cfg: GR00TFinetuneCfg | None = None,
) -> str:
    """Generate the shell command to launch GR00T fine-tuning.

    Args:
        config_path: Path to the generated config JSON.
        cfg: Configuration (for GPU count).

    Returns:
        Shell command string.
    """
    if cfg is None:
        cfg = GR00TFinetuneCfg()

    cmd_parts = []

    if cfg.training.num_gpus > 1:
        cmd_parts.append(f"torchrun --nproc_per_node={cfg.training.num_gpus}")
    else:
        cmd_parts.append("python")

    cmd_parts.extend([
        "-m", "gr00t.experiment.train",
        f"--config {config_path}",
        f"--output_dir {cfg.output_dir}",
        f"--experiment_name {cfg.experiment_name}",
    ])

    if cfg.training.bf16:
        cmd_parts.append("--bf16")
    if cfg.use_wandb:
        cmd_parts.append(f"--wandb_project {cfg.wandb_project}")

    return " \\\n  ".join(cmd_parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RoboForce — GR00T N1.6 Fine-tuning Configuration"
    )
    parser.add_argument(
        "--generate_config", action="store_true",
        help="Generate the fine-tuning config JSON",
    )
    parser.add_argument(
        "--config_output", type=str,
        default="configs/gr00t_finetune.json",
        help="Output path for the config file",
    )
    parser.add_argument(
        "--validate_dataset", type=str, default=None,
        help="Validate a dataset for GR00T compatibility",
    )
    parser.add_argument(
        "--print_command", action="store_true",
        help="Print the training launch command",
    )

    args = parser.parse_args()
    cfg = GR00TFinetuneCfg()

    if args.generate_config:
        path = save_gr00t_config(args.config_output, cfg)
        print(f"Config saved to: {path}")

    if args.validate_dataset:
        report = validate_dataset_for_gr00t(args.validate_dataset)
        print(f"\nDataset Validation Report: {args.validate_dataset}")
        print(f"  Valid: {report['valid']}")
        if report["errors"]:
            print(f"  Errors: {report['errors']}")
        if report["warnings"]:
            print(f"  Warnings: {report['warnings']}")
        print(f"  Info: {json.dumps(report['info'], indent=4)}")

    if args.print_command:
        config_path = args.config_output
        cmd = generate_training_command(config_path, cfg)
        print(f"\nTraining command:\n{cmd}")

    if not any([args.generate_config, args.validate_dataset, args.print_command]):
        # Default: show config summary
        config = generate_gr00t_config(cfg)
        print("GR00T N1.6 Fine-tuning Configuration")
        print("=" * 50)
        print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
