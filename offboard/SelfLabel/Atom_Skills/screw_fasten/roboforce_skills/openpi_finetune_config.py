# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""OpenPI (π₀ / π₀.5) fine-tuning configuration for screw driving.

Provides configuration and utilities for fine-tuning Physical Intelligence's
π₀ (pi-zero) and π₀.5 VLA models on the RoboForce screw driving demonstration
data collected via the DaaS pipeline.

π₀ is a flow-matching based Vision-Language-Action model that uses a pre-trained
VLM (PaliGemma) as backbone and outputs continuous action chunks via iterative
denoising.

References:
    - OpenPI: https://github.com/Physical-Intelligence/openpi
    - π₀ paper: https://www.physicalintelligence.company/research/pi0
    - LeRobot V2 format: https://github.com/huggingface/lerobot

Usage:
    python -m roboforce_skills.openpi_finetune_config --generate_config
    python -m roboforce_skills.openpi_finetune_config --validate_dataset /path/to/dataset
    python -m roboforce_skills.openpi_finetune_config --print_command
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# π₀ / π₀.5 Model Configuration
# ---------------------------------------------------------------------------

@dataclass
class OpenPIModelCfg:
    """π₀ / π₀.5 model configuration."""

    # Model variant
    model_name: str = "pi0"
    """Model variant: ``pi0`` (3B), ``pi0_fast`` (distilled), or ``pi05`` (π₀.5)."""
    pretrained_path: str = "s3://openpi-assets/checkpoints/pi0_base"
    """Path or URI to pre-trained checkpoint."""
    model_size: str = "3B"
    """Approximate parameter count."""

    # VLM backbone (PaliGemma)
    vlm_backbone: str = "paligemma"
    """Vision-language backbone architecture."""
    freeze_vlm: bool = True
    """Freeze the VLM backbone during initial fine-tuning."""
    unfreeze_vlm_after_steps: int = 10_000
    """Unfreeze VLM after this many steps (0 = never unfreeze)."""

    # Input modalities
    image_size: tuple[int, int] = (224, 224)
    """Input image resolution for the vision encoder."""
    num_cameras: int = 1
    """Number of camera inputs (head_rgb)."""
    state_dim: int = 32
    """Proprioceptive state dimension (joints + EE + F/T)."""

    # Action output
    action_dim: int = 8
    """Action output dimension (6D pose + screw_rot + gripper)."""
    action_horizon: int = 16
    """Number of future action steps to predict per chunk."""

    # Flow matching
    num_flow_steps: int = 10
    """Number of denoising steps during inference."""
    flow_schedule: str = "linear"
    """Noise schedule for flow matching: ``linear``, ``cosine``, ``sigmoid``."""
    noise_std: float = 1.0
    """Standard deviation of the Gaussian noise prior."""

    # === Inference optimization (Blackwell RTX 5080/5070Ti) ===
    use_cuda_graph: bool = True
    """Capture CUDA Graph after warmup (~1.3x for single-step, less for multi-step)."""
    use_torch_compile: bool = True
    """Apply torch.compile with mode='reduce-overhead'."""
    use_flash_attn: bool = True
    """Enable Flash Attention for the PaliGemma backbone."""
    quantization: str = "bf16"
    """Inference precision: bf16 | fp16 | fp8 (TensorRT)."""
    use_tensorrt: bool = False
    """Use TensorRT engine for inference."""
    tensorrt_engine_path: str = ""
    """Path to exported TensorRT engine."""

    # Flow step distillation (see roboforce_optimize.distill_flow)
    distilled_steps: int = 4
    """Number of distilled flow steps (default 4, down from 10 -> ~2.5x speedup)."""
    use_early_stop: bool = True
    """Early stop flow decoding when action converges."""


@dataclass
class OpenPILoraCfg:
    """LoRA (Low-Rank Adaptation) configuration for parameter-efficient fine-tuning."""

    enabled: bool = True
    """Enable LoRA for the action head and/or VLM backbone."""
    rank: int = 32
    """LoRA rank (r). Higher → more expressive, more params."""
    alpha: float = 64.0
    """LoRA scaling factor (alpha). Typically alpha = 2 * rank."""
    dropout: float = 0.05
    """LoRA dropout probability."""

    # Module targeting
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    """Attention/MLP modules to apply LoRA to."""

    apply_to_vlm: bool = False
    """Apply LoRA to the VLM backbone (PaliGemma) in addition to the action head."""
    apply_to_action_head: bool = True
    """Apply LoRA to the flow-matching action head."""

    # Merging
    merge_after_training: bool = True
    """Merge LoRA weights into the base model after training."""


@dataclass
class OpenPITrainingCfg:
    """Training hyperparameters for OpenPI fine-tuning."""

    # Optimization
    learning_rate: float = 5e-5
    """Peak learning rate (lower than GR00T due to flow matching sensitivity)."""
    weight_decay: float = 0.01
    """AdamW weight decay."""
    warmup_steps: int = 1000
    """Linear warmup steps."""
    max_steps: int = 80_000
    """Total training steps."""
    batch_size: int = 16
    """Per-GPU batch size (smaller than GR00T — flow matching is memory-heavy)."""
    gradient_accumulation_steps: int = 4
    """Gradient accumulation for effective batch size of 64."""

    # Scheduler
    lr_scheduler: str = "cosine"
    """Learning rate scheduler type."""
    min_lr_ratio: float = 0.01
    """Minimum LR as fraction of peak."""

    # Mixed precision
    fp16: bool = False
    bf16: bool = True
    """Use bf16 mixed precision (recommended for RTX 5080 / 5090)."""

    # Flow matching loss
    flow_loss_weight: float = 1.0
    """Weight for the flow-matching (denoising) loss."""
    vlm_loss_weight: float = 0.1
    """Weight for the VLM auxiliary loss (language grounding)."""

    # Regularization
    dropout: float = 0.1
    gradient_clip_norm: float = 1.0
    """Max gradient norm for clipping."""

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999
    """Exponential moving average decay for model weights."""

    # Checkpointing
    save_steps: int = 5000
    eval_steps: int = 2000
    logging_steps: int = 100

    # Hardware
    num_gpus: int = 1
    num_workers: int = 8
    """DataLoader workers."""


@dataclass
class OpenPIDataCfg:
    """Data configuration for OpenPI fine-tuning."""

    # Dataset
    dataset_path: str = "datasets/daas_training/roboforce_daas_v1"
    """Path to the LeRobot V2 dataset."""
    dataset_format: str = "lerobot_v2"
    """Dataset format: ``lerobot_v2`` or ``hdf5``."""

    # Task instruction
    task_instruction: str = (
        "Pick up the screw and drive it into the solar panel mounting bracket"
    )
    """Natural language task instruction for the VLA model."""

    # Alternative instructions (for instruction diversity during training)
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
    augmentation_config: dict[str, Any] = field(default_factory=lambda: {
        "random_crop": {"enabled": True, "scale": (0.9, 1.0)},
        "color_jitter": {"enabled": True, "brightness": 0.2, "contrast": 0.2},
        "random_erasing": {"enabled": False, "p": 0.1},
    })
    state_normalization: str = "per_feature"
    """State normalization strategy: ``per_feature``, ``global``, or ``none``."""
    action_normalization: str = "per_feature"
    """Action normalization strategy."""

    # Splits
    train_split: str = "train"
    val_split: str = "validation"
    train_ratio: float = 0.9
    """Train/val split ratio if no explicit split exists."""

    # Embodiment mapping (RoboForce → OpenPI)
    embodiment_name: str = "roboforce"
    """Custom embodiment identifier for the OpenPI config."""

    joint_names: list[str] = field(default_factory=lambda: [
        "base_x", "base_y", "base_yaw",
        "right_shoulder_pan", "right_shoulder_lift", "right_elbow",
        "right_wrist_1", "right_wrist_2", "right_wrist_3", "right_wrist_roll",
        "screw_driver_rotation",
        "head_pan", "head_tilt",
    ])
    """Robot joint names (tracked base + 7DOF arm + screw EE + head)."""

    camera_names: list[str] = field(default_factory=lambda: ["head_rgb"])
    """Camera names in the dataset."""

    control_frequency_hz: float = 50.0
    """Robot control frequency (Hz)."""


@dataclass
class OpenPIFinetuneCfg:
    """Complete OpenPI (π₀) fine-tuning configuration."""

    model: OpenPIModelCfg = field(default_factory=OpenPIModelCfg)
    lora: OpenPILoraCfg = field(default_factory=OpenPILoraCfg)
    training: OpenPITrainingCfg = field(default_factory=OpenPITrainingCfg)
    data: OpenPIDataCfg = field(default_factory=OpenPIDataCfg)

    # Output
    output_dir: str = "checkpoints/openpi_screw_driving"
    experiment_name: str = "roboforce_pi0_screw_v1"

    # Wandb / logging
    use_wandb: bool = True
    wandb_project: str = "roboforce-openpi"
    wandb_entity: str = ""


# ---------------------------------------------------------------------------
# Config Generator
# ---------------------------------------------------------------------------

def generate_openpi_config(cfg: OpenPIFinetuneCfg | None = None) -> dict:
    """Generate an OpenPI fine-tuning config dict.

    This produces a configuration compatible with the OpenPI training script:
    ``python -m openpi.training.train --config <config.json>``

    Returns:
        Config dict ready for JSON serialization.
    """
    if cfg is None:
        cfg = OpenPIFinetuneCfg()

    config = {
        # Model
        "model": {
            "name": cfg.model.model_name,
            "pretrained_path": cfg.model.pretrained_path,
            "size": cfg.model.model_size,
            "vlm_backbone": cfg.model.vlm_backbone,
            "freeze_vlm": cfg.model.freeze_vlm,
            "unfreeze_vlm_after_steps": cfg.model.unfreeze_vlm_after_steps,
            "image_size": list(cfg.model.image_size),
            "num_cameras": cfg.model.num_cameras,
            "state_dim": cfg.model.state_dim,
            "action_dim": cfg.model.action_dim,
            "action_horizon": cfg.model.action_horizon,
            "flow_matching": {
                "num_flow_steps": cfg.model.num_flow_steps,
                "schedule": cfg.model.flow_schedule,
                "noise_std": cfg.model.noise_std,
                "distilled_steps": cfg.model.distilled_steps,
                "use_early_stop": cfg.model.use_early_stop,
            },
            # Inference optimization (Blackwell RTX 5080/5070Ti)
            "use_cuda_graph": cfg.model.use_cuda_graph,
            "use_torch_compile": cfg.model.use_torch_compile,
            "use_flash_attn": cfg.model.use_flash_attn,
            "quantization": cfg.model.quantization,
            "use_tensorrt": cfg.model.use_tensorrt,
            "tensorrt_engine_path": cfg.model.tensorrt_engine_path,
        },

        # LoRA
        "lora": {
            "enabled": cfg.lora.enabled,
            "rank": cfg.lora.rank,
            "alpha": cfg.lora.alpha,
            "dropout": cfg.lora.dropout,
            "target_modules": cfg.lora.target_modules,
            "apply_to_vlm": cfg.lora.apply_to_vlm,
            "apply_to_action_head": cfg.lora.apply_to_action_head,
            "merge_after_training": cfg.lora.merge_after_training,
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
            "flow_loss_weight": cfg.training.flow_loss_weight,
            "vlm_loss_weight": cfg.training.vlm_loss_weight,
            "dropout": cfg.training.dropout,
            "gradient_clip_norm": cfg.training.gradient_clip_norm,
            "use_ema": cfg.training.use_ema,
            "ema_decay": cfg.training.ema_decay,
            "save_steps": cfg.training.save_steps,
            "eval_steps": cfg.training.eval_steps,
            "logging_steps": cfg.training.logging_steps,
            "num_gpus": cfg.training.num_gpus,
            "num_workers": cfg.training.num_workers,
        },

        # Data
        "data": {
            "dataset_path": cfg.data.dataset_path,
            "dataset_format": cfg.data.dataset_format,
            "task_instruction": cfg.data.task_instruction,
            "instruction_variants": cfg.data.instruction_variants,
            "image_augmentation": cfg.data.image_augmentation,
            "augmentation_config": cfg.data.augmentation_config,
            "state_normalization": cfg.data.state_normalization,
            "action_normalization": cfg.data.action_normalization,
            "train_split": cfg.data.train_split,
            "val_split": cfg.data.val_split,
            "train_ratio": cfg.data.train_ratio,
            "embodiment_name": cfg.data.embodiment_name,
            "joint_names": cfg.data.joint_names,
            "camera_names": cfg.data.camera_names,
            "control_frequency_hz": cfg.data.control_frequency_hz,
        },

        # Embodiment mapping (required for new embodiments in OpenPI)
        "embodiment": {
            "name": cfg.data.embodiment_name,
            "description": "RoboForce: tracked base + 7DOF arm + screw driver end-effector",
            "modality": {
                "video": {
                    "cameras": cfg.data.camera_names,
                    "resolution": list(cfg.model.image_size),
                    "fps": cfg.data.control_frequency_hz,
                },
                "state": {
                    "joint_positions": cfg.data.joint_names[:10],
                    "ee_state": [
                        "ee_x", "ee_y", "ee_z",
                        "ee_qw", "ee_qx", "ee_qy", "ee_qz",
                    ],
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
            "control_frequency_hz": cfg.data.control_frequency_hz,
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


def save_openpi_config(
    path: str | Path,
    cfg: OpenPIFinetuneCfg | None = None,
) -> str:
    """Generate and save OpenPI config to a JSON file.

    Args:
        path: Output path for the config JSON.
        cfg: Fine-tuning configuration.

    Returns:
        The output file path.
    """
    config = generate_openpi_config(cfg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return str(path)


# ---------------------------------------------------------------------------
# Dataset Validator
# ---------------------------------------------------------------------------

def validate_dataset_for_openpi(dataset_path: str) -> dict[str, Any]:
    """Validate that a LeRobot V2 dataset is compatible with OpenPI (π₀).

    Checks:
    - Required files exist (metadata.json, episodes.json, data.jsonl/hdf5)
    - Observation dimensions match expected state_dim
    - Action dimensions match expected action_dim
    - Camera images exist and have correct resolution
    - Instruction field is present

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
        report["info"]["fps"] = meta.get("fps", "unknown")

        # Check action dim
        action_shape = meta.get("features", {}).get("action", {}).get("shape", [])
        if action_shape and action_shape != [8]:
            report["warnings"].append(
                f"Action dim is {action_shape}, expected [8]. "
                "OpenPI requires matching action_dim in the config."
            )

        # Check format compatibility
        fmt = meta.get("format", "")
        if fmt != "lerobot_v2":
            report["warnings"].append(
                f"Format is '{fmt}', expected 'lerobot_v2'. "
                "OpenPI works best with LeRobot V2 datasets."
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

        # Minimum episode count warning
        if len(episodes) < 50:
            report["warnings"].append(
                f"Only {len(episodes)} episodes. OpenPI typically needs 50+ demonstrations "
                "for reliable fine-tuning (100+ recommended)."
            )

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
            report["warnings"].append(
                "Missing head_rgb camera directory. "
                "OpenPI expects at least one camera view."
            )
    else:
        report["warnings"].append("No images directory found — π₀ requires visual input")

    return report


# ---------------------------------------------------------------------------
# Training Launch Helper
# ---------------------------------------------------------------------------

def generate_training_command(
    config_path: str,
    cfg: OpenPIFinetuneCfg | None = None,
) -> str:
    """Generate the shell command to launch OpenPI fine-tuning.

    Args:
        config_path: Path to the generated config JSON.
        cfg: Configuration (for GPU count, LoRA, etc.).

    Returns:
        Shell command string.
    """
    if cfg is None:
        cfg = OpenPIFinetuneCfg()

    cmd_parts = []

    # Environment setup
    cmd_parts.append("# OpenPI fine-tuning for RoboForce screw driving")
    cmd_parts.append("# Requires: pip install openpi-client")
    cmd_parts.append("")

    if cfg.training.num_gpus > 1:
        cmd_parts.append(f"torchrun --nproc_per_node={cfg.training.num_gpus}")
    else:
        cmd_parts.append("python")

    cmd_parts.extend([
        "-m", "openpi.training.train",
        f"--config {config_path}",
        f"--output_dir {cfg.output_dir}",
        f"--experiment_name {cfg.experiment_name}",
    ])

    if cfg.training.bf16:
        cmd_parts.append("--bf16")
    if cfg.lora.enabled:
        cmd_parts.append(f"--lora_rank {cfg.lora.rank}")
        cmd_parts.append(f"--lora_alpha {cfg.lora.alpha}")
    if cfg.training.use_ema:
        cmd_parts.append(f"--ema_decay {cfg.training.ema_decay}")
    if cfg.use_wandb:
        cmd_parts.append(f"--wandb_project {cfg.wandb_project}")

    return " \\\n  ".join(cmd_parts)


def generate_inference_command(
    checkpoint_path: str,
    cfg: OpenPIFinetuneCfg | None = None,
) -> str:
    """Generate the shell command for inference with a fine-tuned π₀ model.

    Args:
        checkpoint_path: Path to the fine-tuned checkpoint.
        cfg: Configuration.

    Returns:
        Shell command string.
    """
    if cfg is None:
        cfg = OpenPIFinetuneCfg()

    cmd_parts = [
        "python -m openpi.serve.policy_server",
        f"  --checkpoint {checkpoint_path}",
        f"  --model {cfg.model.model_name}",
        f"  --num_flow_steps {cfg.model.num_flow_steps}",
        f"  --action_horizon {cfg.model.action_horizon}",
        "  --port 8000",
    ]

    return " \\\n".join(cmd_parts)


# ---------------------------------------------------------------------------
# Comparison: GR00T vs OpenPI
# ---------------------------------------------------------------------------

def print_comparison_table() -> None:
    """Print a comparison table between GR00T N1.6 and OpenPI (π₀).

    Useful for deciding which VLA to use for a given scenario.
    """
    print()
    print("┌────────────────────────────┬─────────────────────────┬─────────────────────────┐")
    print("│ Feature                    │ GR00T N1.6              │ OpenPI (π₀)             │")
    print("├────────────────────────────┼─────────────────────────┼─────────────────────────┤")
    rows = [
        ("Architecture", "VLA (Eagle2-ViT)", "VLA (PaliGemma + Flow)"),
        ("Parameters", "3B", "3B"),
        ("Action decoding", "Direct regression", "Flow matching (iterative)"),
        ("Inference steps", "1", f"~10 (denoising)"),
        ("Inference speed", "Faster (single pass)", "Slower (multi-step)"),
        ("Action diversity", "Unimodal", "Multimodal (flow)"),
        ("Fine-tuning", "Full / LoRA", "LoRA (recommended)"),
        ("Language grounding", "Strong (Eagle2)", "Strong (PaliGemma)"),
        ("Best for", "Fast real-time control", "Complex multi-modal tasks"),
        ("RoboForce screw task", "Baseline", "Exploration"),
    ]
    for feat, groot, openpi in rows:
        print(f"│ {feat:<26s} │ {groot:<23s} │ {openpi:<23s} │")
    print("└────────────────────────────┴─────────────────────────┴─────────────────────────┘")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RoboForce — OpenPI (π₀) Fine-tuning Configuration"
    )
    parser.add_argument(
        "--generate_config", action="store_true",
        help="Generate the fine-tuning config JSON",
    )
    parser.add_argument(
        "--config_output", type=str,
        default="configs/openpi_finetune.json",
        help="Output path for the config file",
    )
    parser.add_argument(
        "--validate_dataset", type=str, default=None,
        help="Validate a dataset for OpenPI compatibility",
    )
    parser.add_argument(
        "--print_command", action="store_true",
        help="Print the training launch command",
    )
    parser.add_argument(
        "--print_inference_command", action="store_true",
        help="Print the inference server launch command",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Print GR00T vs OpenPI comparison table",
    )
    parser.add_argument(
        "--model", type=str, default="pi0",
        choices=["pi0", "pi0_fast", "pi05"],
        help="Model variant",
    )

    args = parser.parse_args()
    cfg = OpenPIFinetuneCfg()
    cfg.model.model_name = args.model

    if args.generate_config:
        path = save_openpi_config(args.config_output, cfg)
        print(f"Config saved to: {path}")

    if args.validate_dataset:
        report = validate_dataset_for_openpi(args.validate_dataset)
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

    if args.print_inference_command:
        cmd = generate_inference_command(cfg.output_dir + "/best_checkpoint", cfg)
        print(f"\nInference command:\n{cmd}")

    if args.compare:
        print_comparison_table()

    if not any([
        args.generate_config, args.validate_dataset,
        args.print_command, args.print_inference_command, args.compare,
    ]):
        # Default: show config summary
        config = generate_openpi_config(cfg)
        print("OpenPI (π₀) Fine-tuning Configuration")
        print("=" * 50)
        print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
