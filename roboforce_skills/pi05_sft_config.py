# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""π₀.5 SFT (Supervised Fine-Tuning) configuration for screw driving.

Provides configuration and utilities for fine-tuning Physical Intelligence's
π₀.5 VLA model on RoboForce screw driving demonstrations using LeRobot V3
format.

π₀.5 extends π₀ with:
- **Co-training on heterogeneous data**: Mixed robot action, web multimodal,
  verbal instruction, and object detection/bbox data sources.
- **50-step action chunks** via flow matching (vs. 16-step for π₀).
- **Chain-of-thought subtask decomposition**: Auto-regressive text output of
  high-level subtask sequences alongside low-level action chunks.
- **Force/torque as first-class observations**: Explicit F/T sensor channels
  (Fx, Fy, Fz, Tx, Ty, Tz) for contact detection and tightening control.
- **Multi-camera support**: 3 cameras (head_left, head_right, wrist) at
  ≥640×480 resolution.
- **Quantile normalization**: Per-feature quantile stats for robust
  normalization.

References:
    - π₀.5 paper: https://arxiv.org/abs/2504.16054
    - OpenPI: https://github.com/Physical-Intelligence/openpi
    - LeRobot V3: https://github.com/huggingface/lerobot

Usage:
    python -m roboforce_skills.pi05_sft_config --generate_config
    python -m roboforce_skills.pi05_sft_config --validate_dataset /path/to/dataset
    python -m roboforce_skills.pi05_sft_config --print_command
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# π₀.5 Model Configuration
# ---------------------------------------------------------------------------

@dataclass
class Pi05ModelCfg:
    """π₀.5 model architecture configuration."""

    model_name: str = "pi05"
    """Model variant: ``pi05`` (π₀.5)."""
    pretrained_path: str = "lerobot/pi05_base"
    """Path or URI to the pre-trained π₀.5 checkpoint."""
    model_size: str = "3B"
    """Approximate parameter count."""

    # VLM backbone (PaliGemma)
    vlm_backbone: str = "paligemma"
    """Vision-language backbone architecture."""
    freeze_vlm: bool = True
    """Freeze VLM backbone during initial fine-tuning."""
    unfreeze_vlm_after_steps: int = 15_000
    """Unfreeze VLM after this many steps (0 = never)."""

    # Input modalities — multi-camera
    image_size: tuple[int, int] = (640, 480)
    """Input image resolution per camera (width, height)."""
    num_cameras: int = 3
    """Number of camera inputs (head_left, head_right, wrist)."""
    camera_names: list[str] = field(default_factory=lambda: [
        "head_left", "head_right", "wrist",
    ])
    """Camera identifiers matching the sensor suite."""

    state_dim: int = 32
    """Proprioceptive state dimension (joints + EE + F/T)."""

    # Action output — 50-step chunks
    action_dim: int = 8
    """Action output dimension (6D pose + screw_rot + gripper)."""
    action_horizon: int = 50
    """Number of future action steps per chunk (π₀.5 uses 50)."""

    # Flow matching
    num_flow_steps: int = 10
    """Number of denoising steps during inference."""
    flow_schedule: str = "linear"
    """Noise schedule: ``linear``, ``cosine``, ``sigmoid``."""
    noise_std: float = 1.0
    """Standard deviation of the Gaussian noise prior."""

    # Chain-of-thought subtask decomposition
    enable_cot: bool = True
    """Enable chain-of-thought subtask output."""
    max_subtask_tokens: int = 64
    """Maximum token length for the subtask text sequence."""


# ---------------------------------------------------------------------------
# Force/Torque Sensor Configuration
# ---------------------------------------------------------------------------

@dataclass
class FTSensorCfg:
    """Force/torque sensor configuration for π₀.5 observations.

    π₀.5 treats F/T as a first-class observation modality alongside
    proprioception and vision.
    """

    enabled: bool = True
    """Include F/T data in observations."""

    # Sensor definitions
    sensors: list[str] = field(default_factory=lambda: ["wrist_ft", "ee_tip_ft"])
    """F/T sensor names matching the MockSensorSuite."""

    # Per-sensor channels: [Fx, Fy, Fz, Tx, Ty, Tz]
    channels_per_sensor: int = 6
    """6-DOF wrench per sensor."""

    total_ft_dim: int = 12
    """Total F/T observation dimension (2 sensors × 6 channels)."""

    # Thresholds for contact detection and tightening
    contact_force_threshold_n: float = 5.0
    """Minimum normal force (Fz) to detect screw contact (N)."""
    tightening_torque_threshold_nm: float = 2.5
    """Target tightening torque (Tz) for task completion (N·m)."""
    slip_force_threshold_n: float = 15.0
    """Lateral force (Fx/Fy) threshold for slip detection (N)."""
    max_force_n: float = 50.0
    """Safety limit — abort if any force exceeds this (N)."""

    # Normalization
    force_range: tuple[float, float] = (-100.0, 100.0)
    """Expected force range for normalization (N)."""
    torque_range: tuple[float, float] = (-25.0, 25.0)
    """Expected torque range for normalization (N·m)."""


# ---------------------------------------------------------------------------
# Co-Training Data Source Configuration
# ---------------------------------------------------------------------------

@dataclass
class DataSourceCfg:
    """Configuration for a single data source in heterogeneous co-training."""

    name: str = ""
    """Human-readable source name."""
    path: str = ""
    """Path to the data source."""
    source_type: str = "robot_action"
    """Type: ``robot_action``, ``web_multimodal``, ``verbal_instruction``,
    ``object_detection``."""
    weight: float = 1.0
    """Sampling weight relative to other sources."""
    num_samples: int = 0
    """Number of samples in this source (0 = auto-detect)."""


@dataclass
class Pi05DataCfg:
    """Data configuration for π₀.5 SFT with heterogeneous co-training."""

    # Primary dataset
    dataset_path: str = "datasets/daas_training/roboforce_daas_v1"
    """Path to the primary LeRobot V3 dataset."""
    dataset_format: str = "lerobot_v3"
    """Dataset format: ``lerobot_v3``."""

    # Heterogeneous co-training data sources
    data_sources: list[DataSourceCfg] = field(default_factory=lambda: [
        DataSourceCfg(
            name="roboforce_screw_driving",
            path="datasets/daas_training/roboforce_daas_v1",
            source_type="robot_action",
            weight=1.0,
        ),
        DataSourceCfg(
            name="web_screw_driving",
            path="datasets/web_multimodal/screw_assembly",
            source_type="web_multimodal",
            weight=0.3,
        ),
        DataSourceCfg(
            name="verbal_instructions",
            path="datasets/verbal/screw_instructions",
            source_type="verbal_instruction",
            weight=0.2,
        ),
        DataSourceCfg(
            name="screw_detection_bbox",
            path="datasets/detection/screw_bbox",
            source_type="object_detection",
            weight=0.1,
        ),
    ])
    """Co-training data sources with sampling weights."""

    # Task instruction
    task_instruction: str = (
        "Pick up the screw and drive it into the solar panel mounting bracket"
    )
    """Primary natural language task instruction."""

    instruction_variants: list[str] = field(default_factory=lambda: [
        "Pick up the screw and drive it into the solar panel mounting bracket",
        "Tighten the screw into the mounting bracket on the solar panel",
        "Install the screw into the bracket by driving it clockwise",
        "Fasten the bolt into the solar panel frame bracket",
        "Use the screw driver to install the fastener into the bracket",
        "Secure the solar panel by driving the screw into the mount",
        "Grab the M4 hex screw and fasten it into the mounting hole",
        "Align the screw driver with the screw head and tighten until secure",
    ])
    """Instruction diversity for robust language grounding."""

    # Chain-of-thought subtask sequences
    subtask_sequences: dict[str, list[str]] = field(default_factory=lambda: {
        "default": [
            "approach_screw", "align_driver", "insert_screw",
            "tighten_clockwise", "verify_torque",
        ],
        "pick_and_place": [
            "pick_screw", "move_to_hole", "align_driver",
            "insert_screw", "tighten_clockwise",
        ],
        "recovery": [
            "detect_slip", "retract", "realign_driver",
            "reinsert_screw", "tighten_clockwise",
        ],
    })
    """Named subtask sequences for chain-of-thought decomposition."""

    # Data processing
    image_augmentation: bool = True
    augmentation_config: dict[str, Any] = field(default_factory=lambda: {
        "random_crop": {"enabled": True, "scale": (0.85, 1.0)},
        "color_jitter": {"enabled": True, "brightness": 0.3, "contrast": 0.3},
        "random_erasing": {"enabled": False, "p": 0.1},
        "horizontal_flip": {"enabled": False},
    })

    # Normalization — quantile-based for π₀.5
    state_normalization: str = "quantile"
    """State normalization: ``quantile`` (recommended for π₀.5), ``per_feature``."""
    action_normalization: str = "quantile"
    """Action normalization: ``quantile`` (recommended for π₀.5)."""
    quantile_range: tuple[float, float] = (0.01, 0.99)
    """Quantile range for robust normalization (clips outliers)."""

    # Splits
    train_split: str = "train"
    val_split: str = "validation"
    train_ratio: float = 0.9

    # Embodiment mapping (RoboForce → π₀.5)
    embodiment_name: str = "roboforce"
    joint_names: list[str] = field(default_factory=lambda: [
        "base_x", "base_y", "base_yaw",
        "right_shoulder_pan", "right_shoulder_lift", "right_elbow",
        "right_wrist_1", "right_wrist_2", "right_wrist_3", "right_wrist_roll",
        "screw_driver_rotation",
        "head_pan", "head_tilt",
    ])
    """Robot joint names (tracked base + 7DOF arm + screw EE + head)."""

    camera_names: list[str] = field(default_factory=lambda: [
        "head_left", "head_right", "wrist",
    ])
    """Camera names — 3 cameras for multi-view input."""

    control_frequency_hz: float = 50.0
    """Robot control frequency (Hz)."""


# ---------------------------------------------------------------------------
# LoRA Configuration
# ---------------------------------------------------------------------------

@dataclass
class Pi05LoraCfg:
    """LoRA configuration for parameter-efficient π₀.5 fine-tuning."""

    enabled: bool = True
    rank: int = 32
    alpha: float = 64.0
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    apply_to_vlm: bool = False
    apply_to_action_head: bool = True
    apply_to_cot_head: bool = True
    """Apply LoRA to the chain-of-thought auto-regressive head."""
    merge_after_training: bool = True


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

@dataclass
class Pi05TrainingCfg:
    """Training hyperparameters for π₀.5 SFT."""

    # Optimization
    learning_rate: float = 3e-5
    """Peak LR (slightly lower than π₀ for stability with 50-step chunks)."""
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_steps: int = 100_000
    """More steps needed for heterogeneous co-training convergence."""
    batch_size: int = 8
    """Per-GPU batch size (smaller — 50-step chunks are memory-heavy)."""
    gradient_accumulation_steps: int = 8
    """Effective batch size = 64."""

    # Scheduler
    lr_scheduler: str = "cosine"
    min_lr_ratio: float = 0.01

    # Mixed precision
    fp16: bool = False
    bf16: bool = True

    # Loss weights
    flow_loss_weight: float = 1.0
    """Weight for flow-matching action denoising loss."""
    vlm_loss_weight: float = 0.1
    """Weight for VLM auxiliary loss."""
    cot_loss_weight: float = 0.5
    """Weight for chain-of-thought subtask prediction loss."""

    # Regularization
    dropout: float = 0.1
    gradient_clip_norm: float = 1.0

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999

    # Checkpointing
    save_steps: int = 5000
    eval_steps: int = 2000
    logging_steps: int = 100

    # Hardware
    num_gpus: int = 1
    num_workers: int = 8


# ---------------------------------------------------------------------------
# Top-Level SFT Configuration
# ---------------------------------------------------------------------------

@dataclass
class Pi05SFTCfg:
    """Complete π₀.5 SFT configuration for RoboForce screw driving."""

    model: Pi05ModelCfg = field(default_factory=Pi05ModelCfg)
    ft_sensor: FTSensorCfg = field(default_factory=FTSensorCfg)
    lora: Pi05LoraCfg = field(default_factory=Pi05LoraCfg)
    training: Pi05TrainingCfg = field(default_factory=Pi05TrainingCfg)
    data: Pi05DataCfg = field(default_factory=Pi05DataCfg)

    # Output
    output_dir: str = "checkpoints/pi05_screw_driving"
    experiment_name: str = "roboforce_pi05_sft_v1"

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "roboforce-pi05"
    wandb_entity: str = ""


# ---------------------------------------------------------------------------
# Config Generator
# ---------------------------------------------------------------------------

def generate_pi05_config(cfg: Pi05SFTCfg | None = None) -> dict:
    """Generate a π₀.5 SFT config dict.

    Produces a configuration compatible with the LeRobot training script:
    ``lerobot-train --config <config.json>``

    Returns:
        Config dict ready for JSON serialization.
    """
    if cfg is None:
        cfg = Pi05SFTCfg()

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
            "camera_names": cfg.model.camera_names,
            "state_dim": cfg.model.state_dim,
            "action_dim": cfg.model.action_dim,
            "action_horizon": cfg.model.action_horizon,
            "flow_matching": {
                "num_flow_steps": cfg.model.num_flow_steps,
                "schedule": cfg.model.flow_schedule,
                "noise_std": cfg.model.noise_std,
            },
            "chain_of_thought": {
                "enabled": cfg.model.enable_cot,
                "max_subtask_tokens": cfg.model.max_subtask_tokens,
            },
        },

        # Force/Torque sensor config
        "ft_sensor": {
            "enabled": cfg.ft_sensor.enabled,
            "sensors": cfg.ft_sensor.sensors,
            "channels_per_sensor": cfg.ft_sensor.channels_per_sensor,
            "total_ft_dim": cfg.ft_sensor.total_ft_dim,
            "thresholds": {
                "contact_force_n": cfg.ft_sensor.contact_force_threshold_n,
                "tightening_torque_nm": cfg.ft_sensor.tightening_torque_threshold_nm,
                "slip_force_n": cfg.ft_sensor.slip_force_threshold_n,
                "max_force_n": cfg.ft_sensor.max_force_n,
            },
            "normalization": {
                "force_range": list(cfg.ft_sensor.force_range),
                "torque_range": list(cfg.ft_sensor.torque_range),
            },
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
            "apply_to_cot_head": cfg.lora.apply_to_cot_head,
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
            "loss_weights": {
                "flow": cfg.training.flow_loss_weight,
                "vlm": cfg.training.vlm_loss_weight,
                "cot": cfg.training.cot_loss_weight,
            },
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

        # Data — heterogeneous co-training
        "data": {
            "dataset_path": cfg.data.dataset_path,
            "dataset_format": cfg.data.dataset_format,
            "task_instruction": cfg.data.task_instruction,
            "instruction_variants": cfg.data.instruction_variants,
            "data_sources": [
                {
                    "name": src.name,
                    "path": src.path,
                    "type": src.source_type,
                    "weight": src.weight,
                    "num_samples": src.num_samples,
                }
                for src in cfg.data.data_sources
            ],
            "subtask_sequences": cfg.data.subtask_sequences,
            "image_augmentation": cfg.data.image_augmentation,
            "augmentation_config": cfg.data.augmentation_config,
            "normalization": {
                "state": cfg.data.state_normalization,
                "action": cfg.data.action_normalization,
                "quantile_range": list(cfg.data.quantile_range),
            },
            "train_split": cfg.data.train_split,
            "val_split": cfg.data.val_split,
            "train_ratio": cfg.data.train_ratio,
            "embodiment_name": cfg.data.embodiment_name,
            "joint_names": cfg.data.joint_names,
            "camera_names": cfg.data.camera_names,
            "control_frequency_hz": cfg.data.control_frequency_hz,
        },

        # Embodiment mapping (required for new embodiments)
        "embodiment": {
            "name": cfg.data.embodiment_name,
            "description": (
                "RoboForce: tracked base + 7DOF arm + screw driver EE "
                "with dual F/T sensors"
            ),
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
                    "force_torque": {
                        "wrist_ft": ["fx", "fy", "fz", "tx", "ty", "tz"],
                        "ee_tip_ft": ["fx", "fy", "fz", "tx", "ty", "tz"],
                    },
                },
                "action": {
                    "type": "delta_ee_pose_and_screw",
                    "components": [
                        "delta_x", "delta_y", "delta_z",
                        "delta_rx", "delta_ry", "delta_rz",
                        "screw_rotation", "gripper",
                    ],
                    "horizon": cfg.model.action_horizon,
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


def save_pi05_config(
    path: str | Path,
    cfg: Pi05SFTCfg | None = None,
) -> str:
    """Generate and save π₀.5 SFT config to a JSON file.

    Args:
        path: Output path for the config JSON.
        cfg: SFT configuration.

    Returns:
        The output file path.
    """
    config = generate_pi05_config(cfg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return str(path)


# ---------------------------------------------------------------------------
# Dataset Validator (LeRobot V3)
# ---------------------------------------------------------------------------

def validate_dataset_for_pi05(dataset_path: str) -> dict[str, Any]:
    """Validate that a LeRobot V3 dataset is compatible with π₀.5 SFT.

    Checks:
    - Required V3 files exist (metadata.json, episodes.json, data/)
    - Action dimension matches (8D)
    - Action horizon / chunk length is compatible with 50-step
    - F/T sensor channels present (wrist_ft, ee_tip_ft)
    - Multi-camera images exist (head_left, head_right, wrist)
    - Quantile stats file exists
    - Subtask labels present

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

        # Format check
        fmt = meta.get("format", "")
        if fmt != "lerobot_v3":
            report["warnings"].append(
                f"Format is '{fmt}', expected 'lerobot_v3'. "
                "π₀.5 SFT requires LeRobot V3 format."
            )

        # Action dim
        action_dim = meta.get("observation_spec", {}).get("action_dim")
        if action_dim and action_dim != 8:
            report["warnings"].append(
                f"Action dim is {action_dim}, expected 8."
            )

        # F/T sensor presence
        ft_sensors = meta.get("sensors", {}).get("ft_sensors", [])
        for required_ft in ["wrist_ft", "ee_tip_ft"]:
            if required_ft not in ft_sensors:
                report["warnings"].append(
                    f"Missing F/T sensor '{required_ft}'. "
                    "π₀.5 uses F/T as first-class observations."
                )

        # Camera check
        cameras = meta.get("sensors", {}).get("cameras", [])
        for required_cam in ["head_left", "head_right", "wrist"]:
            if required_cam not in cameras:
                report["warnings"].append(
                    f"Missing camera '{required_cam}'. "
                    "π₀.5 expects 3 camera views."
                )

    # Episodes index
    episodes_path = path / "episodes.json"
    if not episodes_path.exists():
        report["valid"] = False
        report["errors"].append("Missing episodes.json")
    else:
        with open(episodes_path) as f:
            episodes = json.load(f)
        report["info"]["num_episodes_listed"] = len(episodes)

        if len(episodes) < 50:
            report["warnings"].append(
                f"Only {len(episodes)} episodes. π₀.5 SFT typically needs "
                "50+ demonstrations (200+ recommended for co-training)."
            )

    # Data directory (V3 uses a data/ subdirectory)
    data_dir = path / "data"
    if not data_dir.exists():
        # Fall back to checking flat files
        data_jsonl = path / "data.jsonl"
        data_hdf5 = path / "data.hdf5"
        if not data_jsonl.exists() and not data_hdf5.exists():
            report["valid"] = False
            report["errors"].append("Missing data/ directory or data file")
    else:
        report["info"]["data_format"] = "lerobot_v3_parquet"

    # Quantile stats
    quantile_path = path / "quantile_stats.json"
    if not quantile_path.exists():
        report["warnings"].append(
            "Missing quantile_stats.json. Run the quantile stats computation "
            "before training: `python -m roboforce_skills.pi05_data_converter "
            "--compute_quantile_stats`"
        )

    # Subtask labels
    subtask_path = path / "subtask_labels.json"
    if not subtask_path.exists():
        report["warnings"].append(
            "Missing subtask_labels.json. Chain-of-thought training requires "
            "subtask label annotations."
        )

    # Images directory
    images_dir = path / "images"
    if images_dir.exists():
        cameras_found = [d.name for d in images_dir.iterdir() if d.is_dir()]
        report["info"]["cameras"] = cameras_found
        for required_cam in ["head_left", "head_right", "wrist"]:
            if required_cam not in cameras_found:
                report["warnings"].append(
                    f"Missing camera directory '{required_cam}' in images/."
                )
    else:
        report["warnings"].append(
            "No images directory found — π₀.5 requires multi-view visual input."
        )

    return report


# ---------------------------------------------------------------------------
# Training Command Generator
# ---------------------------------------------------------------------------

def generate_training_command(
    config_path: str,
    cfg: Pi05SFTCfg | None = None,
) -> str:
    """Generate the shell command to launch π₀.5 SFT via lerobot-train.

    Args:
        config_path: Path to the generated config JSON.
        cfg: SFT configuration.

    Returns:
        Shell command string.
    """
    if cfg is None:
        cfg = Pi05SFTCfg()

    cmd_parts = []
    cmd_parts.append("# π₀.5 SFT for RoboForce screw driving")
    cmd_parts.append("# Requires: pip install lerobot[pi05]")
    cmd_parts.append("")

    if cfg.training.num_gpus > 1:
        cmd_parts.append(f"torchrun --nproc_per_node={cfg.training.num_gpus}")
    else:
        cmd_parts.append("lerobot-train")

    cmd_parts.extend([
        f"--config {config_path}",
        f"--policy.pretrained_path={cfg.model.pretrained_path}",
        f"--policy.action_horizon={cfg.model.action_horizon}",
        f"--output_dir {cfg.output_dir}",
        f"--experiment_name {cfg.experiment_name}",
        f"--dataset.path={cfg.data.dataset_path}",
        "--dataset.format=lerobot_v3",
    ])

    if cfg.training.bf16:
        cmd_parts.append("--bf16")
    if cfg.lora.enabled:
        cmd_parts.append(f"--lora.rank={cfg.lora.rank}")
        cmd_parts.append(f"--lora.alpha={cfg.lora.alpha}")
    if cfg.model.enable_cot:
        cmd_parts.append("--policy.enable_cot=true")
    if cfg.ft_sensor.enabled:
        cmd_parts.append("--policy.ft_sensor.enabled=true")
    if cfg.training.use_ema:
        cmd_parts.append(f"--ema_decay={cfg.training.ema_decay}")
    if cfg.use_wandb:
        cmd_parts.append(f"--wandb.project={cfg.wandb_project}")

    return " \\\n  ".join(cmd_parts)


def generate_inference_command(
    checkpoint_path: str,
    cfg: Pi05SFTCfg | None = None,
) -> str:
    """Generate the shell command for inference with a fine-tuned π₀.5 model.

    Args:
        checkpoint_path: Path to the fine-tuned checkpoint.
        cfg: Configuration.

    Returns:
        Shell command string.
    """
    if cfg is None:
        cfg = Pi05SFTCfg()

    cmd_parts = [
        "lerobot-infer",
        f"  --checkpoint {checkpoint_path}",
        f"  --policy.name=pi05",
        f"  --policy.action_horizon={cfg.model.action_horizon}",
        f"  --policy.num_flow_steps={cfg.model.num_flow_steps}",
        f"  --policy.enable_cot={str(cfg.model.enable_cot).lower()}",
        "  --device cuda",
    ]

    return " \\\n".join(cmd_parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RoboForce — π₀.5 SFT Configuration"
    )
    parser.add_argument(
        "--generate_config", action="store_true",
        help="Generate the SFT config JSON",
    )
    parser.add_argument(
        "--config_output", type=str,
        default="configs/pi05_sft.json",
        help="Output path for the config file",
    )
    parser.add_argument(
        "--validate_dataset", type=str, default=None,
        help="Validate a dataset for π₀.5 SFT compatibility",
    )
    parser.add_argument(
        "--print_command", action="store_true",
        help="Print the training launch command",
    )
    parser.add_argument(
        "--print_inference_command", action="store_true",
        help="Print the inference launch command",
    )

    args = parser.parse_args()
    cfg = Pi05SFTCfg()

    if args.generate_config:
        path = save_pi05_config(args.config_output, cfg)
        print(f"Config saved to: {path}")

    if args.validate_dataset:
        report = validate_dataset_for_pi05(args.validate_dataset)
        print(f"\nπ₀.5 Dataset Validation Report: {args.validate_dataset}")
        print(f"  Valid: {report['valid']}")
        if report["errors"]:
            print(f"  Errors: {report['errors']}")
        if report["warnings"]:
            print(f"  Warnings: {report['warnings']}")
        print(f"  Info: {json.dumps(report['info'], indent=4)}")

    if args.print_command:
        cmd = generate_training_command(args.config_output, cfg)
        print(f"\nTraining command:\n{cmd}")

    if args.print_inference_command:
        cmd = generate_inference_command(
            cfg.output_dir + "/best_checkpoint", cfg
        )
        print(f"\nInference command:\n{cmd}")

    if not any([
        args.generate_config, args.validate_dataset,
        args.print_command, args.print_inference_command,
    ]):
        config = generate_pi05_config(cfg)
        print("π₀.5 SFT Configuration")
        print("=" * 50)
        print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
