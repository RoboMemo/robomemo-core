"""
OpenPI Config Generator
=======================
生成 π₀.5 SFT 训练配置文件。
从 demo_screw_to_pi05_sft.py 提取。
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any


@dataclass
class OpenPIModelCfg:
    """模型配置"""
    model_name: str = "pi05"
    pretrained_path: str = "s3://openpi-assets/checkpoints/pi0_base"
    vlm_backbone: str = "paligemma"
    freeze_vlm: bool = True
    unfreeze_vlm_after_steps: int = 10_000
    image_size: tuple = (224, 224)
    num_cameras: int = 1
    state_dim: int = 32
    action_dim: int = 8  # 6D pose + screw_rot + gripper
    action_horizon: int = 16
    num_flow_steps: int = 10
    flow_schedule: str = "linear"


@dataclass
class OpenPILoraCfg:
    """LoRA配置"""
    enabled: bool = True
    rank: int = 32
    alpha: float = 64.0
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    apply_to_vlm: bool = False
    apply_to_action_head: bool = True
    merge_after_training: bool = True


@dataclass
class OpenPITrainingCfg:
    """训练配置"""
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1_000
    max_steps: int = 80_000
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    lr_scheduler: str = "cosine"
    bf16: bool = True
    flow_loss_weight: float = 1.0
    vlm_loss_weight: float = 0.1
    gradient_clip_norm: float = 1.0
    use_ema: bool = True
    ema_decay: float = 0.9999
    save_steps: int = 5_000
    eval_steps: int = 2_000
    logging_steps: int = 100
    num_gpus: int = 1


@dataclass
class OpenPIDataCfg:
    """数据配置"""
    dataset_path: str = "sft_output/lerobot"
    dataset_format: str = "lerobot_v2"
    task_instruction: str = "Pick up the screw and drive it into the mounting bracket"
    instruction_variants: List[str] = field(default_factory=lambda: [
        "Pick up the screw and drive it into the mounting bracket",
        "Tighten the screw into the bracket",
        "Install the screw by driving it clockwise",
    ])
    image_augmentation: bool = True
    state_normalization: str = "per_feature"
    action_normalization: str = "per_feature"
    train_ratio: float = 0.9
    embodiment_name: str = "roboforce"
    camera_names: List[str] = field(default_factory=lambda: ["head_rgb"])
    control_frequency_hz: float = 50.0


@dataclass
class OpenPIFinetuneCfg:
    """完整配置"""
    model: OpenPIModelCfg = field(default_factory=OpenPIModelCfg)
    lora: OpenPILoraCfg = field(default_factory=OpenPILoraCfg)
    training: OpenPITrainingCfg = field(default_factory=OpenPITrainingCfg)
    data: OpenPIDataCfg = field(default_factory=OpenPIDataCfg)
    output_dir: str = "checkpoints/openpi_screw_driving"
    experiment_name: str = "roboforce_pi05_screw_v1"
    use_wandb: bool = True
    wandb_project: str = "roboforce-openpi"


def generate_openpi_config(cfg: OpenPIFinetuneCfg) -> Dict[str, Any]:
    """生成配置字典"""
    return {
        "model": {
            "name": cfg.model.model_name,
            "pretrained_path": cfg.model.pretrained_path,
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
            },
        },
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
        "training": {
            "learning_rate": cfg.training.learning_rate,
            "weight_decay": cfg.training.weight_decay,
            "warmup_steps": cfg.training.warmup_steps,
            "max_steps": cfg.training.max_steps,
            "batch_size": cfg.training.batch_size,
            "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
            "lr_scheduler": cfg.training.lr_scheduler,
            "bf16": cfg.training.bf16,
            "flow_loss_weight": cfg.training.flow_loss_weight,
            "vlm_loss_weight": cfg.training.vlm_loss_weight,
            "gradient_clip_norm": cfg.training.gradient_clip_norm,
            "use_ema": cfg.training.use_ema,
            "ema_decay": cfg.training.ema_decay,
            "save_steps": cfg.training.save_steps,
            "eval_steps": cfg.training.eval_steps,
            "logging_steps": cfg.training.logging_steps,
            "num_gpus": cfg.training.num_gpus,
        },
        "data": {
            "dataset_path": cfg.data.dataset_path,
            "dataset_format": cfg.data.dataset_format,
            "task_instruction": cfg.data.task_instruction,
            "instruction_variants": cfg.data.instruction_variants,
            "image_augmentation": cfg.data.image_augmentation,
            "state_normalization": cfg.data.state_normalization,
            "action_normalization": cfg.data.action_normalization,
            "train_ratio": cfg.data.train_ratio,
            "embodiment_name": cfg.data.embodiment_name,
            "camera_names": cfg.data.camera_names,
            "control_frequency_hz": cfg.data.control_frequency_hz,
        },
        "embodiment": {
            "name": cfg.data.embodiment_name,
            "modality": {
                "video": {
                    "cameras": cfg.data.camera_names,
                    "resolution": list(cfg.model.image_size),
                    "fps": cfg.data.control_frequency_hz,
                },
                "action": {
                    "type": "delta_ee_pose_and_screw",
                    "components": [
                        "delta_x", "delta_y", "delta_z",
                        "delta_rx", "delta_ry", "delta_rz",
                        "screw_rotation", "gripper",
                    ],
                    "dim": cfg.model.action_dim,
                },
            },
        },
        "output_dir": cfg.output_dir,
        "experiment_name": cfg.experiment_name,
        "wandb": {
            "enabled": cfg.use_wandb,
            "project": cfg.wandb_project,
        },
    }


def save_openpi_config(output_dir: str, cfg: OpenPIFinetuneCfg) -> str:
    """保存配置到文件"""
    config = generate_openpi_config(cfg)
    config_path = Path(output_dir) / "configs" / "openpi_finetune.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 生成训练启动脚本
    cmd = (
        f"python -m openpi.training.train \\\n"
        f"  --config {config_path} \\\n"
        f"  --output_dir {cfg.output_dir} \\\n"
        f"  --experiment_name {cfg.experiment_name} \\\n"
        f"  --bf16 \\\n"
        f"  --lora_rank {cfg.lora.rank} \\\n"
        f"  --lora_alpha {cfg.lora.alpha}"
    )
    
    cmd_path = Path(output_dir) / "configs" / "launch_training.sh"
    with open(cmd_path, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n# Auto-generated by RoboMemo SFT Pipeline\n\n")
        f.write(cmd + "\n")

    return str(config_path)
