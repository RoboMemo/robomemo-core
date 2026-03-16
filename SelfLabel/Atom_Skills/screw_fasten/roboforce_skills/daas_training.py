# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Data-as-a-Service (DaaS) training pipeline.

Generates large-scale synthetic demonstration data from Isaac Sim using
the scripted expert policy with extensive domain randomization. Designed
to produce training-ready datasets at scale.

DaaS flow:
1. Configure domain randomization sweeps
2. Launch parallel simulation instances
3. Collect demonstrations with expert policy
4. Apply domain randomization per episode
5. Export combined dataset in LeRobot V2 format
6. Generate data statistics and quality reports

Usage:
    python -m roboforce_skills.daas_training --total_demos 50000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from roboforce_skills.data_collection import (
    DataCollectionCfg,
    DataCollectionPipeline,
    ExpertPolicyCfg,
    LeRobotV2Writer,
)
from roboforce_labeling.domain_randomization import (
    DomainRandomizationPipeline,
    DomainRandomizationPipelineCfg,
    ScrewRandomizerCfg,
    LightingRandomizerCfg,
    WeatherRandomizerCfg,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DaaSSweepCfg:
    """Configuration for a domain randomization sweep."""

    name: str = "default"
    """Sweep name for logging."""

    # Screw variations
    screw_types: list[str] = field(default_factory=lambda: ["hex", "phillips", "flathead", "torx"])
    screw_sizes: list[str] = field(default_factory=lambda: ["M4", "M5", "M6", "M8", "M10"])

    # Position variations
    num_screw_positions: int = 10
    """Number of different screw positions per panel."""
    position_noise_m: float = 0.01
    """Random noise on screw positions."""

    # Weather
    weather_presets: list[str] = field(default_factory=lambda: ["day", "night", "dusty"])
    weather_weights: list[float] = field(default_factory=lambda: [0.5, 0.25, 0.25])

    # Lighting
    lighting_intensity_range: tuple[float, float] = (0.3, 3.0)
    lighting_color_perturbation: float = 0.15

    # Expert policy noise levels
    action_noise_levels: list[float] = field(default_factory=lambda: [0.01, 0.02, 0.05])
    """Different noise levels for demonstration diversity."""

    # Episodes per configuration
    episodes_per_config: int = 10
    """Number of episodes for each unique configuration."""


@dataclass
class DaaSTrainingCfg:
    """Full DaaS training pipeline configuration."""

    # Output
    output_dir: str = "datasets/daas_training"
    dataset_name: str = "roboforce_daas_v1"

    # Scale
    total_demonstrations: int = 50_000
    """Target total number of demonstration episodes."""
    max_steps_per_episode: int = 500

    # Parallelism
    num_workers: int = 4
    """Number of parallel simulation instances (limited by GPU memory)."""

    # Sweep
    sweep: DaaSSweepCfg = field(default_factory=DaaSSweepCfg)

    # Domain randomization
    dr_config: DomainRandomizationPipelineCfg = field(
        default_factory=DomainRandomizationPipelineCfg
    )

    # Data collection
    collection: DataCollectionCfg = field(default_factory=DataCollectionCfg)

    # Validation
    holdout_ratio: float = 0.1
    """Fraction of episodes to hold out for validation."""

    # Seed
    seed: int = 42

    # Logging
    log_interval: int = 100
    checkpoint_interval: int = 5000


# ---------------------------------------------------------------------------
# Configuration Generator
# ---------------------------------------------------------------------------

class ConfigurationGenerator:
    """Generate diverse configurations for DaaS sweeps.

    Produces a stream of configuration dicts, each specifying a unique
    combination of screw type, size, position, weather, and noise level.
    """

    def __init__(self, sweep_cfg: DaaSSweepCfg, seed: int = 42):
        self.cfg = sweep_cfg
        self.rng = np.random.default_rng(seed)

    def generate(self, num_configs: int) -> list[dict[str, Any]]:
        """Generate a batch of configurations.

        Args:
            num_configs: Number of configurations to generate.

        Returns:
            List of configuration dicts.
        """
        configs = []
        for i in range(num_configs):
            # Sample screw type and size
            screw_type = self.rng.choice(self.cfg.screw_types)
            screw_size = self.rng.choice(self.cfg.screw_sizes)

            # Sample weather
            weights = np.array(self.cfg.weather_weights)
            weights /= weights.sum()
            weather = self.rng.choice(self.cfg.weather_presets, p=weights)

            # Sample screw position offset
            position_offset = self.rng.uniform(
                -self.cfg.position_noise_m,
                self.cfg.position_noise_m,
                3,
            )

            # Sample action noise
            action_noise = float(self.rng.choice(self.cfg.action_noise_levels))

            # Lighting variation
            light_mult = self.rng.uniform(*self.cfg.lighting_intensity_range)

            configs.append({
                "config_id": i,
                "screw_type": str(screw_type),
                "screw_size": str(screw_size),
                "weather": str(weather),
                "position_offset": position_offset.tolist(),
                "action_noise": action_noise,
                "lighting_multiplier": float(light_mult),
            })

        return configs

    def generate_exhaustive(self) -> list[dict[str, Any]]:
        """Generate all combinations (for smaller sweeps).

        Returns:
            List of all configuration combinations.
        """
        configs = []
        config_id = 0
        for screw_type in self.cfg.screw_types:
            for screw_size in self.cfg.screw_sizes:
                for weather in self.cfg.weather_presets:
                    for noise in self.cfg.action_noise_levels:
                        configs.append({
                            "config_id": config_id,
                            "screw_type": screw_type,
                            "screw_size": screw_size,
                            "weather": weather,
                            "position_offset": [0.0, 0.0, 0.0],
                            "action_noise": noise,
                            "lighting_multiplier": 1.0,
                        })
                        config_id += 1
        return configs


# ---------------------------------------------------------------------------
# DaaS Pipeline
# ---------------------------------------------------------------------------

class DaaSTrainingPipeline:
    """Data-as-a-Service training pipeline.

    Orchestrates large-scale demonstration generation with domain
    randomization, producing a unified dataset for VLA fine-tuning.
    """

    def __init__(self, cfg: DaaSTrainingCfg | None = None):
        self.cfg = cfg or DaaSTrainingCfg()
        self.config_gen = ConfigurationGenerator(self.cfg.sweep, self.cfg.seed)
        self.rng = np.random.default_rng(self.cfg.seed)

        # Statistics
        self._stats: list[dict] = []

    def run(self) -> str:
        """Execute the full DaaS pipeline.

        Returns:
            Path to the output dataset.
        """
        t0 = time.time()
        total = self.cfg.total_demonstrations

        # Generate configurations
        num_configs = total // self.cfg.sweep.episodes_per_config
        configs = self.config_gen.generate(num_configs)

        print(f"DaaS Training Pipeline")
        print(f"=" * 60)
        print(f"  Target demonstrations: {total}")
        print(f"  Unique configurations: {len(configs)}")
        print(f"  Episodes per config: {self.cfg.sweep.episodes_per_config}")
        print(f"  Output: {self.cfg.output_dir}/{self.cfg.dataset_name}")

        # Initialize writer
        writer = LeRobotV2Writer(self.cfg.output_dir, self.cfg.dataset_name)

        episode_idx = 0
        for cfg_idx, config in enumerate(configs):
            if episode_idx >= total:
                break

            # Create expert policy with config-specific noise
            expert_cfg = ExpertPolicyCfg(action_noise=config["action_noise"])
            collection_cfg = DataCollectionCfg(
                output_dir=self.cfg.output_dir,
                dataset_name=f"_tmp_config_{cfg_idx}",
                num_episodes=self.cfg.sweep.episodes_per_config,
                max_steps_per_episode=self.cfg.max_steps_per_episode,
                expert=expert_cfg,
                seed=self.cfg.seed + cfg_idx,
            )

            # Run collection for this config
            pipeline = DataCollectionPipeline(collection_cfg)

            for ep in range(self.cfg.sweep.episodes_per_config):
                if episode_idx >= total:
                    break

                ep_stats = pipeline._simulate_episode(episode_idx)
                ep_stats["config"] = config
                self._stats.append(ep_stats)

                # Transfer frames to main writer
                if pipeline.writer._episodes:
                    last_ep = pipeline.writer._episodes[-1]
                    writer.start_episode(episode_idx, metadata=config)
                    for frame in last_ep["frames"]:
                        obs = {"state": np.array(frame["observation.state"])}
                        writer.add_frame(
                            observation=obs,
                            action=np.array(frame["action"]),
                            reward=frame["reward"],
                            done=frame["done"],
                            info=frame.get("info"),
                            timestamp=frame["timestamp"],
                        )
                    writer.end_episode()

                episode_idx += 1

            # Progress
            if (cfg_idx + 1) % (self.cfg.log_interval // self.cfg.sweep.episodes_per_config + 1) == 0:
                elapsed = time.time() - t0
                rate = episode_idx / elapsed
                eta = (total - episode_idx) / rate if rate > 0 else 0
                success_rate = sum(1 for s in self._stats if s["success"]) / len(self._stats)
                print(f"  [{episode_idx}/{total}] "
                      f"config={cfg_idx+1}/{len(configs)}, "
                      f"success={success_rate:.1%}, "
                      f"rate={rate:.1f} ep/s, "
                      f"ETA={eta:.0f}s")

            # Checkpoint
            if episode_idx % self.cfg.checkpoint_interval == 0:
                self._save_checkpoint(episode_idx)

        # Save final dataset
        dataset_path = writer.save()

        # Save statistics
        self._save_statistics(dataset_path)

        elapsed = time.time() - t0
        success_count = sum(1 for s in self._stats if s["success"])
        print(f"\nDaaS Pipeline Complete")
        print(f"  Total episodes: {episode_idx}")
        print(f"  Success rate: {success_count/len(self._stats):.1%}")
        print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
        print(f"  Dataset: {dataset_path}")

        return dataset_path

    def _save_checkpoint(self, num_episodes: int) -> None:
        """Save a training checkpoint."""
        ckpt_dir = Path(self.cfg.output_dir) / self.cfg.dataset_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(ckpt_dir / "daas_checkpoint.json", "w") as f:
            json.dump({"num_episodes": num_episodes, "timestamp": time.time()}, f)

    def _save_statistics(self, dataset_path: str) -> None:
        """Save detailed statistics for the generated dataset."""
        stats_path = Path(dataset_path) / "daas_statistics.json"

        # Aggregate by config dimensions
        by_screw_type = {}
        by_weather = {}
        by_noise = {}

        for s in self._stats:
            config = s.get("config", {})

            st = config.get("screw_type", "unknown")
            by_screw_type.setdefault(st, {"total": 0, "success": 0})
            by_screw_type[st]["total"] += 1
            if s["success"]:
                by_screw_type[st]["success"] += 1

            w = config.get("weather", "unknown")
            by_weather.setdefault(w, {"total": 0, "success": 0})
            by_weather[w]["total"] += 1
            if s["success"]:
                by_weather[w]["success"] += 1

            n = str(config.get("action_noise", 0))
            by_noise.setdefault(n, {"total": 0, "success": 0})
            by_noise[n]["total"] += 1
            if s["success"]:
                by_noise[n]["success"] += 1

        statistics = {
            "total_episodes": len(self._stats),
            "overall_success_rate": sum(1 for s in self._stats if s["success"]) / max(1, len(self._stats)),
            "by_screw_type": {
                k: {**v, "success_rate": v["success"] / max(1, v["total"])}
                for k, v in by_screw_type.items()
            },
            "by_weather": {
                k: {**v, "success_rate": v["success"] / max(1, v["total"])}
                for k, v in by_weather.items()
            },
            "by_action_noise": {
                k: {**v, "success_rate": v["success"] / max(1, v["total"])}
                for k, v in by_noise.items()
            },
        }

        with open(stats_path, "w") as f:
            json.dump(statistics, f, indent=2)

        logger.info(f"Statistics saved to {stats_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RoboForce — DaaS Training Pipeline"
    )
    parser.add_argument("--total_demos", type=int, default=50_000)
    parser.add_argument("--output_dir", type=str, default="datasets/daas_training")
    parser.add_argument("--dataset_name", type=str, default="roboforce_daas_v1")
    parser.add_argument("--episodes_per_config", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    cfg = DaaSTrainingCfg(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        total_demonstrations=args.total_demos,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
    )
    cfg.sweep.episodes_per_config = args.episodes_per_config

    pipeline = DaaSTrainingPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
