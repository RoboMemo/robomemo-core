# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""End-to-end labeling pipeline: render scene → extract labels → save dataset.

Orchestrates the Isaac Sim rendering loop with domain randomization to
produce a large-scale labeled dataset for screw detection training.

Features:
- Headless GPU rendering
- Domain randomization per frame
- COCO-format output with train/val/test splits
- Progress tracking and resumability
- Configurable target count (default: 10K images per configuration)

Usage:
    python -m roboforce_labeling.label_pipeline --num_images 10000
    python -m roboforce_labeling.label_pipeline --config custom_config.yaml
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

from roboforce_labeling.screw_detector import ScrewDetectorCfg, ScrewDetectionDatasetBuilder
from roboforce_labeling.domain_randomization import (
    DomainRandomizationPipeline,
    DomainRandomizationPipelineCfg,
)

try:
    import isaaclab.sim as sim_utils
    ISAACLAB_AVAILABLE = True
except ImportError:
    ISAACLAB_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LabelPipelineCfg:
    """Configuration for the end-to-end labeling pipeline."""

    # Output
    output_dir: str = "datasets/screw_detection"
    dataset_name: str = "roboforce_screw_v1"

    # Generation targets
    num_images: int = 10_000
    """Total number of labeled images to generate."""
    images_per_config: int = 100
    """Number of images per domain randomization configuration."""
    batch_size: int = 16
    """Number of images to render in parallel (GPU batch)."""

    # Camera
    camera_width: int = 640
    camera_height: int = 480
    camera_fov_deg: float = 60.0

    # Scene
    num_screw_positions: int = 12
    """Number of screw positions on the solar rack."""

    # Rendering
    headless: bool = True
    """Run without display (headless mode)."""
    render_samples: int = 64
    """Path-tracer samples per pixel (quality vs. speed)."""
    antialiasing: bool = True

    # Domain randomization
    randomization: DomainRandomizationPipelineCfg = field(
        default_factory=DomainRandomizationPipelineCfg
    )

    # Detector config
    detector: ScrewDetectorCfg = field(default_factory=ScrewDetectorCfg)

    # Checkpoint
    checkpoint_interval: int = 1000
    """Save checkpoint every N images."""
    resume_from: str | None = None
    """Path to checkpoint file to resume from."""

    # Logging
    log_interval: int = 100
    """Log progress every N images."""


# ---------------------------------------------------------------------------
# Simulated Renderer (offline fallback)
# ---------------------------------------------------------------------------

class SimulatedRenderer:
    """Simulated renderer for offline testing without Isaac Sim.

    Generates synthetic RGB images and segmentation masks with randomly
    placed circular "screw" regions.
    """

    def __init__(self, width: int, height: int, num_screws: int, seed: int = 42):
        self.width = width
        self.height = height
        self.num_screws = num_screws
        self.rng = np.random.default_rng(seed)

    def render(self, randomization_params: dict | None = None) -> dict[str, np.ndarray]:
        """Render a synthetic frame.

        Returns:
            Dict with ``rgb``, ``semantic_mask``, ``depth`` arrays.
        """
        # Background
        bg_color = self.rng.integers(60, 120, 3).astype(np.uint8)
        rgb = np.full((self.height, self.width, 3), bg_color, dtype=np.uint8)

        # Add noise
        noise = self.rng.integers(-20, 20, rgb.shape).astype(np.int16)
        rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Semantic mask
        mask = np.zeros((self.height, self.width), dtype=np.int32)

        # Depth
        depth = np.full((self.height, self.width), 2.0, dtype=np.float32)

        # Screw positions and poses
        screw_poses = []

        for i in range(self.num_screws):
            cx = self.rng.integers(80, self.width - 80)
            cy = self.rng.integers(80, self.height - 80)
            r = self.rng.integers(5, 18)

            # Draw screw in RGB (metallic circle)
            yy, xx = np.ogrid[:self.height, :self.width]
            circle = ((xx - cx)**2 + (yy - cy)**2) <= r**2
            screw_color = self.rng.integers(150, 220, 3)
            rgb[circle] = screw_color.astype(np.uint8)

            # Semantic mask (ID = 10 for screws)
            mask[circle] = 10

            # Depth variation
            depth[circle] = 1.5 + self.rng.uniform(-0.1, 0.1)

            # Pose
            screw_poses.append({
                "position": [
                    float(cx - self.width / 2) * 0.002,
                    float(cy - self.height / 2) * 0.002,
                    1.5,
                ],
                "quaternion": [1.0, 0.0, 0.0, 0.0],
            })

        # Solar panel background region
        panel_y_start = self.height // 4
        panel_y_end = 3 * self.height // 4
        panel_x_start = self.width // 6
        panel_x_end = 5 * self.width // 6
        panel_region = np.zeros_like(mask, dtype=bool)
        panel_region[panel_y_start:panel_y_end, panel_x_start:panel_x_end] = True
        # Don't overwrite screws
        panel_region = panel_region & (mask == 0)
        rgb[panel_region] = np.array([15, 15, 60], dtype=np.uint8)  # Dark blue panel

        return {
            "rgb": rgb,
            "semantic_mask": mask,
            "depth": depth,
            "screw_poses": screw_poses,
        }


# ---------------------------------------------------------------------------
# Isaac Sim Renderer
# ---------------------------------------------------------------------------

class IsaacSimRenderer:
    """Render frames from Isaac Sim for labeling.

    Uses the ``SolarPanelEnv`` scene with domain randomization applied
    per frame. Extracts RGB, semantic segmentation, and depth images.
    """

    def __init__(self, cfg: LabelPipelineCfg):
        self.cfg = cfg
        self.width = cfg.camera_width
        self.height = cfg.camera_height
        self._initialized = False

    def initialize(self) -> None:
        """Set up Isaac Sim scene and camera."""
        if not ISAACLAB_AVAILABLE:
            raise RuntimeError("Isaac Lab not available. Use SimulatedRenderer for testing.")

        from roboforce_sim.envs.solar_panel_env import SolarPanelEnv, SolarPanelEnvCfg

        env_cfg = SolarPanelEnvCfg.from_yaml()
        env_cfg.num_envs = 1
        env_cfg.enable_camera = True
        env_cfg.camera_width = self.width
        env_cfg.camera_height = self.height

        self.env = SolarPanelEnv(env_cfg)
        self._initialized = True

    def render(self, randomization_params: dict | None = None) -> dict[str, np.ndarray]:
        """Render a frame with optional domain randomization.

        Args:
            randomization_params: Output from ``DomainRandomizationPipeline.randomize()``.

        Returns:
            Dict with ``rgb``, ``semantic_mask``, ``depth``, ``screw_poses``.
        """
        if not self._initialized:
            self.initialize()

        # Apply randomization to scene
        if randomization_params is not None:
            weather = randomization_params.get("weather", {}).get("preset", "day")
            self.env.set_weather(weather, randomize=True)

        # Step simulation to settle
        # Render camera outputs
        # (In real Isaac Sim, this would use rep.BasicWriter or Camera sensor)

        # Placeholder return — actual implementation depends on Isaac Sim camera API
        return {
            "rgb": np.zeros((self.height, self.width, 3), dtype=np.uint8),
            "semantic_mask": np.zeros((self.height, self.width), dtype=np.int32),
            "depth": np.zeros((self.height, self.width), dtype=np.float32),
            "screw_poses": [],
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class LabelPipeline:
    """End-to-end pipeline: render → label → save.

    Manages the rendering loop, domain randomization scheduling,
    progress tracking, and checkpointing.
    """

    def __init__(self, cfg: LabelPipelineCfg | None = None):
        self.cfg = cfg or LabelPipelineCfg()

        # Detector / dataset builder
        self.cfg.detector.output_dir = self.cfg.output_dir
        self.cfg.detector.dataset_name = self.cfg.dataset_name
        self.dataset_builder = ScrewDetectionDatasetBuilder(self.cfg.detector)

        # Domain randomization
        self.dr_pipeline = DomainRandomizationPipeline(self.cfg.randomization)

        # Renderer
        if ISAACLAB_AVAILABLE and not getattr(self.cfg, '_force_simulated', False):
            self.renderer = IsaacSimRenderer(self.cfg)
        else:
            logger.info("Using simulated renderer (Isaac Sim not available)")
            self.renderer = SimulatedRenderer(
                self.cfg.camera_width,
                self.cfg.camera_height,
                self.cfg.num_screw_positions,
            )

        # Camera intrinsics (approximate)
        fx = self.cfg.camera_width / (2 * np.tan(np.radians(self.cfg.camera_fov_deg / 2)))
        fy = fx
        cx = self.cfg.camera_width / 2
        cy = self.cfg.camera_height / 2
        self.camera_intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=np.float64)

        self.camera_extrinsics = np.eye(4, dtype=np.float64)

        # State
        self._start_idx = 0
        if self.cfg.resume_from:
            self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Resume from a checkpoint file."""
        ckpt_path = Path(self.cfg.resume_from)
        if ckpt_path.exists():
            with open(ckpt_path) as f:
                ckpt = json.load(f)
            self._start_idx = ckpt.get("num_images_generated", 0)
            logger.info(f"Resuming from checkpoint: {self._start_idx} images")

    def _save_checkpoint(self, num_generated: int) -> None:
        """Save a checkpoint."""
        ckpt_dir = Path(self.cfg.output_dir) / self.cfg.dataset_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "checkpoint.json"
        with open(ckpt_path, "w") as f:
            json.dump({
                "num_images_generated": num_generated,
                "timestamp": time.time(),
            }, f)

    def run(self) -> dict[str, str]:
        """Execute the full pipeline.

        Returns:
            Dict of annotation file paths (train/val/test).
        """
        total = self.cfg.num_images
        start = self._start_idx
        t0 = time.time()

        logger.info(f"Starting labeling pipeline: {total} images, starting from {start}")
        print(f"Labeling pipeline: generating {total - start} images")
        print(f"  Output: {self.cfg.output_dir}/{self.cfg.dataset_name}")
        print(f"  Resolution: {self.cfg.camera_width}×{self.cfg.camera_height}")
        print(f"  Domain randomization: enabled")

        for i in range(start, total):
            # Domain randomization
            dr_params = self.dr_pipeline.randomize(
                num_screws=self.cfg.num_screw_positions
            )

            # Render
            frame = self.renderer.render(dr_params)

            # Add to dataset
            self.dataset_builder.add_frame(
                rgb_image=frame["rgb"],
                semantic_mask=frame["semantic_mask"],
                screw_poses=frame.get("screw_poses"),
                camera_intrinsics=self.camera_intrinsics,
                camera_extrinsics=self.camera_extrinsics,
                screw_type=dr_params.get("screw", {}).get("screw_type", "hex"),
                metadata={
                    "weather": dr_params.get("weather", {}).get("preset", "day"),
                    "screw_type": dr_params.get("screw", {}).get("screw_type", "hex"),
                    "screw_size": dr_params.get("screw", {}).get("screw_size", "M6"),
                    "panel_condition": dr_params.get("panel", {}).get("condition", "clean"),
                },
            )

            # Progress logging
            if (i + 1) % self.cfg.log_interval == 0:
                elapsed = time.time() - t0
                rate = (i + 1 - start) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{total}] {rate:.1f} img/s, ETA: {eta:.0f}s")

            # Checkpoint
            if (i + 1) % self.cfg.checkpoint_interval == 0:
                self._save_checkpoint(i + 1)

        # Save final dataset
        elapsed = time.time() - t0
        print(f"\nGeneration complete: {total} images in {elapsed:.1f}s")
        paths = self.dataset_builder.save()

        return paths


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry point for the labeling pipeline."""
    parser = argparse.ArgumentParser(
        description="RoboForce Screw Detection — Labeling Pipeline"
    )
    parser.add_argument(
        "--num_images", type=int, default=10_000,
        help="Total number of images to generate (default: 10000)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="datasets/screw_detection",
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="roboforce_screw_v1",
        help="Dataset name",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Rendering batch size",
    )
    parser.add_argument(
        "--headless", action="store_true", default=True,
        help="Run in headless mode",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint file to resume from",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cfg = LabelPipelineCfg(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        num_images=args.num_images,
        batch_size=args.batch_size,
        headless=args.headless,
        resume_from=args.resume,
    )
    cfg.randomization.seed = args.seed
    cfg._force_simulated = not ISAACLAB_AVAILABLE  # type: ignore[attr-defined]

    pipeline = LabelPipeline(cfg)
    paths = pipeline.run()

    print(f"\nDataset annotation files:")
    for split, path in paths.items():
        print(f"  {split}: {path}")


if __name__ == "__main__":
    main()
