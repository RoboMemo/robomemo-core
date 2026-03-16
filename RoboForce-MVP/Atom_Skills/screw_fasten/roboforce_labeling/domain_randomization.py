# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Domain randomization for synthetic data generation.

Provides randomizers for:
- Lighting (intensity, color, direction)
- Camera viewpoint
- Screw/bolt type and size
- Solar panel condition (clean, dusty, damaged)
- Weather and time of day
- Material properties (texture variation)
- Background clutter

Each randomizer can be used independently or composed into a full
randomization pipeline via ``DomainRandomizationPipeline``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

try:
    import isaaclab.sim as sim_utils
    ISAACLAB_AVAILABLE = True
except ImportError:
    ISAACLAB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Individual Randomizers
# ---------------------------------------------------------------------------

@dataclass
class LightingRandomizerCfg:
    """Configuration for lighting randomization."""
    intensity_range: tuple[float, float] = (0.3, 3.0)
    """Multiplier range for light intensity."""
    color_perturbation: float = 0.15
    """Max random shift per RGB channel."""
    direction_noise_deg: float = 20.0
    """Max angular perturbation for directional lights (degrees)."""
    enabled: bool = True


@dataclass
class CameraRandomizerCfg:
    """Configuration for camera viewpoint randomization."""
    position_noise_m: float = 0.1
    """Max random displacement in each axis (meters)."""
    orientation_noise_deg: float = 5.0
    """Max random rotation perturbation (degrees)."""
    fov_range: tuple[float, float] = (50.0, 70.0)
    """Field of view range (degrees)."""
    enabled: bool = True


@dataclass
class ScrewRandomizerCfg:
    """Configuration for screw type/size randomization."""
    types: list[str] = field(default_factory=lambda: ["hex", "phillips", "flathead", "torx"])
    """Screw types to sample from."""
    sizes: list[str] = field(default_factory=lambda: ["M4", "M5", "M6", "M8"])
    """Metric sizes to sample from."""
    color_perturbation: float = 0.1
    """Max random color shift."""
    position_noise_m: float = 0.005
    """Max random displacement for screw positions (meters)."""
    enabled: bool = True


@dataclass
class PanelConditionRandomizerCfg:
    """Configuration for solar panel surface condition randomization."""
    conditions: list[str] = field(default_factory=lambda: [
        "clean", "dusty", "scratched", "weathered"
    ])
    """Panel conditions to sample from."""
    dust_opacity_range: tuple[float, float] = (0.0, 0.4)
    """Opacity range for dust overlay."""
    scratch_density_range: tuple[int, int] = (0, 10)
    """Number of surface scratches."""
    enabled: bool = True


@dataclass
class WeatherRandomizerCfg:
    """Configuration for weather/time-of-day randomization."""
    presets: list[str] = field(default_factory=lambda: ["day", "night", "dusty"])
    """Weather presets to sample from."""
    weights: list[float] = field(default_factory=lambda: [0.5, 0.25, 0.25])
    """Sampling weights for each preset."""
    apply_continuous_variation: bool = True
    """Apply continuous random variation on top of discrete presets."""
    enabled: bool = True


@dataclass
class MaterialRandomizerCfg:
    """Configuration for material property randomization."""
    roughness_range: tuple[float, float] = (0.1, 0.95)
    """PBR roughness range."""
    metallic_range: tuple[float, float] = (0.0, 1.0)
    """PBR metallic range (for metal surfaces)."""
    color_jitter: float = 0.1
    """Max random color jitter per channel."""
    enabled: bool = True


@dataclass
class BackgroundRandomizerCfg:
    """Configuration for background/environment randomization."""
    terrain_seeds: list[int] = field(default_factory=lambda: list(range(10)))
    """Random seeds for terrain height field generation."""
    num_clutter_objects_range: tuple[int, int] = (0, 5)
    """Range for number of random clutter objects."""
    enabled: bool = True


# ---------------------------------------------------------------------------
# Randomizer Implementations
# ---------------------------------------------------------------------------

class Randomizer:
    """Base class for domain randomizers."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def randomize(self, **kwargs) -> dict[str, Any]:
        """Apply randomization. Returns dict of randomized parameters."""
        raise NotImplementedError


class LightingRandomizer(Randomizer):
    """Randomize scene lighting parameters."""

    def __init__(self, cfg: LightingRandomizerCfg | None = None, seed: int = 42):
        super().__init__(seed)
        self.cfg = cfg or LightingRandomizerCfg()

    def randomize(self, base_intensity: float = 1000.0, base_color: Sequence[float] = (1.0, 1.0, 1.0), **kwargs) -> dict[str, Any]:
        if not self.cfg.enabled:
            return {"intensity": base_intensity, "color": list(base_color), "direction": [0, -0.7, -0.6]}

        # Intensity
        mult = self.rng.uniform(*self.cfg.intensity_range)
        intensity = base_intensity * mult

        # Color
        color = np.array(base_color[:3], dtype=np.float64)
        color += self.rng.uniform(-self.cfg.color_perturbation, self.cfg.color_perturbation, 3)
        color = np.clip(color, 0.0, 1.0)

        # Direction
        noise_rad = math.radians(self.cfg.direction_noise_deg)
        direction = np.array([0.3, -0.7, -0.6], dtype=np.float64)
        direction += self.rng.uniform(-noise_rad, noise_rad, 3)
        direction /= np.linalg.norm(direction)

        return {
            "intensity": float(intensity),
            "color": color.tolist(),
            "direction": direction.tolist(),
            "intensity_multiplier": float(mult),
        }


class CameraRandomizer(Randomizer):
    """Randomize camera viewpoint."""

    def __init__(self, cfg: CameraRandomizerCfg | None = None, seed: int = 42):
        super().__init__(seed)
        self.cfg = cfg or CameraRandomizerCfg()

    def randomize(
        self,
        base_position: Sequence[float] = (0.0, 1.0, 1.8),
        base_orientation_deg: Sequence[float] = (0.0, -15.0, 0.0),
        **kwargs,
    ) -> dict[str, Any]:
        if not self.cfg.enabled:
            return {
                "position": list(base_position),
                "orientation_deg": list(base_orientation_deg),
                "fov_deg": 60.0,
            }

        pos = np.array(base_position) + self.rng.uniform(
            -self.cfg.position_noise_m, self.cfg.position_noise_m, 3
        )
        ori = np.array(base_orientation_deg) + self.rng.uniform(
            -self.cfg.orientation_noise_deg, self.cfg.orientation_noise_deg, 3
        )
        fov = self.rng.uniform(*self.cfg.fov_range)

        return {
            "position": pos.tolist(),
            "orientation_deg": ori.tolist(),
            "fov_deg": float(fov),
        }


class ScrewRandomizer(Randomizer):
    """Randomize screw type, size, and position."""

    def __init__(self, cfg: ScrewRandomizerCfg | None = None, seed: int = 42):
        super().__init__(seed)
        self.cfg = cfg or ScrewRandomizerCfg()

    def randomize(self, num_screws: int = 12, **kwargs) -> dict[str, Any]:
        if not self.cfg.enabled:
            return {
                "screw_type": "hex",
                "screw_size": "M6",
                "position_offsets": [[0, 0, 0]] * num_screws,
                "color_offset": [0, 0, 0],
            }

        screw_type = self.rng.choice(self.cfg.types)
        screw_size = self.rng.choice(self.cfg.sizes)

        offsets = self.rng.uniform(
            -self.cfg.position_noise_m, self.cfg.position_noise_m, (num_screws, 3)
        )
        color_offset = self.rng.uniform(
            -self.cfg.color_perturbation, self.cfg.color_perturbation, 3
        )

        return {
            "screw_type": str(screw_type),
            "screw_size": str(screw_size),
            "position_offsets": offsets.tolist(),
            "color_offset": color_offset.tolist(),
        }


class WeatherRandomizer(Randomizer):
    """Randomize weather preset with continuous variation."""

    def __init__(self, cfg: WeatherRandomizerCfg | None = None, seed: int = 42):
        super().__init__(seed)
        self.cfg = cfg or WeatherRandomizerCfg()

    def randomize(self, **kwargs) -> dict[str, Any]:
        if not self.cfg.enabled:
            return {"preset": "day", "continuous_variation": {}}

        # Sample preset
        weights = np.array(self.cfg.weights)
        weights /= weights.sum()
        preset = self.rng.choice(self.cfg.presets, p=weights)

        variation = {}
        if self.cfg.apply_continuous_variation:
            variation = {
                "intensity_multiplier": float(self.rng.uniform(0.7, 1.3)),
                "color_temperature_offset": float(self.rng.uniform(-500, 500)),
                "fog_density_multiplier": float(self.rng.uniform(0.5, 1.5)),
            }

        return {
            "preset": str(preset),
            "continuous_variation": variation,
        }


class PanelConditionRandomizer(Randomizer):
    """Randomize solar panel surface condition."""

    def __init__(self, cfg: PanelConditionRandomizerCfg | None = None, seed: int = 42):
        super().__init__(seed)
        self.cfg = cfg or PanelConditionRandomizerCfg()

    def randomize(self, **kwargs) -> dict[str, Any]:
        if not self.cfg.enabled:
            return {"condition": "clean", "dust_opacity": 0.0, "scratches": 0}

        condition = self.rng.choice(self.cfg.conditions)
        dust_opacity = self.rng.uniform(*self.cfg.dust_opacity_range)
        scratches = int(self.rng.integers(*self.cfg.scratch_density_range))

        return {
            "condition": str(condition),
            "dust_opacity": float(dust_opacity),
            "scratches": scratches,
        }


# ---------------------------------------------------------------------------
# Composition Pipeline
# ---------------------------------------------------------------------------

@dataclass
class DomainRandomizationPipelineCfg:
    """Full domain randomization pipeline configuration."""
    lighting: LightingRandomizerCfg = field(default_factory=LightingRandomizerCfg)
    camera: CameraRandomizerCfg = field(default_factory=CameraRandomizerCfg)
    screw: ScrewRandomizerCfg = field(default_factory=ScrewRandomizerCfg)
    panel: PanelConditionRandomizerCfg = field(default_factory=PanelConditionRandomizerCfg)
    weather: WeatherRandomizerCfg = field(default_factory=WeatherRandomizerCfg)
    material: MaterialRandomizerCfg = field(default_factory=MaterialRandomizerCfg)
    background: BackgroundRandomizerCfg = field(default_factory=BackgroundRandomizerCfg)
    seed: int = 42


class DomainRandomizationPipeline:
    """Compose all domain randomizers into a single pipeline.

    Usage:
        pipeline = DomainRandomizationPipeline(cfg)
        params = pipeline.randomize()
        # Apply params to simulation scene
    """

    def __init__(self, cfg: DomainRandomizationPipelineCfg | None = None):
        self.cfg = cfg or DomainRandomizationPipelineCfg()

        self.lighting = LightingRandomizer(self.cfg.lighting, self.cfg.seed)
        self.camera = CameraRandomizer(self.cfg.camera, self.cfg.seed + 1)
        self.screw = ScrewRandomizer(self.cfg.screw, self.cfg.seed + 2)
        self.panel = PanelConditionRandomizer(self.cfg.panel, self.cfg.seed + 3)
        self.weather = WeatherRandomizer(self.cfg.weather, self.cfg.seed + 4)

    def randomize(self, **kwargs) -> dict[str, Any]:
        """Run all enabled randomizers and return combined parameters.

        Returns:
            Dict with keys ``lighting``, ``camera``, ``screw``, ``panel``, ``weather``.
        """
        return {
            "lighting": self.lighting.randomize(**kwargs),
            "camera": self.camera.randomize(**kwargs),
            "screw": self.screw.randomize(**kwargs),
            "panel": self.panel.randomize(**kwargs),
            "weather": self.weather.randomize(**kwargs),
        }

    def apply_to_scene(self, params: dict[str, Any], stage=None) -> None:
        """Apply randomized parameters to the Isaac Sim scene.

        Args:
            params: Output from ``randomize()``.
            stage: USD stage.
        """
        if not ISAACLAB_AVAILABLE or stage is None:
            return

        # Apply lighting
        light_params = params.get("lighting", {})
        if light_params:
            dome = sim_utils.DomeLightCfg(
                intensity=light_params.get("intensity", 1000.0),
                color=tuple(light_params.get("color", [1, 1, 1])),
            )
            dome.func("/World/DomeLight", dome)

        # Weather preset
        weather_params = params.get("weather", {})
        preset = weather_params.get("preset", "day")
        # The WeatherManager in solar_panel_env handles preset application


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Domain Randomization Pipeline — Test")
    print("=" * 50)

    pipeline = DomainRandomizationPipeline()

    for i in range(5):
        params = pipeline.randomize(num_screws=12)
        print(f"\nSample {i}:")
        print(f"  Lighting: intensity_mult={params['lighting']['intensity_multiplier']:.2f}, "
              f"color={[f'{c:.2f}' for c in params['lighting']['color']]}")
        print(f"  Camera: fov={params['camera']['fov_deg']:.1f}°")
        print(f"  Screw: type={params['screw']['screw_type']}, size={params['screw']['screw_size']}")
        print(f"  Panel: condition={params['panel']['condition']}, dust={params['panel']['dust_opacity']:.2f}")
        print(f"  Weather: preset={params['weather']['preset']}")
