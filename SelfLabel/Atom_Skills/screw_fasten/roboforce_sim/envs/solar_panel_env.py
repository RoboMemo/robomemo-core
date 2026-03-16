# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Isaac Lab environment for solar panel screw installation.

This module defines the full simulation scene using Isaac Lab's ManagerBasedEnv
API (v2.3.x). It assembles:
- Desert terrain ground plane
- Solar panel rack with mounting brackets
- Screw/bolt objects at mounting points
- RoboForce robot (placeholder articulation)
- Cameras, lights, and weather presets

Usage:
    python -m roboforce_sim.envs.solar_panel_env
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import (
        Articulation,
        ArticulationCfg,
        RigidObject,
        RigidObjectCfg,
    )
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import (
        Camera,
        CameraCfg,
        ContactSensor,
        ContactSensorCfg,
    )
    from isaaclab.sim import SimulationCfg
    from isaaclab.utils import configclass

    ISAACLAB_AVAILABLE = True
except ImportError:
    ISAACLAB_AVAILABLE = False

    # Stub decorators / base classes for offline usage
    def configclass(cls):  # type: ignore[misc]
        return dataclass(cls)

    class ManagerBasedEnvCfg:  # type: ignore[no-redef]
        pass

    class InteractiveSceneCfg:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
_CFG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _load_yaml(name: str) -> dict:
    """Load a YAML config from the configs directory."""
    path = _CFG_DIR / name
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Scene Configuration
# ---------------------------------------------------------------------------

@configclass
class RoboForceSceneCfg(InteractiveSceneCfg if ISAACLAB_AVAILABLE else object):  # type: ignore[misc]
    """Interactive scene configuration for the solar panel environment."""

    # --- Ground Plane ---
    ground_plane: dict = field(default_factory=lambda: {
        "type": "desert_terrain",
        "size": [20.0, 20.0],
        "resolution": 64,
        "max_height": 0.3,
    })

    # --- Solar Panel Rack ---
    solar_rack: dict = field(default_factory=lambda: {
        "panel_width": 2.0,
        "panel_height": 1.0,
        "panel_tilt_deg": 30.0,
        "num_brackets_per_side": 3,
        "screws_per_bracket": 2,
        "position": [0.0, -1.0, 0.0],
    })

    # --- Robot (placeholder articulation) ---
    robot: dict = field(default_factory=lambda: {
        "prim_path": "/World/Robot",
        "initial_position": [0.0, 1.0, 0.0],
        "initial_yaw_deg": 0.0,
    })

    # --- Weather ---
    weather_preset: str = "day"


# ---------------------------------------------------------------------------
# Environment Configuration
# ---------------------------------------------------------------------------

@configclass
class SolarPanelEnvCfg(ManagerBasedEnvCfg if ISAACLAB_AVAILABLE else object):  # type: ignore[misc]
    """Full environment configuration for solar panel screw installation."""

    # Simulation
    sim_dt: float = 0.005
    sim_substeps: int = 2
    decimation: int = 4  # Control at 50 Hz (200 Hz physics / 4)

    # Scene
    scene: RoboForceSceneCfg = field(default_factory=RoboForceSceneCfg)

    # Episode
    episode_length_s: float = 60.0
    num_envs: int = 64
    env_spacing: float = 5.0

    # Observations
    obs_group: str = "policy"
    enable_camera: bool = True
    camera_width: int = 640
    camera_height: int = 480

    # Actions
    action_dim: int = 8  # 6D delta pose + screw rotation + gripper

    # Weather
    weather: str = "day"

    @classmethod
    def from_yaml(cls, yaml_path: str | Path | None = None) -> "SolarPanelEnvCfg":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML config. Defaults to ``configs/base_env.yaml``.

        Returns:
            Populated configuration instance.
        """
        if yaml_path is None:
            yaml_path = _CFG_DIR / "base_env.yaml"
        data = _load_yaml(Path(yaml_path).name) if isinstance(yaml_path, Path) else yaml.safe_load(open(yaml_path))

        cfg = cls()
        # Map YAML fields
        sim = data.get("sim", {})
        cfg.sim_dt = sim.get("dt", cfg.sim_dt)
        cfg.sim_substeps = sim.get("substeps", cfg.sim_substeps)

        env = data.get("env", {})
        cfg.num_envs = env.get("num_envs", cfg.num_envs)
        cfg.env_spacing = env.get("env_spacing", cfg.env_spacing)
        cfg.episode_length_s = env.get("episode_length_s", cfg.episode_length_s)

        scene = data.get("scene", {})
        if "terrain" in scene:
            cfg.scene.ground_plane.update(scene["terrain"])
        if "solar_rack" in scene:
            cfg.scene.solar_rack.update(scene["solar_rack"])

        return cfg


# ---------------------------------------------------------------------------
# Weather system
# ---------------------------------------------------------------------------

class WeatherManager:
    """Manages lighting and atmospheric presets for the simulation scene.

    Reads presets from ``weather_configs.yaml`` and applies them to the USD
    stage lights and atmosphere settings.
    """

    def __init__(self, stage=None):
        self._presets = _load_yaml("weather_configs.yaml").get("presets", {})
        self._randomization = _load_yaml("weather_configs.yaml").get("randomization", {})
        self._stage = stage
        self._current_preset: str = "day"

    @property
    def available_presets(self) -> list[str]:
        return list(self._presets.keys())

    def apply_preset(self, name: str, randomize: bool = False) -> dict:
        """Apply a weather preset to the scene.

        Args:
            name: Preset name (``day``, ``night``, ``dusty``).
            randomize: If True, add random perturbations to the preset.

        Returns:
            The applied lighting configuration dict.
        """
        if name not in self._presets:
            raise ValueError(f"Unknown weather preset '{name}'. Available: {self.available_presets}")

        preset = self._presets[name]
        self._current_preset = name

        if self._stage is not None and ISAACLAB_AVAILABLE:
            self._apply_to_stage(preset, randomize)

        return preset

    def _apply_to_stage(self, preset: dict, randomize: bool) -> None:
        """Apply lighting configuration to the USD stage."""
        lighting = preset.get("lighting", {})
        rng = np.random.default_rng() if randomize else None

        # Dome light
        dome_cfg = lighting.get("dome_light", {})
        intensity = dome_cfg.get("intensity", 1000.0)
        if randomize and rng is not None:
            mult_range = self._randomization.get("dome_light_intensity_range", [1.0, 1.0])
            intensity *= rng.uniform(mult_range[0], mult_range[1])

        dome = sim_utils.DomeLightCfg(
            intensity=intensity,
            color=tuple(dome_cfg.get("color", [1, 1, 1])),
        )
        dome.func("/World/DomeLight", dome)

        # Distant light (sun / moon)
        dist_cfg = lighting.get("distant_light", {})
        dist_intensity = dist_cfg.get("intensity", 10000.0)
        if randomize and rng is not None:
            mult_range = self._randomization.get("distant_light_intensity_range", [1.0, 1.0])
            dist_intensity *= rng.uniform(mult_range[0], mult_range[1])

        distant = sim_utils.DistantLightCfg(
            intensity=dist_intensity,
            color=tuple(dist_cfg.get("color", [1, 1, 1])),
            angle=dist_cfg.get("angle", 0.53),
        )
        distant.func("/World/DistantLight", distant)

        # Robot spotlight (if defined)
        spotlight_cfg = preset.get("robot_spotlight", {})
        if spotlight_cfg.get("enabled", False):
            spot_intensity = spotlight_cfg.get("intensity", 20000.0)
            spot = sim_utils.SphereLightCfg(
                intensity=spot_intensity,
                color=tuple(spotlight_cfg.get("color", [1, 1, 1])),
                radius=0.05,
            )
            spot.func("/World/Robot/head/spotlight", spot)


# ---------------------------------------------------------------------------
# Placeholder Robot Builder
# ---------------------------------------------------------------------------

def _build_placeholder_robot_cfg(robot_cfg: dict) -> dict:
    """Build an ArticulationCfg-compatible dict for the placeholder RoboForce.

    The placeholder models:
    - Tracked base: 3 DOF (prismatic X, prismatic Y, revolute yaw)
    - Right arm: 7 DOF revolute chain
    - Screw driver EE: 1 DOF revolute (continuous rotation)
    - Head: 2 DOF pan-tilt

    Total: 13 DOF (single-arm operation for screw driving)
    """
    robot_yaml = _load_yaml("robot_configs.yaml").get("robot", {})

    joint_names = []
    joint_types = []
    joint_limits_lower = []
    joint_limits_upper = []
    joint_max_velocity = []
    joint_damping = []

    # Base joints
    for jnt in robot_yaml.get("base", {}).get("joints", []):
        joint_names.append(jnt["name"])
        joint_types.append(jnt["type"])
        joint_limits_lower.append(jnt["limits"][0])
        joint_limits_upper.append(jnt["limits"][1])
        joint_max_velocity.append(jnt.get("max_velocity", 1.0))
        joint_damping.append(jnt.get("damping", 10.0))

    # Right arm joints
    for jnt in robot_yaml.get("right_arm", {}).get("joints", []):
        joint_names.append(jnt["name"])
        joint_types.append(jnt["type"])
        joint_limits_lower.append(jnt["limits"][0])
        joint_limits_upper.append(jnt["limits"][1])
        joint_max_velocity.append(jnt.get("max_velocity", 2.0))
        joint_damping.append(jnt.get("damping", 5.0))

    # Screw driver EE
    for jnt in robot_yaml.get("end_effector", {}).get("joints", []):
        joint_names.append(jnt["name"])
        joint_types.append(jnt["type"])
        joint_limits_lower.append(jnt["limits"][0])
        joint_limits_upper.append(jnt["limits"][1])
        joint_max_velocity.append(jnt.get("max_velocity", 30.0))
        joint_damping.append(jnt.get("damping", 0.5))

    # Head joints
    for jnt in robot_yaml.get("head", {}).get("joints", []):
        joint_names.append(jnt["name"])
        joint_types.append(jnt["type"])
        joint_limits_lower.append(jnt["limits"][0])
        joint_limits_upper.append(jnt["limits"][1])
        joint_max_velocity.append(jnt.get("max_velocity", 2.0))
        joint_damping.append(jnt.get("damping", 2.0))

    return {
        "joint_names": joint_names,
        "joint_types": joint_types,
        "joint_limits_lower": joint_limits_lower,
        "joint_limits_upper": joint_limits_upper,
        "joint_max_velocity": joint_max_velocity,
        "joint_damping": joint_damping,
        "num_dof": len(joint_names),
    }


# ---------------------------------------------------------------------------
# Environment Class
# ---------------------------------------------------------------------------

class SolarPanelEnv:
    """Solar panel screw installation environment.

    This is the top-level environment that composes the scene, manages
    weather presets, and provides the standard step/reset interface.

    When Isaac Lab is available, this wraps ``ManagerBasedEnv``. Otherwise
    it serves as a configuration container for offline testing.
    """

    def __init__(self, cfg: SolarPanelEnvCfg | None = None):
        if cfg is None:
            cfg = SolarPanelEnvCfg.from_yaml()
        self.cfg = cfg

        # Weather
        self.weather_manager = WeatherManager()

        # Robot config
        self._robot_info = _build_placeholder_robot_cfg(cfg.scene.robot)

        # Scene objects (populated on build)
        self._screw_holes: list[dict] = []
        self._screw_paths: list[str] = []

        if ISAACLAB_AVAILABLE:
            self._build_scene()

    def _build_scene(self) -> None:
        """Construct the full USD scene in Isaac Sim."""
        from roboforce_sim.assets.solar_panel_rack import (
            SolarPanelRackCfg,
            spawn_solar_panel_rack,
        )
        from roboforce_sim.assets.screw_bolt import (
            ScrewBoltCfg,
            spawn_screws_at_holes,
        )
        from roboforce_sim.assets.desert_terrain import (
            DesertTerrainCfg,
            spawn_desert_terrain,
        )

        # Terrain
        terrain_cfg = DesertTerrainCfg(
            terrain_size=tuple(self.cfg.scene.ground_plane.get("size", [20, 20])),
            resolution=self.cfg.scene.ground_plane.get("resolution", 64),
            max_height=self.cfg.scene.ground_plane.get("max_height", 0.3),
        )
        spawn_desert_terrain("/World/Terrain", terrain_cfg)

        # Solar rack
        rack_pos = self.cfg.scene.solar_rack.get("position", [0, -1, 0])
        rack_cfg = SolarPanelRackCfg(
            panel_width=self.cfg.scene.solar_rack.get("panel_width", 2.0),
            panel_height=self.cfg.scene.solar_rack.get("panel_height", 1.0),
            panel_tilt_deg=self.cfg.scene.solar_rack.get("panel_tilt_deg", 30.0),
            num_brackets_per_side=self.cfg.scene.solar_rack.get("num_brackets_per_side", 3),
            screws_per_bracket=self.cfg.scene.solar_rack.get("screws_per_bracket", 2),
        )
        self._screw_holes = spawn_solar_panel_rack(
            "/World/SolarRack", rack_cfg, translation=rack_pos
        )

        # Screws
        screw_cfg = ScrewBoltCfg()
        self._screw_paths = spawn_screws_at_holes(
            self._screw_holes, screw_cfg, parent_path="/World/Screws"
        )

        # Weather
        self.weather_manager.apply_preset(self.cfg.weather)

    @property
    def num_screws(self) -> int:
        """Total number of screw holes in the scene."""
        return len(self._screw_holes)

    @property
    def screw_positions(self) -> list[tuple[float, float, float]]:
        """World-frame positions of all screw holes."""
        return [h["position"] for h in self._screw_holes]

    @property
    def robot_dof(self) -> int:
        """Number of robot degrees of freedom."""
        return self._robot_info["num_dof"]

    def set_weather(self, preset: str, randomize: bool = False) -> None:
        """Change weather preset at runtime.

        Args:
            preset: One of ``day``, ``night``, ``dusty``.
            randomize: Apply random perturbations.
        """
        self.weather_manager.apply_preset(preset, randomize=randomize)
        self.cfg.weather = preset

    def info(self) -> dict:
        """Return a summary of the environment configuration."""
        return {
            "num_envs": self.cfg.num_envs,
            "episode_length_s": self.cfg.episode_length_s,
            "action_dim": self.cfg.action_dim,
            "robot_dof": self.robot_dof,
            "num_screws": self.num_screws,
            "weather": self.cfg.weather,
            "sim_dt": self.cfg.sim_dt,
            "control_dt": self.cfg.sim_dt * self.cfg.decimation,
            "camera_enabled": self.cfg.enable_camera,
            "camera_resolution": (self.cfg.camera_width, self.cfg.camera_height),
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch the solar panel environment for visual inspection."""
    print("=" * 60)
    print("RoboForce Solar Panel Screw Installation Environment")
    print("=" * 60)

    cfg = SolarPanelEnvCfg.from_yaml()
    print(f"\nConfiguration loaded from: {_CFG_DIR / 'base_env.yaml'}")

    if ISAACLAB_AVAILABLE:
        # Initialize simulation
        sim_cfg = SimulationCfg(dt=cfg.sim_dt, substeps=cfg.sim_substeps)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=[3.0, 3.0, 3.0], target=[0.0, 0.0, 1.0])

        env = SolarPanelEnv(cfg)
        info = env.info()
        print(f"\nEnvironment info:")
        for k, v in info.items():
            print(f"  {k}: {v}")

        # Run simulation loop
        print("\nStarting simulation... (press Ctrl+C to stop)")
        sim.reset()
        while sim.is_running():
            sim.step()
    else:
        print("\n[OFFLINE MODE] Isaac Lab not available — showing config only.")
        env = SolarPanelEnv(cfg)
        info = env.info()
        for k, v in info.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
