# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Sensor configuration for RoboForce mock data pipeline.

Defines the exact sensor specs that RoboForce will provide:
- 3x RGBD cameras (head stereo pair + wrist)
- 2x 6-axis force/torque sensors (wrist + EE tip)

These configs serve as the single source of truth for data format,
so downstream VLA training and the future real-sensor integration
use the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Camera Configuration
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """RGBD camera specification.

    Matches the RGBD cameras RoboForce will provide.
    All units SI (meters, radians) unless noted.
    """

    name: str = "camera"
    """Unique camera identifier."""

    # Resolution
    width: int = 640
    height: int = 480

    # Intrinsics
    fx: float = 615.0
    """Focal length x (pixels). ~60° HFOV at 640px."""
    fy: float = 615.0
    """Focal length y (pixels)."""
    cx: float = 320.0
    """Principal point x (pixels)."""
    cy: float = 240.0
    """Principal point y (pixels)."""

    # Depth
    depth_min: float = 0.1
    """Minimum depth range (meters)."""
    depth_max: float = 10.0
    """Maximum depth range (meters)."""
    depth_dtype: str = "float32"
    """Depth map data type. float32 = meters, uint16 = millimeters."""

    # Noise model
    rgb_noise_std: float = 3.0
    """Gaussian noise std on RGB channels (0-255 scale)."""
    depth_noise_std: float = 0.005
    """Gaussian noise std on depth (meters)."""

    # Frame rate
    fps: float = 30.0
    """Capture frame rate (Hz)."""

    # Mounting (relative to parent link)
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset from parent link (meters)."""
    orientation_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Orientation offset roll/pitch/yaw (radians)."""
    parent_link: str = ""
    """Parent link name on robot."""

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """3x3 camera intrinsic matrix K."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)


# ---------------------------------------------------------------------------
# Force/Torque Sensor Configuration
# ---------------------------------------------------------------------------

@dataclass
class FTSensorConfig:
    """6-axis force/torque sensor specification.

    Matches the FT sensors RoboForce will provide.
    """

    name: str = "ft_sensor"
    """Unique sensor identifier."""

    # Measurement range (±)
    force_range: Tuple[float, float, float] = (100.0, 100.0, 200.0)
    """Max measurable force Fx, Fy, Fz (Newtons)."""
    torque_range: Tuple[float, float, float] = (25.0, 25.0, 25.0)
    """Max measurable torque Tx, Ty, Tz (N·m)."""

    # Noise model
    force_noise_std: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    """Gaussian noise std on force channels (N)."""
    torque_noise_std: Tuple[float, float, float] = (0.005, 0.005, 0.005)
    """Gaussian noise std on torque channels (N·m)."""

    # Bias (sensor zero offset)
    force_bias: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Static bias on force channels (N). Simulates imperfect calibration."""
    torque_bias: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Static bias on torque channels (N·m)."""

    # Sample rate
    fps: float = 500.0
    """Sample rate (Hz). Typical industrial FT sensors: 500-1000 Hz."""

    # Data format
    output_dim: int = 6
    """Output dimension: [Fx, Fy, Fz, Tx, Ty, Tz]."""

    # Mounting
    parent_link: str = ""
    """Parent link on robot where sensor is mounted."""

    @property
    def full_range(self) -> np.ndarray:
        """Combined force+torque range as (6,) array."""
        return np.array(list(self.force_range) + list(self.torque_range), dtype=np.float32)

    @property
    def full_noise_std(self) -> np.ndarray:
        """Combined force+torque noise std as (6,) array."""
        return np.array(list(self.force_noise_std) + list(self.torque_noise_std), dtype=np.float32)


# ---------------------------------------------------------------------------
# Full Sensor Suite Config
# ---------------------------------------------------------------------------

@dataclass
class SensorConfig:
    """Complete sensor suite for RoboForce.

    3 RGBD cameras + 2 FT sensors as specified by RoboForce.
    """

    # ---- 3 RGBD Cameras ----

    head_left_cam: CameraConfig = field(default_factory=lambda: CameraConfig(
        name="head_left",
        width=640,
        height=480,
        fx=615.0, fy=615.0, cx=320.0, cy=240.0,
        depth_min=0.1, depth_max=10.0,
        fps=30.0,
        position=(0.05, 0.04, 1.80),
        orientation_rpy=(0.0, -0.26, 0.0),  # ~-15° pitch (looking down)
        parent_link="head_link",
    ))
    """Left stereo camera on robot head."""

    head_right_cam: CameraConfig = field(default_factory=lambda: CameraConfig(
        name="head_right",
        width=640,
        height=480,
        fx=615.0, fy=615.0, cx=320.0, cy=240.0,
        depth_min=0.1, depth_max=10.0,
        fps=30.0,
        position=(0.05, -0.04, 1.80),
        orientation_rpy=(0.0, -0.26, 0.0),
        parent_link="head_link",
    ))
    """Right stereo camera on robot head. Baseline ~8cm from left."""

    wrist_cam: CameraConfig = field(default_factory=lambda: CameraConfig(
        name="wrist",
        width=640,
        height=480,
        fx=615.0, fy=615.0, cx=320.0, cy=240.0,
        depth_min=0.05, depth_max=3.0,
        fps=30.0,
        position=(0.0, 0.0, -0.05),
        orientation_rpy=(0.0, -1.57, 0.0),  # Looking along EE axis
        parent_link="right_wrist_roll_link",
    ))
    """Wrist-mounted camera for close-up screw alignment."""

    # ---- 2 FT Sensors ----

    wrist_ft: FTSensorConfig = field(default_factory=lambda: FTSensorConfig(
        name="wrist_ft",
        force_range=(100.0, 100.0, 200.0),
        torque_range=(25.0, 25.0, 25.0),
        force_noise_std=(0.1, 0.1, 0.1),
        torque_noise_std=(0.005, 0.005, 0.005),
        fps=500.0,
        parent_link="right_wrist_3_link",
    ))
    """6D FT sensor at wrist (between last arm joint and EE)."""

    ee_tip_ft: FTSensorConfig = field(default_factory=lambda: FTSensorConfig(
        name="ee_tip_ft",
        force_range=(50.0, 50.0, 100.0),
        torque_range=(15.0, 15.0, 15.0),
        force_noise_std=(0.05, 0.05, 0.05),
        torque_noise_std=(0.003, 0.003, 0.003),
        fps=500.0,
        parent_link="right_ee_link",
    ))
    """6D FT sensor at screw driver tip (measures screw engagement forces)."""

    @property
    def cameras(self) -> list[CameraConfig]:
        """All camera configs as a list."""
        return [self.head_left_cam, self.head_right_cam, self.wrist_cam]

    @property
    def ft_sensors(self) -> list[FTSensorConfig]:
        """All FT sensor configs as a list."""
        return [self.wrist_ft, self.ee_tip_ft]

    @property
    def camera_names(self) -> list[str]:
        return [c.name for c in self.cameras]

    @property
    def ft_sensor_names(self) -> list[str]:
        return [f.name for f in self.ft_sensors]
