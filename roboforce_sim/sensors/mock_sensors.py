# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Mock sensor implementations for RoboForce.

Generates synthetic but physically-plausible sensor data for:
- 3x RGBD cameras (structured scenes with noise)
- 2x 6D force/torque sensors (task-phase-aware profiles)

This allows the full VLA training pipeline to run before real
sensor hardware is integrated. The mock data follows the same
interface as future real-sensor wrappers.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple

import numpy as np

from roboforce_sim.sensors.sensor_config import (
    CameraConfig,
    FTSensorConfig,
    SensorConfig,
)


# ---------------------------------------------------------------------------
# Mock RGBD Camera
# ---------------------------------------------------------------------------

class MockRGBDCamera:
    """Generate synthetic RGBD images with task-relevant structure.

    Produces images with:
    - A sky gradient (upper) + ground/panel (lower)
    - Procedural screw-head-like circles in the wrist camera view
    - Configurable Gaussian noise
    - Depth maps with distance-based structure

    The images are NOT photorealistic — they provide the right shapes,
    noise profiles, and data format for pipeline integration testing.
    """

    def __init__(self, config: CameraConfig, seed: int = 42):
        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self._frame_count = 0
        self._t0 = time.monotonic()

    def capture(
        self,
        ee_position: Optional[np.ndarray] = None,
        screw_positions: Optional[list[np.ndarray]] = None,
        robot_state: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Capture one RGBD frame.

        Args:
            ee_position: Current end-effector position (3,). Used for
                wrist camera to generate close-up screw views.
            screw_positions: List of screw world positions for projection.
            robot_state: Robot joint state for state-dependent rendering.

        Returns:
            Dict with keys:
                ``rgb``: uint8 array (H, W, 3)
                ``depth``: float32 array (H, W) in meters
                ``timestamp``: capture timestamp (seconds)
                ``frame_id``: monotonic frame counter
                ``intrinsics``: 3x3 camera matrix
        """
        H, W = self.cfg.height, self.cfg.width

        # --- RGB ---
        rgb = self._render_rgb(H, W, ee_position, screw_positions)

        # Add noise
        if self.cfg.rgb_noise_std > 0:
            noise = self.rng.normal(0, self.cfg.rgb_noise_std, rgb.shape)
            rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # --- Depth ---
        depth = self._render_depth(H, W, ee_position, screw_positions)

        if self.cfg.depth_noise_std > 0:
            depth_noise = self.rng.normal(0, self.cfg.depth_noise_std, depth.shape).astype(np.float32)
            depth = np.clip(depth + depth_noise, self.cfg.depth_min, self.cfg.depth_max)

        self._frame_count += 1
        timestamp = time.monotonic() - self._t0

        return {
            "rgb": rgb,
            "depth": depth,
            "timestamp": timestamp,
            "frame_id": self._frame_count,
            "intrinsics": self.cfg.intrinsic_matrix,
        }

    def _render_rgb(
        self,
        H: int,
        W: int,
        ee_position: Optional[np.ndarray],
        screw_positions: Optional[list[np.ndarray]],
    ) -> np.ndarray:
        """Render a synthetic RGB image."""
        img = np.zeros((H, W, 3), dtype=np.uint8)

        is_wrist = "wrist" in self.cfg.name

        if is_wrist:
            # Wrist camera: close-up metallic surface with screw hole
            # Gray metallic background
            base_gray = self.rng.integers(120, 160)
            img[:, :] = [base_gray, base_gray, base_gray + 10]

            # Add subtle texture (brushed metal effect)
            for row in range(0, H, 2):
                offset = self.rng.integers(-5, 5)
                img[row, :] = np.clip(img[row, :].astype(int) + offset, 0, 255).astype(np.uint8)

            # Draw screw head (circle) near center
            cy, cx = H // 2 + self.rng.integers(-20, 20), W // 2 + self.rng.integers(-20, 20)
            radius = self.rng.integers(25, 45)
            yy, xx = np.ogrid[:H, :W]
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

            # Screw head (darker circle)
            mask = dist < radius
            img[mask] = [80, 80, 85]

            # Hex socket (hexagonal pattern inside)
            hex_r = radius * 0.5
            for angle in range(0, 360, 60):
                rad = math.radians(angle)
                hx = int(cx + hex_r * 0.7 * math.cos(rad))
                hy = int(cy + hex_r * 0.7 * math.sin(rad))
                if 0 <= hy < H and 0 <= hx < W:
                    # Small dark spot for hex vertices
                    spot_mask = ((yy - hy) ** 2 + (xx - hx) ** 2) < (3 ** 2)
                    img[spot_mask] = [40, 40, 45]

            # Screw center hole
            center_mask = dist < (radius * 0.15)
            img[center_mask] = [20, 20, 25]

        else:
            # Head camera: landscape with sky + solar panels
            horizon = int(H * 0.35)

            # Sky gradient (blue to light blue)
            for row in range(horizon):
                t = row / max(horizon, 1)
                r = int(135 + t * 50)
                g = int(180 + t * 40)
                b = int(230 + t * 20)
                img[row, :] = [r, g, min(b, 255)]

            # Ground (sandy desert)
            for row in range(horizon, H):
                t = (row - horizon) / max(H - horizon, 1)
                r = int(190 - t * 30)
                g = int(170 - t * 25)
                b = int(130 - t * 20)
                img[row, :] = [r, g, b]

            # Solar panel (dark rectangle in lower-mid area)
            panel_top = int(H * 0.4)
            panel_bot = int(H * 0.7)
            panel_left = int(W * 0.2)
            panel_right = int(W * 0.8)
            img[panel_top:panel_bot, panel_left:panel_right] = [30, 30, 50]

            # Panel grid lines
            for gx in np.linspace(panel_left, panel_right, 8, dtype=int):
                if 0 <= gx < W:
                    img[panel_top:panel_bot, gx] = [60, 60, 80]
            for gy in np.linspace(panel_top, panel_bot, 5, dtype=int):
                if 0 <= gy < H:
                    img[gy, panel_left:panel_right] = [60, 60, 80]

            # Mounting bracket screws (small circles along panel edges)
            for sx in np.linspace(panel_left + 20, panel_right - 20, 6):
                for sy in [panel_top + 10, panel_bot - 10]:
                    sx_i, sy_i = int(sx), int(sy)
                    yy, xx = np.ogrid[:H, :W]
                    screw_mask = ((yy - sy_i) ** 2 + (xx - sx_i) ** 2) < (5 ** 2)
                    img[screw_mask] = [100, 100, 110]

        return img

    def _render_depth(
        self,
        H: int,
        W: int,
        ee_position: Optional[np.ndarray],
        screw_positions: Optional[list[np.ndarray]],
    ) -> np.ndarray:
        """Render a synthetic depth map (float32, meters)."""
        is_wrist = "wrist" in self.cfg.name

        if is_wrist:
            # Close-up: mostly flat surface at ~0.15m with screw indentation
            base_depth = 0.15 + self.rng.uniform(-0.02, 0.02)
            depth = np.full((H, W), base_depth, dtype=np.float32)

            # Screw hole depression
            cy, cx = H // 2, W // 2
            yy, xx = np.ogrid[:H, :W]
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            screw_r = 35.0
            depression = np.clip(1.0 - dist / screw_r, 0, 1) * 0.008
            depth += depression

        else:
            # Head camera: gradient depth (sky=far, ground=mid, panel=close)
            depth = np.full((H, W), 5.0, dtype=np.float32)
            horizon = int(H * 0.35)

            # Sky = max range
            depth[:horizon, :] = self.cfg.depth_max

            # Ground gradient
            for row in range(horizon, H):
                t = (row - horizon) / max(H - horizon, 1)
                depth[row, :] = 5.0 - t * 3.0  # 5m → 2m

            # Solar panel closer
            panel_top = int(H * 0.4)
            panel_bot = int(H * 0.7)
            panel_left = int(W * 0.2)
            panel_right = int(W * 0.8)
            depth[panel_top:panel_bot, panel_left:panel_right] = 1.5 + self.rng.uniform(-0.1, 0.1)

        depth = np.clip(depth, self.cfg.depth_min, self.cfg.depth_max)
        return depth

    def reset(self) -> None:
        """Reset frame counter and timestamp."""
        self._frame_count = 0
        self._t0 = time.monotonic()

    @property
    def info(self) -> dict:
        return {
            "name": self.cfg.name,
            "resolution": self.cfg.resolution,
            "fps": self.cfg.fps,
            "depth_range": (self.cfg.depth_min, self.cfg.depth_max),
            "parent_link": self.cfg.parent_link,
        }


# ---------------------------------------------------------------------------
# Mock FT Sensor
# ---------------------------------------------------------------------------

class TaskPhase(Enum):
    """Task phases that affect FT sensor readings."""
    IDLE = auto()
    APPROACH = auto()
    CONTACT = auto()
    SCREW_DRIVING = auto()
    DONE = auto()


@dataclass
class FTProfile:
    """Force/torque profile for a task phase."""
    force_mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    force_std: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    torque_mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    torque_std: Tuple[float, float, float] = (0.01, 0.01, 0.01)


# Phase-specific FT profiles (physically plausible values)
WRIST_FT_PROFILES: Dict[TaskPhase, FTProfile] = {
    TaskPhase.IDLE: FTProfile(
        force_mean=(0.0, 0.0, -2.0),    # Gravity of EE
        force_std=(0.2, 0.2, 0.2),
        torque_mean=(0.0, 0.0, 0.0),
        torque_std=(0.01, 0.01, 0.01),
    ),
    TaskPhase.APPROACH: FTProfile(
        force_mean=(0.0, 0.0, -2.0),    # Still mostly gravity
        force_std=(0.3, 0.3, 0.3),
        torque_mean=(0.0, 0.0, 0.0),
        torque_std=(0.02, 0.02, 0.02),
    ),
    TaskPhase.CONTACT: FTProfile(
        force_mean=(0.5, 0.3, -8.0),    # Pushing against panel
        force_std=(1.0, 1.0, 2.0),
        torque_mean=(0.1, 0.1, 0.0),
        torque_std=(0.05, 0.05, 0.02),
    ),
    TaskPhase.SCREW_DRIVING: FTProfile(
        force_mean=(0.3, 0.2, -12.0),   # Maintaining contact + driving
        force_std=(0.8, 0.8, 1.5),
        torque_mean=(0.1, 0.1, 5.0),    # Screw driving torque on Tz
        torque_std=(0.1, 0.1, 1.5),
    ),
    TaskPhase.DONE: FTProfile(
        force_mean=(0.0, 0.0, -2.0),
        force_std=(0.2, 0.2, 0.2),
        torque_mean=(0.0, 0.0, 0.0),
        torque_std=(0.01, 0.01, 0.01),
    ),
}

EE_TIP_FT_PROFILES: Dict[TaskPhase, FTProfile] = {
    TaskPhase.IDLE: FTProfile(
        force_mean=(0.0, 0.0, 0.0),
        force_std=(0.05, 0.05, 0.05),
        torque_mean=(0.0, 0.0, 0.0),
        torque_std=(0.005, 0.005, 0.005),
    ),
    TaskPhase.APPROACH: FTProfile(
        force_mean=(0.0, 0.0, 0.0),
        force_std=(0.05, 0.05, 0.05),
        torque_mean=(0.0, 0.0, 0.0),
        torque_std=(0.005, 0.005, 0.005),
    ),
    TaskPhase.CONTACT: FTProfile(
        force_mean=(0.2, 0.1, -5.0),    # Direct contact with screw
        force_std=(0.5, 0.5, 1.0),
        torque_mean=(0.02, 0.02, 0.0),
        torque_std=(0.02, 0.02, 0.01),
    ),
    TaskPhase.SCREW_DRIVING: FTProfile(
        force_mean=(0.1, 0.1, -6.0),    # Maintaining engagement
        force_std=(0.3, 0.3, 0.8),
        torque_mean=(0.05, 0.05, 8.0),  # Screw resistance torque
        torque_std=(0.05, 0.05, 2.0),
    ),
    TaskPhase.DONE: FTProfile(
        force_mean=(0.0, 0.0, 0.0),
        force_std=(0.05, 0.05, 0.05),
        torque_mean=(0.0, 0.0, 0.0),
        torque_std=(0.005, 0.005, 0.005),
    ),
}


class MockFTSensor:
    """Generate synthetic 6D force/torque readings.

    Produces task-phase-aware readings with:
    - Phase-specific mean + std profiles
    - Smooth transitions between phases
    - Configurable noise and bias
    - Ramp-up behavior for screw driving torque
    """

    def __init__(
        self,
        config: FTSensorConfig,
        profiles: Optional[Dict[TaskPhase, FTProfile]] = None,
        seed: int = 42,
    ):
        self.cfg = config
        self.profiles = profiles or WRIST_FT_PROFILES
        self.rng = np.random.default_rng(seed)
        self._phase = TaskPhase.IDLE
        self._step = 0
        self._screw_progress = 0.0  # 0.0 to 1.0
        self._prev_reading = np.zeros(6, dtype=np.float32)
        self._bias = np.array(
            list(config.force_bias) + list(config.torque_bias),
            dtype=np.float32,
        )

    def set_phase(self, phase: TaskPhase) -> None:
        """Set the current task phase (affects FT profile)."""
        self._phase = phase

    def set_screw_progress(self, progress: float) -> None:
        """Set screw driving progress [0, 1] for torque ramp-up."""
        self._screw_progress = np.clip(progress, 0.0, 1.0)

    def read(self) -> Dict[str, Any]:
        """Read one FT sample.

        Returns:
            Dict with keys:
                ``force``: (3,) array [Fx, Fy, Fz] in Newtons
                ``torque``: (3,) array [Tx, Ty, Tz] in N·m
                ``wrench``: (6,) array [Fx, Fy, Fz, Tx, Ty, Tz]
                ``timestamp``: sample timestamp
                ``step``: monotonic step counter
                ``phase``: current task phase name
        """
        self._step += 1

        profile = self.profiles[self._phase]

        # Base reading from profile
        force = np.array(profile.force_mean, dtype=np.float32)
        torque = np.array(profile.torque_mean, dtype=np.float32)

        # Scale screw driving torque by progress (ramp up)
        if self._phase == TaskPhase.SCREW_DRIVING:
            # Torque increases as screw tightens (exponential ramp)
            torque_scale = 0.3 + 0.7 * (self._screw_progress ** 1.5)
            torque *= torque_scale
            # Force also increases slightly
            force[2] *= (0.8 + 0.2 * self._screw_progress)

        # Add noise
        force_noise = self.rng.normal(0, profile.force_std, 3).astype(np.float32)
        torque_noise = self.rng.normal(0, profile.torque_std, 3).astype(np.float32)

        force += force_noise
        torque += torque_noise

        # Combine into wrench
        wrench = np.concatenate([force, torque])

        # Add bias
        wrench += self._bias

        # Add sensor noise
        sensor_noise = self.rng.normal(0, self.cfg.full_noise_std, 6).astype(np.float32)
        wrench += sensor_noise

        # Clip to sensor range
        wrench = np.clip(wrench, -self.cfg.full_range, self.cfg.full_range)

        # Smooth with previous reading (low-pass filter, alpha=0.7)
        alpha = 0.7
        wrench = alpha * wrench + (1 - alpha) * self._prev_reading
        self._prev_reading = wrench.copy()

        return {
            "force": wrench[:3].copy(),
            "torque": wrench[3:].copy(),
            "wrench": wrench.copy(),
            "timestamp": self._step / self.cfg.fps,
            "step": self._step,
            "phase": self._phase.name,
        }

    def reset(self) -> None:
        """Reset sensor state."""
        self._step = 0
        self._screw_progress = 0.0
        self._phase = TaskPhase.IDLE
        self._prev_reading = np.zeros(6, dtype=np.float32)

    @property
    def info(self) -> dict:
        return {
            "name": self.cfg.name,
            "fps": self.cfg.fps,
            "force_range": self.cfg.force_range,
            "torque_range": self.cfg.torque_range,
            "parent_link": self.cfg.parent_link,
        }


# ---------------------------------------------------------------------------
# Combined Sensor Suite
# ---------------------------------------------------------------------------

class MockSensorSuite:
    """All mock sensors for RoboForce in one bundle.

    Provides a single `step()` call that returns all sensor data
    in a unified dict, matching the format expected by the VLA
    training pipeline.

    Data format per step:
    ```python
    {
        "cameras": {
            "head_left":  {"rgb": (480,640,3), "depth": (480,640), ...},
            "head_right": {"rgb": (480,640,3), "depth": (480,640), ...},
            "wrist":      {"rgb": (480,640,3), "depth": (480,640), ...},
        },
        "ft_sensors": {
            "wrist_ft":  {"wrench": (6,), "force": (3,), "torque": (3,), ...},
            "ee_tip_ft": {"wrench": (6,), "force": (3,), "torque": (3,), ...},
        },
        "timestamp": float,
        "step": int,
    }
    ```
    """

    def __init__(self, config: Optional[SensorConfig] = None, seed: int = 42):
        self.cfg = config or SensorConfig()

        # Create cameras
        self.cameras: Dict[str, MockRGBDCamera] = {}
        for cam_cfg in self.cfg.cameras:
            self.cameras[cam_cfg.name] = MockRGBDCamera(cam_cfg, seed=seed)

        # Create FT sensors
        self.ft_sensors: Dict[str, MockFTSensor] = {}
        wrist_ft_cfg = self.cfg.wrist_ft
        self.ft_sensors[wrist_ft_cfg.name] = MockFTSensor(
            wrist_ft_cfg, profiles=WRIST_FT_PROFILES, seed=seed + 1
        )
        ee_ft_cfg = self.cfg.ee_tip_ft
        self.ft_sensors[ee_ft_cfg.name] = MockFTSensor(
            ee_ft_cfg, profiles=EE_TIP_FT_PROFILES, seed=seed + 2
        )

        self._step_count = 0

    def set_task_phase(self, phase: TaskPhase) -> None:
        """Set task phase for all FT sensors."""
        for ft in self.ft_sensors.values():
            ft.set_phase(phase)

    def set_screw_progress(self, progress: float) -> None:
        """Set screw driving progress for all FT sensors."""
        for ft in self.ft_sensors.values():
            ft.set_screw_progress(progress)

    def step(
        self,
        ee_position: Optional[np.ndarray] = None,
        screw_positions: Optional[list[np.ndarray]] = None,
        robot_state: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Capture all sensor data for one timestep.

        Args:
            ee_position: EE position for context-aware rendering.
            screw_positions: Screw positions for camera projection.
            robot_state: Joint state for state-dependent FT profiles.

        Returns:
            Unified sensor data dict (see class docstring for format).
        """
        self._step_count += 1

        cam_data = {}
        for name, cam in self.cameras.items():
            cam_data[name] = cam.capture(ee_position, screw_positions, robot_state)

        ft_data = {}
        for name, ft in self.ft_sensors.items():
            ft_data[name] = ft.read()

        return {
            "cameras": cam_data,
            "ft_sensors": ft_data,
            "timestamp": time.monotonic(),
            "step": self._step_count,
        }

    def reset(self) -> None:
        """Reset all sensors."""
        self._step_count = 0
        for cam in self.cameras.values():
            cam.reset()
        for ft in self.ft_sensors.values():
            ft.reset()

    @property
    def info(self) -> dict:
        return {
            "cameras": {n: c.info for n, c in self.cameras.items()},
            "ft_sensors": {n: f.info for n, f in self.ft_sensors.items()},
        }
