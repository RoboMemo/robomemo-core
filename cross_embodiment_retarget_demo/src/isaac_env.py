"""
Isaac Lab simulation environment wrapper for the Unitree G1.

Provides two backends:
  1. IsaacLabEnv — Full Isaac Lab / Isaac Sim physics (requires Isaac Sim installed)
  2. MockPhysicsEnv — Lightweight numpy-based physics for testing without Isaac Sim

Both backends:
  - Accept joint target positions (29D)
  - Apply PD control
  - Return joint states (position + velocity)
  - Provide optional visualisation
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class SimEnv(ABC):
    """Abstract simulation environment interface."""

    @abstractmethod
    def init(self, cfg: dict) -> None:
        ...

    @abstractmethod
    def step(self, joint_targets: np.ndarray) -> None:
        """Advance simulation by one timestep with the given joint targets."""
        ...

    @abstractmethod
    def get_joint_pos(self) -> np.ndarray:
        ...

    @abstractmethod
    def get_joint_vel(self) -> np.ndarray:
        ...

    @abstractmethod
    def get_gravity_vector(self) -> np.ndarray:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


# ── Mock physics (no Isaac Sim required) ──────────────────────

class MockPhysicsEnv(SimEnv):
    """Simple PD-controlled rigid-body simulation.

    No collision or contact — just joint dynamics with damping.
    Good enough to verify the retarget pipeline end-to-end.
    """

    def __init__(self):
        self._num_joints = 29
        self._dt = 0.02
        self._kp = 100.0
        self._kd = 2.0
        self._joint_pos = np.zeros(29, dtype=np.float32)
        self._joint_vel = np.zeros(29, dtype=np.float32)
        self._gravity = np.array([0, 0, -9.81], dtype=np.float32)
        self._step_count = 0

        # Visualisation (optional)
        self._viz_enabled = False
        self._viz_thread: Optional[threading.Thread] = None
        self._viz_running = False

    def init(self, cfg: dict) -> None:
        robot_cfg = cfg["robot"]["unitree_g1"]
        sim_cfg = cfg["simulation"]["mock_physics"]

        self._num_joints = robot_cfg.get("num_joints", 29)
        self._kp = robot_cfg.get("kp", 100.0)
        self._kd = robot_cfg.get("kd", 2.0)
        self._dt = sim_cfg.get("dt", 0.02)

        default_pose = robot_cfg.get("default_pose", [0.0] * self._num_joints)
        self._joint_pos = np.array(default_pose, dtype=np.float32)
        self._joint_vel = np.zeros(self._num_joints, dtype=np.float32)

        grav = sim_cfg.get("gravity", [0, 0, -9.81])
        self._gravity = np.array(grav, dtype=np.float32)

        self._viz_enabled = sim_cfg.get("enable_visualization", False)
        logger.info(
            f"MockPhysicsEnv initialised: {self._num_joints} joints, "
            f"dt={self._dt}s, kp={self._kp}, kd={self._kd}"
        )

    def step(self, joint_targets: np.ndarray) -> None:
        targets = np.asarray(joint_targets, dtype=np.float32)
        if len(targets) != self._num_joints:
            targets = np.resize(targets, self._num_joints)

        # PD control with gravity compensation (simplified)
        error = targets - self._joint_pos
        torque = self._kp * error - self._kd * self._joint_vel

        # Simple Euler integration (mass = 1 for each joint)
        accel = torque  # assuming unit inertia
        self._joint_vel += accel * self._dt
        # Damping
        self._joint_vel *= 0.98
        self._joint_pos += self._joint_vel * self._dt

        self._step_count += 1

    def get_joint_pos(self) -> np.ndarray:
        return self._joint_pos.copy()

    def get_joint_vel(self) -> np.ndarray:
        return self._joint_vel.copy()

    def get_gravity_vector(self) -> np.ndarray:
        return self._gravity.copy()

    def close(self) -> None:
        self._viz_running = False
        logger.info(f"MockPhysicsEnv closed after {self._step_count} steps")


# ── Isaac Lab environment (requires Isaac Sim) ────────────────

class IsaacLabEnv(SimEnv):
    """Full Isaac Lab DirectRLEnv wrapper for Unitree G1.

    Requires Isaac Sim 4.5+ and Isaac Lab 2.3+ installed.
    """

    def __init__(self):
        self._env = None
        self._num_joints = 29
        self._step_count = 0

    def init(self, cfg: dict) -> None:
        try:
            # Isaac Lab imports (only available when Isaac Sim is running)
            import torch
            from omni.isaac.lab.envs import DirectRLEnvCfg
            from omni.isaac.lab.scene import InteractiveSceneCfg
            from omni.isaac.lab.sim import SimulationCfg
            from omni.isaac.lab.utils import configclass
        except ImportError as e:
            raise ImportError(
                f"Isaac Lab not available: {e}\n"
                "Install Isaac Sim 4.5+ and Isaac Lab 2.3+ first.\n"
                "See: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html"
            ) from e

        isaac_cfg = cfg["simulation"]["isaac_lab"]
        robot_cfg = cfg["robot"]["unitree_g1"]
        self._num_joints = robot_cfg.get("num_joints", 29)

        # TODO: Create proper DirectRLEnv subclass with G1 USD asset
        # For now, initialise Isaac Sim and load basic scene
        logger.info("IsaacLabEnv: Initialising Isaac Sim scene…")

        try:
            from omni.isaac.lab.app import AppLauncher
            launcher = AppLauncher(headless=isaac_cfg.get("headless", False))
            simulation_app = launcher.app

            import omni.isaac.lab.sim as sim_utils
            from omni.isaac.lab.assets import ArticulationCfg, Articulation
            from omni.isaac.lab_assets.unitree import G1_29DOF_CFG

            # Create simulation context
            sim_cfg = SimulationCfg(
                dt=isaac_cfg.get("sim_dt", 0.02),
                render_interval=isaac_cfg.get("render_interval", 2),
                device="cuda:0" if isaac_cfg.get("gpu_physics", True) else "cpu",
            )

            # Spawn G1 robot
            robot_cfg_isaac = G1_29DOF_CFG.replace(prim_path="/World/G1")
            self._robot = Articulation(robot_cfg_isaac)

            # Build scene
            sim = sim_utils.SimulationContext(sim_cfg)
            sim.set_camera_view([2.0, 0, 1.5], [0, 0, 0.8])

            self._sim = sim
            self._sim.reset()
            self._robot.reset()
            logger.info("IsaacLabEnv: G1 robot loaded and scene ready")

        except Exception as e:
            raise RuntimeError(f"Failed to initialise Isaac Lab: {e}") from e

    def step(self, joint_targets: np.ndarray) -> None:
        import torch
        targets_t = torch.tensor(
            joint_targets, dtype=torch.float32, device="cuda:0"
        ).unsqueeze(0)
        self._robot.set_joint_position_target(targets_t)
        self._sim.step()
        self._robot.update(self._sim.get_physics_dt())
        self._step_count += 1

    def get_joint_pos(self) -> np.ndarray:
        return self._robot.data.joint_pos[0].cpu().numpy()

    def get_joint_vel(self) -> np.ndarray:
        return self._robot.data.joint_vel[0].cpu().numpy()

    def get_gravity_vector(self) -> np.ndarray:
        return np.array([0, 0, -9.81], dtype=np.float32)

    def close(self) -> None:
        if self._sim is not None:
            self._sim.close()
        logger.info(f"IsaacLabEnv closed after {self._step_count} steps")


# ── Factory ───────────────────────────────────────────────────

def create_sim_env(cfg: dict) -> SimEnv:
    """Create the appropriate simulation backend."""
    backend = cfg["simulation"]["backend"]
    if backend == "isaac_lab":
        env = IsaacLabEnv()
    elif backend == "mock_physics":
        env = MockPhysicsEnv()
    else:
        raise ValueError(f"Unknown simulation backend: {backend}")
    env.init(cfg)
    return env
