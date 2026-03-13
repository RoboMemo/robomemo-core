"""
Isaac Lab simulation environment wrapper for humanoid robots.

Provides two backends:
  1. IsaacLabEnv — Full Isaac Lab / Isaac Sim physics (requires Isaac Sim 4.5+)
  2. MockPhysicsEnv — Lightweight numpy-based physics for testing without Isaac Sim

Supported robots: Unitree G1 (29 DOF), Unitree H1 (19 DOF), Fourier GR1T2 (32 DOF)

Both backends:
  - Accept joint target positions (N-D, depends on robot)
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
        robot_type = cfg["robot"]["type"]
        robot_cfg = cfg["robot"][robot_type]
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

# Robot config class mapping for Isaac Lab 2.x
_ISAAC_ROBOT_CONFIGS = {
    "unitree_g1": ("isaaclab_assets.robots.unitree", "G1_29DOF_CFG"),
    "unitree_h1": ("isaaclab_assets.robots.unitree", "H1_CFG"),
    "fourier_gr1t2": ("isaaclab_assets.robots.fourier", "GR1T2_CFG"),
}


class IsaacLabEnv(SimEnv):
    """Full Isaac Lab physics environment for humanoid robots.

    Requires Isaac Sim 4.5+ and Isaac Lab 2.x installed.
    Supports: Unitree G1, Unitree H1, Fourier GR1T2.
    """

    def __init__(self):
        self._sim = None
        self._robot = None
        self._num_joints = 29
        self._step_count = 0
        self._robot_type = "unitree_g1"

    def init(self, cfg: dict) -> None:
        try:
            import torch  # noqa: F401
            from isaacsim import SimulationApp
        except ImportError as e:
            raise ImportError(
                f"Isaac Sim not available: {e}\n"
                "Install: pip install 'isaacsim==4.5.0.0' "
                "'isaacsim-extscache-physics==4.5.0.0' "
                "'isaacsim-extscache-kit-sdk==4.5.0.0' "
                "--extra-index-url https://pypi.nvidia.com"
            ) from e

        isaac_cfg = cfg["simulation"]["isaac_lab"]
        self._robot_type = cfg["robot"]["type"]
        robot_cfg = cfg["robot"][self._robot_type]
        self._num_joints = robot_cfg.get("num_joints", 29)

        # Launch Isaac Sim headless or with GUI
        headless = isaac_cfg.get("headless", True)
        logger.info(f"IsaacLabEnv: Launching SimulationApp (headless={headless})…")
        self._simulation_app = SimulationApp({"headless": headless})

        try:
            import importlib
            import omni.isaac.lab.sim as sim_utils
            from omni.isaac.lab.sim import SimulationCfg, SimulationContext
            from omni.isaac.lab.assets import Articulation
            from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
            from omni.isaac.lab.utils import configclass

            # Dynamically load robot config from isaaclab_assets
            if self._robot_type not in _ISAAC_ROBOT_CONFIGS:
                raise ValueError(
                    f"No Isaac Lab asset config for robot '{self._robot_type}'. "
                    f"Available: {list(_ISAAC_ROBOT_CONFIGS.keys())}"
                )
            module_path, cfg_name = _ISAAC_ROBOT_CONFIGS[self._robot_type]
            mod = importlib.import_module(module_path)
            robot_asset_cfg = getattr(mod, cfg_name)
            logger.info(f"IsaacLabEnv: Loaded {cfg_name} from {module_path}")

            # Create simulation context
            sim_cfg = SimulationCfg(
                dt=isaac_cfg.get("sim_dt", 0.02),
                render_interval=isaac_cfg.get("render_interval", 2),
                device="cuda:0" if isaac_cfg.get("gpu_physics", True) else "cpu",
            )
            self._sim = SimulationContext(sim_cfg)
            self._sim.set_camera_view([2.5, 2.5, 2.0], [0.0, 0.0, 0.8])

            # Spawn ground plane
            ground_cfg = sim_utils.GroundPlaneCfg()
            ground_cfg.func("/World/ground", ground_cfg)

            # Spawn robot
            robot_cfg_spawned = robot_asset_cfg.replace(
                prim_path="/World/Robot"
            )
            self._robot = Articulation(robot_cfg_spawned)

            # Initialize simulation
            self._sim.reset()
            self._robot.reset()

            # Get actual DOF count from loaded robot
            actual_dof = self._robot.data.joint_pos.shape[1]
            if actual_dof != self._num_joints:
                logger.warning(
                    f"Config says {self._num_joints} joints but robot has "
                    f"{actual_dof} — using actual count"
                )
                self._num_joints = actual_dof

            logger.info(
                f"IsaacLabEnv: {self._robot_type} loaded successfully "
                f"({self._num_joints} DOF)"
            )

        except Exception as e:
            if self._simulation_app is not None:
                self._simulation_app.close()
            raise RuntimeError(f"Failed to initialise Isaac Lab: {e}") from e

    def step(self, joint_targets: np.ndarray) -> None:
        import torch
        targets = np.asarray(joint_targets, dtype=np.float32)
        if len(targets) != self._num_joints:
            targets = np.resize(targets, self._num_joints)
        targets_t = torch.tensor(
            targets, dtype=torch.float32, device="cuda:0"
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
        if hasattr(self, '_simulation_app') and self._simulation_app is not None:
            self._simulation_app.close()
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
