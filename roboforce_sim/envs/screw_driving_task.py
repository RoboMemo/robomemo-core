# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Screw driving task definition for Isaac Lab ManagerBasedEnv.

Defines the observation space, action space, reward function, and termination
conditions for the screw installation task. This module is designed to work
with the ``SolarPanelEnv`` scene.

Observation space (policy group):
    - Joint positions (N_DOF)
    - Joint velocities (N_DOF)
    - EE position (3)
    - EE orientation quaternion (4)
    - EE force/torque (6)
    - Relative vector EE → target screw (3)
    - Screw rotation progress (1)
    Total: 2 * N_DOF + 17

Action space (8D):
    - EE delta position (3)
    - EE delta orientation (3, axis-angle)
    - Screw rotation command (1)
    - Gripper command (1)

Reward components:
    - Approach: dense reward for reducing EE-screw distance
    - Contact: reward when EE is within engagement threshold
    - Screw driving: reward proportional to rotation progress
    - Completion: large bonus for fully tightened screw
    - Force penalty: penalize excessive contact forces
    - Collision penalty: penalize body collisions with rack
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.utils import configclass

    ISAACLAB_AVAILABLE = True
except ImportError:
    ISAACLAB_AVAILABLE = False

    def configclass(cls):  # type: ignore[misc]
        return dataclass(cls)


# ---------------------------------------------------------------------------
# Task Configuration
# ---------------------------------------------------------------------------

@configclass
class ScrewDrivingRewardCfg:
    """Reward function weights and parameters."""

    # Approach reward: encourage EE to move toward screw
    approach_weight: float = 1.0
    approach_distance_threshold: float = 0.05
    """Distance (m) below which approach reward saturates."""
    approach_temperature: float = 0.02
    """Exponential decay scale for approach reward."""

    # Contact reward: reward when EE touches the screw head
    contact_weight: float = 2.0
    contact_threshold: float = 0.01
    """Distance (m) below which contact is considered established."""

    # Screw driving reward: proportional to rotation progress
    screw_driving_weight: float = 5.0
    progress_scale: float = 10.0
    """Reward per unit of normalized rotation progress (0→1)."""

    # Completion bonus
    completion_weight: float = 100.0
    """One-time bonus when screw is fully tightened."""

    # Force penalty: discourage excessive force
    force_penalty_weight: float = -0.1
    force_threshold: float = 40.0
    """Force (N) above which penalty is applied."""

    # Collision penalty: penalize collisions with rack structure
    collision_penalty_weight: float = -10.0


@configclass
class ScrewDrivingTerminationCfg:
    """Termination conditions."""

    terminate_on_success: bool = True
    terminate_on_timeout: bool = True
    terminate_on_collision: bool = True

    max_collision_force: float = 100.0
    """Force (N) threshold for collision-based termination."""

    timeout_steps: int = 3000
    """Max steps before timeout (at 50 Hz control → 60s)."""


@configclass
class ScrewDrivingTaskCfg:
    """Full task configuration for screw driving."""

    # Reward
    rewards: ScrewDrivingRewardCfg = field(default_factory=ScrewDrivingRewardCfg)

    # Termination
    terminations: ScrewDrivingTerminationCfg = field(
        default_factory=ScrewDrivingTerminationCfg
    )

    # Action scaling
    ee_delta_position_scale: float = 0.01
    """Max EE position change per step (meters)."""
    ee_delta_orientation_scale: float = 0.05
    """Max EE orientation change per step (radians)."""
    screw_rotation_scale: float = 0.5
    """Max screw rotation per step (radians)."""
    gripper_scale: float = 1.0

    # Screw driving target
    target_turns: float = 8.0
    """Number of full turns to tighten the screw."""
    target_torque_nm: float = 10.0
    """Target tightening torque (N·m)."""

    # Observation
    include_camera: bool = True
    camera_width: int = 640
    camera_height: int = 480


# ---------------------------------------------------------------------------
# Task Implementation
# ---------------------------------------------------------------------------

class ScrewDrivingTask:
    """Screw driving task manager.

    Computes observations, rewards, and terminations for the screw
    installation task. Designed to be used inside a ManagerBasedEnv
    or as a standalone task module.

    Args:
        cfg: Task configuration.
        num_envs: Number of parallel environments.
        device: Torch device string.
    """

    def __init__(
        self,
        cfg: ScrewDrivingTaskCfg | None = None,
        num_envs: int = 64,
        device: str = "cuda:0",
    ):
        self.cfg = cfg or ScrewDrivingTaskCfg()
        self.num_envs = num_envs
        self.device = device

        # State buffers (per-environment)
        self._use_torch = ISAACLAB_AVAILABLE
        if self._use_torch:
            self.screw_progress = torch.zeros(num_envs, device=device)
            self.cumulative_rotation = torch.zeros(num_envs, device=device)
            self.episode_step = torch.zeros(num_envs, dtype=torch.long, device=device)
            self.is_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
            self.is_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        else:
            self.screw_progress = np.zeros(num_envs)
            self.cumulative_rotation = np.zeros(num_envs)
            self.episode_step = np.zeros(num_envs, dtype=np.int64)
            self.is_done = np.zeros(num_envs, dtype=bool)
            self.is_success = np.zeros(num_envs, dtype=bool)

        # Target rotation for full tightening (radians)
        self.target_rotation = self.cfg.target_turns * 2.0 * math.pi

    # ----- Observations -----

    def compute_observations(
        self,
        joint_positions: Any,
        joint_velocities: Any,
        ee_position: Any,
        ee_orientation: Any,
        ee_force_torque: Any,
        target_screw_position: Any,
    ) -> dict[str, Any]:
        """Compute the observation dictionary.

        All inputs should be tensors of shape ``(num_envs, dim)`` (or numpy
        arrays in offline mode).

        Args:
            joint_positions: Joint angles, shape ``(N, num_dof)``.
            joint_velocities: Joint velocities, shape ``(N, num_dof)``.
            ee_position: EE world position, shape ``(N, 3)``.
            ee_orientation: EE quaternion ``(w, x, y, z)``, shape ``(N, 4)``.
            ee_force_torque: Force/torque at EE, shape ``(N, 6)``.
            target_screw_position: Target screw world position, shape ``(N, 3)``.

        Returns:
            Dict with ``"policy"`` key containing the flat observation vector,
            and optionally ``"images"`` for camera data.
        """
        # Relative vector: EE → screw
        relative_pos = target_screw_position - ee_position  # (N, 3)

        # Screw progress (0 to 1)
        if self._use_torch:
            progress = self.screw_progress.unsqueeze(-1)  # (N, 1)
            obs_flat = torch.cat([
                joint_positions,
                joint_velocities,
                ee_position,
                ee_orientation,
                ee_force_torque,
                relative_pos,
                progress,
            ], dim=-1)
        else:
            progress = self.screw_progress[:, np.newaxis]
            obs_flat = np.concatenate([
                joint_positions,
                joint_velocities,
                ee_position,
                ee_orientation,
                ee_force_torque,
                relative_pos,
                progress,
            ], axis=-1)

        return {"policy": obs_flat}

    # ----- Actions -----

    def process_actions(self, raw_actions: Any) -> dict[str, Any]:
        """Process raw actions ([-1, 1]) into scaled commands.

        Args:
            raw_actions: Shape ``(N, 8)`` — clipped to [-1, 1].

        Returns:
            Dict with:
            - ``"ee_delta_pos"``: (N, 3) position deltas
            - ``"ee_delta_ori"``: (N, 3) orientation deltas (axis-angle)
            - ``"screw_rotation"``: (N, 1) rotation command
            - ``"gripper"``: (N, 1) gripper command
        """
        if self._use_torch:
            actions = torch.clamp(raw_actions, -1.0, 1.0)
        else:
            actions = np.clip(raw_actions, -1.0, 1.0)

        return {
            "ee_delta_pos": actions[..., :3] * self.cfg.ee_delta_position_scale,
            "ee_delta_ori": actions[..., 3:6] * self.cfg.ee_delta_orientation_scale,
            "screw_rotation": actions[..., 6:7] * self.cfg.screw_rotation_scale,
            "gripper": actions[..., 7:8] * self.cfg.gripper_scale,
        }

    # ----- Rewards -----

    def compute_rewards(
        self,
        ee_position: Any,
        target_screw_position: Any,
        ee_force_torque: Any,
        screw_rotation_delta: Any,
        collision_force: Any | None = None,
    ) -> dict[str, Any]:
        """Compute reward components.

        Args:
            ee_position: EE world position, shape ``(N, 3)``.
            target_screw_position: Target screw position, shape ``(N, 3)``.
            ee_force_torque: Force/torque at EE, shape ``(N, 6)``.
            screw_rotation_delta: Rotation applied this step, shape ``(N,)``.
            collision_force: Collision force magnitude, shape ``(N,)`` or None.

        Returns:
            Dict of named reward components + ``"total"`` key.
        """
        rcfg = self.cfg.rewards

        # Distance EE → screw
        diff = target_screw_position - ee_position
        if self._use_torch:
            distance = torch.norm(diff, dim=-1)
            force_mag = torch.norm(ee_force_torque[..., :3], dim=-1)
        else:
            distance = np.linalg.norm(diff, axis=-1)
            force_mag = np.linalg.norm(ee_force_torque[..., :3], axis=-1)

        # 1. Approach reward (exponential decay)
        if self._use_torch:
            approach = torch.exp(-distance / rcfg.approach_temperature)
        else:
            approach = np.exp(-distance / rcfg.approach_temperature)
        approach_reward = rcfg.approach_weight * approach

        # 2. Contact reward (binary: within threshold)
        if self._use_torch:
            in_contact = (distance < rcfg.contact_threshold).float()
        else:
            in_contact = (distance < rcfg.contact_threshold).astype(float)
        contact_reward = rcfg.contact_weight * in_contact

        # 3. Screw driving reward (incremental progress)
        # Update cumulative rotation
        if self._use_torch:
            self.cumulative_rotation += torch.abs(screw_rotation_delta) * in_contact
            self.screw_progress = torch.clamp(
                self.cumulative_rotation / self.target_rotation, 0.0, 1.0
            )
            driving_reward = rcfg.screw_driving_weight * (
                torch.abs(screw_rotation_delta) * in_contact / self.target_rotation
            ) * rcfg.progress_scale
        else:
            self.cumulative_rotation += np.abs(screw_rotation_delta) * in_contact
            self.screw_progress = np.clip(
                self.cumulative_rotation / self.target_rotation, 0.0, 1.0
            )
            driving_reward = rcfg.screw_driving_weight * (
                np.abs(screw_rotation_delta) * in_contact / self.target_rotation
            ) * rcfg.progress_scale

        # 4. Completion bonus
        if self._use_torch:
            just_completed = (self.screw_progress >= 1.0) & (~self.is_success)
            completion_reward = rcfg.completion_weight * just_completed.float()
            self.is_success = self.is_success | (self.screw_progress >= 1.0)
        else:
            just_completed = (self.screw_progress >= 1.0) & (~self.is_success)
            completion_reward = rcfg.completion_weight * just_completed.astype(float)
            self.is_success = self.is_success | (self.screw_progress >= 1.0)

        # 5. Force penalty
        if self._use_torch:
            excess_force = torch.clamp(force_mag - rcfg.force_threshold, min=0.0)
        else:
            excess_force = np.maximum(force_mag - rcfg.force_threshold, 0.0)
        force_penalty = rcfg.force_penalty_weight * excess_force

        # 6. Collision penalty
        if collision_force is not None:
            if self._use_torch:
                collision_penalty = rcfg.collision_penalty_weight * (
                    collision_force > rcfg.force_threshold
                ).float()
            else:
                collision_penalty = rcfg.collision_penalty_weight * (
                    collision_force > rcfg.force_threshold
                ).astype(float)
        else:
            if self._use_torch:
                collision_penalty = torch.zeros(self.num_envs, device=self.device)
            else:
                collision_penalty = np.zeros(self.num_envs)

        total = (
            approach_reward
            + contact_reward
            + driving_reward
            + completion_reward
            + force_penalty
            + collision_penalty
        )

        return {
            "approach": approach_reward,
            "contact": contact_reward,
            "screw_driving": driving_reward,
            "completion": completion_reward,
            "force_penalty": force_penalty,
            "collision_penalty": collision_penalty,
            "total": total,
        }

    # ----- Termination -----

    def compute_terminations(
        self,
        collision_force: Any | None = None,
    ) -> dict[str, Any]:
        """Compute termination signals.

        Args:
            collision_force: Collision force magnitude, shape ``(N,)`` or None.

        Returns:
            Dict with ``"terminated"`` (bool tensor/array) and ``"reason"`` info.
        """
        tcfg = self.cfg.terminations

        if self._use_torch:
            terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            terminated = np.zeros(self.num_envs, dtype=bool)
            truncated = np.zeros(self.num_envs, dtype=bool)

        # Success termination
        if tcfg.terminate_on_success:
            terminated = terminated | self.is_success

        # Timeout termination
        self.episode_step += 1
        if tcfg.terminate_on_timeout:
            if self._use_torch:
                truncated = truncated | (self.episode_step >= tcfg.timeout_steps)
            else:
                truncated = truncated | (self.episode_step >= tcfg.timeout_steps)

        # Collision termination
        if tcfg.terminate_on_collision and collision_force is not None:
            if self._use_torch:
                terminated = terminated | (collision_force > tcfg.max_collision_force)
            else:
                terminated = terminated | (collision_force > tcfg.max_collision_force)

        self.is_done = terminated | truncated

        return {
            "terminated": terminated,
            "truncated": truncated,
            "done": self.is_done,
        }

    # ----- Reset -----

    def reset(self, env_ids: Any | None = None) -> None:
        """Reset task state for the given environment IDs.

        Args:
            env_ids: Indices of environments to reset. If *None*, reset all.
        """
        if env_ids is None:
            if self._use_torch:
                env_ids = torch.arange(self.num_envs, device=self.device)
            else:
                env_ids = np.arange(self.num_envs)

        if self._use_torch:
            self.screw_progress[env_ids] = 0.0
            self.cumulative_rotation[env_ids] = 0.0
            self.episode_step[env_ids] = 0
            self.is_done[env_ids] = False
            self.is_success[env_ids] = False
        else:
            self.screw_progress[env_ids] = 0.0
            self.cumulative_rotation[env_ids] = 0.0
            self.episode_step[env_ids] = 0
            self.is_done[env_ids] = False
            self.is_success[env_ids] = False

    # ----- Metrics -----

    def get_metrics(self) -> dict[str, float]:
        """Compute aggregate metrics across all environments.

        Returns:
            Dict with success rate, mean progress, mean episode steps.
        """
        if self._use_torch:
            return {
                "success_rate": self.is_success.float().mean().item(),
                "mean_progress": self.screw_progress.mean().item(),
                "mean_episode_steps": self.episode_step.float().mean().item(),
            }
        else:
            return {
                "success_rate": float(self.is_success.mean()),
                "mean_progress": float(self.screw_progress.mean()),
                "mean_episode_steps": float(self.episode_step.mean()),
            }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Screw Driving Task — Offline Test")
    print("=" * 50)

    cfg = ScrewDrivingTaskCfg()
    task = ScrewDrivingTask(cfg, num_envs=4, device="cpu")

    # Simulate a few steps
    num_dof = 13
    for step in range(20):
        # Fake observations
        joint_pos = np.random.randn(4, num_dof) * 0.1
        joint_vel = np.random.randn(4, num_dof) * 0.01
        ee_pos = np.array([[0.0, -0.5, 1.5]] * 4) + np.random.randn(4, 3) * 0.02
        ee_ori = np.array([[1.0, 0, 0, 0]] * 4)
        ee_ft = np.random.randn(4, 6) * 5.0
        screw_pos = np.array([[0.0, -0.5, 1.5]] * 4)

        obs = task.compute_observations(joint_pos, joint_vel, ee_pos, ee_ori, ee_ft, screw_pos)

        # Fake actions
        actions = np.random.randn(4, 8) * 0.5
        processed = task.process_actions(actions)

        # Fake screw rotation from action
        screw_rot = processed["screw_rotation"].squeeze(-1)

        rewards = task.compute_rewards(
            ee_pos, screw_pos, ee_ft, screw_rot, collision_force=None
        )

        terms = task.compute_terminations()

        if step % 5 == 0:
            metrics = task.get_metrics()
            print(f"Step {step:3d}: reward={rewards['total'].mean():.3f}, "
                  f"progress={metrics['mean_progress']:.3f}, "
                  f"success={metrics['success_rate']:.1%}")

    print(f"\nObs shape: {obs['policy'].shape}")
    print(f"Action keys: {list(processed.keys())}")
    print(f"Reward keys: {list(rewards.keys())}")
    print(f"Final metrics: {task.get_metrics()}")
