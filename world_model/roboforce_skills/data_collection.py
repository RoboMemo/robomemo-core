# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Demonstration data collection for screw driving skill training.

Implements a scripted expert policy for screw driving and collects
demonstrations in LeRobot V2 format, compatible with both GR00T N1.6
and OpenPI (π₀) fine-tuning pipelines.

The expert policy follows a 4-phase approach:
1. **Approach** — Move EE toward the target screw
2. **Align** — Orient the screw driver socket to match screw head
3. **Insert** — Lower the socket onto the screw head
4. **Drive** — Rotate to tighten the screw to target torque

Usage:
    python -m roboforce_skills.data_collection --num_episodes 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Expert Policy
# ---------------------------------------------------------------------------

class ScrewDrivingPhase(Enum):
    """Phases of the scripted expert policy."""
    APPROACH = auto()
    ALIGN = auto()
    INSERT = auto()
    DRIVE = auto()
    DONE = auto()


@dataclass
class ExpertPolicyCfg:
    """Configuration for the scripted expert policy."""

    # Approach phase
    approach_speed: float = 0.008
    """EE movement speed during approach (m/step)."""
    approach_threshold: float = 0.05
    """Distance threshold to transition from approach to align (m)."""

    # Align phase
    align_speed: float = 0.003
    """Fine approach speed (m/step)."""
    align_threshold: float = 0.015
    """Distance threshold for alignment completion (m)."""
    align_orientation_threshold: float = 0.05
    """Orientation error threshold (radians)."""

    # Insert phase
    insert_speed: float = 0.002
    """Insertion speed (m/step)."""
    insert_force_threshold: float = 5.0
    """Force threshold indicating screw engagement (N)."""
    insert_depth: float = 0.01
    """Target insertion depth (m)."""

    # Drive phase
    drive_rotation_speed: float = 0.4
    """Screw rotation speed (rad/step)."""
    drive_target_turns: float = 8.0
    """Number of full turns for complete tightening."""
    drive_torque_limit: float = 15.0
    """Max torque before stopping (N·m) — safety limit."""

    # Noise (for demonstration diversity)
    position_noise: float = 0.001
    """Gaussian noise on position commands (m)."""
    orientation_noise: float = 0.005
    """Gaussian noise on orientation commands (rad)."""
    action_noise: float = 0.02
    """Gaussian noise on all action dimensions."""


class ScrewDrivingExpertPolicy:
    """Scripted expert policy for screw driving demonstrations.

    Generates near-optimal actions for each phase of the screw driving task.
    Includes configurable noise for demonstration diversity.
    """

    def __init__(self, cfg: ExpertPolicyCfg | None = None, seed: int = 42):
        self.cfg = cfg or ExpertPolicyCfg()
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        """Reset the policy state for a new episode."""
        self.phase = ScrewDrivingPhase.APPROACH
        self.cumulative_rotation = 0.0
        self.insert_progress = 0.0
        self.step_count = 0

    def get_action(
        self,
        ee_position: np.ndarray,
        ee_orientation: np.ndarray,
        target_position: np.ndarray,
        target_normal: np.ndarray,
        force_torque: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Compute the expert action for the current state.

        Args:
            ee_position: Current EE position, shape ``(3,)``.
            ee_orientation: Current EE quaternion ``(w,x,y,z)``, shape ``(4,)``.
            target_position: Target screw position, shape ``(3,)``.
            target_normal: Target screw surface normal, shape ``(3,)``.
            force_torque: Force/torque at EE, shape ``(6,)``.

        Returns:
            Tuple of:
            - ``action``: 8D action vector ``[dx, dy, dz, drx, dry, drz, screw_rot, gripper]``
            - ``info``: Dict with phase, distance, progress info.
        """
        self.step_count += 1
        action = np.zeros(8, dtype=np.float32)

        # Vector from EE to target
        to_target = target_position - ee_position
        distance = np.linalg.norm(to_target)
        direction = to_target / (distance + 1e-8)

        # Force magnitude
        force_mag = np.linalg.norm(force_torque[:3])
        torque_mag = np.linalg.norm(force_torque[3:])

        info = {
            "phase": self.phase.name,
            "distance": float(distance),
            "force": float(force_mag),
            "torque": float(torque_mag),
            "rotation_progress": self.cumulative_rotation / (self.cfg.drive_target_turns * 2 * np.pi),
        }

        if self.phase == ScrewDrivingPhase.APPROACH:
            # Move toward target
            speed = min(self.cfg.approach_speed, distance)
            action[:3] = direction * speed

            # Keep gripper open
            action[7] = 1.0

            if distance < self.cfg.approach_threshold:
                self.phase = ScrewDrivingPhase.ALIGN

        elif self.phase == ScrewDrivingPhase.ALIGN:
            # Fine approach with orientation alignment
            speed = min(self.cfg.align_speed, distance)
            action[:3] = direction * speed

            # Align EE axis with screw normal
            # Compute orientation correction (simplified: target = -normal)
            target_axis = -target_normal
            current_axis = np.array([0, 0, -1])  # Default EE axis
            cross = np.cross(current_axis, target_axis)
            action[3:6] = cross * 0.1  # Proportional orientation control

            action[7] = 0.5  # Partially close gripper

            if distance < self.cfg.align_threshold:
                self.phase = ScrewDrivingPhase.INSERT

        elif self.phase == ScrewDrivingPhase.INSERT:
            # Push EE down onto screw
            action[:3] = -target_normal * self.cfg.insert_speed
            action[7] = 0.0  # Close gripper

            self.insert_progress += self.cfg.insert_speed
            if force_mag > self.cfg.insert_force_threshold or self.insert_progress > self.cfg.insert_depth:
                self.phase = ScrewDrivingPhase.DRIVE

        elif self.phase == ScrewDrivingPhase.DRIVE:
            # Maintain position, rotate screw
            action[:3] = -target_normal * 0.0005  # Slight push to maintain contact
            action[6] = self.cfg.drive_rotation_speed  # Screw rotation
            action[7] = 0.0  # Gripper closed

            self.cumulative_rotation += abs(self.cfg.drive_rotation_speed)
            target_rotation = self.cfg.drive_target_turns * 2 * np.pi

            if self.cumulative_rotation >= target_rotation:
                self.phase = ScrewDrivingPhase.DONE
                action[6] = 0.0

            # Safety: stop if torque too high
            if torque_mag > self.cfg.drive_torque_limit:
                self.phase = ScrewDrivingPhase.DONE
                action[6] = 0.0

        elif self.phase == ScrewDrivingPhase.DONE:
            # Retract slightly
            action[:3] = target_normal * self.cfg.approach_speed
            action[7] = 1.0  # Open gripper

        # Add noise for diversity
        noise = self.rng.normal(0, self.cfg.action_noise, 8).astype(np.float32)
        action += noise

        # Clip to [-1, 1]
        action = np.clip(action, -1.0, 1.0)

        info["phase"] = self.phase.name
        return action, info


# ---------------------------------------------------------------------------
# Data Collection Configuration
# ---------------------------------------------------------------------------

@dataclass
class DataCollectionCfg:
    """Configuration for demonstration data collection."""

    # Output
    output_dir: str = "datasets/demonstrations"
    dataset_name: str = "roboforce_screw_demos_v1"

    # Collection parameters
    num_episodes: int = 1000
    max_steps_per_episode: int = 500
    """Maximum steps per episode."""

    # Expert policy
    expert: ExpertPolicyCfg = field(default_factory=ExpertPolicyCfg)

    # Camera
    camera_width: int = 640
    camera_height: int = 480

    # Data format
    format: str = "lerobot_v2"
    """Output format: ``lerobot_v2`` (HuggingFace) or ``hdf5``."""

    # Domain randomization (applied per episode)
    randomize_screw_position: bool = True
    randomize_lighting: bool = True
    randomize_screw_type: bool = True

    # Robot
    num_dof: int = 13
    """Number of robot DOF (for observation dimensions)."""

    # Seed
    seed: int = 42

    # Logging
    log_interval: int = 50


# ---------------------------------------------------------------------------
# LeRobot V2 Data Writer
# ---------------------------------------------------------------------------

class LeRobotV2Writer:
    """Write demonstration data in LeRobot V2 format.

    LeRobot V2 uses HuggingFace Datasets format with the following structure:
    - ``observation.images.head_rgb``: Camera images
    - ``observation.state``: Joint positions + EE state
    - ``action``: Robot actions
    - ``episode_index``: Episode identifier
    - ``frame_index``: Frame within episode
    - ``timestamp``: Simulation timestamp

    Compatible with GR00T N1.6 and OpenPI (π₀) fine-tuning.
    """

    def __init__(self, output_dir: str, dataset_name: str):
        self.output_dir = Path(output_dir) / dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Accumulate data
        self._episodes: list[dict] = []
        self._current_episode: dict[str, list] | None = None

    def start_episode(self, episode_idx: int, metadata: dict | None = None) -> None:
        """Start recording a new episode."""
        self._current_episode = {
            "episode_index": episode_idx,
            "metadata": metadata or {},
            "frames": [],
        }

    def add_frame(
        self,
        observation: dict[str, Any],
        action: np.ndarray,
        reward: float = 0.0,
        done: bool = False,
        info: dict | None = None,
        timestamp: float = 0.0,
    ) -> None:
        """Add a single frame to the current episode.

        Args:
            observation: Dict with ``state`` (np.ndarray), and optionally
                ``images`` dict with camera name → image (np.ndarray) mappings.
            action: Action array, shape ``(action_dim,)``.
            reward: Scalar reward.
            done: Whether the episode ended.
            info: Additional info dict.
            timestamp: Simulation time.
        """
        if self._current_episode is None:
            raise RuntimeError("Call start_episode() first")

        frame_idx = len(self._current_episode["frames"])
        frame = {
            "frame_index": frame_idx,
            "timestamp": timestamp,
            "observation.state": observation.get("state", np.array([])).tolist(),
            "action": action.tolist(),
            "reward": reward,
            "done": done,
        }

        # Camera images (stored as file paths, not in the JSON directly)
        if "images" in observation:
            for cam_name, img in observation["images"].items():
                img_dir = self.output_dir / "images" / cam_name
                img_dir.mkdir(parents=True, exist_ok=True)
                ep_idx = self._current_episode["episode_index"]
                img_path = f"ep{ep_idx:06d}_frame{frame_idx:06d}.npy"
                np.save(str(img_dir / img_path), img)
                frame[f"observation.images.{cam_name}"] = f"images/{cam_name}/{img_path}"

        if info:
            frame["info"] = info

        self._current_episode["frames"].append(frame)

    def end_episode(self) -> int:
        """End the current episode and store it.

        Returns:
            Number of frames in the episode.
        """
        if self._current_episode is None:
            return 0

        num_frames = len(self._current_episode["frames"])
        self._episodes.append(self._current_episode)
        self._current_episode = None
        return num_frames

    def save(self) -> str:
        """Save all episodes to disk in LeRobot V2 format.

        Returns:
            Path to the saved dataset.
        """
        # Metadata
        metadata = {
            "format": "lerobot_v2",
            "num_episodes": len(self._episodes),
            "total_frames": sum(len(ep["frames"]) for ep in self._episodes),
            "robot": "RoboForce",
            "task": "screw_driving",
            "fps": 50,  # 50 Hz control
            "features": {
                "observation.state": {"dtype": "float32", "shape": "variable"},
                "action": {"dtype": "float32", "shape": [8]},
                "reward": {"dtype": "float32"},
                "done": {"dtype": "bool"},
            },
        }

        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Episodes index
        episodes_index = []
        for ep in self._episodes:
            episodes_index.append({
                "episode_index": ep["episode_index"],
                "num_frames": len(ep["frames"]),
                "metadata": ep["metadata"],
            })

        index_path = self.output_dir / "episodes.json"
        with open(index_path, "w") as f:
            json.dump(episodes_index, f, indent=2)

        # Save frames as JSON-lines (compact)
        data_path = self.output_dir / "data.jsonl"
        with open(data_path, "w") as f:
            for ep in self._episodes:
                for frame in ep["frames"]:
                    frame_out = {**frame}
                    frame_out["episode_index"] = ep["episode_index"]
                    f.write(json.dumps(frame_out) + "\n")

        # HDF5 (optional, for faster loading)
        if H5PY_AVAILABLE:
            self._save_hdf5()

        print(f"Dataset saved to {self.output_dir}")
        print(f"  Episodes: {len(self._episodes)}")
        print(f"  Total frames: {metadata['total_frames']}")

        return str(self.output_dir)

    def _save_hdf5(self) -> None:
        """Save a compact HDF5 version of the dataset."""
        h5_path = self.output_dir / "data.hdf5"
        with h5py.File(h5_path, "w") as f:
            for ep in self._episodes:
                ep_idx = ep["episode_index"]
                grp = f.create_group(f"episode_{ep_idx:06d}")

                frames = ep["frames"]
                if not frames:
                    continue

                states = np.array([fr["observation.state"] for fr in frames], dtype=np.float32)
                actions = np.array([fr["action"] for fr in frames], dtype=np.float32)
                rewards = np.array([fr["reward"] for fr in frames], dtype=np.float32)
                dones = np.array([fr["done"] for fr in frames], dtype=bool)

                grp.create_dataset("observation.state", data=states, compression="gzip")
                grp.create_dataset("action", data=actions, compression="gzip")
                grp.create_dataset("reward", data=rewards)
                grp.create_dataset("done", data=dones)

                if ep["metadata"]:
                    grp.attrs["metadata"] = json.dumps(ep["metadata"])

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def total_frames(self) -> int:
        return sum(len(ep["frames"]) for ep in self._episodes)


# ---------------------------------------------------------------------------
# Data Collection Pipeline
# ---------------------------------------------------------------------------

class DataCollectionPipeline:
    """Collect demonstration data using the scripted expert policy.

    Runs the expert policy in the simulation environment (or simulated
    offline) and records the trajectories.
    """

    def __init__(self, cfg: DataCollectionCfg | None = None):
        self.cfg = cfg or DataCollectionCfg()
        self.expert = ScrewDrivingExpertPolicy(self.cfg.expert, seed=self.cfg.seed)
        self.writer = LeRobotV2Writer(self.cfg.output_dir, self.cfg.dataset_name)
        self.rng = np.random.default_rng(self.cfg.seed)

    def _simulate_episode(self, episode_idx: int) -> dict:
        """Run one episode of the expert policy with simulated dynamics.

        Returns:
            Episode statistics dict.
        """
        self.expert.reset()
        self.writer.start_episode(episode_idx)

        # Randomize initial conditions
        screw_position = np.array([0.0, -1.0, 1.5])
        if self.cfg.randomize_screw_position:
            screw_position += self.rng.uniform(-0.1, 0.1, 3)
        screw_normal = np.array([0.0, -np.sin(np.radians(30)), np.cos(np.radians(30))])

        ee_position = screw_position + np.array([0.0, 0.5, 0.3])
        ee_orientation = np.array([1.0, 0.0, 0.0, 0.0])

        joint_positions = self.rng.normal(0, 0.1, self.cfg.num_dof).astype(np.float32)
        joint_velocities = np.zeros(self.cfg.num_dof, dtype=np.float32)

        total_reward = 0.0
        success = False
        dt = 0.02  # 50 Hz control

        for step in range(self.cfg.max_steps_per_episode):
            # Simulate force/torque
            distance = np.linalg.norm(screw_position - ee_position)
            force_torque = np.zeros(6, dtype=np.float32)
            if distance < 0.02:
                force_torque[:3] = self.rng.normal(0, 2.0, 3)
                if self.expert.phase == ScrewDrivingPhase.DRIVE:
                    force_torque[5] = min(self.expert.cumulative_rotation * 0.2, 12.0)

            # Expert action
            action, info = self.expert.get_action(
                ee_position, ee_orientation, screw_position, screw_normal, force_torque
            )

            # Observation
            state = np.concatenate([joint_positions, joint_velocities, ee_position, ee_orientation, force_torque])
            observation = {"state": state}

            # Camera (synthetic placeholder)
            if self.cfg.camera_width > 0:
                fake_image = self.rng.integers(
                    0, 255, (self.cfg.camera_height, self.cfg.camera_width, 3), dtype=np.uint8
                )
                observation["images"] = {"head_rgb": fake_image}

            # Reward (simplified)
            reward = -distance * 0.1
            if info["phase"] == "DRIVE":
                reward += info["rotation_progress"] * 5.0
            if info["phase"] == "DONE":
                reward += 100.0
                success = True

            total_reward += reward
            done = (info["phase"] == "DONE") or (step >= self.cfg.max_steps_per_episode - 1)

            self.writer.add_frame(
                observation=observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
                timestamp=step * dt,
            )

            # Simulated dynamics (simplified)
            ee_position += action[:3] * 0.01
            joint_positions += self.rng.normal(0, 0.01, self.cfg.num_dof).astype(np.float32)

            if done:
                break

        num_frames = self.writer.end_episode()

        return {
            "episode_idx": episode_idx,
            "num_frames": num_frames,
            "total_reward": float(total_reward),
            "success": success,
            "final_phase": self.expert.phase.name,
        }

    def run(self) -> str:
        """Run the full data collection.

        Returns:
            Path to the saved dataset.
        """
        t0 = time.time()
        stats = []

        print(f"Collecting {self.cfg.num_episodes} demonstration episodes...")
        print(f"  Expert policy: approach → align → insert → drive")
        print(f"  Output: {self.cfg.output_dir}/{self.cfg.dataset_name}")

        for ep in range(self.cfg.num_episodes):
            ep_stats = self._simulate_episode(ep)
            stats.append(ep_stats)

            if (ep + 1) % self.cfg.log_interval == 0:
                recent = stats[-self.cfg.log_interval:]
                success_rate = sum(1 for s in recent if s["success"]) / len(recent)
                avg_frames = sum(s["num_frames"] for s in recent) / len(recent)
                elapsed = time.time() - t0
                print(f"  [{ep+1}/{self.cfg.num_episodes}] "
                      f"success={success_rate:.1%}, "
                      f"avg_frames={avg_frames:.0f}, "
                      f"elapsed={elapsed:.0f}s")

        # Save
        dataset_path = self.writer.save()

        # Summary
        elapsed = time.time() - t0
        success_count = sum(1 for s in stats if s["success"])
        print(f"\nCollection complete in {elapsed:.1f}s")
        print(f"  Success rate: {success_count}/{len(stats)} ({success_count/len(stats):.1%})")
        print(f"  Total frames: {self.writer.total_frames}")

        # Save stats
        stats_path = Path(dataset_path) / "collection_stats.json"
        with open(stats_path, "w") as f:
            json.dump({
                "num_episodes": len(stats),
                "success_rate": success_count / len(stats),
                "total_frames": self.writer.total_frames,
                "collection_time_s": elapsed,
                "episodes": stats,
            }, f, indent=2)

        return dataset_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RoboForce — Screw Driving Demonstration Collection"
    )
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="datasets/demonstrations")
    parser.add_argument("--dataset_name", type=str, default="roboforce_screw_demos_v1")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cfg = DataCollectionCfg(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
    )

    pipeline = DataCollectionPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
