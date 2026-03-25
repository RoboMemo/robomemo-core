# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Mock sensor-aware data collection for screw driving skill training.

Extends the existing data collection pipeline to use the mock sensor suite
(3 RGBD cameras + 2 FT sensors), producing training data in the format
expected by both GR00T N1.6 and OpenPI VLA pipelines.

This is the bridge between the mock sensor layer and VLA training.
When real sensors come online, swap MockSensorSuite → RealSensorSuite
with the same interface.

Usage:
    python -m roboforce_skills.data_collection_v2 --num_episodes 100 --output_dir datasets/mock_demos_v2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from roboforce_sim.sensors.sensor_config import SensorConfig
from roboforce_sim.sensors.mock_sensors import MockSensorSuite, TaskPhase
from roboforce_skills.data_collection import (
    ScrewDrivingExpertPolicy,
    ExpertPolicyCfg,
    ScrewDrivingPhase,
)

logger = logging.getLogger(__name__)


# Phase mapping: expert policy phase → sensor task phase
PHASE_MAP = {
    ScrewDrivingPhase.APPROACH: TaskPhase.APPROACH,
    ScrewDrivingPhase.ALIGN: TaskPhase.APPROACH,
    ScrewDrivingPhase.INSERT: TaskPhase.CONTACT,
    ScrewDrivingPhase.DRIVE: TaskPhase.SCREW_DRIVING,
    ScrewDrivingPhase.DONE: TaskPhase.DONE,
}


@dataclass
class DataCollectionV2Cfg:
    """Configuration for v2 data collection with full sensor suite."""

    output_dir: str = "datasets/mock_demos_v2"
    dataset_name: str = "roboforce_screw_3rgbd_2ft_v1"

    num_episodes: int = 100
    max_steps_per_episode: int = 500

    expert: ExpertPolicyCfg = field(default_factory=ExpertPolicyCfg)
    sensors: SensorConfig = field(default_factory=SensorConfig)

    # Save images as .npy (True) or skip to save disk space (False)
    save_images: bool = True

    # FT sensor decimation: how many control steps per FT sample saved
    # (FT runs at 500Hz internally, control at 50Hz, so 1:1 is fine)
    ft_decimation: int = 1

    num_dof: int = 13
    seed: int = 42
    log_interval: int = 20


class SensorAwareDataWriter:
    """Write demo data with full sensor observations.

    Output structure per episode:
    ```
    ep_000000/
        metadata.json          # Episode-level metadata
        observations.jsonl     # Per-step state + FT + image paths
        images/
            head_left/frame_000000_rgb.npy, frame_000000_depth.npy
            head_right/...
            wrist/...
    ```

    Also produces a flat `all_episodes.jsonl` for fast sequential loading.
    """

    def __init__(self, output_dir: str, dataset_name: str, save_images: bool = True):
        self.root = Path(output_dir) / dataset_name
        self.root.mkdir(parents=True, exist_ok=True)
        self.save_images = save_images
        self._episodes: list[dict] = []
        self._current: dict | None = None
        self._flat_file = open(self.root / "all_episodes.jsonl", "w")

    def start_episode(self, ep_idx: int, meta: dict | None = None) -> None:
        ep_dir = self.root / f"ep_{ep_idx:06d}"
        ep_dir.mkdir(exist_ok=True)
        if self.save_images:
            for cam in ["head_left", "head_right", "wrist"]:
                (ep_dir / "images" / cam).mkdir(parents=True, exist_ok=True)
        self._current = {
            "ep_idx": ep_idx,
            "ep_dir": ep_dir,
            "meta": meta or {},
            "frames": [],
        }

    def add_frame(
        self,
        step: int,
        action: np.ndarray,
        robot_state: np.ndarray,
        sensor_data: dict,
        phase: str,
        reward: float = 0.0,
        done: bool = False,
        timestamp: float = 0.0,
    ) -> None:
        if self._current is None:
            raise RuntimeError("start_episode first")

        ep_dir = self._current["ep_dir"]
        ep_idx = self._current["ep_idx"]

        frame = {
            "episode_index": ep_idx,
            "frame_index": step,
            "timestamp": timestamp,
            "phase": phase,
            "action": action.tolist(),
            "observation.state": robot_state.tolist(),
            "reward": reward,
            "done": done,
        }

        # FT sensor data (always saved inline — small)
        for ft_name, ft_data in sensor_data.get("ft_sensors", {}).items():
            frame[f"observation.ft.{ft_name}.wrench"] = ft_data["wrench"].tolist()
            frame[f"observation.ft.{ft_name}.force"] = ft_data["force"].tolist()
            frame[f"observation.ft.{ft_name}.torque"] = ft_data["torque"].tolist()

        # Camera images (saved as .npy files, paths stored in frame)
        if self.save_images:
            for cam_name, cam_data in sensor_data.get("cameras", {}).items():
                rgb_path = f"images/{cam_name}/frame_{step:06d}_rgb.npy"
                depth_path = f"images/{cam_name}/frame_{step:06d}_depth.npy"
                np.save(str(ep_dir / rgb_path), cam_data["rgb"])
                np.save(str(ep_dir / depth_path), cam_data["depth"])
                frame[f"observation.images.{cam_name}.rgb"] = rgb_path
                frame[f"observation.images.{cam_name}.depth"] = depth_path
        else:
            # Just record that images exist but weren't saved
            for cam_name in sensor_data.get("cameras", {}):
                frame[f"observation.images.{cam_name}.rgb"] = None
                frame[f"observation.images.{cam_name}.depth"] = None

        self._current["frames"].append(frame)

        # Write to flat file
        self._flat_file.write(json.dumps(frame) + "\n")

    def end_episode(self) -> int:
        if self._current is None:
            return 0

        n = len(self._current["frames"])
        ep_dir = self._current["ep_dir"]

        # Episode metadata
        meta = {
            "episode_index": self._current["ep_idx"],
            "num_frames": n,
            **self._current["meta"],
        }
        with open(ep_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Per-episode observations
        with open(ep_dir / "observations.jsonl", "w") as f:
            for frame in self._current["frames"]:
                f.write(json.dumps(frame) + "\n")

        self._episodes.append(meta)
        self._current = None
        return n

    def save(self) -> str:
        self._flat_file.close()

        # Dataset-level metadata
        dataset_meta = {
            "format": "roboforce_v2",
            "num_episodes": len(self._episodes),
            "total_frames": sum(ep["num_frames"] for ep in self._episodes),
            "robot": "RoboForce",
            "task": "screw_driving",
            "fps": 50,
            "sensors": {
                "cameras": ["head_left", "head_right", "wrist"],
                "ft_sensors": ["wrist_ft", "ee_tip_ft"],
                "camera_resolution": [640, 480],
                "camera_channels": {"rgb": 3, "depth": 1},
            },
            "observation_spec": {
                "state_dim": "variable (joint_pos + joint_vel + ee_pos + ee_quat + ft_wrench)",
                "ft_wrist_dim": 6,
                "ft_ee_dim": 6,
                "image_shape": [480, 640, 3],
                "depth_shape": [480, 640],
                "action_dim": 8,
            },
        }
        with open(self.root / "dataset_meta.json", "w") as f:
            json.dump(dataset_meta, f, indent=2)

        # Episodes index
        with open(self.root / "episodes.json", "w") as f:
            json.dump(self._episodes, f, indent=2)

        print(f"\n📁 Dataset saved: {self.root}")
        print(f"   Episodes: {len(self._episodes)}")
        print(f"   Total frames: {dataset_meta['total_frames']}")
        return str(self.root)


class DataCollectionV2Pipeline:
    """Collect demonstrations with full sensor suite (3 RGBD + 2 FT)."""

    def __init__(self, cfg: DataCollectionV2Cfg | None = None):
        self.cfg = cfg or DataCollectionV2Cfg()
        self.expert = ScrewDrivingExpertPolicy(self.cfg.expert, seed=self.cfg.seed)
        self.sensors = MockSensorSuite(self.cfg.sensors, seed=self.cfg.seed + 100)
        self.writer = SensorAwareDataWriter(
            self.cfg.output_dir, self.cfg.dataset_name, self.cfg.save_images
        )
        self.rng = np.random.default_rng(self.cfg.seed)

    def _run_episode(self, ep_idx: int) -> dict:
        self.expert.reset()
        self.sensors.reset()
        self.writer.start_episode(ep_idx)

        # Randomize
        screw_pos = np.array([0.0, -1.0, 1.5])
        screw_pos += self.rng.uniform(-0.1, 0.1, 3)
        screw_normal = np.array([0.0, -np.sin(np.radians(30)), np.cos(np.radians(30))])

        ee_pos = screw_pos + np.array([0.0, 0.5, 0.3])
        ee_ori = np.array([1.0, 0.0, 0.0, 0.0])

        joint_pos = self.rng.normal(0, 0.1, self.cfg.num_dof).astype(np.float32)
        joint_vel = np.zeros(self.cfg.num_dof, dtype=np.float32)

        screw_positions = [screw_pos]
        total_reward = 0.0
        success = False
        dt = 0.02

        for step in range(self.cfg.max_steps_per_episode):
            # FT from expert phase
            distance = np.linalg.norm(screw_pos - ee_pos)
            force_torque = np.zeros(6, dtype=np.float32)
            if distance < 0.02:
                force_torque[:3] = self.rng.normal(0, 2.0, 3)
                if self.expert.phase == ScrewDrivingPhase.DRIVE:
                    force_torque[5] = min(self.expert.cumulative_rotation * 0.2, 12.0)

            # Expert action
            action, info = self.expert.get_action(
                ee_pos, ee_ori, screw_pos, screw_normal, force_torque
            )

            # Update sensors phase
            sensor_phase = PHASE_MAP.get(self.expert.phase, TaskPhase.IDLE)
            self.sensors.set_task_phase(sensor_phase)
            if self.expert.phase == ScrewDrivingPhase.DRIVE:
                self.sensors.set_screw_progress(info["rotation_progress"])

            # Capture sensors
            sensor_data = self.sensors.step(
                ee_position=ee_pos,
                screw_positions=screw_positions,
            )

            # Build robot state (extended with real FT from sensors)
            wrist_wrench = sensor_data["ft_sensors"]["wrist_ft"]["wrench"]
            ee_wrench = sensor_data["ft_sensors"]["ee_tip_ft"]["wrench"]
            robot_state = np.concatenate([
                joint_pos, joint_vel, ee_pos, ee_ori, wrist_wrench, ee_wrench
            ])

            # Reward
            reward = -distance * 0.1
            if info["phase"] == "DRIVE":
                reward += info["rotation_progress"] * 5.0
            if info["phase"] == "DONE":
                reward += 100.0
                success = True
            total_reward += reward

            done = (info["phase"] == "DONE") or (step >= self.cfg.max_steps_per_episode - 1)

            self.writer.add_frame(
                step=step,
                action=action,
                robot_state=robot_state,
                sensor_data=sensor_data,
                phase=info["phase"],
                reward=reward,
                done=done,
                timestamp=step * dt,
            )

            # Simulate dynamics
            ee_pos += action[:3] * 0.01
            joint_pos += self.rng.normal(0, 0.01, self.cfg.num_dof).astype(np.float32)

            if done:
                break

        num_frames = self.writer.end_episode()
        return {
            "episode_idx": ep_idx,
            "num_frames": num_frames,
            "total_reward": float(total_reward),
            "success": success,
            "final_phase": self.expert.phase.name,
        }

    def run(self) -> str:
        t0 = time.time()
        stats = []

        print(f"🚀 Collecting {self.cfg.num_episodes} episodes with 3 RGBD + 2 FT sensors")
        print(f"   Output: {self.cfg.output_dir}/{self.cfg.dataset_name}")
        print(f"   Save images: {self.cfg.save_images}")

        for ep in range(self.cfg.num_episodes):
            ep_stats = self._run_episode(ep)
            stats.append(ep_stats)

            if (ep + 1) % self.cfg.log_interval == 0:
                recent = stats[-self.cfg.log_interval:]
                sr = sum(1 for s in recent if s["success"]) / len(recent)
                avg_f = sum(s["num_frames"] for s in recent) / len(recent)
                elapsed = time.time() - t0
                print(f"  [{ep+1}/{self.cfg.num_episodes}] success={sr:.0%}, "
                      f"avg_frames={avg_f:.0f}, elapsed={elapsed:.0f}s")

        dataset_path = self.writer.save()

        elapsed = time.time() - t0
        success_count = sum(1 for s in stats if s["success"])
        print(f"\n✅ Collection done in {elapsed:.1f}s")
        print(f"   Success: {success_count}/{len(stats)} ({success_count/len(stats):.0%})")

        # Stats file
        with open(Path(dataset_path) / "collection_stats.json", "w") as f:
            json.dump({
                "num_episodes": len(stats),
                "success_rate": success_count / len(stats),
                "collection_time_s": elapsed,
                "episodes": stats,
            }, f, indent=2)

        return dataset_path


def main() -> None:
    parser = argparse.ArgumentParser(description="RoboForce V2 Data Collection (3 RGBD + 2 FT)")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="datasets/mock_demos_v2")
    parser.add_argument("--dataset_name", type=str, default="roboforce_screw_3rgbd_2ft_v1")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--no_images", action="store_true", help="Skip saving camera images (faster)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cfg = DataCollectionV2Cfg(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        save_images=not args.no_images,
        seed=args.seed,
    )

    pipeline = DataCollectionV2Pipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
