#!/usr/bin/env python3
# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Mock sensor pipeline integration test + sample data generator.

Validates the full mock sensor suite (3 RGBD + 2 FT) and generates
sample data in the format expected by downstream VLA training.

Usage:
    python -m roboforce_sim.sensors.test_mock_pipeline [--num_steps 200] [--output_dir /tmp/roboforce_mock_data]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from roboforce_sim.sensors.sensor_config import SensorConfig
from roboforce_sim.sensors.mock_sensors import MockSensorSuite, TaskPhase


def run_mock_pipeline(num_steps: int = 200, output_dir: str = "/tmp/roboforce_mock_data") -> dict:
    """Run the mock sensor pipeline and save sample data.

    Simulates a complete screw driving episode:
    - Steps 0-30:    IDLE
    - Steps 30-80:   APPROACH
    - Steps 80-120:  CONTACT
    - Steps 120-180: SCREW_DRIVING (with ramp-up)
    - Steps 180+:    DONE

    Returns:
        Summary stats dict.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = SensorConfig()
    suite = MockSensorSuite(config, seed=42)

    print("=" * 60)
    print("RoboForce Mock Sensor Pipeline Test")
    print("=" * 60)
    print(f"\nSensor suite:")
    info = suite.info
    for cam_name, cam_info in info["cameras"].items():
        print(f"  📷 {cam_name}: {cam_info['resolution'][0]}x{cam_info['resolution'][1]} "
              f"@ {cam_info['fps']}Hz, depth [{cam_info['depth_range'][0]}-{cam_info['depth_range'][1]}m]")
    for ft_name, ft_info in info["ft_sensors"].items():
        print(f"  🔧 {ft_name}: F±{ft_info['force_range']} N, T±{ft_info['torque_range']} N·m "
              f"@ {ft_info['fps']}Hz")

    print(f"\nRunning {num_steps} steps...")

    # Phase schedule
    phase_schedule = [
        (30, TaskPhase.IDLE),
        (80, TaskPhase.APPROACH),
        (120, TaskPhase.CONTACT),
        (180, TaskPhase.SCREW_DRIVING),
        (num_steps, TaskPhase.DONE),
    ]

    # Mock EE trajectory (approach → contact → hold)
    ee_start = np.array([0.0, 0.5, 1.8], dtype=np.float32)
    ee_target = np.array([0.0, -1.0, 1.5], dtype=np.float32)
    screw_positions = [
        np.array([0.0, -1.0, 1.5]),
        np.array([0.3, -1.0, 1.5]),
        np.array([-0.3, -1.0, 1.5]),
    ]

    # Storage for stats
    all_wrist_forces = []
    all_ee_forces = []
    all_wrist_torques = []
    all_ee_torques = []
    rgb_shapes = []
    depth_ranges = []
    step_times = []

    # Sample frames to save (first, last, and every 50th)
    save_steps = set([0, num_steps - 1] + list(range(0, num_steps, 50)))

    t0 = time.monotonic()

    for step in range(num_steps):
        # Determine phase
        current_phase = TaskPhase.IDLE
        for threshold, phase in phase_schedule:
            if step < threshold:
                current_phase = phase
                break

        suite.set_task_phase(current_phase)

        # Screw progress ramps during SCREW_DRIVING
        if current_phase == TaskPhase.SCREW_DRIVING:
            drive_start = 120
            drive_end = 180
            progress = (step - drive_start) / max(drive_end - drive_start, 1)
            suite.set_screw_progress(progress)
        else:
            suite.set_screw_progress(0.0)

        # EE position interpolation
        t = min(step / 120.0, 1.0)
        ee_pos = ee_start + t * (ee_target - ee_start)

        step_t0 = time.monotonic()
        data = suite.step(ee_position=ee_pos, screw_positions=screw_positions)
        step_dt = time.monotonic() - step_t0
        step_times.append(step_dt)

        # Collect stats
        wrist_ft = data["ft_sensors"]["wrist_ft"]
        ee_ft = data["ft_sensors"]["ee_tip_ft"]
        all_wrist_forces.append(wrist_ft["force"].copy())
        all_ee_forces.append(ee_ft["force"].copy())
        all_wrist_torques.append(wrist_ft["torque"].copy())
        all_ee_torques.append(ee_ft["torque"].copy())

        for cam_name, cam_data in data["cameras"].items():
            rgb_shapes.append(cam_data["rgb"].shape)
            depth_ranges.append((cam_data["depth"].min(), cam_data["depth"].max()))

        # Save sample frames
        if step in save_steps:
            step_dir = out / f"step_{step:06d}"
            step_dir.mkdir(exist_ok=True)

            # Save camera data
            for cam_name, cam_data in data["cameras"].items():
                np.save(str(step_dir / f"{cam_name}_rgb.npy"), cam_data["rgb"])
                np.save(str(step_dir / f"{cam_name}_depth.npy"), cam_data["depth"])
                np.save(str(step_dir / f"{cam_name}_intrinsics.npy"), cam_data["intrinsics"])

            # Save FT data
            for ft_name, ft_data in data["ft_sensors"].items():
                np.save(str(step_dir / f"{ft_name}_wrench.npy"), ft_data["wrench"])

            # Save metadata
            meta = {
                "step": step,
                "phase": current_phase.name,
                "ee_position": ee_pos.tolist(),
                "wrist_ft_wrench": wrist_ft["wrench"].tolist(),
                "ee_tip_ft_wrench": ee_ft["wrench"].tolist(),
            }
            with open(step_dir / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

        # Progress
        if (step + 1) % 50 == 0:
            elapsed = time.monotonic() - t0
            fps = (step + 1) / elapsed
            print(f"  [{step+1}/{num_steps}] phase={current_phase.name:15s} "
                  f"wrist_Fz={wrist_ft['force'][2]:+.2f}N "
                  f"ee_Tz={ee_ft['torque'][2]:+.3f}N·m "
                  f"| {fps:.0f} steps/s")

    total_time = time.monotonic() - t0

    # Compute stats
    wrist_forces = np.array(all_wrist_forces)
    ee_forces = np.array(all_ee_forces)
    wrist_torques = np.array(all_wrist_torques)
    ee_torques = np.array(all_ee_torques)

    stats = {
        "num_steps": num_steps,
        "total_time_s": total_time,
        "steps_per_second": num_steps / total_time,
        "mean_step_ms": np.mean(step_times) * 1000,
        "p99_step_ms": np.percentile(step_times, 99) * 1000,
        "cameras": {
            "count": 3,
            "names": config.camera_names,
            "rgb_shape": list(rgb_shapes[0]),
            "depth_range": [float(np.min([r[0] for r in depth_ranges])),
                            float(np.max([r[1] for r in depth_ranges]))],
        },
        "wrist_ft": {
            "force_mean": wrist_forces.mean(axis=0).tolist(),
            "force_std": wrist_forces.std(axis=0).tolist(),
            "force_min": wrist_forces.min(axis=0).tolist(),
            "force_max": wrist_forces.max(axis=0).tolist(),
            "torque_mean": wrist_torques.mean(axis=0).tolist(),
            "torque_std": wrist_torques.std(axis=0).tolist(),
        },
        "ee_tip_ft": {
            "force_mean": ee_forces.mean(axis=0).tolist(),
            "force_std": ee_forces.std(axis=0).tolist(),
            "torque_mean": ee_torques.mean(axis=0).tolist(),
            "torque_std": ee_torques.std(axis=0).tolist(),
        },
        "saved_sample_steps": sorted(save_steps),
    }

    # Save stats
    with open(out / "pipeline_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Save sensor config
    config_dump = {
        "cameras": [
            {
                "name": c.name, "width": c.width, "height": c.height,
                "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy,
                "depth_min": c.depth_min, "depth_max": c.depth_max,
                "fps": c.fps, "parent_link": c.parent_link,
                "position": list(c.position),
                "orientation_rpy": list(c.orientation_rpy),
            }
            for c in config.cameras
        ],
        "ft_sensors": [
            {
                "name": f.name, "fps": f.fps,
                "force_range": list(f.force_range),
                "torque_range": list(f.torque_range),
                "force_noise_std": list(f.force_noise_std),
                "torque_noise_std": list(f.torque_noise_std),
                "parent_link": f.parent_link,
            }
            for f in config.ft_sensors
        ],
    }
    with open(out / "sensor_config.json", "w") as f:
        json.dump(config_dump, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"✅ Pipeline test PASSED")
    print(f"{'=' * 60}")
    print(f"  Steps:       {num_steps}")
    print(f"  Time:        {total_time:.2f}s ({num_steps/total_time:.0f} steps/s)")
    print(f"  Step latency: mean={np.mean(step_times)*1000:.2f}ms, P99={np.percentile(step_times, 99)*1000:.2f}ms")
    print(f"\n  📷 Cameras ({len(config.cameras)}):")
    for c in config.cameras:
        print(f"     {c.name}: {c.width}×{c.height} RGBD @ {c.fps}Hz, depth [{c.depth_min}-{c.depth_max}m]")
    print(f"\n  🔧 FT Sensors ({len(config.ft_sensors)}):")
    print(f"     wrist_ft:  Fz mean={wrist_forces[:,2].mean():+.2f}N, Tz mean={wrist_torques[:,2].mean():+.3f}N·m")
    print(f"     ee_tip_ft: Fz mean={ee_forces[:,2].mean():+.2f}N, Tz mean={ee_torques[:,2].mean():+.3f}N·m")
    print(f"\n  📁 Output: {out}")
    print(f"     sensor_config.json    — sensor specifications")
    print(f"     pipeline_stats.json   — run statistics")
    print(f"     step_*/               — sample frames ({len(save_steps)} saved)")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="RoboForce Mock Sensor Pipeline Test")
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="/tmp/roboforce_mock_data")
    args = parser.parse_args()

    stats = run_mock_pipeline(args.num_steps, args.output_dir)

    # Validate
    assert stats["cameras"]["count"] == 3, "Expected 3 cameras"
    assert stats["cameras"]["rgb_shape"] == [480, 640, 3], f"Unexpected RGB shape: {stats['cameras']['rgb_shape']}"
    assert stats["num_steps"] == args.num_steps
    print("\n✅ All assertions passed!")


if __name__ == "__main__":
    main()
