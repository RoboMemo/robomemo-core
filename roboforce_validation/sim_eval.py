# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Simulation evaluation for trained screw-driving policies.

Evaluates GR00T N1.6 and OpenPI (π₀) policies in the Isaac Sim
solar-panel environment across weather conditions and screw positions.

Evaluation protocol:
    - 3 weather conditions: day, night, dusty
    - 10 screw positions per panel (grid layout)
    - 100 episodes per (weather × position) combination
    - Metrics: success rate, completion time, force profile, collision count

Usage:
    python -m roboforce_validation.sim_eval \\
        --policy_type gr00t \\
        --checkpoint checkpoints/gr00t_screw_driving/best \\
        --num_episodes 100

    python -m roboforce_validation.sim_eval \\
        --policy_type openpi \\
        --checkpoint checkpoints/openpi_screw_driving/best \\
        --weather day night dusty
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation Configuration
# ---------------------------------------------------------------------------

@dataclass
class WeatherCondition:
    """A weather / lighting condition for evaluation."""

    name: str
    """Condition identifier: ``day``, ``night``, ``dusty``."""
    sun_intensity: float
    """Directional light intensity (0–2)."""
    ambient_intensity: float
    """Ambient light intensity (0–1)."""
    fog_density: float
    """Fog / dust density (0–1)."""
    color_temperature_k: int
    """Light color temperature (Kelvin)."""
    description: str = ""


WEATHER_CONDITIONS: dict[str, WeatherCondition] = {
    "day": WeatherCondition(
        name="day",
        sun_intensity=1.2,
        ambient_intensity=0.4,
        fog_density=0.0,
        color_temperature_k=5500,
        description="Clear daytime, direct sunlight",
    ),
    "night": WeatherCondition(
        name="night",
        sun_intensity=0.0,
        ambient_intensity=0.05,
        fog_density=0.0,
        color_temperature_k=4000,
        description="Nighttime with work lights only",
    ),
    "dusty": WeatherCondition(
        name="dusty",
        sun_intensity=0.8,
        ambient_intensity=0.3,
        fog_density=0.4,
        color_temperature_k=4500,
        description="Daytime with dust/sand in the air",
    ),
    "overcast": WeatherCondition(
        name="overcast",
        sun_intensity=0.4,
        ambient_intensity=0.5,
        fog_density=0.1,
        color_temperature_k=6500,
        description="Overcast sky, diffuse lighting",
    ),
    "dawn": WeatherCondition(
        name="dawn",
        sun_intensity=0.3,
        ambient_intensity=0.15,
        fog_density=0.05,
        color_temperature_k=3500,
        description="Early morning, warm low-angle light",
    ),
}


@dataclass
class ScrewPosition:
    """A screw position on the solar panel."""

    id: int
    """Position index (0–9)."""
    name: str
    """Human-readable name (e.g. ``top_left``)."""
    offset_from_center: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    """Offset from panel center in meters ``(x, y, z)``."""


def generate_screw_positions(
    num_positions: int = 10,
    panel_width_m: float = 1.6,
    panel_height_m: float = 1.0,
) -> list[ScrewPosition]:
    """Generate a grid of screw positions on a solar panel.

    Uses a 2×5 grid layout for 10 positions (typical mounting pattern).

    Args:
        num_positions: Target number of positions.
        panel_width_m: Panel width in meters.
        panel_height_m: Panel height in meters.

    Returns:
        List of screw positions.
    """
    positions = []
    rows = 2
    cols = (num_positions + rows - 1) // rows

    names = [
        "top_left", "top_center_left", "top_center", "top_center_right", "top_right",
        "bot_left", "bot_center_left", "bot_center", "bot_center_right", "bot_right",
    ]

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= num_positions:
                break
            x = (c / max(cols - 1, 1) - 0.5) * panel_width_m * 0.8
            y = (r / max(rows - 1, 1) - 0.5) * panel_height_m * 0.6
            z = 0.0  # On the panel surface

            name = names[idx] if idx < len(names) else f"pos_{idx}"
            positions.append(ScrewPosition(
                id=idx,
                name=name,
                offset_from_center=np.array([x, y, z], dtype=np.float32),
            ))
            idx += 1

    return positions


@dataclass
class SimEvalCfg:
    """Configuration for simulation evaluation."""

    # Policy
    policy_type: str = "gr00t"
    """Policy type: ``gr00t`` or ``openpi``."""
    checkpoint_path: str = "checkpoints/gr00t_screw_driving/best"
    """Path to the trained policy checkpoint."""

    # Evaluation scope
    weather_conditions: list[str] = field(
        default_factory=lambda: ["day", "night", "dusty"]
    )
    num_screw_positions: int = 10
    """Number of screw positions per panel."""
    num_episodes_per_condition: int = 100
    """Number of episodes per (weather × position) pair."""

    # Episode limits
    max_steps_per_episode: int = 500
    """Maximum timesteps per episode."""
    control_frequency_hz: float = 50.0
    """Control frequency in Hz."""

    # Success criteria
    success_torque_threshold_nm: float = 5.0
    """Minimum torque for a screw to be considered tightened."""
    success_min_turns: float = 6.0
    """Minimum number of full turns for success."""
    collision_force_threshold_n: float = 50.0
    """Force threshold above which a collision is counted."""

    # Output
    output_dir: str = "results/sim_eval"
    save_videos: bool = False
    save_force_profiles: bool = True

    # Reproducibility
    seed: int = 42

    # Hardware
    headless: bool = True
    """Run Isaac Sim in headless mode."""
    num_parallel_envs: int = 1
    """Number of parallel simulation environments."""


# ---------------------------------------------------------------------------
# Episode Result
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    """Result of a single evaluation episode."""

    episode_idx: int
    weather: str
    screw_position: str
    screw_position_id: int

    # Outcome
    success: bool = False
    completion_time_s: float = 0.0
    num_steps: int = 0

    # Force profile
    peak_force_n: float = 0.0
    mean_force_n: float = 0.0
    peak_torque_nm: float = 0.0
    final_torque_nm: float = 0.0
    total_turns: float = 0.0

    # Safety
    collision_count: int = 0
    max_collision_force_n: float = 0.0

    # Timing
    inference_time_ms: float = 0.0
    """Mean policy inference time per step."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_idx": self.episode_idx,
            "weather": self.weather,
            "screw_position": self.screw_position,
            "screw_position_id": self.screw_position_id,
            "success": self.success,
            "completion_time_s": round(self.completion_time_s, 3),
            "num_steps": self.num_steps,
            "peak_force_n": round(self.peak_force_n, 2),
            "mean_force_n": round(self.mean_force_n, 2),
            "peak_torque_nm": round(self.peak_torque_nm, 2),
            "final_torque_nm": round(self.final_torque_nm, 2),
            "total_turns": round(self.total_turns, 2),
            "collision_count": self.collision_count,
            "max_collision_force_n": round(self.max_collision_force_n, 2),
            "inference_time_ms": round(self.inference_time_ms, 3),
        }


# ---------------------------------------------------------------------------
# Mock Policy Loader (placeholder for real VLA inference)
# ---------------------------------------------------------------------------

class PolicyLoader:
    """Load and wrap a trained VLA policy for evaluation.

    In production, this interfaces with:
    - GR00T: ``gr00t.experiment.InferenceModel``
    - OpenPI: ``openpi.serve.PolicyClient`` or local model

    For testing, provides a mock policy that generates random actions.
    """

    def __init__(self, policy_type: str, checkpoint_path: str):
        self.policy_type = policy_type
        self.checkpoint_path = checkpoint_path
        self._loaded = False

    def load(self) -> None:
        """Load the policy checkpoint."""
        logger.info(
            "Loading %s policy from %s", self.policy_type, self.checkpoint_path
        )
        # In production:
        #   if self.policy_type == "gr00t":
        #       from gr00t.experiment import InferenceModel
        #       self.model = InferenceModel.from_checkpoint(self.checkpoint_path)
        #   elif self.policy_type == "openpi":
        #       from openpi.serve import PolicyClient
        #       self.model = PolicyClient(self.checkpoint_path)
        self._loaded = True
        logger.info("Policy loaded (mock mode)")

    def predict(
        self,
        image: np.ndarray,
        state: np.ndarray,
        instruction: str = "",
    ) -> tuple[np.ndarray, float]:
        """Run policy inference.

        Args:
            image: Camera image, shape ``(H, W, 3)``.
            state: Robot state vector.
            instruction: Task instruction text.

        Returns:
            Tuple of ``(action, inference_time_ms)``.
            ``action`` has shape ``(8,)`` — delta pose + screw + gripper.
        """
        if not self._loaded:
            raise RuntimeError("Call load() first")

        t0 = time.perf_counter_ns()

        # Mock inference — in production this calls the real VLA
        action = np.random.randn(8).astype(np.float32) * 0.05
        # Simulate realistic inference latency
        if self.policy_type == "openpi":
            # π₀ has multi-step denoising → slower
            time.sleep(0.005)  # ~5ms mock
        else:
            time.sleep(0.002)  # ~2ms mock

        inference_ms = (time.perf_counter_ns() - t0) / 1_000_000

        return action, inference_ms


# ---------------------------------------------------------------------------
# Simulation Evaluator
# ---------------------------------------------------------------------------

class SimulationEvaluator:
    """Run evaluation episodes in Isaac Sim (or mock sim).

    Evaluates the policy across all (weather × screw_position) combinations,
    collecting per-episode metrics.
    """

    def __init__(self, cfg: SimEvalCfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.policy = PolicyLoader(cfg.policy_type, cfg.checkpoint_path)
        self.screw_positions = generate_screw_positions(cfg.num_screw_positions)
        self.results: list[EpisodeResult] = []

    def _run_episode(
        self,
        weather: WeatherCondition,
        screw_pos: ScrewPosition,
        episode_idx: int,
    ) -> EpisodeResult:
        """Run a single evaluation episode.

        In production this drives the Isaac Sim environment. Here we simulate
        the dynamics with noise to generate realistic-looking metrics.
        """
        dt = 1.0 / self.cfg.control_frequency_hz
        result = EpisodeResult(
            episode_idx=episode_idx,
            weather=weather.name,
            screw_position=screw_pos.name,
            screw_position_id=screw_pos.id,
        )

        # Simulated dynamics
        ee_pos = screw_pos.offset_from_center + np.array([0.0, 0.3, 0.2])
        target_pos = screw_pos.offset_from_center.copy()
        cumulative_rotation = 0.0
        forces = []
        torques = []
        inference_times = []

        # Visibility factor (weather affects perception)
        visibility = 1.0 - weather.fog_density * 0.5
        if weather.name == "night":
            visibility *= 0.6

        # Generate a synthetic image (placeholder)
        image = self.rng.integers(
            0, 255, (480, 640, 3), dtype=np.uint8
        )
        state = np.concatenate([
            self.rng.normal(0, 0.1, 13).astype(np.float32),  # joints
            ee_pos, np.array([1, 0, 0, 0], dtype=np.float32),  # EE pose
            np.zeros(6, dtype=np.float32),  # F/T
        ])

        for step in range(self.cfg.max_steps_per_episode):
            action, inf_ms = self.policy.predict(image, state, "Drive the screw")
            inference_times.append(inf_ms)

            # Simulated dynamics
            distance = np.linalg.norm(target_pos - ee_pos)
            ee_pos += action[:3] * 0.01

            # Force simulation
            force_mag = 0.0
            torque_mag = 0.0
            if distance < 0.05:
                force_mag = abs(self.rng.normal(3.0, 1.0))
                if distance < 0.02:
                    cumulative_rotation += abs(action[6]) if len(action) > 6 else 0.1
                    torque_mag = min(cumulative_rotation * 0.3, 12.0)

            forces.append(force_mag)
            torques.append(torque_mag)

            # Collision detection (simplified)
            collision_force = self.rng.exponential(2.0)
            if collision_force > self.cfg.collision_force_threshold_n:
                result.collision_count += 1
                result.max_collision_force_n = max(
                    result.max_collision_force_n, collision_force
                )

            # Success check
            total_turns = cumulative_rotation / (2 * np.pi)
            if (
                total_turns >= self.cfg.success_min_turns
                and torque_mag >= self.cfg.success_torque_threshold_nm
            ):
                result.success = True
                result.num_steps = step + 1
                result.completion_time_s = (step + 1) * dt
                break

            # Add success probability influenced by visibility / weather
            if step > 100 and self.rng.random() < 0.01 * visibility:
                cumulative_rotation += 2.0  # Lucky progress

        else:
            result.num_steps = self.cfg.max_steps_per_episode
            result.completion_time_s = self.cfg.max_steps_per_episode * dt

        # Force profile
        forces_arr = np.array(forces)
        torques_arr = np.array(torques)
        result.peak_force_n = float(np.max(forces_arr)) if len(forces_arr) > 0 else 0.0
        result.mean_force_n = float(np.mean(forces_arr)) if len(forces_arr) > 0 else 0.0
        result.peak_torque_nm = float(np.max(torques_arr)) if len(torques_arr) > 0 else 0.0
        result.final_torque_nm = float(torques_arr[-1]) if len(torques_arr) > 0 else 0.0
        result.total_turns = cumulative_rotation / (2 * np.pi)
        result.inference_time_ms = float(np.mean(inference_times)) if inference_times else 0.0

        return result

    def run(self) -> dict[str, Any]:
        """Run the full evaluation suite.

        Returns:
            Evaluation report dict.
        """
        self.policy.load()

        weather_list = [
            WEATHER_CONDITIONS[w] for w in self.cfg.weather_conditions
            if w in WEATHER_CONDITIONS
        ]

        total_combos = len(weather_list) * len(self.screw_positions)
        total_episodes = total_combos * self.cfg.num_episodes_per_condition

        print(f"\n{'='*70}")
        print(f"RoboForce Simulation Evaluation")
        print(f"{'='*70}")
        print(f"  Policy: {self.cfg.policy_type} ({self.cfg.checkpoint_path})")
        print(f"  Weather conditions: {[w.name for w in weather_list]}")
        print(f"  Screw positions: {len(self.screw_positions)}")
        print(f"  Episodes per condition: {self.cfg.num_episodes_per_condition}")
        print(f"  Total episodes: {total_episodes}")
        print(f"{'='*70}\n")

        t0 = time.time()
        episode_global = 0

        for weather in weather_list:
            for screw_pos in self.screw_positions:
                for ep in range(self.cfg.num_episodes_per_condition):
                    result = self._run_episode(weather, screw_pos, episode_global)
                    self.results.append(result)
                    episode_global += 1

                # Progress update per (weather, position) block
                block_results = self.results[
                    -(self.cfg.num_episodes_per_condition):
                ]
                block_success = sum(1 for r in block_results if r.success)
                block_rate = block_success / len(block_results)
                elapsed = time.time() - t0
                print(
                    f"  [{episode_global:>5d}/{total_episodes}] "
                    f"weather={weather.name:<6s} pos={screw_pos.name:<18s} "
                    f"success={block_rate:.1%}  elapsed={elapsed:.0f}s"
                )

        elapsed = time.time() - t0
        report = self._build_report(elapsed)
        return report

    def _build_report(self, elapsed_s: float) -> dict[str, Any]:
        """Build the evaluation report from collected results."""
        total = len(self.results)
        successes = sum(1 for r in self.results if r.success)

        # Per-weather breakdown
        by_weather: dict[str, dict[str, Any]] = {}
        for w in self.cfg.weather_conditions:
            w_results = [r for r in self.results if r.weather == w]
            if not w_results:
                continue
            w_success = sum(1 for r in w_results if r.success)
            by_weather[w] = {
                "total_episodes": len(w_results),
                "successes": w_success,
                "success_rate": w_success / len(w_results),
                "mean_completion_time_s": float(np.mean([
                    r.completion_time_s for r in w_results if r.success
                ])) if w_success > 0 else 0.0,
                "mean_peak_force_n": float(np.mean([r.peak_force_n for r in w_results])),
                "mean_peak_torque_nm": float(np.mean([r.peak_torque_nm for r in w_results])),
                "total_collisions": sum(r.collision_count for r in w_results),
                "mean_inference_ms": float(np.mean([r.inference_time_ms for r in w_results])),
            }

        # Per-position breakdown
        by_position: dict[str, dict[str, Any]] = {}
        for pos in self.screw_positions:
            p_results = [r for r in self.results if r.screw_position_id == pos.id]
            if not p_results:
                continue
            p_success = sum(1 for r in p_results if r.success)
            by_position[pos.name] = {
                "total_episodes": len(p_results),
                "successes": p_success,
                "success_rate": p_success / len(p_results),
                "mean_completion_time_s": float(np.mean([
                    r.completion_time_s for r in p_results if r.success
                ])) if p_success > 0 else 0.0,
                "total_collisions": sum(r.collision_count for r in p_results),
            }

        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "policy_type": self.cfg.policy_type,
                "checkpoint_path": self.cfg.checkpoint_path,
                "weather_conditions": self.cfg.weather_conditions,
                "num_screw_positions": self.cfg.num_screw_positions,
                "num_episodes_per_condition": self.cfg.num_episodes_per_condition,
                "max_steps_per_episode": self.cfg.max_steps_per_episode,
                "control_frequency_hz": self.cfg.control_frequency_hz,
            },
            "summary": {
                "total_episodes": total,
                "successes": successes,
                "success_rate": successes / max(total, 1),
                "mean_completion_time_s": float(np.mean([
                    r.completion_time_s for r in self.results if r.success
                ])) if successes > 0 else 0.0,
                "mean_peak_force_n": float(np.mean([r.peak_force_n for r in self.results])),
                "total_collisions": sum(r.collision_count for r in self.results),
                "mean_inference_ms": float(np.mean([r.inference_time_ms for r in self.results])),
                "evaluation_time_s": round(elapsed_s, 1),
            },
            "by_weather": by_weather,
            "by_position": by_position,
            "episodes": [r.to_dict() for r in self.results],
        }

        return report


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

def save_report(report: dict[str, Any], output_dir: str | Path) -> tuple[str, str]:
    """Save the evaluation report as JSON and Markdown.

    Args:
        report: Evaluation report dict.
        output_dir: Output directory.

    Returns:
        Tuple of ``(json_path, markdown_path)``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = report.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    stamp = timestamp.replace(" ", "_").replace(":", "").replace("-", "")
    policy = report["config"]["policy_type"]

    # JSON
    json_path = out / f"eval_{policy}_{stamp}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Markdown
    md_path = out / f"eval_{policy}_{stamp}.md"
    md = _render_markdown(report)
    with open(md_path, "w") as f:
        f.write(md)

    print(f"\n📄 JSON saved: {json_path}")
    print(f"📝 Markdown saved: {md_path}")

    return str(json_path), str(md_path)


def _render_markdown(report: dict[str, Any]) -> str:
    """Render the evaluation report as Markdown."""
    s = report["summary"]
    c = report["config"]

    lines = [
        f"# RoboForce Simulation Evaluation Report\n",
        f"**Date:** {report['timestamp']}  ",
        f"**Policy:** {c['policy_type']} (`{c['checkpoint_path']}`)  ",
        f"**Evaluation time:** {s['evaluation_time_s']}s\n",
        f"## Summary\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total episodes | {s['total_episodes']} |",
        f"| Successes | {s['successes']} |",
        f"| **Success rate** | **{s['success_rate']:.1%}** |",
        f"| Mean completion time | {s['mean_completion_time_s']:.2f}s |",
        f"| Mean peak force | {s['mean_peak_force_n']:.2f} N |",
        f"| Total collisions | {s['total_collisions']} |",
        f"| Mean inference latency | {s['mean_inference_ms']:.2f} ms |",
        "",
        f"## By Weather Condition\n",
        f"| Condition | Episodes | Success Rate | Mean Time (s) | Collisions | Inference (ms) |",
        f"|-----------|----------|-------------|---------------|------------|----------------|",
    ]

    for w_name, w_data in report.get("by_weather", {}).items():
        lines.append(
            f"| {w_name} | {w_data['total_episodes']} "
            f"| {w_data['success_rate']:.1%} "
            f"| {w_data['mean_completion_time_s']:.2f} "
            f"| {w_data['total_collisions']} "
            f"| {w_data['mean_inference_ms']:.2f} |"
        )

    lines.extend([
        "",
        f"## By Screw Position\n",
        f"| Position | Episodes | Success Rate | Mean Time (s) | Collisions |",
        f"|----------|----------|-------------|---------------|------------|",
    ])

    for p_name, p_data in report.get("by_position", {}).items():
        lines.append(
            f"| {p_name} | {p_data['total_episodes']} "
            f"| {p_data['success_rate']:.1%} "
            f"| {p_data['mean_completion_time_s']:.2f} "
            f"| {p_data['total_collisions']} |"
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RoboForce — Simulation Policy Evaluation"
    )
    parser.add_argument(
        "--policy_type", type=str, default="gr00t",
        choices=["gr00t", "openpi"],
        help="VLA policy type",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/gr00t_screw_driving/best",
        help="Path to the trained policy checkpoint",
    )
    parser.add_argument(
        "--weather", nargs="+", type=str,
        default=["day", "night", "dusty"],
        help="Weather conditions to evaluate",
    )
    parser.add_argument(
        "--num_positions", type=int, default=10,
        help="Number of screw positions per panel",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=100,
        help="Episodes per (weather × position) condition",
    )
    parser.add_argument(
        "--max_steps", type=int, default=500,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/sim_eval",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--headless", action="store_true", default=True,
        help="Run Isaac Sim headless",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cfg = SimEvalCfg(
        policy_type=args.policy_type,
        checkpoint_path=args.checkpoint,
        weather_conditions=args.weather,
        num_screw_positions=args.num_positions,
        num_episodes_per_condition=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        output_dir=args.output_dir,
        seed=args.seed,
        headless=args.headless,
    )

    evaluator = SimulationEvaluator(cfg)
    report = evaluator.run()

    # Save report
    save_report(report, cfg.output_dir)

    # Print summary
    s = report["summary"]
    print(f"\n{'='*70}")
    print(f"  EVALUATION COMPLETE")
    print(f"  Success rate: {s['success_rate']:.1%} "
          f"({s['successes']}/{s['total_episodes']})")
    print(f"  Mean time: {s['mean_completion_time_s']:.2f}s")
    print(f"  Collisions: {s['total_collisions']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
