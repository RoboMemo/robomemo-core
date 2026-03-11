"""
Demo Runner — main orchestration for the Cross-Embodiment Retarget pipeline.

Coordinates:
  Input Source → SONIC Retarget → Simulation Env → Visualisation

Supports all four input modes: mock, webcam, xreal, pico.
Uses threading for concurrent input capture, inference, and sim stepping.
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .body_types import BodyPose, InputSource
from .isaac_env import create_sim_env, SimEnv
from .mock_motion import MockMotionSource
from .sonic_retarget import SonicRetarget
from .visualiser import LiveVisualiser, HAS_MPL

logger = logging.getLogger(__name__)


def load_config(path: str | Path = None) -> dict:
    """Load YAML config file."""
    if path is None:
        path = Path(__file__).parent.parent / "configs" / "demo_config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


class InputManager:
    """Unified interface to all input sources."""

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._source_name = cfg["input"]["source"]
        self._source = None

    def start(self) -> None:
        # Re-read source from config (may have been overridden after __init__)
        source_name = self._cfg["input"]["source"]
        self._source_name = source_name
        if source_name == "mock":
            from .mock_motion import MockMotionSource
            self._source = MockMotionSource(self._cfg)
        elif source_name == "webcam":
            from .webcam_capture import WebcamCapture
            self._source = WebcamCapture(self._cfg)
        elif source_name == "xreal":
            from .xreal_receiver import XrealReceiver
            self._source = XrealReceiver(self._cfg)
        elif source_name == "pico":
            from .pico_receiver import PicoReceiver
            self._source = PicoReceiver(self._cfg)
        else:
            raise ValueError(f"Unknown input source: {source_name}")
        self._source.start()
        logger.info(f"Input source '{source_name}' started")

    @property
    def latest_pose(self) -> Optional[BodyPose]:
        if self._source is None:
            return None
        return self._source.latest_pose

    def stop(self) -> None:
        if self._source is not None:
            self._source.stop()


class DemoRunner:
    """Main demo orchestrator."""

    def __init__(self, config_path: str | Path = None):
        self._cfg = load_config(config_path)
        self._setup_logging()

        self._input_mgr = InputManager(self._cfg)
        self._retarget = SonicRetarget(self._cfg)
        self._sim_env: Optional[SimEnv] = None
        self._viz: Optional[LiveVisualiser] = None

        self._running = False
        self._step_count = 0
        self._fps_counter = _FpsCounter()

        # Graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _setup_logging(self):
        level_str = self._cfg.get("logging", {}).get("level", "INFO")
        level = getattr(logging, level_str, logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    def _handle_signal(self, signum, frame):
        logger.info(f"Signal {signum} received — shutting down…")
        self._running = False

    def run(self):
        """Main loop — blocking call."""
        logger.info("=" * 60)
        logger.info("  Cross-Embodiment Retarget Demo")
        logger.info(f"  Input: {self._cfg['input']['source']}")
        logger.info(f"  Simulation: {self._cfg['simulation']['backend']}")
        logger.info(f"  Robot: {self._cfg['robot']['type']}")
        logger.info("=" * 60)

        # Init components
        self._sim_env = create_sim_env(self._cfg)
        self._input_mgr.start()

        # Init visualiser
        sim_backend = self._cfg["simulation"]["backend"]
        viz_enabled = (
            sim_backend == "mock_physics"
            and self._cfg["simulation"]["mock_physics"].get("enable_visualization", False)
            and HAS_MPL
        )
        if viz_enabled:
            self._viz = LiveVisualiser()
            self._viz.init()

        target_dt = self._cfg["simulation"].get(
            sim_backend, {}
        ).get("dt", self._cfg["simulation"].get("mock_physics", {}).get("dt", 0.02))

        self._running = True
        logger.info(f"Demo loop starting at {1/target_dt:.0f} Hz target…")

        print_interval = self._cfg.get("logging", {}).get("print_fps_interval_sec", 2.0)
        last_print = time.time()

        try:
            while self._running:
                t0 = time.time()

                # 1. Get input
                body_pose = self._input_mgr.latest_pose

                # 2. SONIC retarget
                joint_targets = self._retarget.infer(body_pose)

                # 3. Sim step
                self._sim_env.step(joint_targets)

                # 4. Update proprioception
                self._retarget.update_proprioception(
                    self._sim_env.get_joint_pos(),
                    self._sim_env.get_joint_vel(),
                    self._sim_env.get_gravity_vector(),
                )

                # 5. Visualise
                self._fps_counter.tick()
                if self._viz is not None and self._step_count % 2 == 0:
                    src_name = self._cfg["input"]["source"]
                    self._viz.update(
                        self._sim_env.get_joint_pos(),
                        fps=self._fps_counter.fps,
                        source=src_name,
                    )

                self._step_count += 1

                # Print stats
                now = time.time()
                if now - last_print >= print_interval:
                    fps = self._fps_counter.fps
                    latency = (now - t0) * 1000
                    pose_status = "LIVE" if body_pose is not None else "FALLBACK"
                    print(
                        f"[Step {self._step_count:6d}]  "
                        f"FPS: {fps:5.1f}  "
                        f"Latency: {latency:5.1f}ms  "
                        f"Input: {pose_status}"
                    )
                    last_print = now

                # Timing
                elapsed = time.time() - t0
                sleep_time = target_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _shutdown(self):
        logger.info("Shutting down…")
        self._running = False
        self._input_mgr.stop()
        if self._sim_env:
            self._sim_env.close()
        if self._viz:
            self._viz.close()
        logger.info(f"Demo ended after {self._step_count} steps.")


class _FpsCounter:
    """Simple FPS tracker."""

    def __init__(self, window: int = 60):
        self._times: list[float] = []
        self._window = window

    def tick(self):
        now = time.time()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times = self._times[-self._window:]

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1] - self._times[0]
        if dt <= 0:
            return 0.0
        return (len(self._times) - 1) / dt


# ── CLI entry point ───────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Embodiment Retarget Demo")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML")
    parser.add_argument("--input", type=str, default=None,
                        choices=["mock", "webcam", "xreal", "pico"],
                        help="Override input source")
    parser.add_argument("--motion", type=str, default=None,
                        choices=["walk", "wave", "squat", "stand"],
                        help="Mock motion type")
    parser.add_argument("--sim", type=str, default=None,
                        choices=["mock_physics", "isaac_lab"],
                        help="Override simulation backend")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable visualisation")
    args = parser.parse_args()

    runner = DemoRunner(config_path=args.config)

    # Apply overrides
    if args.input:
        runner._cfg["input"]["source"] = args.input
    if args.motion:
        runner._cfg["input"]["mock"]["motion_type"] = args.motion
    if args.sim:
        runner._cfg["simulation"]["backend"] = args.sim
    if args.no_viz:
        runner._cfg["simulation"]["mock_physics"]["enable_visualization"] = False

    runner.run()


if __name__ == "__main__":
    main()
