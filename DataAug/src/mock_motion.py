"""
Mock motion data generator for testing without any hardware.

Generates synthetic BodyPose sequences: walk, wave, squat, stand.
Publishes on a ZMQ PUB socket so it can also test the PICO receiver path.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from typing import Optional

import numpy as np

from .body_types import BodyPose, InputSource, Pose7D

logger = logging.getLogger(__name__)

try:
    import zmq
except ImportError:
    zmq = None  # type: ignore


# ── Motion generators ──────────────────────────────────────────

def _generate_stand(t: float) -> BodyPose:
    """Static standing T-pose."""
    return BodyPose(
        head=Pose7D(np.array([0, 0, 1.6], dtype=np.float32),
                     np.array([1, 0, 0, 0], dtype=np.float32)),
        left_hand=Pose7D(np.array([-0.35, 0, 1.0], dtype=np.float32),
                          np.array([1, 0, 0, 0], dtype=np.float32)),
        right_hand=Pose7D(np.array([0.35, 0, 1.0], dtype=np.float32),
                           np.array([1, 0, 0, 0], dtype=np.float32)),
        left_ankle=Pose7D(np.array([-0.1, 0, 0.0], dtype=np.float32),
                           np.array([1, 0, 0, 0], dtype=np.float32)),
        right_ankle=Pose7D(np.array([0.1, 0, 0.0], dtype=np.float32),
                            np.array([1, 0, 0, 0], dtype=np.float32)),
        waist=Pose7D(np.array([0, 0, 0.9], dtype=np.float32),
                      np.array([1, 0, 0, 0], dtype=np.float32)),
        source=InputSource.MOCK,
        has_leg_tracking=True,
        has_waist_tracking=True,
    )


def _generate_walk(t: float) -> BodyPose:
    """Simulated walking cycle (~1 Hz stride)."""
    freq = 1.0  # Hz
    phase = 2 * math.pi * freq * t

    # Head bobs slightly
    head_z = 1.6 + 0.02 * math.sin(2 * phase)
    # Arms swing opposite to legs
    arm_swing = 0.15 * math.sin(phase)
    # Legs stride
    stride = 0.15 * math.sin(phase)
    foot_lift = max(0, 0.05 * math.sin(phase))
    foot_lift_r = max(0, 0.05 * math.sin(phase + math.pi))
    # Forward progress
    fwd = 0.3 * t

    return BodyPose(
        head=Pose7D(np.array([0, fwd, head_z], dtype=np.float32),
                     np.array([1, 0, 0, 0], dtype=np.float32)),
        left_hand=Pose7D(np.array([-0.25, fwd - arm_swing, 0.95], dtype=np.float32),
                          np.array([1, 0, 0, 0], dtype=np.float32)),
        right_hand=Pose7D(np.array([0.25, fwd + arm_swing, 0.95], dtype=np.float32),
                           np.array([1, 0, 0, 0], dtype=np.float32)),
        left_ankle=Pose7D(np.array([-0.1, fwd + stride, foot_lift], dtype=np.float32),
                           np.array([1, 0, 0, 0], dtype=np.float32)),
        right_ankle=Pose7D(np.array([0.1, fwd - stride, foot_lift_r], dtype=np.float32),
                            np.array([1, 0, 0, 0], dtype=np.float32)),
        waist=Pose7D(np.array([0, fwd, 0.9 + 0.01 * math.sin(2 * phase)], dtype=np.float32),
                      np.array([1, 0, 0, 0], dtype=np.float32)),
        source=InputSource.MOCK,
        has_leg_tracking=True,
        has_waist_tracking=True,
    )


def _generate_wave(t: float) -> BodyPose:
    """Standing with right hand waving."""
    base = _generate_stand(t)
    wave_angle = 0.3 * math.sin(4 * math.pi * t)
    base.right_hand = Pose7D(
        np.array([0.35 + 0.1 * math.sin(4 * math.pi * t),
                   0.0,
                   1.5 + 0.15 * math.cos(4 * math.pi * t)], dtype=np.float32),
        np.array([math.cos(wave_angle / 2), 0, 0, math.sin(wave_angle / 2)],
                  dtype=np.float32),
    )
    return base


def _generate_squat(t: float) -> BodyPose:
    """Squatting motion cycle (~0.5 Hz)."""
    phase = math.sin(math.pi * t)
    squat_depth = 0.3 * max(0, phase)

    return BodyPose(
        head=Pose7D(np.array([0, 0, 1.6 - squat_depth], dtype=np.float32),
                     np.array([1, 0, 0, 0], dtype=np.float32)),
        left_hand=Pose7D(np.array([-0.3, 0, 1.0 - squat_depth * 0.6], dtype=np.float32),
                          np.array([1, 0, 0, 0], dtype=np.float32)),
        right_hand=Pose7D(np.array([0.3, 0, 1.0 - squat_depth * 0.6], dtype=np.float32),
                           np.array([1, 0, 0, 0], dtype=np.float32)),
        left_ankle=Pose7D(np.array([-0.15, 0, 0.0], dtype=np.float32),
                           np.array([1, 0, 0, 0], dtype=np.float32)),
        right_ankle=Pose7D(np.array([0.15, 0, 0.0], dtype=np.float32),
                            np.array([1, 0, 0, 0], dtype=np.float32)),
        waist=Pose7D(np.array([0, 0, 0.9 - squat_depth * 0.8], dtype=np.float32),
                      np.array([1, 0, 0, 0], dtype=np.float32)),
        source=InputSource.MOCK,
        has_leg_tracking=True,
        has_waist_tracking=True,
    )


MOTION_GENERATORS = {
    "stand": _generate_stand,
    "walk": _generate_walk,
    "wave": _generate_wave,
    "squat": _generate_squat,
}


class MockMotionSource:
    """In-process mock motion source (no ZMQ needed)."""

    def __init__(self, cfg: dict):
        mock_cfg = cfg["input"]["mock"]
        self._motion_type = mock_cfg.get("motion_type", "walk")
        self._loop = mock_cfg.get("loop", True)
        self._duration = mock_cfg.get("duration_sec", 30.0)
        self._gen = MOTION_GENERATORS.get(self._motion_type, _generate_walk)
        self._start_time = 0.0

    def start(self):
        self._start_time = time.time()
        logger.info(f"Mock motion source started: {self._motion_type}")

    def stop(self):
        logger.info("Mock motion source stopped")

    @property
    def latest_pose(self) -> Optional[BodyPose]:
        t = time.time() - self._start_time
        if not self._loop and t > self._duration:
            return None
        if self._loop:
            t = t % self._duration
        return self._gen(t)


# ── ZMQ publisher for testing PICO receiver path ──────────────

def run_mock_zmq_publisher(
    motion_type: str = "walk",
    port: int = 5555,
    fps: float = 50.0,
    duration: float = 30.0,
    loop: bool = True,
):
    """Publish mock data over ZMQ PUB socket (standalone test utility)."""
    if zmq is None:
        raise ImportError("pyzmq required. Install: pip install pyzmq")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{port}")
    print(f"Mock ZMQ publisher on port {port}, motion={motion_type}, fps={fps}")

    gen = MOTION_GENERATORS.get(motion_type, _generate_walk)
    dt = 1.0 / fps
    start = time.time()

    try:
        while True:
            t = time.time() - start
            if not loop and t > duration:
                break
            if loop:
                t = t % duration

            bp = gen(t)
            msg = {
                "head": bp.head.to_array().tolist(),
                "left_hand": bp.left_hand.to_array().tolist(),
                "right_hand": bp.right_hand.to_array().tolist(),
                "left_ankle": bp.left_ankle.to_array().tolist(),
                "right_ankle": bp.right_ankle.to_array().tolist(),
            }
            if bp.waist:
                msg["waist"] = bp.waist.to_array().tolist()

            sock.send_string(json.dumps(msg))
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        ctx.term()
        print("Mock ZMQ publisher stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock PICO data sender")
    parser.add_argument("--motion_type", default="walk",
                        choices=list(MOTION_GENERATORS.keys()))
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--no-loop", action="store_true")
    args = parser.parse_args()

    run_mock_zmq_publisher(
        motion_type=args.motion_type,
        port=args.port,
        fps=args.fps,
        duration=args.duration,
        loop=not args.no_loop,
    )
