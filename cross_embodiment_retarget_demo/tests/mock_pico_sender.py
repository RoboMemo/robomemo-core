"""
Mock PICO data sender — standalone script.

Generates synthetic motion data and publishes over ZMQ PUB socket,
simulating the XRoboToolkit PC service from a PICO 4 Ultra headset
with leg and waist trackers.

Usage:
  python -m tests.mock_pico_sender --motion_type walk --port 5555
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mock_motion import run_mock_zmq_publisher

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mock PICO ZMQ data sender")
    parser.add_argument("--motion_type", default="walk",
                        choices=["walk", "wave", "squat", "stand"])
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
