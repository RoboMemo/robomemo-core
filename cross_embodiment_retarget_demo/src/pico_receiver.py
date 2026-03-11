"""
PICO 4 Ultra input receiver via XRoboToolkit ZMQ protocol.

Captures:
  - Head pose (from PICO headset) — 7D
  - Left/Right hand (from PICO controllers) — 7D each
  - Left/Right ankle (from PICO Motion Trackers) — 7D each
  - Waist (from PICO Motion Tracker, optional) — 7D

The XRoboToolkit PC service publishes JSON messages over ZMQ PUB socket.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Optional

import numpy as np

try:
    import zmq
except ImportError:
    zmq = None  # type: ignore

from .body_types import BodyPose, InputSource, Pose7D, convert_body_pose_to_z_up

logger = logging.getLogger(__name__)


def _parse_pose_7d(data: dict, key: str) -> Optional[Pose7D]:
    """Parse a 7D pose [x,y,z,qw,qx,qy,qz] from a JSON dict."""
    if key not in data:
        return None
    vals = data[key]
    if vals is None or len(vals) < 7:
        return None
    arr = np.array(vals[:7], dtype=np.float32)
    return Pose7D(position=arr[:3], quaternion=arr[3:7])


class PicoReceiver:
    """ZMQ subscriber that receives PICO 4 Ultra tracking data
    from XRoboToolkit PC service."""

    def __init__(self, cfg: dict):
        if zmq is None:
            raise ImportError("pyzmq is required for PICO input. Install: pip install pyzmq")

        pico_cfg = cfg["input"]["pico"]
        self._host = pico_cfg.get("zmq_host", "0.0.0.0")
        self._port = pico_cfg.get("zmq_port", 5555)
        self._topic = pico_cfg.get("zmq_topic", "")
        self._enable_legs = pico_cfg.get("enable_leg_trackers", True)
        self._enable_waist = pico_cfg.get("enable_waist_tracker", True)
        self._disconnect_timeout = cfg["safety"]["disconnect_timeout_sec"]

        self._latest_pose: Optional[BodyPose] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_recv_time = 0.0

    @property
    def latest_pose(self) -> Optional[BodyPose]:
        with self._lock:
            if self._latest_pose is None:
                return None
            # Check timeout
            if time.time() - self._last_recv_time > self._disconnect_timeout:
                logger.warning("PICO data timeout — returning safety fallback")
                return None
            return self._latest_pose

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        logger.info(f"PICO receiver started on tcp://{self._host}:{self._port}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("PICO receiver stopped")

    def _recv_loop(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, self._topic.encode())
        sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1 s timeout for clean shutdown
        # Connect to the XRoboToolkit PUB socket
        addr = f"tcp://{self._host}:{self._port}"
        sock.connect(addr)
        logger.info(f"ZMQ SUB connected to {addr}")

        while self._running:
            try:
                raw = sock.recv_string(flags=0)
            except zmq.Again:
                continue
            except zmq.ZMQError as e:
                logger.error(f"ZMQ error: {e}")
                time.sleep(0.1)
                continue

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Failed to parse PICO JSON message")
                continue

            body = self._parse_body_pose(data)
            if body is not None:
                with self._lock:
                    self._latest_pose = body
                    self._last_recv_time = time.time()

        sock.close()
        ctx.term()

    def _parse_body_pose(self, data: dict) -> Optional[BodyPose]:
        head = _parse_pose_7d(data, "head")
        left_hand = _parse_pose_7d(data, "left_hand")
        right_hand = _parse_pose_7d(data, "right_hand")

        if head is None or left_hand is None or right_hand is None:
            return None

        left_ankle = _parse_pose_7d(data, "left_ankle")
        right_ankle = _parse_pose_7d(data, "right_ankle")
        waist = _parse_pose_7d(data, "waist") if self._enable_waist else None

        has_legs = left_ankle is not None and right_ankle is not None
        has_waist = waist is not None

        # Fallback: generate standing ankle poses if trackers not present
        if not has_legs:
            left_ankle = Pose7D(
                position=np.array([0.0, 0.1, 0.0], dtype=np.float32),
                quaternion=np.array([1, 0, 0, 0], dtype=np.float32),
            )
            right_ankle = Pose7D(
                position=np.array([0.0, -0.1, 0.0], dtype=np.float32),
                quaternion=np.array([1, 0, 0, 0], dtype=np.float32),
            )

        bp = BodyPose(
            head=head,
            left_hand=left_hand,
            right_hand=right_hand,
            left_ankle=left_ankle,
            right_ankle=right_ankle,
            waist=waist,
            source=InputSource.PICO,
            has_leg_tracking=has_legs and self._enable_legs,
            has_waist_tracking=has_waist,
        )
        # PICO XRoboToolkit sends Y-up right-hand coords
        return convert_body_pose_to_z_up(bp)
