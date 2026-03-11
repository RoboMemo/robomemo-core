"""
Xreal Air 2 Ultra + Beam Pro input receiver.

The Beam Pro streams head and hand tracking data over WebSocket.
Xreal Air 2 Ultra tracks head (6DoF via SLAM) and optionally hands
(via passthrough cameras). No leg tracking — lower body auto-generated
by kinematic planner.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Optional

import numpy as np

from .body_types import BodyPose, InputSource, Pose7D, convert_body_pose_to_z_up

logger = logging.getLogger(__name__)

try:
    import websockets
    import websockets.sync.client as ws_sync
except ImportError:
    websockets = None  # type: ignore
    ws_sync = None


class XrealReceiver:
    """WebSocket client that receives Xreal Air 2 Ultra tracking data
    relayed through Beam Pro."""

    def __init__(self, cfg: dict):
        xreal_cfg = cfg["input"]["xreal"]
        self._ip = xreal_cfg.get("beam_pro_ip", "192.168.1.100")
        self._port = xreal_cfg.get("beam_pro_port", 8765)
        self._enable_hands = xreal_cfg.get("enable_hand_tracking", True)
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
            if time.time() - self._last_recv_time > self._disconnect_timeout:
                logger.warning("Xreal data timeout — returning safety fallback")
                return None
            return self._latest_pose

    def start(self):
        if websockets is None:
            raise ImportError(
                "websockets is required for Xreal input. Install: pip install websockets"
            )
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        logger.info(f"Xreal receiver started, connecting to ws://{self._ip}:{self._port}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Xreal receiver stopped")

    def _recv_loop(self):
        uri = f"ws://{self._ip}:{self._port}"
        while self._running:
            try:
                with ws_sync.connect(uri, open_timeout=3) as ws:
                    logger.info(f"Connected to Beam Pro at {uri}")
                    while self._running:
                        try:
                            raw = ws.recv(timeout=1.0)
                        except TimeoutError:
                            continue
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        body = self._parse_body_pose(data)
                        if body:
                            with self._lock:
                                self._latest_pose = body
                                self._last_recv_time = time.time()
            except Exception as e:
                if self._running:
                    logger.warning(f"Xreal connection failed: {e}, retrying in 2s…")
                    time.sleep(2.0)

    def _parse_body_pose(self, data: dict) -> Optional[BodyPose]:
        def _p7(key: str) -> Optional[Pose7D]:
            if key not in data:
                return None
            v = data[key]
            if v is None or len(v) < 7:
                return None
            arr = np.array(v[:7], dtype=np.float32)
            return Pose7D(position=arr[:3], quaternion=arr[3:7])

        head = _p7("head")
        if head is None:
            return None

        left_hand = _p7("left_hand") or Pose7D(
            position=np.array([-0.2, 0.0, 0.9], dtype=np.float32),
            quaternion=np.array([1, 0, 0, 0], dtype=np.float32),
        )
        right_hand = _p7("right_hand") or Pose7D(
            position=np.array([0.2, 0.0, 0.9], dtype=np.float32),
            quaternion=np.array([1, 0, 0, 0], dtype=np.float32),
        )

        # Xreal has no leg tracking — provide standing defaults
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
            source=InputSource.XREAL,
            has_leg_tracking=False,
            has_waist_tracking=False,
        )
        return convert_body_pose_to_z_up(bp)
