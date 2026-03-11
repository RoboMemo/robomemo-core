"""
Webcam-based full-body motion capture using MediaPipe Pose.

Produces BodyPose at ~30Hz from a standard USB/laptop webcam.
Provides head, hands, and ankles (with depth estimation heuristics).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from .body_types import BodyPose, InputSource, Pose7D, convert_body_pose_to_z_up

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

try:
    import mediapipe as mp
except ImportError:
    mp = None  # type: ignore


# MediaPipe Pose landmark indices
_NOSE = 0
_LEFT_WRIST = 15
_RIGHT_WRIST = 16
_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28
_LEFT_HIP = 23
_RIGHT_HIP = 24


def _landmark_to_pos(lm, w: int, h: int) -> np.ndarray:
    """Convert a MediaPipe landmark to a 3D position (metres, Y-up)."""
    # MediaPipe provides x,y in [0,1] image coords and z as relative depth
    # We scale to a rough metric space assuming person is ~1.7m tall at ~2m distance
    return np.array([
        (lm.x - 0.5) * 2.0,  # x: left-right (centred)
        -(lm.y - 0.5) * 2.0,  # y: up (MediaPipe y goes down)
        -lm.z * 2.0,          # z: depth (MediaPipe z is towards camera)
    ], dtype=np.float32)


class WebcamCapture:
    """MediaPipe Pose full-body capture from webcam."""

    def __init__(self, cfg: dict):
        if cv2 is None:
            raise ImportError("opencv-python is required for webcam. Install: pip install opencv-python")
        if mp is None:
            raise ImportError("mediapipe is required for webcam. Install: pip install mediapipe")

        cam_cfg = cfg["input"]["webcam"]
        self._device_id = cam_cfg.get("device_id", 0)
        self._width, self._height = cam_cfg.get("resolution", [1280, 720])
        self._fps = cam_cfg.get("fps", 30)
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
                return None
            return self._latest_pose

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"Webcam capture started (device {self._device_id})")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("Webcam capture stopped")

    def _capture_loop(self):
        cap = cv2.VideoCapture(self._device_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, self._fps)

        pose_detector = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose_detector.process(rgb)

                if result.pose_landmarks:
                    lms = result.pose_landmarks.landmark
                    h, w = frame.shape[:2]

                    head_pos = _landmark_to_pos(lms[_NOSE], w, h)
                    head_pos[1] += 0.15  # Offset nose to approximate head centre
                    lh_pos = _landmark_to_pos(lms[_LEFT_WRIST], w, h)
                    rh_pos = _landmark_to_pos(lms[_RIGHT_WRIST], w, h)
                    la_pos = _landmark_to_pos(lms[_LEFT_ANKLE], w, h)
                    ra_pos = _landmark_to_pos(lms[_RIGHT_ANKLE], w, h)

                    ident_q = np.array([1, 0, 0, 0], dtype=np.float32)
                    bp = BodyPose(
                        head=Pose7D(position=head_pos, quaternion=ident_q.copy()),
                        left_hand=Pose7D(position=lh_pos, quaternion=ident_q.copy()),
                        right_hand=Pose7D(position=rh_pos, quaternion=ident_q.copy()),
                        left_ankle=Pose7D(position=la_pos, quaternion=ident_q.copy()),
                        right_ankle=Pose7D(position=ra_pos, quaternion=ident_q.copy()),
                        source=InputSource.WEBCAM,
                        has_leg_tracking=True,
                        has_waist_tracking=False,
                    )
                    # MediaPipe is Y-up
                    bp = convert_body_pose_to_z_up(bp)

                    with self._lock:
                        self._latest_pose = bp
                        self._last_recv_time = time.time()
        finally:
            pose_detector.close()
            cap.release()
