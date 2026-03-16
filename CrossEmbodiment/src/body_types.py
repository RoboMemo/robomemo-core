"""
Shared data types and coordinate utilities for the retarget pipeline.
All input sources produce BodyPose, which SONIC retarget consumes.
"""

from __future__ import annotations

import dataclasses
import enum
import time
from typing import Optional

import numpy as np


class InputSource(enum.Enum):
    MOCK = "mock"
    WEBCAM = "webcam"
    XREAL = "xreal"
    PICO = "pico"


@dataclasses.dataclass
class Pose7D:
    """Position (x,y,z) + quaternion (qw,qx,qy,qz)."""
    position: np.ndarray  # (3,) float32
    quaternion: np.ndarray  # (4,) float32 — [qw, qx, qy, qz]

    @staticmethod
    def identity() -> Pose7D:
        return Pose7D(
            position=np.zeros(3, dtype=np.float32),
            quaternion=np.array([1, 0, 0, 0], dtype=np.float32),
        )

    def to_array(self) -> np.ndarray:
        return np.concatenate([self.position, self.quaternion])

    @staticmethod
    def from_array(arr: np.ndarray) -> Pose7D:
        arr = np.asarray(arr, dtype=np.float32)
        return Pose7D(position=arr[:3], quaternion=arr[3:7])


@dataclasses.dataclass
class BodyPose:
    """Standardised full-body pose — the universal interface between
    input sources and the SONIC retarget engine.

    Coordinate convention after normalisation: Z-up, right-hand (Isaac Lab).
    """
    head: Pose7D
    left_hand: Pose7D
    right_hand: Pose7D
    left_ankle: Pose7D
    right_ankle: Pose7D
    waist: Optional[Pose7D] = None  # Optional, from PICO waist tracker
    timestamp: float = dataclasses.field(default_factory=time.time)
    source: InputSource = InputSource.MOCK

    # Flags indicating which parts are real tracking vs generated
    has_leg_tracking: bool = False
    has_waist_tracking: bool = False

    def to_flat_array(self) -> np.ndarray:
        """Pack into (35,) or (42,) float32 array for SONIC encoder."""
        parts = [
            self.head.to_array(),       # 7
            self.left_hand.to_array(),   # 7
            self.right_hand.to_array(),  # 7
            self.left_ankle.to_array(),  # 7
            self.right_ankle.to_array(), # 7
        ]
        if self.waist is not None:
            parts.append(self.waist.to_array())  # 7
        return np.concatenate(parts).astype(np.float32)


def y_up_to_z_up(pose: Pose7D) -> Pose7D:
    """Convert Y-up right-hand → Z-up right-hand coordinate system.

    Rotation: x' = x, y' = -z, z' = y
    Quaternion: apply the same 90° rotation about X-axis.
    """
    x, y, z = pose.position
    pos_new = np.array([x, -z, y], dtype=np.float32)

    qw, qx, qy, qz = pose.quaternion
    # 90° rotation about X:  q_rot = (cos45, sin45, 0, 0)
    c = np.float32(np.cos(np.pi / 4))
    s = np.float32(np.sin(np.pi / 4))
    # Hamilton product: q_rot * q_orig
    nw = c * qw - s * qx
    nx = c * qx + s * qw
    ny = c * qy - s * qz
    nz = c * qz + s * qy
    quat_new = np.array([nw, nx, ny, nz], dtype=np.float32)
    norm = np.linalg.norm(quat_new)
    if norm > 1e-6:
        quat_new /= norm

    return Pose7D(position=pos_new, quaternion=quat_new)


def convert_body_pose_to_z_up(bp: BodyPose) -> BodyPose:
    """Convert an entire BodyPose from Y-up to Z-up."""
    return BodyPose(
        head=y_up_to_z_up(bp.head),
        left_hand=y_up_to_z_up(bp.left_hand),
        right_hand=y_up_to_z_up(bp.right_hand),
        left_ankle=y_up_to_z_up(bp.left_ankle),
        right_ankle=y_up_to_z_up(bp.right_ankle),
        waist=y_up_to_z_up(bp.waist) if bp.waist else None,
        timestamp=bp.timestamp,
        source=bp.source,
        has_leg_tracking=bp.has_leg_tracking,
        has_waist_tracking=bp.has_waist_tracking,
    )
