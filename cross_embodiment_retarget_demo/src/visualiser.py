"""
Matplotlib-based 3D stick-figure visualiser for the mock physics env.

Renders a live 3D skeleton + joint angle bar chart at ~20 FPS.
Non-blocking: runs rendering in the main thread via matplotlib's
interactive backend, or in a separate thread with Agg+Tkinter.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# Simplified skeleton: joint index → (parent_index, name)
# Maps joint angles to 3D positions using a fixed kinematic chain
_JOINT_NAMES = [
    "waist_yaw", "waist_pitch", "waist_roll",
    "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw", "L_elbow", "L_wrist",
    "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow", "R_wrist",
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle_pitch", "R_ankle_roll",
    "L_hand_grip", "L_hand_wrist", "R_hand_grip", "R_hand_wrist",
]


def _joint_angles_to_keypoints(joint_pos: np.ndarray) -> dict[str, np.ndarray]:
    """Convert 29 joint angles to approximate 3D keypoint positions.

    This is a rough forward kinematics for visualisation only.
    """
    jp = joint_pos if len(joint_pos) >= 29 else np.zeros(29)

    waist = np.array([0, 0, 0.9])
    torso = waist + np.array([0, jp[1] * 0.2, 0.4])  # pitch → lean
    head = torso + np.array([jp[0] * 0.1, 0, 0.25])  # yaw → turn

    # Arms
    l_shoulder = torso + np.array([-0.2, 0, 0.05])
    l_elbow = l_shoulder + np.array([
        -0.15 - jp[4] * 0.15,
        jp[3] * 0.15,
        -0.15 + jp[3] * 0.1
    ])
    l_hand = l_elbow + np.array([0, jp[6] * 0.15, -0.2 - jp[6] * 0.1])

    r_shoulder = torso + np.array([0.2, 0, 0.05])
    r_elbow = r_shoulder + np.array([
        0.15 + jp[9] * 0.15,
        jp[8] * 0.15,
        -0.15 + jp[8] * 0.1
    ])
    r_hand = r_elbow + np.array([0, jp[11] * 0.15, -0.2 - jp[11] * 0.1])

    # Legs
    l_hip = waist + np.array([-0.1, 0, 0])
    l_knee_pos = l_hip + np.array([0, jp[15] * 0.2, -0.35 + jp[16] * 0.1])
    l_ankle = l_knee_pos + np.array([0, -jp[15] * 0.1, -0.35 + jp[16] * 0.15])

    r_hip = waist + np.array([0.1, 0, 0])
    r_knee_pos = r_hip + np.array([0, jp[21] * 0.2, -0.35 + jp[22] * 0.1])
    r_ankle = r_knee_pos + np.array([0, -jp[21] * 0.1, -0.35 + jp[22] * 0.15])

    return {
        "waist": waist, "torso": torso, "head": head,
        "l_shoulder": l_shoulder, "l_elbow": l_elbow, "l_hand": l_hand,
        "r_shoulder": r_shoulder, "r_elbow": r_elbow, "r_hand": r_hand,
        "l_hip": l_hip, "l_knee": l_knee_pos, "l_ankle": l_ankle,
        "r_hip": r_hip, "r_knee": r_knee_pos, "r_ankle": r_ankle,
    }


_BONES = [
    ("waist", "torso"), ("torso", "head"),
    ("torso", "l_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_hand"),
    ("torso", "r_shoulder"), ("r_shoulder", "r_elbow"), ("r_elbow", "r_hand"),
    ("waist", "l_hip"), ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
    ("waist", "r_hip"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
]


class LiveVisualiser:
    """Non-blocking 3D skeleton visualiser."""

    def __init__(self):
        self._fig = None
        self._ax3d = None
        self._ax_bar = None
        self._running = False

    def init(self):
        if not HAS_MPL:
            logger.warning("matplotlib not available — visualisation disabled")
            return

        plt.ion()
        self._fig = plt.figure(figsize=(14, 6))
        self._ax3d = self._fig.add_subplot(121, projection="3d")
        self._ax_bar = self._fig.add_subplot(122)
        self._fig.suptitle("Cross-Embodiment Retarget Demo — Unitree G1")
        self._running = True
        logger.info("Live visualiser initialised")

    def update(self, joint_pos: np.ndarray, fps: float = 0.0, source: str = ""):
        if not self._running or self._fig is None:
            return

        kps = _joint_angles_to_keypoints(joint_pos)

        # ── 3D skeleton ──
        self._ax3d.cla()
        self._ax3d.set_xlim(-0.8, 0.8)
        self._ax3d.set_ylim(-0.8, 0.8)
        self._ax3d.set_zlim(0, 2.0)
        self._ax3d.set_xlabel("X")
        self._ax3d.set_ylabel("Y")
        self._ax3d.set_zlabel("Z")
        self._ax3d.set_title(f"Unitree G1  |  {source}  |  {fps:.0f} Hz")

        # Draw bones
        for a, b in _BONES:
            pa, pb = kps[a], kps[b]
            self._ax3d.plot(
                [pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                "b-", linewidth=2
            )

        # Draw joints
        for name, pos in kps.items():
            color = "red" if "hand" in name or "ankle" in name else "darkblue"
            self._ax3d.scatter(*pos, c=color, s=30)

        # ── Joint angle bar chart ──
        self._ax_bar.cla()
        n = min(len(joint_pos), 29)
        names = _JOINT_NAMES[:n]
        vals = joint_pos[:n]
        colors = ["#1f77b4"] * 3 + ["#2ca02c"] * 5 + ["#d62728"] * 5 + \
                 ["#ff7f0e"] * 6 + ["#9467bd"] * 6 + ["#8c564b"] * 4
        colors = colors[:n]
        self._ax_bar.barh(range(n), vals, color=colors)
        self._ax_bar.set_yticks(range(n))
        self._ax_bar.set_yticklabels(names, fontsize=7)
        self._ax_bar.set_xlim(-2, 2)
        self._ax_bar.set_title("Joint Angles (rad)")
        self._ax_bar.invert_yaxis()

        self._fig.tight_layout()
        try:
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
        except Exception:
            pass

    def close(self):
        self._running = False
        if self._fig is not None:
            plt.close(self._fig)
        logger.info("Live visualiser closed")
