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


# Joint name tables per robot type
_JOINT_NAMES_G1 = [
    "waist_yaw", "waist_pitch", "waist_roll",
    "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw", "L_elbow", "L_wrist",
    "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow", "R_wrist",
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle_pitch", "L_ankle_roll",
    "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle_pitch", "R_ankle_roll",
    "L_hand_grip", "L_hand_wrist", "R_hand_grip", "R_hand_wrist",
]

_JOINT_NAMES_H1 = [
    "waist_yaw",
    "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw", "L_elbow",
    "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow",
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle_pitch",
    "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle_pitch",
]

_JOINT_NAMES_GR1T2 = [
    "waist_yaw", "waist_pitch", "waist_roll",
    "L_shoulder_P", "L_shoulder_R", "L_shoulder_Y", "L_elbow_P", "L_elbow_R", "L_wrist_P", "L_wrist_Y",
    "R_shoulder_P", "R_shoulder_R", "R_shoulder_Y", "R_elbow_P", "R_elbow_R", "R_wrist_P", "R_wrist_Y",
    "L_hip_Y", "L_hip_R", "L_hip_P", "L_knee", "L_ankle_P", "L_ankle_R",
    "R_hip_Y", "R_hip_R", "R_hip_P", "R_knee", "R_ankle_P", "R_ankle_R",
    "head_yaw", "head_pitch", "head_roll",
]

_ROBOT_JOINT_NAMES = {
    "unitree_g1": _JOINT_NAMES_G1,
    "unitree_h1": _JOINT_NAMES_H1,
    "fourier_gr1t2": _JOINT_NAMES_GR1T2,
}

_ROBOT_DISPLAY_NAMES = {
    "unitree_g1": "Unitree G1",
    "unitree_h1": "Unitree H1",
    "fourier_gr1t2": "Fourier GR1T2",
}


def _joint_angles_to_keypoints(joint_pos: np.ndarray, robot_type: str = "unitree_g1") -> dict[str, np.ndarray]:
    """Convert joint angles to approximate 3D keypoint positions.

    This is a rough forward kinematics for visualisation only.
    Works for all supported robots by mapping to a common skeleton.
    """
    # Pad to expected length for safety
    n = {"unitree_g1": 29, "unitree_h1": 19, "fourier_gr1t2": 32}.get(robot_type, 29)
    jp = joint_pos if len(joint_pos) >= n else np.zeros(n)

    waist = np.array([0, 0, 0.9])

    if robot_type == "unitree_h1":
        # H1: [0] waist_yaw, [1:5] L arm, [5:9] R arm, [9:14] L leg, [14:19] R leg
        torso = waist + np.array([0, 0, 0.4])
        head = torso + np.array([jp[0] * 0.1, 0, 0.25])
        l_shoulder = torso + np.array([-0.2, 0, 0.05])
        l_elbow = l_shoulder + np.array([-0.15 - jp[2] * 0.15, jp[1] * 0.15, -0.15 + jp[1] * 0.1])
        l_hand = l_elbow + np.array([0, jp[4] * 0.15, -0.2 - jp[4] * 0.1])
        r_shoulder = torso + np.array([0.2, 0, 0.05])
        r_elbow = r_shoulder + np.array([0.15 + jp[6] * 0.15, jp[5] * 0.15, -0.15 + jp[5] * 0.1])
        r_hand = r_elbow + np.array([0, jp[8] * 0.15, -0.2 - jp[8] * 0.1])
        l_hip = waist + np.array([-0.1, 0, 0])
        l_knee_pos = l_hip + np.array([0, jp[11] * 0.2, -0.35 + jp[12] * 0.1])
        l_ankle = l_knee_pos + np.array([0, -jp[11] * 0.1, -0.35 + jp[12] * 0.15])
        r_hip = waist + np.array([0.1, 0, 0])
        r_knee_pos = r_hip + np.array([0, jp[16] * 0.2, -0.35 + jp[17] * 0.1])
        r_ankle = r_knee_pos + np.array([0, -jp[16] * 0.1, -0.35 + jp[17] * 0.15])

    elif robot_type == "fourier_gr1t2":
        # GR1T2: [0:3] waist, [3:10] L arm, [10:17] R arm, [17:23] L leg, [23:29] R leg, [29:32] head
        torso = waist + np.array([0, jp[1] * 0.2, 0.4])
        head = torso + np.array([jp[29] * 0.1 if len(jp) > 29 else 0, 0, 0.25])
        l_shoulder = torso + np.array([-0.2, 0, 0.05])
        l_elbow = l_shoulder + np.array([-0.15 - jp[4] * 0.15, jp[3] * 0.15, -0.15 + jp[3] * 0.1])
        l_hand = l_elbow + np.array([0, jp[6] * 0.15, -0.2 - jp[6] * 0.1])
        r_shoulder = torso + np.array([0.2, 0, 0.05])
        r_elbow = r_shoulder + np.array([0.15 + jp[11] * 0.15, jp[10] * 0.15, -0.15 + jp[10] * 0.1])
        r_hand = r_elbow + np.array([0, jp[13] * 0.15, -0.2 - jp[13] * 0.1])
        l_hip = waist + np.array([-0.1, 0, 0])
        l_knee_pos = l_hip + np.array([0, jp[19] * 0.2, -0.35 + jp[20] * 0.1])
        l_ankle = l_knee_pos + np.array([0, -jp[19] * 0.1, -0.35 + jp[20] * 0.15])
        r_hip = waist + np.array([0.1, 0, 0])
        r_knee_pos = r_hip + np.array([0, jp[25] * 0.2, -0.35 + jp[26] * 0.1])
        r_ankle = r_knee_pos + np.array([0, -jp[25] * 0.1, -0.35 + jp[26] * 0.15])

    else:
        # G1 (default)
        torso = waist + np.array([0, jp[1] * 0.2, 0.4])
        head = torso + np.array([jp[0] * 0.1, 0, 0.25])
        l_shoulder = torso + np.array([-0.2, 0, 0.05])
        l_elbow = l_shoulder + np.array([-0.15 - jp[4] * 0.15, jp[3] * 0.15, -0.15 + jp[3] * 0.1])
        l_hand = l_elbow + np.array([0, jp[6] * 0.15, -0.2 - jp[6] * 0.1])
        r_shoulder = torso + np.array([0.2, 0, 0.05])
        r_elbow = r_shoulder + np.array([0.15 + jp[9] * 0.15, jp[8] * 0.15, -0.15 + jp[8] * 0.1])
        r_hand = r_elbow + np.array([0, jp[11] * 0.15, -0.2 - jp[11] * 0.1])
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

    def __init__(self, robot_type: str = "unitree_g1"):
        self._fig = None
        self._ax3d = None
        self._ax_bar = None
        self._running = False
        self._robot_type = robot_type
        self._display_name = _ROBOT_DISPLAY_NAMES.get(robot_type, robot_type)

    def init(self):
        if not HAS_MPL:
            logger.warning("matplotlib not available — visualisation disabled")
            return

        plt.ion()
        self._fig = plt.figure(figsize=(14, 6))
        self._ax3d = self._fig.add_subplot(121, projection="3d")
        self._ax_bar = self._fig.add_subplot(122)
        self._fig.suptitle(f"Cross-Embodiment Retarget Demo — {self._display_name}")
        self._running = True
        logger.info("Live visualiser initialised")

    def update(self, joint_pos: np.ndarray, fps: float = 0.0, source: str = ""):
        if not self._running or self._fig is None:
            return

        kps = _joint_angles_to_keypoints(joint_pos, self._robot_type)

        # ── 3D skeleton ──
        self._ax3d.cla()
        self._ax3d.set_xlim(-0.8, 0.8)
        self._ax3d.set_ylim(-0.8, 0.8)
        self._ax3d.set_zlim(0, 2.0)
        self._ax3d.set_xlabel("X")
        self._ax3d.set_ylabel("Y")
        self._ax3d.set_zlabel("Z")
        self._ax3d.set_title(f"{self._display_name}  |  {source}  |  {fps:.0f} Hz")

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
        joint_names = _ROBOT_JOINT_NAMES.get(self._robot_type, _JOINT_NAMES_G1)
        n = min(len(joint_pos), len(joint_names))
        names = joint_names[:n]
        vals = joint_pos[:n]
        colors = plt.cm.tab10(np.linspace(0, 1, n)).tolist()
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
