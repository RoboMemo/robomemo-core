"""
AMASS Motion Capture Data Loader.

Loads AMASS .npz files (SMPL format) and provides:
  - Forward kinematics: axis-angle poses → 3D joint positions
  - FPS resampling to match SONIC control frequency (50Hz)
  - Synthetic AMASS generator for testing without real data

SMPL skeleton (24 joints):
    0: Pelvis, 1: L_Hip, 2: R_Hip, 3: Spine1, 4: L_Knee, 5: R_Knee,
    6: Spine2, 7: L_Ankle, 8: R_Ankle, 9: Spine3, 10: L_Foot, 11: R_Foot,
    12: Neck, 13: L_Collar, 14: R_Collar, 15: Head, 16: L_Shoulder, 17: R_Shoulder,
    18: L_Elbow, 19: R_Elbow, 20: L_Wrist, 21: R_Wrist, 22: L_Hand, 23: R_Hand
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── SMPL Kinematic Tree ───────────────────────────────────────

SMPL_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]

# Parent joint index for each joint (-1 = root)
SMPL_PARENTS = [
    -1,  # 0: Pelvis (root)
    0,   # 1: L_Hip → Pelvis
    0,   # 2: R_Hip → Pelvis
    0,   # 3: Spine1 → Pelvis
    1,   # 4: L_Knee → L_Hip
    2,   # 5: R_Knee → R_Hip
    3,   # 6: Spine2 → Spine1
    4,   # 7: L_Ankle → L_Knee
    5,   # 8: R_Ankle → R_Knee
    6,   # 9: Spine3 → Spine2
    7,   # 10: L_Foot → L_Ankle
    8,   # 11: R_Foot → R_Ankle
    9,   # 12: Neck → Spine3
    9,   # 13: L_Collar → Spine3
    9,   # 14: R_Collar → Spine3
    12,  # 15: Head → Neck
    13,  # 16: L_Shoulder → L_Collar
    14,  # 17: R_Shoulder → R_Collar
    16,  # 18: L_Elbow → L_Shoulder
    17,  # 19: R_Elbow → R_Shoulder
    18,  # 20: L_Wrist → L_Elbow
    19,  # 21: R_Wrist → R_Elbow
    20,  # 22: L_Hand → L_Wrist
    21,  # 23: R_Hand → R_Wrist
]

# Approximate bone lengths in meters (from average SMPL body)
SMPL_BONE_OFFSETS = np.array([
    [0.0, 0.0, 0.0],         # 0: Pelvis (root)
    [0.065, -0.02, -0.08],   # 1: L_Hip
    [-0.065, -0.02, -0.08],  # 2: R_Hip
    [0.0, 0.02, 0.12],       # 3: Spine1
    [0.0, -0.01, -0.40],     # 4: L_Knee
    [0.0, -0.01, -0.40],     # 5: R_Knee
    [0.0, 0.01, 0.14],       # 6: Spine2
    [0.0, -0.02, -0.42],     # 7: L_Ankle
    [0.0, -0.02, -0.42],     # 8: R_Ankle
    [0.0, 0.01, 0.14],       # 9: Spine3
    [0.0, 0.10, -0.05],      # 10: L_Foot
    [0.0, 0.10, -0.05],      # 11: R_Foot
    [0.0, 0.01, 0.12],       # 12: Neck
    [0.04, 0.01, 0.06],      # 13: L_Collar
    [-0.04, 0.01, 0.06],     # 14: R_Collar
    [0.0, 0.02, 0.12],       # 15: Head
    [0.10, -0.01, 0.02],     # 16: L_Shoulder
    [-0.10, -0.01, 0.02],    # 17: R_Shoulder
    [0.26, 0.0, 0.0],        # 18: L_Elbow
    [-0.26, 0.0, 0.0],       # 19: R_Elbow
    [0.25, 0.0, 0.0],        # 20: L_Wrist
    [-0.25, 0.0, 0.0],       # 21: R_Wrist
    [0.10, 0.0, 0.0],        # 22: L_Hand
    [-0.10, 0.0, 0.0],       # 23: R_Hand
], dtype=np.float32)

NUM_SMPL_JOINTS = 24


# ── Rotation utilities ────────────────────────────────────────

def rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle (3,) to rotation matrix (3, 3) via Rodrigues formula."""
    theta = np.linalg.norm(axis_angle)
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    k = axis_angle / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def rodrigues_batch(axis_angles: np.ndarray) -> np.ndarray:
    """Batch convert axis-angle (N, 3) to rotation matrices (N, 3, 3)."""
    N = axis_angles.shape[0]
    result = np.zeros((N, 3, 3), dtype=np.float32)
    for i in range(N):
        result[i] = rodrigues(axis_angles[i])
    return result


def rotation_matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix (3, 3) to 6D rotation representation.

    Takes the first two columns of the rotation matrix.
    """
    return R[:, :2].T.flatten()  # (6,) — [r1x, r1y, r1z, r2x, r2y, r2z]


# ── Forward Kinematics ────────────────────────────────────────

def smpl_forward_kinematics(
    poses: np.ndarray,
    trans: np.ndarray,
    bone_offsets: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute 3D joint positions from SMPL axis-angle poses.

    Args:
        poses: (N_frames, 72) or (N_frames, 24, 3) axis-angle per joint.
               First 3 values are global rotation (joint 0 = pelvis).
        trans: (N_frames, 3) root translation.
        bone_offsets: (24, 3) bone offset vectors. Uses defaults if None.

    Returns:
        positions: (N_frames, 24, 3) joint world positions.
    """
    if bone_offsets is None:
        bone_offsets = SMPL_BONE_OFFSETS.copy()

    N = poses.shape[0]
    if poses.ndim == 2:
        poses = poses[:, :72].reshape(N, 24, 3)  # Only use first 24 joints

    positions = np.zeros((N, 24, 3), dtype=np.float32)

    for frame_idx in range(N):
        # Build world transforms for each joint
        world_transforms = np.zeros((24, 4, 4), dtype=np.float32)

        for j in range(24):
            R_local = rodrigues(poses[frame_idx, j])

            # Build local transform
            T_local = np.eye(4, dtype=np.float32)
            T_local[:3, :3] = R_local
            T_local[:3, 3] = bone_offsets[j]

            parent = SMPL_PARENTS[j]
            if parent == -1:
                # Root joint: apply global translation
                T_local[:3, 3] += trans[frame_idx]
                world_transforms[j] = T_local
            else:
                world_transforms[j] = world_transforms[parent] @ T_local

            positions[frame_idx, j] = world_transforms[j][:3, 3]

    return positions


# ── AMASS Data Container ──────────────────────────────────────

@dataclass
class AMASSMotion:
    """Processed AMASS motion data ready for SONIC consumption."""
    joint_positions: np.ndarray    # (N_frames, 24, 3) — world positions
    root_translation: np.ndarray   # (N_frames, 3)
    root_orientation: np.ndarray   # (N_frames, 3, 3) — rotation matrices
    raw_poses: np.ndarray          # (N_frames, 72) — axis-angle
    fps: float                     # frames per second
    duration: float                # total duration in seconds
    source: str                    # "amass" or "synthetic"

    @property
    def n_frames(self) -> int:
        return self.joint_positions.shape[0]

    @property
    def wrist_positions(self) -> np.ndarray:
        """Extract left and right wrist positions. (N_frames, 2, 3)"""
        return self.joint_positions[:, [20, 21], :]  # L_Wrist=20, R_Wrist=21

    @property
    def anchor_orientations_6d(self) -> np.ndarray:
        """Root/pelvis orientation in 6D representation. (N_frames, 6)"""
        N = self.root_orientation.shape[0]
        result = np.zeros((N, 6), dtype=np.float32)
        for i in range(N):
            result[i] = rotation_matrix_to_6d(self.root_orientation[i])
        return result


# ── AMASS File Loader ─────────────────────────────────────────

def load_amass(
    path: str | Path,
    target_fps: float = 50.0,
) -> AMASSMotion:
    """Load an AMASS .npz file and compute joint positions.

    Args:
        path: Path to .npz file.
        target_fps: Resample to this FPS for SONIC (default 50Hz).

    Returns:
        AMASSMotion with computed joint positions.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"AMASS file not found: {path}")

    data = np.load(str(path), allow_pickle=True)

    # AMASS format fields
    poses = data["poses"].astype(np.float32)     # (N, 156) or (N, 72)
    trans = data["trans"].astype(np.float32)      # (N, 3)
    source_fps = float(data.get("mocap_framerate", 120.0))

    logger.info(
        f"Loaded AMASS: {path.name}, frames={poses.shape[0]}, "
        f"fps={source_fps}, pose_dim={poses.shape[1]}"
    )

    # Only use first 72 dims (24 joints × 3 axis-angle)
    poses_72 = poses[:, :72]

    # Resample to target FPS
    if abs(source_fps - target_fps) > 0.5:
        poses_72, trans = _resample(poses_72, trans, source_fps, target_fps)
        logger.info(f"Resampled {source_fps}Hz → {target_fps}Hz: {poses_72.shape[0]} frames")

    # Forward kinematics
    joint_positions = smpl_forward_kinematics(poses_72, trans)

    # Extract root orientations
    root_orientations = rodrigues_batch(poses_72[:, :3])

    duration = joint_positions.shape[0] / target_fps

    return AMASSMotion(
        joint_positions=joint_positions,
        root_translation=trans,
        root_orientation=root_orientations,
        raw_poses=poses_72,
        fps=target_fps,
        duration=duration,
        source="amass",
    )


def _resample(
    poses: np.ndarray,
    trans: np.ndarray,
    source_fps: float,
    target_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample motion data from source_fps to target_fps via linear interpolation."""
    N_source = poses.shape[0]
    duration = N_source / source_fps
    N_target = int(duration * target_fps)

    t_source = np.linspace(0, duration, N_source)
    t_target = np.linspace(0, duration, N_target)

    # Interpolate each dimension
    poses_resampled = np.zeros((N_target, poses.shape[1]), dtype=np.float32)
    trans_resampled = np.zeros((N_target, 3), dtype=np.float32)

    for dim in range(poses.shape[1]):
        poses_resampled[:, dim] = np.interp(t_target, t_source, poses[:, dim])
    for dim in range(3):
        trans_resampled[:, dim] = np.interp(t_target, t_source, trans[:, dim])

    return poses_resampled, trans_resampled


# ── Synthetic AMASS Generator ─────────────────────────────────

class SyntheticAMASS:
    """Generate synthetic SMPL-format motion data for testing.

    Creates realistic-looking walking/waving motions using simple
    sinusoidal joint trajectories, without needing real AMASS downloads.
    """

    MOTIONS = {
        "walk": "_generate_walk",
        "wave": "_generate_wave",
        "squat": "_generate_squat",
        "dance": "_generate_dance",
    }

    def __init__(self, fps: float = 50.0):
        self.fps = fps

    def generate(
        self,
        motion_type: str = "walk",
        duration: float = 5.0,
    ) -> AMASSMotion:
        """Generate synthetic motion.

        Args:
            motion_type: One of "walk", "wave", "squat", "dance".
            duration: Duration in seconds.

        Returns:
            AMASSMotion with synthetic joint positions.
        """
        if motion_type not in self.MOTIONS:
            raise ValueError(
                f"Unknown motion type '{motion_type}'. "
                f"Available: {list(self.MOTIONS.keys())}"
            )

        method = getattr(self, self.MOTIONS[motion_type])
        poses, trans = method(duration)

        joint_positions = smpl_forward_kinematics(poses, trans)
        root_orientations = rodrigues_batch(poses[:, :3])

        logger.info(
            f"Generated synthetic '{motion_type}': "
            f"{joint_positions.shape[0]} frames, {duration:.1f}s @ {self.fps}Hz"
        )

        return AMASSMotion(
            joint_positions=joint_positions,
            root_translation=trans,
            root_orientation=root_orientations,
            raw_poses=poses,
            fps=self.fps,
            duration=duration,
            source=f"synthetic_{motion_type}",
        )

    def _generate_walk(self, duration: float) -> tuple[np.ndarray, np.ndarray]:
        """Walking motion: alternating leg swings + arm counter-swing."""
        N = int(duration * self.fps)
        t = np.linspace(0, duration, N)
        poses = np.zeros((N, 72), dtype=np.float32)
        trans = np.zeros((N, 3), dtype=np.float32)

        walk_freq = 1.2  # Hz (steps per second)
        stride_length = 0.5

        for i in range(N):
            phase = 2 * np.pi * walk_freq * t[i]

            # Root translation (forward movement)
            trans[i] = [0, stride_length * t[i] / duration * 2, 0.92]

            # Slight root sway
            poses[i, 0:3] = [0, 0, 0.03 * np.sin(phase)]  # pelvis rotation

            # Leg swings (hip pitch)
            hip_amplitude = 0.4
            poses[i, 1*3+2] = hip_amplitude * np.sin(phase)      # L_Hip pitch (z-axis)
            poses[i, 2*3+2] = -hip_amplitude * np.sin(phase)     # R_Hip pitch

            # Knee bends
            knee_amp = 0.5
            l_knee_phase = max(0, np.sin(phase))
            r_knee_phase = max(0, -np.sin(phase))
            poses[i, 4*3+2] = -knee_amp * l_knee_phase   # L_Knee
            poses[i, 5*3+2] = -knee_amp * r_knee_phase   # R_Knee

            # Ankle compensation
            poses[i, 7*3+2] = 0.2 * np.sin(phase)   # L_Ankle
            poses[i, 8*3+2] = -0.2 * np.sin(phase)  # R_Ankle

            # Arm counter-swing (shoulder pitch)
            arm_amp = 0.3
            poses[i, 16*3+2] = -arm_amp * np.sin(phase)  # L_Shoulder pitch
            poses[i, 17*3+2] = arm_amp * np.sin(phase)   # R_Shoulder pitch

            # Slight elbow bend
            poses[i, 18*3+1] = -0.3   # L_Elbow slightly bent
            poses[i, 19*3+1] = 0.3    # R_Elbow slightly bent

            # Spine twist
            poses[i, 3*3+2] = 0.05 * np.sin(phase)   # Spine1
            poses[i, 9*3+2] = 0.03 * np.sin(phase)   # Spine3

        return poses, trans

    def _generate_wave(self, duration: float) -> tuple[np.ndarray, np.ndarray]:
        """Waving motion: right arm waves while standing."""
        N = int(duration * self.fps)
        t = np.linspace(0, duration, N)
        poses = np.zeros((N, 72), dtype=np.float32)
        trans = np.zeros((N, 3), dtype=np.float32)

        wave_freq = 2.0

        for i in range(N):
            phase = 2 * np.pi * wave_freq * t[i]
            trans[i] = [0, 0, 0.92]

            # Right arm raised and waving
            poses[i, 17*3+0] = -0.3   # R_Shoulder abduction
            poses[i, 17*3+2] = -2.2   # R_Shoulder pitch (raised)
            poses[i, 19*3+1] = 1.8 + 0.4 * np.sin(phase)  # R_Elbow bend + wave
            poses[i, 21*3+2] = 0.3 * np.sin(phase * 2)    # R_Wrist

            # Left arm relaxed at side
            poses[i, 18*3+1] = -0.2   # L_Elbow slightly bent

            # Slight body sway
            poses[i, 0*3+0] = 0.02 * np.sin(phase * 0.5)
            poses[i, 9*3+0] = 0.03 * np.sin(phase * 0.5)

        return poses, trans

    def _generate_squat(self, duration: float) -> tuple[np.ndarray, np.ndarray]:
        """Squatting motion: periodic deep squats."""
        N = int(duration * self.fps)
        t = np.linspace(0, duration, N)
        poses = np.zeros((N, 72), dtype=np.float32)
        trans = np.zeros((N, 3), dtype=np.float32)

        squat_freq = 0.5

        for i in range(N):
            phase = 2 * np.pi * squat_freq * t[i]
            squat_depth = 0.5 * (1 - np.cos(phase)) * 0.5  # 0..0.5 range

            # Root drops during squat
            trans[i] = [0, 0, 0.92 - squat_depth * 0.35]

            # Hip flexion (both legs)
            hip_angle = squat_depth * 1.5
            poses[i, 1*3+2] = hip_angle    # L_Hip pitch
            poses[i, 2*3+2] = hip_angle    # R_Hip pitch

            # Knee flexion
            knee_angle = squat_depth * 2.5
            poses[i, 4*3+2] = -knee_angle  # L_Knee
            poses[i, 5*3+2] = -knee_angle  # R_Knee

            # Ankle dorsiflexion
            poses[i, 7*3+2] = squat_depth * 0.6
            poses[i, 8*3+2] = squat_depth * 0.6

            # Arms forward for balance
            poses[i, 16*3+2] = -squat_depth * 1.0   # L_Shoulder
            poses[i, 17*3+2] = -squat_depth * 1.0   # R_Shoulder

        return poses, trans

    def _generate_dance(self, duration: float) -> tuple[np.ndarray, np.ndarray]:
        """Dance-like motion: rhythmic arm and body movement."""
        N = int(duration * self.fps)
        t = np.linspace(0, duration, N)
        poses = np.zeros((N, 72), dtype=np.float32)
        trans = np.zeros((N, 3), dtype=np.float32)

        beat_freq = 1.5  # beats per second

        for i in range(N):
            phase = 2 * np.pi * beat_freq * t[i]
            trans[i] = [0.1 * np.sin(phase * 0.5), 0, 0.92 + 0.03 * np.sin(phase * 2)]

            # Body groove
            poses[i, 0*3+0] = 0.1 * np.sin(phase)       # pelvis yaw
            poses[i, 0*3+2] = 0.05 * np.sin(phase * 2)  # pelvis roll
            poses[i, 3*3+0] = 0.08 * np.sin(phase * 1.5)  # spine twist
            poses[i, 9*3+0] = 0.06 * np.sin(phase * 1.5)  # upper spine

            # Alternating arm raises
            l_phase = np.sin(phase)
            r_phase = np.sin(phase + np.pi * 0.5)

            poses[i, 16*3+2] = -1.0 - 0.8 * max(0, l_phase)   # L_Shoulder up
            poses[i, 18*3+1] = -0.5 - 0.8 * max(0, l_phase)   # L_Elbow
            poses[i, 17*3+2] = -1.0 - 0.8 * max(0, r_phase)   # R_Shoulder up
            poses[i, 19*3+1] = 0.5 + 0.8 * max(0, r_phase)    # R_Elbow

            # Slight leg movement
            poses[i, 1*3+2] = 0.15 * np.sin(phase)     # L_Hip
            poses[i, 2*3+2] = -0.15 * np.sin(phase)    # R_Hip
            poses[i, 4*3+2] = -0.2 * max(0, np.sin(phase))   # L_Knee
            poses[i, 5*3+2] = -0.2 * max(0, -np.sin(phase))  # R_Knee

        return poses, trans
