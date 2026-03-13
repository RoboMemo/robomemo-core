"""
SONIC ↔ AMASS Bridge.

Converts AMASS SMPL data into the flat observation vectors required
by the SONIC encoder (1762-dim) and decoder (994-dim) ONNX models.

Encoder observation layout (total = 1762):
  [  0:   4] encoder_mode_4                                — one-hot mode (4)
  [  4: 294] motion_joint_positions_10frame_step5          — 29 × 10  (290)
  [294: 584] motion_joint_velocities_10frame_step5         — 29 × 10  (290)
  [584: 594] motion_root_z_position_10frame_step5          — 1 × 10   (10)
  [594: 595] motion_root_z_position                        — 1        (1)
  [595: 655] motion_anchor_orientation_10frame_step5       — 6 × 10   (60)
  [655: 775] motion_joint_positions_lowerbody_10frame_step5 — 12 × 10 (120)
  [775: 895] motion_joint_velocities_lowerbody_10frame_step5— 12 × 10 (120)
  [895: 904] vr_3point_local_target                        — 3 × 3    (9)
  [904: 922] vr_3point_local_orn_target                    — 3 × 6    (18)
  [922:1642] smpl_joints_10frame_step1                     — 24 × 3 × 10 (720)
  [1642:1702] smpl_anchor_orientation_10frame_step1        — 6 × 10   (60)
  [1702:1762] motion_joint_positions_wrists_10frame_step1  — 2 × 3 × 10 (60)

Decoder observation layout (total = 994):
  [  0:  64] token_state        — encoder output tokens (64)
  [ 64: 994] proprioception     — 10 frames × 93 dims per frame (930)
    Per frame (93):
      [0:29]  body_joint_positions    (29)
      [29:58] body_joint_velocities   (29)
      [58:87] last_actions            (29)
      [87:90] base_angular_velocity   (3)
      [90:93] gravity_direction       (3)

NOTE: Testing shows this ONNX checkpoint only has a trained G1 mode path.
SMPL regions [922:1762] are dead (produce zero gradient). We therefore use
G1 mode (mode_id=0) and convert SMPL joint data to G1-compatible format:
  - Map SMPL 24 joints → G1 29 joint positions via analytical retarget
  - Compute velocities via finite differencing
  - Pack into g1 observation regions [4:294] and [294:584]
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np

from .amass_loader import AMASSMotion, rotation_matrix_to_6d

logger = logging.getLogger(__name__)

# ── Observation dimensions ────────────────────────────────────

ENCODER_DIM = 1762
DECODER_DIM = 994
ENCODER_TOKEN_DIM = 64
NUM_G1_JOINTS = 29
HISTORY_FRAMES = 10
HISTORY_STEP5 = 5  # step5 means sample every 5th frame
PROPRIOCEPTION_PER_FRAME = 93  # 29 + 29 + 29 + 3 + 3

# Encoder offsets (validated against ONNX model)
ENC_MODE_START = 0
ENC_MODE_DIM = 4
ENC_G1_JOINT_POS_START = 4       # 29 × 10 = 290
ENC_G1_JOINT_VEL_START = 294     # 29 × 10 = 290
ENC_ROOT_Z_HIST_START = 584      # 1 × 10 = 10
ENC_ROOT_Z_CURR_START = 594      # 1
ENC_ANCHOR_ORIENT_START = 595    # 6 × 10 = 60
ENC_LOWERBODY_POS_START = 655    # 12 × 10 = 120
ENC_LOWERBODY_VEL_START = 775    # 12 × 10 = 120
ENC_VR_TARGET_START = 895        # 3 × 3 = 9
ENC_VR_ORN_START = 904           # 3 × 6 = 18
ENC_SMPL_JOINTS_START = 922      # 24 × 3 × 10 = 720
ENC_SMPL_ANCHOR_START = 1642     # 6 × 10 = 60
ENC_WRIST_POS_START = 1702       # 2 × 3 × 10 = 60

# G1 mode one-hot (the only mode with trained weights in this checkpoint)
G1_MODE_ONEHOT = np.array([1, 0, 0, 0], dtype=np.float32)   # mode_id=0

# G1 lower body joint indices (12 joints: 6 per leg)
# From G1 joint layout: [13:19] left leg, [19:25] right leg
G1_LOWER_BODY_INDICES = list(range(13, 25))


# ── SMPL → G1 joint mapping ──────────────────────────────────

def smpl_to_g1_joint_positions(smpl_joints: np.ndarray) -> np.ndarray:
    """Map SMPL 24-joint positions to G1 29-joint angle-like values.

    Uses geometric relationships between SMPL keypoints to estimate
    corresponding G1 joint angles. This is an analytical approximation,
    not a learned mapping.

    Args:
        smpl_joints: (24, 3) SMPL world joint positions.

    Returns:
        (29,) approximate G1 joint angle targets.
    """
    g1 = np.zeros(29, dtype=np.float32)

    # Reference vectors
    pelvis = smpl_joints[0]
    spine1 = smpl_joints[3]
    spine3 = smpl_joints[9]
    neck = smpl_joints[12]

    # Spine direction
    spine_dir = spine3 - pelvis
    spine_len = max(np.linalg.norm(spine_dir), 0.01)
    spine_dir_norm = spine_dir / spine_len

    # Torso angles (approximate from spine direction)
    g1[0] = np.arctan2(spine_dir[0], spine_dir[2]) * 0.5   # yaw
    g1[1] = np.arctan2(spine_dir[1], spine_dir[2]) * 0.5   # pitch
    g1[2] = np.arctan2(spine_dir[0], spine_dir[1]) * 0.3   # roll

    # Left arm
    l_shoulder = smpl_joints[16]
    l_elbow = smpl_joints[18]
    l_wrist = smpl_joints[20]
    l_hand = smpl_joints[22]

    l_upper_arm = l_elbow - l_shoulder
    l_forearm = l_wrist - l_elbow

    g1[3] = np.arctan2(-l_upper_arm[2], np.sqrt(l_upper_arm[0]**2 + l_upper_arm[1]**2))  # shoulder pitch
    g1[4] = np.arctan2(l_upper_arm[0], -l_upper_arm[2])  # shoulder roll
    g1[5] = np.arctan2(l_upper_arm[1], l_upper_arm[0])   # shoulder yaw

    l_elbow_angle = np.arccos(np.clip(
        np.dot(l_upper_arm, l_forearm) /
        (max(np.linalg.norm(l_upper_arm), 0.01) * max(np.linalg.norm(l_forearm), 0.01)),
        -1, 1
    ))
    g1[6] = -(np.pi - l_elbow_angle)  # elbow (negative = flexion)
    g1[7] = 0.0  # wrist

    # Right arm (mirror of left)
    r_shoulder = smpl_joints[17]
    r_elbow = smpl_joints[19]
    r_wrist = smpl_joints[21]

    r_upper_arm = r_elbow - r_shoulder
    r_forearm = r_wrist - r_elbow

    g1[8] = np.arctan2(-r_upper_arm[2], np.sqrt(r_upper_arm[0]**2 + r_upper_arm[1]**2))
    g1[9] = np.arctan2(r_upper_arm[0], -r_upper_arm[2])
    g1[10] = np.arctan2(r_upper_arm[1], r_upper_arm[0])

    r_elbow_angle = np.arccos(np.clip(
        np.dot(r_upper_arm, r_forearm) /
        (max(np.linalg.norm(r_upper_arm), 0.01) * max(np.linalg.norm(r_forearm), 0.01)),
        -1, 1
    ))
    g1[11] = -(np.pi - r_elbow_angle)
    g1[12] = 0.0

    # Left leg
    l_hip = smpl_joints[1]
    l_knee = smpl_joints[4]
    l_ankle = smpl_joints[7]

    l_thigh = l_knee - l_hip
    l_shin = l_ankle - l_knee

    g1[13] = np.arctan2(l_thigh[0], -l_thigh[2]) * 0.5   # hip yaw
    g1[14] = np.arctan2(-l_thigh[0], -l_thigh[2]) * 0.5  # hip roll
    g1[15] = np.arctan2(l_thigh[1], -l_thigh[2])          # hip pitch

    l_knee_angle = np.arccos(np.clip(
        np.dot(l_thigh, l_shin) /
        (max(np.linalg.norm(l_thigh), 0.01) * max(np.linalg.norm(l_shin), 0.01)),
        -1, 1
    ))
    g1[16] = np.pi - l_knee_angle  # knee (positive = extension)
    g1[17] = np.arctan2(l_shin[1], -l_shin[2]) * 0.5  # ankle pitch
    g1[18] = np.arctan2(l_shin[0], -l_shin[2]) * 0.3  # ankle roll

    # Right leg
    r_hip = smpl_joints[2]
    r_knee = smpl_joints[5]
    r_ankle = smpl_joints[8]

    r_thigh = r_knee - r_hip
    r_shin = r_ankle - r_knee

    g1[19] = np.arctan2(r_thigh[0], -r_thigh[2]) * 0.5
    g1[20] = np.arctan2(-r_thigh[0], -r_thigh[2]) * 0.5
    g1[21] = np.arctan2(r_thigh[1], -r_thigh[2])

    r_knee_angle = np.arccos(np.clip(
        np.dot(r_thigh, r_shin) /
        (max(np.linalg.norm(r_thigh), 0.01) * max(np.linalg.norm(r_shin), 0.01)),
        -1, 1
    ))
    g1[22] = np.pi - r_knee_angle
    g1[23] = np.arctan2(r_shin[1], -r_shin[2]) * 0.5
    g1[24] = np.arctan2(r_shin[0], -r_shin[2]) * 0.3

    # Hands (25-28) — no direct SMPL mapping, leave at zero
    return g1


class HistoryBuffer:
    """Ring buffer for maintaining frame history windows."""

    def __init__(self, max_frames: int, feature_dim: int):
        self.max_frames = max_frames
        self.feature_dim = feature_dim
        self._buffer: deque[np.ndarray] = deque(maxlen=max_frames)
        # Initialize with zeros
        for _ in range(max_frames):
            self._buffer.append(np.zeros(feature_dim, dtype=np.float32))

    def push(self, frame: np.ndarray):
        """Add a new frame to the history."""
        self._buffer.append(frame.astype(np.float32).flatten()[:self.feature_dim])

    def get_flat(self) -> np.ndarray:
        """Get all frames as a flat vector. Oldest first."""
        return np.concatenate(list(self._buffer))

    def get_flat_step(self, step: int, n_samples: int = 10) -> np.ndarray:
        """Get n_samples frames sampled at given step interval. Returns flat vector.

        Samples from newest frame backwards. If buffer is too small, repeats oldest.
        """
        buf = list(self._buffer)
        n_buf = len(buf)
        # Sample indices from newest backwards with given step
        indices = []
        for i in range(n_samples):
            idx = n_buf - 1 - i * step
            indices.append(max(0, idx))
        indices.reverse()
        return np.concatenate([buf[i] for i in indices])

    @property
    def latest(self) -> np.ndarray:
        """Most recent frame."""
        return self._buffer[-1].copy()


class SonicAMASSBridge:
    """Bridge between AMASS SMPL data and SONIC ONNX model inputs.

    Uses G1 mode (mode_id=0) since the available ONNX checkpoint only has
    trained weights for the G1 observation path. SMPL joint positions are
    converted to G1-compatible format via analytical retargeting.
    """

    def __init__(self, dt: float = 0.02):
        """
        Args:
            dt: Simulation timestep in seconds (default 0.02 = 50Hz).
        """
        self.dt = dt

        # History buffers for encoder (G1 mode, step5 = sample every 5 frames)
        # We keep a deeper buffer so step5 sampling works properly
        self._g1_joint_pos_history = HistoryBuffer(HISTORY_FRAMES * HISTORY_STEP5, NUM_G1_JOINTS)
        self._g1_joint_vel_history = HistoryBuffer(HISTORY_FRAMES * HISTORY_STEP5, NUM_G1_JOINTS)
        self._root_z_history = HistoryBuffer(HISTORY_FRAMES * HISTORY_STEP5, 1)
        self._anchor_orient_history = HistoryBuffer(HISTORY_FRAMES * HISTORY_STEP5, 6)
        self._lowerbody_pos_history = HistoryBuffer(HISTORY_FRAMES * HISTORY_STEP5, 12)
        self._lowerbody_vel_history = HistoryBuffer(HISTORY_FRAMES * HISTORY_STEP5, 12)

        # Previous G1 joint positions for velocity computation
        self._prev_g1_joints = np.zeros(NUM_G1_JOINTS, dtype=np.float32)
        self._prev_lowerbody = np.zeros(12, dtype=np.float32)

        # History buffers for decoder proprioception (10 frames, step1)
        self._joint_pos_history = HistoryBuffer(HISTORY_FRAMES, NUM_G1_JOINTS)
        self._joint_vel_history = HistoryBuffer(HISTORY_FRAMES, NUM_G1_JOINTS)
        self._last_actions_history = HistoryBuffer(HISTORY_FRAMES, NUM_G1_JOINTS)
        self._angular_vel_history = HistoryBuffer(HISTORY_FRAMES, 3)
        self._gravity_history = HistoryBuffer(HISTORY_FRAMES, 3)

        # Current robot state (mock physics)
        self.joint_positions = np.zeros(NUM_G1_JOINTS, dtype=np.float32)
        self.joint_velocities = np.zeros(NUM_G1_JOINTS, dtype=np.float32)
        self.last_actions = np.zeros(NUM_G1_JOINTS, dtype=np.float32)
        self.base_angular_velocity = np.zeros(3, dtype=np.float32)
        self.gravity_direction = np.array([0, 0, -1], dtype=np.float32)

        self._frame_count = 0

    def reset(self):
        """Reset all state to initial conditions."""
        self.__init__(self.dt)

    def pack_encoder_input(
        self,
        motion: AMASSMotion,
        frame_idx: int,
    ) -> np.ndarray:
        """Pack encoder input vector (1762-dim) for G1 mode.

        Converts SMPL joints to G1 joint format and fills the G1 observation
        regions that the encoder actually responds to.

        Args:
            motion: AMASSMotion data.
            frame_idx: Current frame index.

        Returns:
            (1, 1762) float32 array ready for ONNX inference.
        """
        # Extract current frame SMPL data
        smpl_joints = motion.joint_positions[frame_idx]   # (24, 3)
        root_orient = motion.root_orientation[frame_idx]  # (3, 3)
        root_z = motion.root_translation[frame_idx, 2]    # scalar

        # Convert SMPL → G1 joint positions
        g1_joints = smpl_to_g1_joint_positions(smpl_joints)

        # Compute velocities via finite differencing
        g1_vel = (g1_joints - self._prev_g1_joints) / self.dt
        self._prev_g1_joints = g1_joints.copy()

        # Extract lower body (12 joints)
        lowerbody_pos = g1_joints[G1_LOWER_BODY_INDICES]
        lowerbody_vel = (lowerbody_pos - self._prev_lowerbody) / self.dt
        self._prev_lowerbody = lowerbody_pos.copy()

        # Anchor orientation in 6D
        anchor_6d = rotation_matrix_to_6d(root_orient)

        # Update all history buffers
        self._g1_joint_pos_history.push(g1_joints)
        self._g1_joint_vel_history.push(g1_vel)
        self._root_z_history.push(np.array([root_z]))
        self._anchor_orient_history.push(anchor_6d)
        self._lowerbody_pos_history.push(lowerbody_pos)
        self._lowerbody_vel_history.push(lowerbody_vel)

        # Build the flat encoder input
        obs = np.zeros(ENCODER_DIM, dtype=np.float32)

        # Mode one-hot: G1 mode (the only responsive mode in this checkpoint)
        obs[ENC_MODE_START:ENC_MODE_START + ENC_MODE_DIM] = G1_MODE_ONEHOT

        # G1 joint positions (step5 sampling, 10 entries × 29 = 290)
        obs[ENC_G1_JOINT_POS_START:ENC_G1_JOINT_POS_START + 290] = (
            self._g1_joint_pos_history.get_flat_step(HISTORY_STEP5)
        )

        # G1 joint velocities (step5 sampling, 10 × 29 = 290)
        obs[ENC_G1_JOINT_VEL_START:ENC_G1_JOINT_VEL_START + 290] = (
            self._g1_joint_vel_history.get_flat_step(HISTORY_STEP5)
        )

        # Root z position history (step5, 10 × 1 = 10)
        obs[ENC_ROOT_Z_HIST_START:ENC_ROOT_Z_HIST_START + 10] = (
            self._root_z_history.get_flat_step(HISTORY_STEP5)
        )

        # Root z position current
        obs[ENC_ROOT_Z_CURR_START] = root_z

        # Anchor orientation history (step5, 10 × 6 = 60)
        obs[ENC_ANCHOR_ORIENT_START:ENC_ANCHOR_ORIENT_START + 60] = (
            self._anchor_orient_history.get_flat_step(HISTORY_STEP5)
        )

        # Lower body joint positions (step5, 10 × 12 = 120)
        obs[ENC_LOWERBODY_POS_START:ENC_LOWERBODY_POS_START + 120] = (
            self._lowerbody_pos_history.get_flat_step(HISTORY_STEP5)
        )

        # Lower body joint velocities (step5, 10 × 12 = 120)
        obs[ENC_LOWERBODY_VEL_START:ENC_LOWERBODY_VEL_START + 120] = (
            self._lowerbody_vel_history.get_flat_step(HISTORY_STEP5)
        )

        return obs.reshape(1, ENCODER_DIM)

    def pack_decoder_input(
        self,
        encoded_tokens: np.ndarray,
    ) -> np.ndarray:
        """Pack decoder input vector (994-dim).

        Args:
            encoded_tokens: (64,) or (1, 64) from encoder output.

        Returns:
            (1, 994) float32 array ready for ONNX inference.
        """
        tokens = encoded_tokens.flatten()[:ENCODER_TOKEN_DIM]

        # Update proprioception history with current state
        self._joint_pos_history.push(self.joint_positions)
        self._joint_vel_history.push(self.joint_velocities)
        self._last_actions_history.push(self.last_actions)
        self._angular_vel_history.push(self.base_angular_velocity)
        self._gravity_history.push(self.gravity_direction)

        # Build proprioception: interleave per-frame features
        # Layout: for each of 10 frames: [joint_pos(29), joint_vel(29),
        #          last_actions(29), angular_vel(3), gravity(3)]
        prop_frames = []
        jp_list = list(self._joint_pos_history._buffer)
        jv_list = list(self._joint_vel_history._buffer)
        la_list = list(self._last_actions_history._buffer)
        av_list = list(self._angular_vel_history._buffer)
        gd_list = list(self._gravity_history._buffer)

        for f in range(HISTORY_FRAMES):
            frame_data = np.concatenate([
                jp_list[f],   # 29
                jv_list[f],   # 29
                la_list[f],   # 29
                av_list[f],   # 3
                gd_list[f],   # 3
            ])  # = 93
            prop_frames.append(frame_data)

        proprioception = np.concatenate(prop_frames)  # 930

        # Concatenate: tokens (64) + proprioception (930) = 994
        obs = np.concatenate([tokens, proprioception]).astype(np.float32)
        return obs.reshape(1, DECODER_DIM)

    def step_mock_physics(
        self,
        action: np.ndarray,
        velocity_limit: float = 10.0,
    ):
        """Update mock robot state based on action output.

        Simulates simple PD-like position tracking without a real physics engine.

        Args:
            action: (29,) joint target positions from decoder.
            velocity_limit: Maximum joint velocity (rad/s).
        """
        action = action.flatten()[:NUM_G1_JOINTS]

        # Compute desired velocity
        pos_error = action - self.joint_positions
        desired_vel = pos_error / self.dt

        # Velocity limiting
        desired_vel = np.clip(desired_vel, -velocity_limit, velocity_limit)

        # Update state
        self.joint_velocities = desired_vel.astype(np.float32)
        self.joint_positions = (
            self.joint_positions + self.joint_velocities * self.dt
        ).astype(np.float32)

        # Clamp joint positions to reasonable range
        self.joint_positions = np.clip(self.joint_positions, -3.14, 3.14)

        # Update last actions
        self.last_actions = action.astype(np.float32)

        self._frame_count += 1

    def get_joint_targets_history(self) -> np.ndarray:
        """Get the last_actions history for debugging. (10, 29)"""
        return np.array(list(self._last_actions_history._buffer))
