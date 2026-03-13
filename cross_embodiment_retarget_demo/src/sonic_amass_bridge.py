"""
SONIC ↔ AMASS Bridge (v2 — fixed observation layout).

Converts AMASS SMPL data into the flat observation vectors required
by the SONIC encoder (1762-dim) and decoder (994-dim) ONNX models.

Key insight from the official GR00T-WholeBodyControl documentation:

  The observation vector is formed by **concatenating observations in YAML
  order**, where each multi-frame observation packs ALL its frames
  contiguously (not interleaved per-frame).

Encoder input (1762-dim) — observations concatenated in config order:
  [   0:   4] encoder_mode_4                                 (4)
  [   4: 294] motion_joint_positions_10frame_step5           (290 = 29 × 10)
  [ 294: 584] motion_joint_velocities_10frame_step5          (290 = 29 × 10)
  [ 584: 594] motion_root_z_position_10frame_step5           (10 = 1 × 10)
  [ 594: 595] motion_root_z_position                         (1)
  [ 595: 601] motion_anchor_orientation                      (6)
  [ 601: 661] motion_anchor_orientation_10frame_step5        (60 = 6 × 10)
  [ 661: 781] motion_joint_positions_lowerbody_10frame_step5 (120 = 12 × 10)
  [ 781: 901] motion_joint_velocities_lowerbody_10frame_step5(120 = 12 × 10)
  [ 901: 910] vr_3point_local_target                         (9 = 3 × 3)
  [ 910: 922] vr_3point_local_orn_target                     (12 = 3 × 4)
  [ 922:1642] smpl_joints_10frame_step1                      (720 = 24 × 3 × 10)
  [1642:1702] smpl_anchor_orientation_10frame_step1          (60 = 6 × 10)
  [1702:1762] motion_joint_positions_wrists_10frame_step1    (60 = 6 × 10)
  Total: 4+290+290+10+1+6+60+120+120+9+12+720+60+60 = 1762 ✓

Decoder input (994-dim) — concatenated in config order:
  [  0:  64] token_state                                     (64)
  [ 64:  94] his_base_angular_velocity_10frame_step1         (30 = 3 × 10)
  [ 94: 384] his_body_joint_positions_10frame_step1          (290 = 29 × 10)
  [384: 674] his_body_joint_velocities_10frame_step1         (290 = 29 × 10)
  [674: 964] his_last_actions_10frame_step1                  (290 = 29 × 10)
  [964: 994] his_gravity_dir_10frame_step1                   (30 = 3 × 10)
  Total: 64+30+290+290+290+30 = 994 ✓

The action output is **normalized** (typically [-1, +1] range). In the
official C++ deploy it's mapped to joint targets via:
    target_dof_pos = action * action_scale + default_dof_pos

For G1, typical action_scale ≈ 0.25 rad and default_dof_pos is the
standing pose. We use the Unitree G1 defaults from IsaacLab.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np

from .amass_loader import AMASSMotion, rotation_matrix_to_6d

logger = logging.getLogger(__name__)

# ── Dimensions ────────────────────────────────────────────────

ENCODER_DIM = 1762
DECODER_DIM = 994
ENCODER_TOKEN_DIM = 64
NUM_G1_JOINTS = 29
NUM_LOWER_BODY_JOINTS = 12  # 6 per leg
NUM_SMPL_JOINTS = 24

# ── G1 Default Standing Pose (IsaacLab convention, radians) ──

G1_DEFAULT_DOF_POS = np.array([
    # Torso: yaw, pitch, roll
    0.0, 0.0, 0.0,
    # Left arm: shoulder pitch, roll, yaw, elbow, wrist
    0.0, 0.2, 0.0, -0.4, 0.0,
    # Right arm: shoulder pitch, roll, yaw, elbow, wrist
    0.0, -0.2, 0.0, -0.4, 0.0,
    # Left leg: hip yaw, roll, pitch, knee, ankle pitch, roll
    0.0, 0.0, -0.3, 0.6, -0.3, 0.0,
    # Right leg: hip yaw, roll, pitch, knee, ankle pitch, roll
    0.0, 0.0, -0.3, 0.6, -0.3, 0.0,
    # Hands: left grip, left wrist, right grip, right wrist
    0.0, 0.0, 0.0, 0.0,
], dtype=np.float32)

# Action scale: how much each unit of action corresponds to in radians
G1_ACTION_SCALE = 0.25  # rad per action unit

# G1 joint limits (conservative, in radians)
G1_JOINT_LIMITS_LOW = np.array([
    -2.6, -0.4, -0.4,        # torso
    -2.9, -0.3, -1.6, -1.6, -0.5,  # left arm
    -2.9, -1.7, -1.6, -1.6, -0.5,  # right arm
    -0.5, -0.3, -1.6, -0.1, -0.9, -0.3,  # left leg
    -0.5, -0.3, -1.6, -0.1, -0.9, -0.3,  # right leg
    -0.5, -0.5, -0.5, -0.5,  # hands
], dtype=np.float32)

G1_JOINT_LIMITS_HIGH = np.array([
    2.6, 0.4, 0.4,           # torso
    2.9, 1.7, 1.6, 0.0, 0.5,  # left arm
    2.9, 0.3, 1.6, 0.0, 0.5,  # right arm
    0.5, 0.3, 0.5, 2.4, 0.7, 0.3,  # left leg
    0.5, 0.3, 0.5, 2.4, 0.7, 0.3,  # right leg
    0.5, 0.5, 0.5, 0.5,  # hands
], dtype=np.float32)

# G1 lower body indices (IsaacLab order): left leg (13-18) + right leg (19-24)
G1_LOWER_BODY_INDICES = list(range(13, 25))

# ── SMPL → G1 Joint Retarget ─────────────────────────────────

# SMPL joint indices
SMPL_PELVIS = 0
SMPL_L_HIP, SMPL_R_HIP = 1, 2
SMPL_SPINE1 = 3
SMPL_L_KNEE, SMPL_R_KNEE = 4, 5
SMPL_SPINE2 = 6
SMPL_L_ANKLE, SMPL_R_ANKLE = 7, 8
SMPL_SPINE3 = 9
SMPL_NECK = 12
SMPL_L_SHOULDER, SMPL_R_SHOULDER = 16, 17
SMPL_L_ELBOW, SMPL_R_ELBOW = 18, 19
SMPL_L_WRIST, SMPL_R_WRIST = 20, 21


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two vectors (radians)."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return float(np.arccos(cos_angle))


def smpl_to_g1_joint_positions(smpl_joints: np.ndarray) -> np.ndarray:
    """Convert SMPL 24-joint positions to G1 29-joint angles (radians).

    This is an analytical approximation that extracts joint angles from
    the geometric relationships between SMPL keypoints.

    Args:
        smpl_joints: (24, 3) SMPL joint world positions.

    Returns:
        (29,) G1 joint angles in radians, centered around default pose.
    """
    g1 = G1_DEFAULT_DOF_POS.copy()

    pelvis = smpl_joints[SMPL_PELVIS]
    spine3 = smpl_joints[SMPL_SPINE3]

    # -- Torso (joints 0-2): lean from SMPL spine --
    spine_dir = spine3 - pelvis
    spine_len = max(np.linalg.norm(spine_dir), 1e-4)
    spine_unit = spine_dir / spine_len

    # Small torso adjustments relative to upright
    g1[0] = np.arctan2(spine_unit[0], max(spine_unit[2], 0.1)) * 0.3  # yaw
    g1[1] = np.arctan2(-spine_unit[1], max(spine_unit[2], 0.1)) * 0.3  # pitch
    g1[2] = 0.0  # roll

    # -- Arms --
    for side, (sh_idx, el_idx, wr_idx, base) in enumerate([
        (SMPL_L_SHOULDER, SMPL_L_ELBOW, SMPL_L_WRIST, 3),   # left arm starts at g1[3]
        (SMPL_R_SHOULDER, SMPL_R_ELBOW, SMPL_R_WRIST, 8),   # right arm starts at g1[8]
    ]):
        shoulder = smpl_joints[sh_idx]
        elbow = smpl_joints[el_idx]
        wrist = smpl_joints[wr_idx]

        upper_arm = elbow - shoulder
        forearm = wrist - elbow
        ua_len = max(np.linalg.norm(upper_arm), 1e-4)
        fa_len = max(np.linalg.norm(forearm), 1e-4)
        ua_unit = upper_arm / ua_len

        # Shoulder pitch: angle of upper arm from horizontal
        g1[base + 0] = np.arctan2(-ua_unit[2], np.sqrt(ua_unit[0]**2 + ua_unit[1]**2))

        # Shoulder roll: lateral spread
        sign = 1.0 if side == 0 else -1.0  # left vs right
        g1[base + 1] = sign * np.arctan2(abs(ua_unit[0]), max(abs(ua_unit[2]), 0.1)) * 0.5

        # Shoulder yaw
        g1[base + 2] = np.arctan2(ua_unit[1], ua_unit[0]) * 0.3

        # Elbow: flexion angle (negative = flexed)
        elbow_angle = _angle_between(upper_arm, forearm)
        g1[base + 3] = -(np.pi - elbow_angle) * 0.5  # damped

        # Wrist: leave at default
        g1[base + 4] = 0.0

    # -- Legs --
    for side, (hip_idx, knee_idx, ankle_idx, base) in enumerate([
        (SMPL_L_HIP, SMPL_L_KNEE, SMPL_L_ANKLE, 13),   # left leg
        (SMPL_R_HIP, SMPL_R_KNEE, SMPL_R_ANKLE, 19),   # right leg
    ]):
        hip = smpl_joints[hip_idx]
        knee = smpl_joints[knee_idx]
        ankle = smpl_joints[ankle_idx]

        thigh = knee - hip
        shin = ankle - knee
        th_len = max(np.linalg.norm(thigh), 1e-4)
        th_unit = thigh / th_len

        # Hip yaw
        g1[base + 0] = np.arctan2(th_unit[0], max(-th_unit[2], 0.1)) * 0.3

        # Hip roll
        sign = 1.0 if side == 0 else -1.0
        g1[base + 1] = sign * np.arctan2(abs(th_unit[0]), max(-th_unit[2], 0.1)) * 0.2

        # Hip pitch: forward/backward swing
        g1[base + 2] = np.arctan2(th_unit[1], -th_unit[2])

        # Knee: extension angle
        knee_angle = _angle_between(thigh, shin)
        g1[base + 3] = max(np.pi - knee_angle, 0.0)

        # Ankle pitch
        g1[base + 4] = np.arctan2(shin[1] / max(np.linalg.norm(shin), 1e-4),
                                   -shin[2] / max(np.linalg.norm(shin), 1e-4)) * 0.5

        # Ankle roll
        g1[base + 5] = 0.0

    # Clamp to joint limits
    g1 = np.clip(g1, G1_JOINT_LIMITS_LOW, G1_JOINT_LIMITS_HIGH)

    return g1


class RingBuffer:
    """Simple ring buffer for multi-frame history observations.

    The SONIC system packs multi-frame observations as:
      [frame_oldest, ..., frame_newest] each of dim `feature_dim`
    All frames of one observation type are contiguous (not interleaved).
    """

    def __init__(self, max_frames: int, feature_dim: int):
        self.max_frames = max_frames
        self.feature_dim = feature_dim
        self._buffer: deque[np.ndarray] = deque(maxlen=max_frames)
        for _ in range(max_frames):
            self._buffer.append(np.zeros(feature_dim, dtype=np.float32))

    def push(self, frame: np.ndarray):
        """Add newest frame."""
        self._buffer.append(frame.astype(np.float32).ravel()[:self.feature_dim])

    def get_contiguous(self, n_frames: int = 10, step: int = 1) -> np.ndarray:
        """Get n_frames sampled at step interval, packed contiguously.

        Returns shape (n_frames * feature_dim,) with oldest first.
        For step=5, samples every 5th tick backward from newest.
        """
        buf = list(self._buffer)
        n_buf = len(buf)
        indices = []
        for i in range(n_frames - 1, -1, -1):
            idx = n_buf - 1 - i * step
            indices.append(max(0, idx))
        return np.concatenate([buf[i] for i in indices])

    @property
    def latest(self) -> np.ndarray:
        return self._buffer[-1].copy()


class SonicAMASSBridge:
    """Bridge between AMASS SMPL data and SONIC ONNX models.

    Fixed in v2:
    - Correct decoder observation layout (each obs type contiguous, not interleaved)
    - Action interpreted as normalized (action * scale + default_dof_pos → target angle)
    - Proper velocity computation with smoothing
    - Joint angle clamping to physical limits
    """

    def __init__(self, dt: float = 0.02, action_scale: float = G1_ACTION_SCALE):
        self.dt = dt
        self.action_scale = action_scale
        self._frame_count = 0

        # Keep enough history for step5 × 10 frames = 50 ticks
        history_depth = 50

        # Encoder histories (motion reference, sampled at step5)
        self._motion_joint_pos_buf = RingBuffer(history_depth, NUM_G1_JOINTS)    # 29
        self._motion_joint_vel_buf = RingBuffer(history_depth, NUM_G1_JOINTS)    # 29
        self._motion_root_z_buf = RingBuffer(history_depth, 1)                    # 1
        self._motion_anchor_orient_buf = RingBuffer(history_depth, 6)             # 6
        self._motion_lower_pos_buf = RingBuffer(history_depth, NUM_LOWER_BODY_JOINTS)  # 12
        self._motion_lower_vel_buf = RingBuffer(history_depth, NUM_LOWER_BODY_JOINTS)  # 12
        self._smpl_joints_buf = RingBuffer(history_depth, NUM_SMPL_JOINTS * 3)    # 72
        self._smpl_anchor_buf = RingBuffer(history_depth, 6)                      # 6
        self._motion_wrist_pos_buf = RingBuffer(history_depth, 6)                 # 6 (wrist joints)

        # Decoder histories (robot state, sampled at step1)
        self._his_angular_vel = RingBuffer(10, 3)                                 # 3
        self._his_joint_pos = RingBuffer(10, NUM_G1_JOINTS)                       # 29
        self._his_joint_vel = RingBuffer(10, NUM_G1_JOINTS)                       # 29
        self._his_last_actions = RingBuffer(10, NUM_G1_JOINTS)                    # 29
        self._his_gravity = RingBuffer(10, 3)                                     # 3

        # Previous frame for velocity computation
        self._prev_g1_joints: Optional[np.ndarray] = None
        self._prev_lower_joints: Optional[np.ndarray] = None

        # Current robot state (simulated)
        self.joint_positions = G1_DEFAULT_DOF_POS.copy()
        self.joint_velocities = np.zeros(NUM_G1_JOINTS, dtype=np.float32)
        self.last_actions = np.zeros(NUM_G1_JOINTS, dtype=np.float32)
        self.base_angular_velocity = np.zeros(3, dtype=np.float32)
        self.gravity_direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    def reset(self):
        """Reset all state."""
        self.__init__(self.dt, self.action_scale)

    def pack_encoder_input(
        self,
        motion: AMASSMotion,
        frame_idx: int,
    ) -> np.ndarray:
        """Pack 1762-dim encoder input in the exact layout from observation_config.yaml.

        Uses G1 mode (mode_id=0) since AMASS SMPL data is converted to
        G1 joint angles via analytical retarget.
        """
        smpl_joints = motion.joint_positions[frame_idx]    # (24, 3)
        root_orient = motion.root_orientation[frame_idx]   # (3, 3)
        root_z = float(motion.root_translation[frame_idx, 2])

        # Convert SMPL → G1 joint positions (reference motion)
        g1_ref_joints = smpl_to_g1_joint_positions(smpl_joints)

        # Compute reference velocities via finite differencing
        if self._prev_g1_joints is not None:
            g1_ref_vel = (g1_ref_joints - self._prev_g1_joints) / self.dt
            # Clamp velocity to prevent explosion
            g1_ref_vel = np.clip(g1_ref_vel, -10.0, 10.0)
        else:
            g1_ref_vel = np.zeros(NUM_G1_JOINTS, dtype=np.float32)
        self._prev_g1_joints = g1_ref_joints.copy()

        # Lower body subset
        lower_pos = g1_ref_joints[G1_LOWER_BODY_INDICES]
        if self._prev_lower_joints is not None:
            lower_vel = (lower_pos - self._prev_lower_joints) / self.dt
            lower_vel = np.clip(lower_vel, -10.0, 10.0)
        else:
            lower_vel = np.zeros(NUM_LOWER_BODY_JOINTS, dtype=np.float32)
        self._prev_lower_joints = lower_pos.copy()

        # Anchor orientation (6D from rotation matrix)
        anchor_6d = rotation_matrix_to_6d(root_orient)

        # SMPL joints (24 × 3 = 72, local to pelvis)
        smpl_local = smpl_joints - smpl_joints[SMPL_PELVIS]  # pelvis-relative
        smpl_flat = smpl_local.flatten()  # 72

        # Wrist positions from G1 reference (6 values: 3 wrist-related joints per side)
        # From observation docs: "Wrist joints only (6 joints)"
        # This likely means left wrist region (5,6,7) + right wrist region (10,11,12) 
        # But the dim is 60/10 = 6 per frame, which matches 6 joints
        wrist_joints = np.array([
            g1_ref_joints[5], g1_ref_joints[6], g1_ref_joints[7],   # left: yaw, elbow, wrist
            g1_ref_joints[10], g1_ref_joints[11], g1_ref_joints[12], # right: yaw, elbow, wrist
        ], dtype=np.float32)

        # Update all history buffers
        self._motion_joint_pos_buf.push(g1_ref_joints)
        self._motion_joint_vel_buf.push(g1_ref_vel)
        self._motion_root_z_buf.push(np.array([root_z], dtype=np.float32))
        self._motion_anchor_orient_buf.push(anchor_6d)
        self._motion_lower_pos_buf.push(lower_pos)
        self._motion_lower_vel_buf.push(lower_vel)
        self._smpl_joints_buf.push(smpl_flat)
        self._smpl_anchor_buf.push(anchor_6d)
        self._motion_wrist_pos_buf.push(wrist_joints)

        # ── Build encoder observation vector ──
        obs = np.zeros(ENCODER_DIM, dtype=np.float32)
        offset = 0

        # [0:4] encoder_mode_4 — G1 mode: [1, 0, 0, 0]
        obs[offset] = 1.0  # mode_id=0
        offset += 4

        # [4:294] motion_joint_positions_10frame_step5 (290)
        obs[offset:offset + 290] = self._motion_joint_pos_buf.get_contiguous(10, step=5)
        offset += 290

        # [294:584] motion_joint_velocities_10frame_step5 (290)
        obs[offset:offset + 290] = self._motion_joint_vel_buf.get_contiguous(10, step=5)
        offset += 290

        # [584:594] motion_root_z_position_10frame_step5 (10)
        obs[offset:offset + 10] = self._motion_root_z_buf.get_contiguous(10, step=5)
        offset += 10

        # [594:595] motion_root_z_position (1)
        obs[offset] = root_z
        offset += 1

        # [595:601] motion_anchor_orientation (6) — current frame only
        obs[offset:offset + 6] = anchor_6d
        offset += 6

        # [601:661] motion_anchor_orientation_10frame_step5 (60)
        obs[offset:offset + 60] = self._motion_anchor_orient_buf.get_contiguous(10, step=5)
        offset += 60

        # [661:781] motion_joint_positions_lowerbody_10frame_step5 (120)
        obs[offset:offset + 120] = self._motion_lower_pos_buf.get_contiguous(10, step=5)
        offset += 120

        # [781:901] motion_joint_velocities_lowerbody_10frame_step5 (120)
        obs[offset:offset + 120] = self._motion_lower_vel_buf.get_contiguous(10, step=5)
        offset += 120

        # [901:910] vr_3point_local_target (9) — zeros for non-VR mode
        offset += 9

        # [910:922] vr_3point_local_orn_target (12) — zeros for non-VR mode
        offset += 12

        # [922:1642] smpl_joints_10frame_step1 (720) — zeros for G1 mode
        # (SMPL regions are zero-filled in G1 mode per encoder_modes config)
        offset += 720

        # [1642:1702] smpl_anchor_orientation_10frame_step1 (60) — zeros for G1 mode
        offset += 60

        # [1702:1762] motion_joint_positions_wrists_10frame_step1 (60) — zeros for G1 mode
        offset += 60

        assert offset == ENCODER_DIM, f"Encoder offset mismatch: {offset} != {ENCODER_DIM}"

        return obs.reshape(1, ENCODER_DIM)

    def pack_decoder_input(self, encoded_tokens: np.ndarray) -> np.ndarray:
        """Pack 994-dim decoder input.

        Layout (from observation_config.yaml, concatenated in order):
          [  0: 64] token_state
          [ 64: 94] his_base_angular_velocity_10frame_step1  (30)
          [ 94:384] his_body_joint_positions_10frame_step1   (290)
          [384:674] his_body_joint_velocities_10frame_step1  (290)
          [674:964] his_last_actions_10frame_step1           (290)
          [964:994] his_gravity_dir_10frame_step1            (30)
        """
        tokens = encoded_tokens.ravel()[:ENCODER_TOKEN_DIM]

        # Push current robot state into history buffers
        self._his_angular_vel.push(self.base_angular_velocity)
        self._his_joint_pos.push(self.joint_positions)
        self._his_joint_vel.push(self.joint_velocities)
        self._his_last_actions.push(self.last_actions)
        self._his_gravity.push(self.gravity_direction)

        obs = np.zeros(DECODER_DIM, dtype=np.float32)
        offset = 0

        # [0:64] token_state
        obs[offset:offset + 64] = tokens
        offset += 64

        # [64:94] his_base_angular_velocity_10frame_step1 (30)
        obs[offset:offset + 30] = self._his_angular_vel.get_contiguous(10, step=1)
        offset += 30

        # [94:384] his_body_joint_positions_10frame_step1 (290)
        obs[offset:offset + 290] = self._his_joint_pos.get_contiguous(10, step=1)
        offset += 290

        # [384:674] his_body_joint_velocities_10frame_step1 (290)
        obs[offset:offset + 290] = self._his_joint_vel.get_contiguous(10, step=1)
        offset += 290

        # [674:964] his_last_actions_10frame_step1 (290)
        obs[offset:offset + 290] = self._his_last_actions.get_contiguous(10, step=1)
        offset += 290

        # [964:994] his_gravity_dir_10frame_step1 (30)
        obs[offset:offset + 30] = self._his_gravity.get_contiguous(10, step=1)
        offset += 30

        assert offset == DECODER_DIM, f"Decoder offset mismatch: {offset} != {DECODER_DIM}"

        return obs.reshape(1, DECODER_DIM)

    def action_to_joint_targets(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized action to target joint positions.

        target = action * action_scale + default_dof_pos

        Args:
            action: (29,) raw decoder output.

        Returns:
            (29,) target joint angles in radians, clamped to limits.
        """
        action = action.ravel()[:NUM_G1_JOINTS]
        targets = action * self.action_scale + G1_DEFAULT_DOF_POS
        targets = np.clip(targets, G1_JOINT_LIMITS_LOW, G1_JOINT_LIMITS_HIGH)
        return targets

    def step_mock_physics(
        self,
        action: np.ndarray,
        velocity_limit: float = 5.0,
    ):
        """Update simulated robot state from action.

        Uses PD-like tracking of the target joint positions derived from
        the normalized action output.

        Args:
            action: (29,) raw normalized action from decoder.
            velocity_limit: Max joint velocity (rad/s).
        """
        raw_action = action.ravel()[:NUM_G1_JOINTS]

        # Convert action to target position
        target_pos = self.action_to_joint_targets(raw_action)

        # PD control
        pos_error = target_pos - self.joint_positions
        desired_vel = pos_error / self.dt
        desired_vel = np.clip(desired_vel, -velocity_limit, velocity_limit)

        # Update state
        self.joint_velocities = desired_vel.astype(np.float32)
        self.joint_positions = (
            self.joint_positions + self.joint_velocities * self.dt
        ).astype(np.float32)
        self.joint_positions = np.clip(
            self.joint_positions, G1_JOINT_LIMITS_LOW, G1_JOINT_LIMITS_HIGH
        )

        # Store raw action (this is what goes into his_last_actions)
        self.last_actions = raw_action.astype(np.float32)

        self._frame_count += 1
