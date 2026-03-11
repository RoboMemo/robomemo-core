"""
SONIC Latent Retarget Engine.

Implements the SONIC pipeline:
  BodyPose → Hybrid Encoder → FSQ Quantizer → Robot Control Decoder → Joint targets

Supports two backends:
  1. ONNX Runtime (primary) — loads encoder/decoder ONNX models
  2. Mock (testing) — simple IK-like mapping for pipeline validation

When the real SONIC ONNX models are available, swap the MockSonicBackend
with OnnxSonicBackend.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from .body_types import BodyPose, Pose7D

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore


class SonicBackend(ABC):
    """Abstract interface for SONIC inference."""

    @abstractmethod
    def init(self, cfg: dict) -> None:
        ...

    @abstractmethod
    def infer(self, body_pose: BodyPose, proprioception: np.ndarray) -> np.ndarray:
        """Run retarget inference.

        Args:
            body_pose: Current full-body pose from input source.
            proprioception: Current robot state [joint_pos(29), joint_vel(29), gravity(3)] = (61,)

        Returns:
            Joint target positions (num_joints,) float32
        """
        ...

    @abstractmethod
    def num_joints(self) -> int:
        ...


# ── Mock backend for testing ──────────────────────────────────

class MockSonicBackend(SonicBackend):
    """Analytical retarget: maps body keypoints to joint angles
    via simplified inverse kinematics. Good enough to validate
    the full pipeline without real SONIC weights."""

    def __init__(self):
        self._num_joints = 29
        self._default_pose: Optional[np.ndarray] = None

    def init(self, cfg: dict) -> None:
        robot_cfg = cfg["robot"]["unitree_g1"]
        self._num_joints = robot_cfg.get("num_joints", 29)
        self._default_pose = np.array(
            robot_cfg.get("default_pose", [0.0] * self._num_joints),
            dtype=np.float32,
        )
        logger.info(f"MockSonicBackend initialised ({self._num_joints} joints)")

    def num_joints(self) -> int:
        return self._num_joints

    def infer(self, body_pose: BodyPose, proprioception: np.ndarray) -> np.ndarray:
        """Simple analytical mapping from body keypoints to joint targets.

        Joint layout (Unitree G1, 29 DOF):
          [0:3]   torso (waist yaw, pitch, roll)
          [3:8]   left arm (shoulder pitch/roll/yaw, elbow, wrist)
          [8:13]  right arm (shoulder pitch/roll/yaw, elbow, wrist)
          [13:19] left leg (hip yaw/roll/pitch, knee, ankle pitch/roll)
          [19:25] right leg (hip yaw/roll/pitch, knee, ankle pitch/roll)
          [25:29] hands (left grip, left wrist, right grip, right wrist)
        """
        joints = self._default_pose.copy()

        # ── Torso from head/waist ──
        head_pos = body_pose.head.position
        # Torso yaw from head x offset
        joints[0] = np.clip(head_pos[0] * 0.5, -0.5, 0.5)
        # Torso pitch from head forward lean
        joints[1] = np.clip((head_pos[1] - 0.0) * 0.3, -0.3, 0.3)

        if body_pose.waist is not None:
            waist_pos = body_pose.waist.position
            joints[2] = np.clip(waist_pos[0] * 0.3, -0.3, 0.3)  # roll

        # ── Left arm ──
        lh = body_pose.left_hand.position
        joints[3] = np.clip(-(lh[2] - 1.0) * 1.5, -1.5, 1.5)   # shoulder pitch
        joints[4] = np.clip((lh[0] + 0.3) * 1.0, -1.0, 1.0)     # shoulder roll
        joints[5] = np.clip(lh[1] * 0.5, -0.8, 0.8)              # shoulder yaw
        # Elbow: distance from shoulder to hand
        arm_len = np.linalg.norm(lh - np.array([-0.2, 0.0, 1.3]))
        joints[6] = np.clip(-(arm_len - 0.4) * 3.0, -2.0, 0.0)  # elbow flexion

        # ── Right arm ──
        rh = body_pose.right_hand.position
        joints[8] = np.clip(-(rh[2] - 1.0) * 1.5, -1.5, 1.5)
        joints[9] = np.clip((rh[0] - 0.3) * 1.0, -1.0, 1.0)
        joints[10] = np.clip(rh[1] * 0.5, -0.8, 0.8)
        arm_len_r = np.linalg.norm(rh - np.array([0.2, 0.0, 1.3]))
        joints[11] = np.clip(-(arm_len_r - 0.4) * 3.0, -2.0, 0.0)

        # ── Legs (if tracked) ──
        if body_pose.has_leg_tracking:
            la = body_pose.left_ankle.position
            ra = body_pose.right_ankle.position

            # Left leg
            joints[13] = np.clip(la[0] * 0.5, -0.5, 0.5)          # hip yaw
            joints[14] = np.clip(-(la[0] + 0.1) * 0.8, -0.5, 0.5) # hip roll
            joints[15] = np.clip(la[1] * 1.0, -1.0, 1.0)          # hip pitch
            joints[16] = np.clip(la[2] * 4.0, 0.0, 2.0)           # knee
            joints[17] = np.clip(-la[1] * 0.8, -0.8, 0.8)         # ankle pitch

            # Right leg
            joints[19] = np.clip(ra[0] * 0.5, -0.5, 0.5)
            joints[20] = np.clip(-(ra[0] - 0.1) * 0.8, -0.5, 0.5)
            joints[21] = np.clip(ra[1] * 1.0, -1.0, 1.0)
            joints[22] = np.clip(ra[2] * 4.0, 0.0, 2.0)
            joints[23] = np.clip(-ra[1] * 0.8, -0.8, 0.8)

        return joints


# ── ONNX backend (for real SONIC models) ──────────────────────

class OnnxSonicBackend(SonicBackend):
    """ONNX Runtime backend for real SONIC encoder + decoder models.

    Expected model files (from HuggingFace nvidia/GEAR-SONIC):
      - model_encoder.onnx  → Hybrid Encoder + FSQ Quantizer
      - model_decoder.onnx  → Robot Control Decoder
      - observation_config.yaml → Input tensor layout
    """

    def __init__(self):
        self._encoder_session: Optional[object] = None
        self._decoder_session: Optional[object] = None
        self._num_joints = 29

    def init(self, cfg: dict) -> None:
        if ort is None:
            raise ImportError(
                "onnxruntime-gpu is required for ONNX backend. "
                "Install: pip install onnxruntime-gpu"
            )

        sonic_cfg = cfg["sonic"]
        model_dir = Path(sonic_cfg["model_dir"]).expanduser()
        encoder_path = model_dir / "model_encoder.onnx"
        decoder_path = model_dir / "model_decoder.onnx"

        if not encoder_path.exists() or not decoder_path.exists():
            raise FileNotFoundError(
                f"SONIC ONNX models not found in {model_dir}. "
                "Run: python setup/download_checkpoints.py"
            )

        device = sonic_cfg.get("inference_device", "cuda")
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._encoder_session = ort.InferenceSession(
            str(encoder_path), sess_options=sess_opts, providers=providers
        )
        self._decoder_session = ort.InferenceSession(
            str(decoder_path), sess_options=sess_opts, providers=providers
        )

        robot_cfg = cfg["robot"]["unitree_g1"]
        self._num_joints = robot_cfg.get("num_joints", 29)
        logger.info(
            f"OnnxSonicBackend initialised: encoder={encoder_path.name}, "
            f"decoder={decoder_path.name}, device={device}"
        )

    def num_joints(self) -> int:
        return self._num_joints

    def infer(self, body_pose: BodyPose, proprioception: np.ndarray) -> np.ndarray:
        # TODO: Verify exact input tensor shapes from observation_config.yaml
        # The shapes below are best estimates from the SONIC paper/code.

        # Pack body pose into encoder input
        upper_body = np.concatenate([
            body_pose.head.to_array(),       # 7
            body_pose.left_hand.to_array(),  # 7
            body_pose.right_hand.to_array(), # 7
        ]).reshape(1, -1).astype(np.float32)  # (1, 21)

        lower_body = np.concatenate([
            body_pose.left_ankle.to_array(),  # 7
            body_pose.right_ankle.to_array(), # 7
        ]).reshape(1, -1).astype(np.float32)  # (1, 14)

        prop = proprioception.reshape(1, -1).astype(np.float32)  # (1, 61)

        # Encoder: body keypoints + proprioception → latent tokens
        enc_inputs = {
            "upper_body": upper_body,
            "lower_body": lower_body,
            "proprioception": prop,
        }
        # TODO: Actual input names from model — adjust after loading real model
        try:
            enc_names = [i.name for i in self._encoder_session.get_inputs()]
            enc_feed = {}
            for name in enc_names:
                if "upper" in name.lower():
                    enc_feed[name] = upper_body
                elif "lower" in name.lower():
                    enc_feed[name] = lower_body
                elif "prop" in name.lower():
                    enc_feed[name] = prop
                else:
                    # Generic fallback: full body
                    full = body_pose.to_flat_array().reshape(1, -1)
                    enc_feed[name] = full
            tokens = self._encoder_session.run(None, enc_feed)[0]
        except Exception as e:
            logger.error(f"Encoder inference failed: {e}")
            return np.zeros(self._num_joints, dtype=np.float32)

        # Decoder: tokens + proprioception → joint targets
        try:
            dec_names = [i.name for i in self._decoder_session.get_inputs()]
            dec_feed = {}
            for name in dec_names:
                if "token" in name.lower() or "latent" in name.lower():
                    dec_feed[name] = tokens
                elif "prop" in name.lower():
                    dec_feed[name] = prop
                else:
                    dec_feed[name] = tokens
            joint_targets = self._decoder_session.run(None, dec_feed)[0]
        except Exception as e:
            logger.error(f"Decoder inference failed: {e}")
            return np.zeros(self._num_joints, dtype=np.float32)

        return joint_targets.flatten()[:self._num_joints].astype(np.float32)


# ── Main retarget class ───────────────────────────────────────

class SonicRetarget:
    """High-level SONIC retarget wrapper.

    Manages backend selection, proprioception state, safety limits,
    and joint velocity smoothing.
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg
        sonic_cfg = cfg["sonic"]
        robot_cfg = cfg["robot"]["unitree_g1"]
        safety_cfg = cfg["safety"]

        # Select backend
        model_dir = Path(sonic_cfg["model_dir"]).expanduser()
        has_onnx_models = (
            (model_dir / "model_encoder.onnx").exists()
            and (model_dir / "model_decoder.onnx").exists()
        )

        if has_onnx_models and ort is not None:
            logger.info("Using ONNX SONIC backend (real models found)")
            self._backend: SonicBackend = OnnxSonicBackend()
        else:
            logger.info("Using Mock SONIC backend (analytical IK)")
            self._backend = MockSonicBackend()

        self._backend.init(cfg)

        self._num_joints = self._backend.num_joints()
        self._kp = robot_cfg.get("kp", 100.0)
        self._kd = robot_cfg.get("kd", 2.0)
        default = robot_cfg.get("default_pose", [0.0] * self._num_joints)
        self._default_pose = np.array(default, dtype=np.float32)
        self._vel_scale = safety_cfg.get("max_joint_velocity_scale", 0.8)

        # State
        self._joint_pos = self._default_pose.copy()
        self._joint_vel = np.zeros(self._num_joints, dtype=np.float32)
        self._gravity = np.array([0, 0, -9.81], dtype=np.float32)
        self._prev_targets = self._default_pose.copy()
        self._last_infer_time = 0.0

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def default_pose(self) -> np.ndarray:
        return self._default_pose.copy()

    def update_proprioception(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        gravity: Optional[np.ndarray] = None,
    ):
        """Update robot state for next inference call."""
        self._joint_pos = joint_pos.astype(np.float32)
        self._joint_vel = joint_vel.astype(np.float32)
        if gravity is not None:
            self._gravity = gravity.astype(np.float32)

    def infer(self, body_pose: Optional[BodyPose]) -> np.ndarray:
        """Run retarget inference.

        Returns smoothed joint target positions (num_joints,).
        If body_pose is None (disconnected), returns default standing pose.
        """
        t0 = time.time()

        if body_pose is None:
            # Safety fallback: smoothly return to default pose
            alpha = 0.1  # Slow blend to standing
            targets = self._prev_targets * (1 - alpha) + self._default_pose * alpha
            self._prev_targets = targets
            return targets

        prop = np.concatenate([self._joint_pos, self._joint_vel, self._gravity])
        raw_targets = self._backend.infer(body_pose, prop)

        # Velocity limiting
        dt = max(t0 - self._last_infer_time, 0.001) if self._last_infer_time > 0 else 0.02
        max_delta = self._vel_scale * 10.0 * dt  # ~10 rad/s max
        delta = np.clip(raw_targets - self._prev_targets, -max_delta, max_delta)
        targets = self._prev_targets + delta

        # Joint limit clamping
        pos_min = self._cfg["robot"]["unitree_g1"]["joint_limits"]["position_min"]
        pos_max = self._cfg["robot"]["unitree_g1"]["joint_limits"]["position_max"]
        targets = np.clip(targets, pos_min, pos_max)

        self._prev_targets = targets
        self._last_infer_time = t0

        latency_ms = (time.time() - t0) * 1000
        if latency_ms > 20:
            logger.warning(f"SONIC inference latency: {latency_ms:.1f}ms (target <20ms)")

        return targets
