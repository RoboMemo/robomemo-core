"""
Unit tests for the retarget pipeline.

Tests:
  1. BodyPose creation & coordinate conversion
  2. Mock motion generators produce valid data
  3. SonicRetarget (mock backend) produces valid joint targets
  4. Joint targets stay within limits
  5. Safety fallback on None input
  6. Full pipeline integration (mock → retarget → mock physics)
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.body_types import BodyPose, Pose7D, InputSource, y_up_to_z_up, convert_body_pose_to_z_up
from src.mock_motion import MockMotionSource, MOTION_GENERATORS
from src.sonic_retarget import SonicRetarget, MockSonicBackend
from src.isaac_env import MockPhysicsEnv


def _make_test_config() -> dict:
    """Minimal config for testing."""
    return {
        "input": {
            "source": "mock",
            "mock": {"motion_type": "walk", "duration_sec": 10.0, "loop": True},
            "webcam": {"device_id": 0, "resolution": [640, 480], "fps": 30},
            "xreal": {"beam_pro_ip": "127.0.0.1", "beam_pro_port": 8765, "enable_hand_tracking": True},
            "pico": {
                "zmq_host": "0.0.0.0", "zmq_port": 5555, "zmq_topic": "",
                "enable_leg_trackers": True, "enable_waist_tracker": True,
                "tracker_roles": {"waist": "waist", "left_ankle": "left_ankle", "right_ankle": "right_ankle"},
            },
        },
        "sonic": {
            "model_dir": "/tmp/sonic_models_nonexistent",
            "onnx_policy_path": "/tmp/sonic_models_nonexistent/policy.onnx",
            "use_groot_bindings": False,
            "groot_repo_path": "~/GR00T-WholeBodyControl",
            "inference_device": "cpu",
            "target_fps": 50,
            "max_inference_latency_ms": 20,
            "hybrid_encoder": {"upper_body_keypoints": ["head", "left_hand", "right_hand"], "lower_body_future_frames": 10},
            "fsq_quantizer": {"codebook_size": 1024, "latent_dim": 256},
            "control_decoder": {"output_dim": 29},
        },
        "robot": {
            "type": "unitree_g1",
            "unitree_g1": {
                "num_joints": 29,
                "kp": 100.0, "kd": 2.0,
                "joint_limits": {
                    "position_min": -3.14, "position_max": 3.14,
                    "velocity_max": 10.0, "torque_max": 200.0,
                },
                "default_pose": [0.0] * 29,
            },
        },
        "simulation": {
            "backend": "mock_physics",
            "mock_physics": {"dt": 0.02, "enable_visualization": False, "gravity": [0, 0, -9.81]},
            "isaac_lab": {"headless": True, "gpu_physics": False, "sim_dt": 0.02, "render_interval": 2},
        },
        "safety": {
            "disconnect_timeout_sec": 2.0,
            "max_joint_velocity_scale": 0.8,
            "estop_key": "q",
        },
        "logging": {"level": "WARNING", "log_to_file": False, "print_fps_interval_sec": 5.0},
    }


class TestBodyTypes(unittest.TestCase):
    def test_pose7d_identity(self):
        p = Pose7D.identity()
        np.testing.assert_array_almost_equal(p.position, [0, 0, 0])
        np.testing.assert_array_almost_equal(p.quaternion, [1, 0, 0, 0])

    def test_pose7d_roundtrip(self):
        arr = np.array([1, 2, 3, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        p = Pose7D.from_array(arr)
        np.testing.assert_array_almost_equal(p.to_array(), arr)

    def test_body_pose_flat_array(self):
        bp = BodyPose(
            head=Pose7D.identity(), left_hand=Pose7D.identity(),
            right_hand=Pose7D.identity(), left_ankle=Pose7D.identity(),
            right_ankle=Pose7D.identity(),
        )
        flat = bp.to_flat_array()
        self.assertEqual(flat.shape, (35,))

    def test_body_pose_with_waist(self):
        bp = BodyPose(
            head=Pose7D.identity(), left_hand=Pose7D.identity(),
            right_hand=Pose7D.identity(), left_ankle=Pose7D.identity(),
            right_ankle=Pose7D.identity(), waist=Pose7D.identity(),
        )
        flat = bp.to_flat_array()
        self.assertEqual(flat.shape, (42,))

    def test_y_up_to_z_up(self):
        # Point at (0, 1, 0) in Y-up = (0, 0, 1) in Z-up
        p = Pose7D(
            position=np.array([0, 1, 0], dtype=np.float32),
            quaternion=np.array([1, 0, 0, 0], dtype=np.float32),
        )
        p2 = y_up_to_z_up(p)
        self.assertAlmostEqual(p2.position[2], 1.0, places=3)
        self.assertAlmostEqual(p2.position[1], 0.0, places=3)


class TestMockMotion(unittest.TestCase):
    def test_all_generators_run(self):
        for name, gen in MOTION_GENERATORS.items():
            bp = gen(0.0)
            self.assertIsInstance(bp, BodyPose, f"Generator '{name}' failed")
            self.assertEqual(bp.source, InputSource.MOCK)

    def test_walk_changes_over_time(self):
        bp0 = MOTION_GENERATORS["walk"](0.0)
        bp1 = MOTION_GENERATORS["walk"](0.5)
        # Head should bob or move forward
        self.assertFalse(
            np.allclose(bp0.head.position, bp1.head.position),
            "Walk should produce different poses at different times"
        )

    def test_mock_source(self):
        cfg = _make_test_config()
        src = MockMotionSource(cfg)
        src.start()
        time.sleep(0.05)
        pose = src.latest_pose
        self.assertIsNotNone(pose)
        src.stop()


class TestSonicRetarget(unittest.TestCase):
    def setUp(self):
        self.cfg = _make_test_config()
        self.retarget = SonicRetarget(self.cfg)

    def test_infer_returns_correct_shape(self):
        bp = MOTION_GENERATORS["stand"](0.0)
        joints = self.retarget.infer(bp)
        self.assertEqual(joints.shape, (29,))

    def test_joints_within_limits(self):
        for t in [0.0, 0.5, 1.0, 2.0, 5.0]:
            bp = MOTION_GENERATORS["walk"](t)
            joints = self.retarget.infer(bp)
            self.assertTrue(
                np.all(joints >= -3.14) and np.all(joints <= 3.14),
                f"Joints out of range at t={t}: min={joints.min():.3f}, max={joints.max():.3f}"
            )

    def test_none_input_returns_to_default(self):
        # First set a non-default pose
        bp = MOTION_GENERATORS["wave"](1.0)
        self.retarget.infer(bp)
        # Then feed None (disconnect)
        for _ in range(100):
            joints = self.retarget.infer(None)
        # Should converge toward default (all zeros)
        np.testing.assert_array_less(np.abs(joints), 0.5)

    def test_velocity_limiting(self):
        """Sudden large input change should be smoothed."""
        bp1 = MOTION_GENERATORS["stand"](0.0)
        j1 = self.retarget.infer(bp1)

        bp2 = MOTION_GENERATORS["squat"](1.0)
        j2 = self.retarget.infer(bp2)

        # The change should be limited (not jump instantly)
        delta = np.abs(j2 - j1)
        max_allowed = 0.8 * 10.0 * 0.02 + 0.01  # vel_scale * max_vel * dt + tolerance
        # At least most joints should be within limit
        within_limit = np.sum(delta <= max_allowed * 2)  # generous tolerance
        self.assertGreater(within_limit, 20, "Most joints should be velocity-limited")


class TestMockPhysics(unittest.TestCase):
    def setUp(self):
        self.cfg = _make_test_config()
        self.env = MockPhysicsEnv()
        self.env.init(self.cfg)

    def test_step_returns_valid_state(self):
        targets = np.zeros(29, dtype=np.float32)
        self.env.step(targets)
        pos = self.env.get_joint_pos()
        vel = self.env.get_joint_vel()
        self.assertEqual(pos.shape, (29,))
        self.assertEqual(vel.shape, (29,))

    def test_pd_control_converges(self):
        target = np.ones(29, dtype=np.float32) * 0.5
        for _ in range(500):
            self.env.step(target)
        pos = self.env.get_joint_pos()
        np.testing.assert_array_less(np.abs(pos - target), 0.05)

    def tearDown(self):
        self.env.close()


class TestIntegration(unittest.TestCase):
    """End-to-end pipeline: mock input → retarget → mock physics."""

    def test_full_pipeline_100_steps(self):
        cfg = _make_test_config()
        retarget = SonicRetarget(cfg)
        env = MockPhysicsEnv()
        env.init(cfg)
        gen = MOTION_GENERATORS["walk"]

        t = 0.0
        dt = 0.02
        for step in range(100):
            bp = gen(t)
            joints = retarget.infer(bp)
            env.step(joints)

            retarget.update_proprioception(
                env.get_joint_pos(),
                env.get_joint_vel(),
                env.get_gravity_vector(),
            )

            pos = env.get_joint_pos()
            self.assertTrue(np.all(np.isfinite(pos)), f"Non-finite joints at step {step}")
            t += dt

        env.close()

    def test_pipeline_with_disconnect_recovery(self):
        """Simulate input disconnect and verify safe recovery."""
        cfg = _make_test_config()
        retarget = SonicRetarget(cfg)
        env = MockPhysicsEnv()
        env.init(cfg)
        gen = MOTION_GENERATORS["walk"]

        # Normal operation
        for i in range(50):
            bp = gen(i * 0.02)
            joints = retarget.infer(bp)
            env.step(joints)
            retarget.update_proprioception(env.get_joint_pos(), env.get_joint_vel())

        # Disconnect (None input)
        for _ in range(100):
            joints = retarget.infer(None)
            env.step(joints)
            retarget.update_proprioception(env.get_joint_pos(), env.get_joint_vel())

        # Should have returned close to default
        final_pos = env.get_joint_pos()
        self.assertTrue(np.all(np.abs(final_pos) < 1.0), "Should converge toward default")
        env.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
