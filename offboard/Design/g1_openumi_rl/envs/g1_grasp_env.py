"""
G1 + OpenUMI gripper bimanual grasping environment for MuJoCo RL.

Joint index layout (matches g1_29dof_dex1.py + 4 gripper DOF):
  0- 5  : left  legs  (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
  6-11  : right legs
 12-14  : waist (yaw, roll, pitch)
 15-21  : left  arm   (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
 22-28  : right arm
 29-30  : left  gripper (right_jaw [0→0.044 m], left_jaw [-0.044→0 m])
 31-32  : right gripper

Observation space (flat vector, dim=84):
  [0:33]  joint positions  (29 arm/leg/waist + 4 gripper)
  [33:66] joint velocities (same ordering)
  [66:69] left  EE position  (world frame)
  [69:72] right EE position  (world frame)
  [72:75] target object position
  [75:78] pelvis linear velocity (from IMU site)
  [78:81] pelvis orientation quaternion (w,x,y,z → cos/sin/…, simplified to 3 Euler here)
  [81:84] pelvis angular velocity

Action space (dim=33 → normalised torques):
  Arm+waist joints only (no legs by default); gripper open/close ∈ [-1,1] → scaled.
  See ARM_ACT_IDX and GRIPPER_ACT_IDX below.

Reward:
  r = w_grasp * grasp_success
    + w_dist  * (-dist_ee_to_obj)
    + w_energy* (-||tau||^2)
    + w_alive * 1.0
    - w_fall  * fall_penalty
"""

from __future__ import annotations

import os
import numpy as np
import mujoco

# ── Joint index constants ──────────────────────────────────────────────────
# These correspond to the *actuator* (ctrl) ordering in g1_with_openumi.xml.
LEFT_LEG_IDX   = list(range(0, 6))
RIGHT_LEG_IDX  = list(range(6, 12))
WAIST_IDX      = list(range(12, 15))
LEFT_ARM_IDX   = list(range(15, 22))   # shoulder_pitch … wrist_yaw
RIGHT_ARM_IDX  = list(range(22, 29))
L_GRIPPER_IDX  = [29, 30]              # right_jaw, left_jaw
R_GRIPPER_IDX  = [31, 32]

# Actuated DOFs for policy output (waist + both arms + both grippers)
ARM_ACT_IDX     = WAIST_IDX + LEFT_ARM_IDX + RIGHT_ARM_IDX   # 17 joints
GRIPPER_ACT_IDX = L_GRIPPER_IDX + R_GRIPPER_IDX              # 4  joints
POLICY_ACT_IDX  = ARM_ACT_IDX + GRIPPER_ACT_IDX              # 21 joints total

N_CTRL = 33   # total actuators in xml (12 leg + 3 waist + 7*2 arm + 4 gripper)

# Torque limits for normalisation (from xml ctrlrange)
CTRL_LIMITS = np.array([
    88, 88, 88, 139, 50, 50,   # left  leg
    88, 88, 88, 139, 50, 50,   # right leg
    88, 50, 50,                 # waist
    25, 25, 25, 25, 25, 5, 5,  # left  arm
    25, 25, 25, 25, 25, 5, 5,  # right arm
    100, 100,                   # left  gripper
    100, 100,                   # right gripper
], dtype=np.float64)


class G1GraspEnv:
    """
    Minimal MuJoCo environment for G1 + OpenUMI bimanual grasping.

    Parameters
    ----------
    xml_path : str
        Path to g1_with_openumi.xml (absolute or relative to cwd).
    sim_dt   : float
        MuJoCo physics timestep (seconds).
    control_dt : float
        Policy control timestep (seconds); number of physics steps per action
        = round(control_dt / sim_dt).
    max_episode_steps : int
        Episode length in *control* steps.
    reward_weights : dict
        Override default reward weights.
    """

    OBS_DIM = 84
    ACT_DIM = len(POLICY_ACT_IDX)  # 21

    DEFAULT_REWARD_WEIGHTS = dict(
        grasp   = 10.0,
        dist    = 1.0,
        energy  = 1e-4,
        alive   = 0.05,
        fall    = 5.0,
    )

    def __init__(
        self,
        xml_path: str = "models/g1_with_openumi.xml",
        sim_dt: float = 0.002,
        control_dt: float = 0.02,
        max_episode_steps: int = 500,
        reward_weights: dict | None = None,
    ):
        self.xml_path = os.path.abspath(xml_path)
        self.sim_dt   = sim_dt
        self.control_dt = control_dt
        self.n_substeps = max(1, round(control_dt / sim_dt))
        self.max_episode_steps = max_episode_steps
        self.rw = {**self.DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}

        # Load model
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.model.opt.timestep = sim_dt
        self.data  = mujoco.MjData(self.model)

        # Sensor / site IDs
        self._l_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "l_gripper_ee")
        self._r_ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "r_gripper_ee")
        self._imu_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "imu")

        # Target object (a free body placed in the scene at reset)
        # We add it programmatically as a simple sphere geom attached to a mocap body.
        # For a richer task, add a graspable object body to the XML instead.
        self._target_pos  = np.zeros(3)
        self._step_count  = 0

        # Observation / action spaces (numpy arrays for gym-like interface)
        self.observation_space_shape = (self.OBS_DIM,)
        self.action_space_shape      = (self.ACT_DIM,)

    # ── Reset ─────────────────────────────────────────────────────────────
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset simulation and return initial observation."""
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)

        # Spawn robot at a comfortable standing pose (init from g1_29dof_dex1.py)
        self._set_initial_pose()

        # Randomise target object position in front of robot
        self._target_pos = np.array([
            np.random.uniform(0.3, 0.6),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(0.7, 1.0),   # height roughly at waist
        ])

        self._step_count = 0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _set_initial_pose(self):
        """Set robot to a standing pose with arms at rest."""
        # Floating-base joint: qpos[0:7] = [x, y, z, qw, qx, qy, qz]
        self.data.qpos[0:3]  = [0.0, 0.0, 0.793]   # z-height
        self.data.qpos[3:7]  = [1.0, 0.0, 0.0, 0.0]  # upright quaternion

        # Named joint positions (subset matching init_state in g1_29dof_dex1.py)
        def _set_joint(name: str, val: float):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self.data.qpos[self.model.jnt_qposadr[jid]] = val

        # Legs
        for side in ("left", "right"):
            _set_joint(f"{side}_hip_pitch_joint",  -0.20)
            _set_joint(f"{side}_knee_joint",         0.42)
            _set_joint(f"{side}_ankle_pitch_joint", -0.23)
            _set_joint(f"{side}_elbow_joint",        0.87)

        _set_joint("left_shoulder_roll_joint",   0.18)
        _set_joint("left_shoulder_pitch_joint",  0.35)
        _set_joint("right_shoulder_roll_joint",  -0.18)
        _set_joint("right_shoulder_pitch_joint",  0.35)

        # Grippers: fully open
        for name in (
            "l_gripper_right_jaw_joint",
            "r_gripper_right_jaw_joint",
        ):
            _set_joint(name, 0.044339)
        for name in (
            "l_gripper_left_jaw_joint",
            "r_gripper_left_jaw_joint",
        ):
            _set_joint(name, -0.044339)

    # ── Step ──────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        """
        Apply action and advance simulation.

        Parameters
        ----------
        action : np.ndarray, shape (21,)
            Normalised torques in [-1, 1] for POLICY_ACT_IDX joints.
            Legs are held at their current qpos via PD control (not controlled by policy).

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        assert action.shape == (self.ACT_DIM,), \
            f"Expected action dim {self.ACT_DIM}, got {action.shape}"

        # Build full ctrl vector
        ctrl = np.zeros(N_CTRL)

        # Scale policy actions by torque limits
        for i, idx in enumerate(POLICY_ACT_IDX):
            ctrl[idx] = np.clip(action[i], -1.0, 1.0) * CTRL_LIMITS[idx]

        # Step physics
        for _ in range(self.n_substeps):
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        reward, info = self._compute_reward()
        terminated = self._is_terminated()
        truncated  = self._step_count >= self.max_episode_steps

        if terminated:
            info["fall"] = True

        return obs, reward, terminated, truncated, info

    # ── Observation ───────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        """Return flat observation vector of shape (OBS_DIM,)."""
        qpos = self.data.qpos   # includes floating base (7 DOF) + joints
        qvel = self.data.qvel

        # Extract joint positions (skip floating-base 7 qpos / 6 qvel)
        jnt_pos = qpos[7 : 7 + N_CTRL]    # 33 joints
        jnt_vel = qvel[6 : 6 + N_CTRL]

        # End-effector positions (world frame)
        mujoco.mj_fwdPosition(self.model, self.data)
        l_ee = self.data.site_xpos[self._l_ee_id].copy()
        r_ee = self.data.site_xpos[self._r_ee_id].copy()

        # Pelvis state (from IMU site)
        imu_pos = self.data.site_xpos[self._imu_id].copy()
        pel_linvel = self.data.qvel[0:3].copy()     # pelvis linear vel
        pel_angvel = self.data.qvel[3:6].copy()     # pelvis angular vel

        obs = np.concatenate([
            jnt_pos,            # 33
            jnt_vel,            # 33
            l_ee,               #  3
            r_ee,               #  3
            self._target_pos,   #  3
            pel_linvel,         #  3
            imu_pos,            #  3  (proxy for orientation, replace with euler if needed)
            pel_angvel,         #  3
        ])
        assert obs.shape == (self.OBS_DIM,), f"obs shape mismatch: {obs.shape}"
        return obs.astype(np.float32)

    # ── Reward ────────────────────────────────────────────────────────────
    def _compute_reward(self):
        l_ee = self.data.site_xpos[self._l_ee_id]
        r_ee = self.data.site_xpos[self._r_ee_id]

        # Distance from each gripper to target
        dist_l = float(np.linalg.norm(l_ee - self._target_pos))
        dist_r = float(np.linalg.norm(r_ee - self._target_pos))
        dist   = (dist_l + dist_r) * 0.5

        # Grasp: both grippers within 5 cm of target
        grasp = 1.0 if (dist_l < 0.05 and dist_r < 0.05) else 0.0

        # Energy penalty: sum of squared torques
        energy = float(np.sum(self.data.actuator_force ** 2))

        # Alive: robot pelvis above 0.4 m
        alive = 1.0 if self.data.qpos[2] > 0.4 else 0.0

        reward = (
            self.rw["grasp"]  * grasp
          - self.rw["dist"]   * dist
          - self.rw["energy"] * energy
          + self.rw["alive"]  * alive
        )

        info = dict(grasp=grasp, dist=dist, energy=energy, alive=alive)
        return float(reward), info

    # ── Termination ───────────────────────────────────────────────────────
    def _is_terminated(self) -> bool:
        """Terminate if robot falls (pelvis below 0.4 m)."""
        return bool(self.data.qpos[2] < 0.4)

    # ── Render helpers ────────────────────────────────────────────────────
    def render_viewer(self):
        """Launch MuJoCo passive viewer (blocking)."""
        with mujoco.viewer.launch_passive(self.model, self.data) as v:
            while v.is_running():
                mujoco.mj_step(self.model, self.data)
                v.sync()

    def close(self):
        pass
