"""
train.py — Training entry point for G1 + OpenUMI bimanual grasping.

Usage:
  # PPO with stable-baselines3:
  python train.py --config configs/g1_grasp.yaml

  # Render a trained policy:
  python train.py --config configs/g1_grasp.yaml --render --load checkpoints/best_model

  # Quick env sanity check (no training):
  python train.py --check

Dependencies:
  pip install mujoco stable-baselines3[extra] pyyaml

Notes:
  - The environment wraps G1GraspEnv in a gym-compatible interface using
    a thin adapter so it works with SB3 out of the box.
  - For mjlab's native runner, replace the SB3 boilerplate below with
    mjlab's env-registration and runner calls.
"""

from __future__ import annotations

import argparse
import os
import sys
import yaml
import numpy as np

# ── Gym-compatible wrapper ─────────────────────────────────────────────────
try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    print("[warn] gymnasium not installed; gym wrapper disabled.")

# ── SB3 ───────────────────────────────────────────────────────────────────
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize
    from stable_baselines3.common.callbacks import (
        EvalCallback, CheckpointCallback
    )
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("[warn] stable-baselines3 not installed; PPO training disabled.")

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))
from envs.g1_grasp_env import G1GraspEnv


# ── Gymnasium adapter ──────────────────────────────────────────────────────
if GYM_AVAILABLE:
    class G1GraspGymEnv(gym.Env):
        """Thin gymnasium wrapper around G1GraspEnv."""

        metadata = {"render_modes": ["human"]}

        def __init__(self, xml_path: str, sim_dt: float, control_dt: float,
                     max_episode_steps: int, reward_weights: dict | None = None):
            super().__init__()
            self._env = G1GraspEnv(
                xml_path=xml_path,
                sim_dt=sim_dt,
                control_dt=control_dt,
                max_episode_steps=max_episode_steps,
                reward_weights=reward_weights,
            )
            obs_dim = self._env.OBS_DIM
            act_dim = self._env.ACT_DIM

            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
            )

        def reset(self, *, seed=None, options=None):
            obs = self._env.reset(seed=seed)
            return obs, {}

        def step(self, action):
            obs, rew, terminated, truncated, info = self._env.step(action)
            return obs, rew, terminated, truncated, info

        def render(self):
            self._env.render_viewer()

        def close(self):
            self._env.close()


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train G1 + OpenUMI grasping policy")
    p.add_argument("--config", default="configs/g1_grasp.yaml")
    p.add_argument("--load",   default=None, help="Path to checkpoint to continue training")
    p.add_argument("--render", action="store_true", help="Render with loaded policy")
    p.add_argument("--check",  action="store_true", help="Quick environment sanity check")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Sanity check ───────────────────────────────────────────────────────────
def run_env_check(cfg: dict):
    """Run one episode with random actions and print stats."""
    env_cfg  = cfg["env"]
    xml_path = os.path.abspath(env_cfg["xml_path"])
    print(f"[check] Loading model from: {xml_path}")

    env = G1GraspEnv(
        xml_path=xml_path,
        sim_dt=env_cfg["sim_dt"],
        control_dt=env_cfg["control_dt"],
        max_episode_steps=env_cfg["max_episode_steps"],
        reward_weights=env_cfg.get("reward_weights"),
    )

    obs = env.reset(seed=0)
    print(f"[check] obs shape = {obs.shape}, dtype = {obs.dtype}")
    print(f"[check] act dim   = {env.ACT_DIM}")

    total_reward = 0.0
    for step_i in range(50):
        action = np.random.uniform(-0.1, 0.1, size=(env.ACT_DIM,)).astype(np.float32)
        obs, rew, terminated, truncated, info = env.step(action)
        total_reward += rew
        if terminated or truncated:
            print(f"[check] episode ended at step {step_i + 1}")
            break

    print(f"[check] total reward over {step_i+1} steps = {total_reward:.4f}")
    print(f"[check] last info = {info}")
    print("[check] PASSED")
    env.close()


# ── Training ───────────────────────────────────────────────────────────────
def train(cfg: dict, load_path: str | None = None):
    if not SB3_AVAILABLE or not GYM_AVAILABLE:
        raise RuntimeError(
            "stable-baselines3 and gymnasium are required for training.\n"
            "  pip install mujoco stable-baselines3[extra] gymnasium pyyaml"
        )

    env_cfg   = cfg["env"]
    agent_cfg = cfg["agent"]
    train_cfg = cfg["training"]

    xml_path = os.path.abspath(env_cfg["xml_path"])
    os.makedirs(train_cfg["log_dir"],  exist_ok=True)
    os.makedirs(train_cfg["save_dir"], exist_ok=True)

    def make_env():
        return G1GraspGymEnv(
            xml_path=xml_path,
            sim_dt=env_cfg["sim_dt"],
            control_dt=env_cfg["control_dt"],
            max_episode_steps=env_cfg["max_episode_steps"],
            reward_weights=env_cfg.get("reward_weights"),
        )

    n_envs = train_cfg["n_envs"]
    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=train_cfg["seed"])

    if train_cfg.get("normalize_obs") or train_cfg.get("normalize_reward"):
        vec_env = VecNormalize(
            vec_env,
            norm_obs=train_cfg.get("normalize_obs", True),
            norm_reward=train_cfg.get("normalize_reward", True),
            clip_obs=10.0,
        )

    policy_kwargs = dict(
        net_arch=dict(
            pi=agent_cfg["policy_net"],
            vf=agent_cfg["value_net"],
        ),
        activation_fn=__import__("torch").nn.Tanh
            if agent_cfg.get("activation") == "tanh"
            else __import__("torch").nn.ReLU,
    )

    if load_path is not None:
        print(f"[train] Loading checkpoint: {load_path}")
        model = PPO.load(load_path, env=vec_env)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=agent_cfg["learning_rate"],
            n_steps=agent_cfg["n_steps"],
            batch_size=agent_cfg["batch_size"],
            n_epochs=agent_cfg["n_epochs"],
            gamma=agent_cfg["gamma"],
            gae_lambda=agent_cfg["gae_lambda"],
            clip_range=agent_cfg["clip_range"],
            ent_coef=agent_cfg["ent_coef"],
            vf_coef=agent_cfg["vf_coef"],
            max_grad_norm=agent_cfg["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=train_cfg["log_dir"],
            seed=train_cfg["seed"],
        )

    eval_env = make_vec_env(make_env, n_envs=1, seed=train_cfg["seed"] + 1)
    if isinstance(vec_env, VecNormalize):
        eval_env = VecNormalize(eval_env, training=False, norm_reward=False)

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=train_cfg["save_dir"],
            log_path=train_cfg["log_dir"],
            eval_freq=max(train_cfg["eval_freq"] // n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=max(train_cfg["save_freq"] // n_envs, 1),
            save_path=train_cfg["save_dir"],
            name_prefix="g1_grasp",
        ),
    ]

    print(f"[train] Starting PPO training for {train_cfg['total_timesteps']:,} steps …")
    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=callbacks,
        reset_num_timesteps=(load_path is None),
    )

    final_path = os.path.join(train_cfg["save_dir"], "g1_grasp_final")
    model.save(final_path)
    print(f"[train] Saved final model to {final_path}")
    vec_env.close()


# ── Render ────────────────────────────────────────────────────────────────
def render_policy(cfg: dict, load_path: str):
    if not SB3_AVAILABLE or not GYM_AVAILABLE:
        raise RuntimeError("stable-baselines3 and gymnasium are required for rendering.")

    env_cfg = cfg["env"]
    xml_path = os.path.abspath(env_cfg["xml_path"])

    env   = G1GraspGymEnv(
        xml_path=xml_path,
        sim_dt=env_cfg["sim_dt"],
        control_dt=env_cfg["control_dt"],
        max_episode_steps=env_cfg["max_episode_steps"],
    )
    model = PPO.load(load_path)

    obs, _ = env.reset()
    import mujoco
    with mujoco.viewer.launch_passive(env._env.model, env._env.data) as v:
        while v.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, _ = env.step(action)
            v.sync()
            if terminated or truncated:
                obs, _ = env.reset()

    env.close()


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args   = parse_args()
    cfg    = load_config(args.config)

    if args.check:
        run_env_check(cfg)
    elif args.render:
        if args.load is None:
            print("[error] --load <checkpoint_path> required for --render")
            sys.exit(1)
        render_policy(cfg, args.load)
    else:
        train(cfg, load_path=args.load)
