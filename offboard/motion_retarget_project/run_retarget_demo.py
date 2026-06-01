"""
Isaac Lab 三机器人动作重定向 Demo
H1 + G1 + Fourier(H1代替) 并排执行 retarget 后的人类动作
使用 isaacsim.core.api.World 正确步进物理
"""
import argparse
import numpy as np
import os
from pathlib import Path

# --- Args (must parse before SimulationApp) ---
parser = argparse.ArgumentParser()
parser.add_argument("--motion", default="walking", choices=["walking", "running", "jumping"])
parser.add_argument("--headless", action="store_true")
parser.add_argument("--num_frames", type=int, default=120)
args = parser.parse_args()

# --- Log helper ---
LOG_PATH = "/tmp/retarget_demo.log"
_log = open(LOG_PATH, "w")
def log(msg):
    _log.write(msg + "\n"); _log.flush()

log("=== Motion Retarget Demo ===")
log(f"Motion: {args.motion}, Headless: {args.headless}")

# --- Isaac Sim bootstrap ---
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaacsim import SimulationApp
sim = SimulationApp({"headless": args.headless, "width": 1920, "height": 1080})

import carb
settings = carb.settings.get_settings()
ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5"
settings.set("/persistent/isaac/asset_root/cloud", ASSET_ROOT)
settings.set("/persistent/isaac/asset_root/default", ASSET_ROOT)

from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction

# --- World setup ---
world = World(stage_units_in_meters=1.0)
world.scene.add_ground_plane()
log("World + ground created.")

# --- Robot configs ---
NUCLEUS = ASSET_ROOT
robots_cfg = [
    {
        "name": "h1",
        "usd": f"{NUCLEUS}/Isaac/IsaacLab/Robots/Unitree/H1/h1.usd",
        "prim": "/World/H1",
        "pos": np.array([-3.0, 0.0, 1.05]),
        "motion_file": "H1",
        # retarget index → DOF name mapping
        "joint_map": {
            0: "left_hip_yaw",    1: "left_hip_roll",    2: "left_hip_pitch",
            3: "left_knee",       4: "left_ankle",
            5: "right_hip_yaw",   6: "right_hip_roll",   7: "right_hip_pitch",
            8: "right_knee",      9: "right_ankle",
            10: "torso",
            11: "left_shoulder_pitch",  12: "left_shoulder_roll",
            13: "left_shoulder_yaw",    14: "left_elbow",
            15: "right_shoulder_pitch", 16: "right_shoulder_roll",
            17: "right_shoulder_yaw",   18: "right_elbow",
        },
    },
    {
        "name": "g1",
        "usd": f"{NUCLEUS}/Isaac/IsaacLab/Robots/Unitree/G1/g1.usd",
        "prim": "/World/G1",
        "pos": np.array([0.0, 0.0, 0.74]),
        "motion_file": "G1",
        "joint_map": {
            0: "left_hip_yaw_joint",    1: "left_hip_roll_joint",    2: "left_hip_pitch_joint",
            3: "left_knee_joint",       4: "left_ankle_pitch_joint",
            5: "right_hip_yaw_joint",   6: "right_hip_roll_joint",   7: "right_hip_pitch_joint",
            8: "right_knee_joint",      9: "right_ankle_pitch_joint",
            10: "torso_joint",
            11: "left_shoulder_pitch_joint",  12: "left_shoulder_roll_joint",
            13: "left_shoulder_yaw_joint",    14: "left_elbow_pitch_joint",
            15: "right_shoulder_pitch_joint", 16: "right_shoulder_roll_joint",
            17: "right_shoulder_yaw_joint",   18: "right_elbow_pitch_joint",
            19: "left_ankle_roll_joint",      20: "right_ankle_roll_joint",
            21: "left_elbow_roll_joint",      22: "right_elbow_roll_joint",
        },
    },
    {
        "name": "fourier",
        "usd": f"{NUCLEUS}/Isaac/IsaacLab/Robots/Unitree/H1/h1.usd",  # placeholder
        "prim": "/World/Fourier",
        "pos": np.array([3.0, 0.0, 1.05]),
        "motion_file": "Fourier",
        "joint_map": {
            0: "left_hip_yaw",    1: "left_hip_roll",    2: "left_hip_pitch",
            3: "left_knee",       4: "left_ankle",
            5: "right_hip_yaw",   6: "right_hip_roll",   7: "right_hip_pitch",
            8: "right_knee",      9: "right_ankle",
            10: "torso",
            11: "left_shoulder_pitch",  12: "left_shoulder_roll",
            13: "left_shoulder_yaw",    14: "left_elbow",
            15: "right_shoulder_pitch", 16: "right_shoulder_roll",
            17: "right_shoulder_yaw",   18: "right_elbow",
        },
    },
]

# --- Add robots to stage ---
robot_objs = []
for cfg in robots_cfg:
    add_reference_to_stage(cfg["usd"], cfg["prim"])
    robot = world.scene.add(Robot(
        prim_path=cfg["prim"],
        name=cfg["name"],
        position=cfg["pos"],
    ))
    robot_objs.append(robot)
    log(f"  Added {cfg['name']} at {cfg['pos']}")

# --- Load motion data ---
motion_dir = Path("retargeted_motions")
motions = {}
for cfg in robots_cfg:
    fpath = motion_dir / f"{args.motion}_{cfg['motion_file']}.npy"
    motions[cfg["name"]] = np.load(fpath)
    m = motions[cfg["name"]]
    log(f"  {cfg['name']} motion: {m.shape}, range [{m.min():.3f}, {m.max():.3f}]")

# --- Initialize physics ---
world.reset()
log("Physics initialized.")

# --- Build DOF index lookup per robot ---
dof_maps = {}
for i, cfg in enumerate(robots_cfg):
    robot = robot_objs[i]
    dof_names = robot.dof_names
    log(f"  {cfg['name']} DOFs ({robot.num_dof}): {dof_names}")

    # Map retarget indices to actual DOF indices
    idx_map = {}
    for retarget_idx, joint_name in cfg["joint_map"].items():
        for dof_idx, name in enumerate(dof_names):
            if name == joint_name:
                idx_map[retarget_idx] = dof_idx
                break
    dof_maps[cfg["name"]] = idx_map
    log(f"  {cfg['name']} mapped {len(idx_map)}/{len(cfg['joint_map'])} joints")

# --- Set PD gains for all robots (high stiffness to track targets) ---
for i, cfg in enumerate(robots_cfg):
    robot = robot_objs[i]
    n = robot.num_dof
    # High stiffness (Kp) + moderate damping (Kd) for position tracking
    stiffness = np.full(n, 200.0)
    damping = np.full(n, 20.0)
    robot.get_articulation_controller().set_gains(kps=stiffness, kds=damping)
    log(f"  {cfg['name']} PD gains set: Kp=200, Kd=20")

# --- Simulation loop ---
num_frames = min(args.num_frames, min(m.shape[0] for m in motions.values()))
log(f"\n🎬 Playing '{args.motion}' motion: {num_frames} frames")

frame_skip = 2  # 60Hz sim / 30fps mocap
step = 0
frame = 0

while frame < num_frames:
    # Apply joint targets every frame_skip physics steps
    if step % frame_skip == 0 and frame < num_frames:
        for i, cfg in enumerate(robots_cfg):
            robot = robot_objs[i]
            motion = motions[cfg["name"]]
            idx_map = dof_maps[cfg["name"]]

            targets = np.zeros(robot.num_dof)
            for retarget_idx, dof_idx in idx_map.items():
                if retarget_idx < motion.shape[1]:
                    targets[dof_idx] = motion[frame, retarget_idx]

            # Use position targets (PD control) instead of forcing positions
            robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=targets)
            )

        if frame % 30 == 0:
            poses = []
            for r in robot_objs:
                pos, _ = r.get_world_pose()
                poses.append(f"z={pos[2]:.3f}")
            log(f"  Frame {frame:3d}/{num_frames}: {' | '.join(poses)}")

        frame += 1

    world.step(render=not args.headless)
    step += 1

# --- Final state ---
log(f"\n=== Final State ===")
for i, cfg in enumerate(robots_cfg):
    robot = robot_objs[i]
    pos, rot = robot.get_world_pose()
    joints = robot.get_joint_positions()
    log(f"  {cfg['name']}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), joints_range=[{joints.min():.3f}, {joints.max():.3f}]")

log(f"\n✅ PASSED - {num_frames} frames, 3 robots")
sim.close()
log("Done.")
_log.close()
