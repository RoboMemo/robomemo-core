"""
Full Isaac Sim test: ground plane + 3 placeholder robots + retargeted motion playback.
All output goes to /tmp/isaac_full.log.
"""
import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

import numpy as np
from pathlib import Path

LOG_FILE = "/tmp/isaac_full.log"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# Clear previous log
with open(LOG_FILE, "w") as f:
    f.write("")

log("=== Isaac Sim Full Test ===")
log("Creating SimulationApp (headless=True)...")

from isaacsim import SimulationApp
sim = SimulationApp({"headless": True})

log("SimulationApp created.")

# USD imports (must be after SimulationApp creation)
import omni.usd
from pxr import UsdGeom, UsdPhysics, Gf, Sdf, PhysxSchema

stage = omni.usd.get_context().get_stage()

# --- Create ground plane ---
log("\nCreating ground plane...")
ground_prim = UsdGeom.Mesh.Define(stage, "/World/GroundPlane")
ground_prim.CreatePointsAttr([
    Gf.Vec3f(-50, -50, 0), Gf.Vec3f(50, -50, 0),
    Gf.Vec3f(50, 50, 0), Gf.Vec3f(-50, 50, 0),
])
ground_prim.CreateFaceVertexCountsAttr([4])
ground_prim.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
ground_prim.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)

# Add collision to ground
UsdPhysics.CollisionAPI.Apply(ground_prim.GetPrim())

# Enable physics scene
physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
physics_scene.CreateGravityMagnitudeAttr(9.81)

log("Ground plane and physics scene created.")

# --- Create 3 placeholder robots (capsules) ---
robots_config = {
    "H1":      {"joints": 19, "pos": Gf.Vec3d(-2.0, 0.0, 1.0), "color": Gf.Vec3f(0.8, 0.2, 0.2)},
    "G1":      {"joints": 23, "pos": Gf.Vec3d(0.0, 0.0, 1.0),  "color": Gf.Vec3f(0.2, 0.8, 0.2)},
    "Fourier": {"joints": 21, "pos": Gf.Vec3d(2.0, 0.0, 1.0),  "color": Gf.Vec3f(0.2, 0.2, 0.8)},
}

robot_xforms = {}
for name, cfg in robots_config.items():
    # Create an Xform (transform) for each robot
    xform_path = f"/World/{name}"
    xform = UsdGeom.Xform.Define(stage, xform_path)
    xform.AddTranslateOp().Set(cfg["pos"])
    robot_xforms[name] = xform

    # Body capsule
    capsule = UsdGeom.Capsule.Define(stage, f"{xform_path}/Body")
    capsule.CreateHeightAttr(0.8)
    capsule.CreateRadiusAttr(0.15)
    capsule.CreateAxisAttr("Z")
    capsule.CreateDisplayColorAttr([cfg["color"]])

    # Head sphere
    head = UsdGeom.Sphere.Define(stage, f"{xform_path}/Head")
    head.CreateRadiusAttr(0.12)
    head.CreateDisplayColorAttr([cfg["color"]])
    head_xform = UsdGeom.Xformable(head.GetPrim())
    head_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.55))

    log(f"  Created placeholder robot: {name} at {cfg['pos']}")

log(f"Created {len(robots_config)} placeholder robots.")

# --- Load retargeted motion data ---
motion_dir = Path(__file__).parent / "retargeted_motions"
motion_name = "walking"
motions = {}

for name, cfg in robots_config.items():
    fpath = motion_dir / f"{motion_name}_{name}.npy"
    data = np.load(fpath)
    motions[name] = data
    log(f"  Loaded {name} motion: shape={data.shape}")

# --- Run 60 frames with motion applied ---
num_frames = 60
log(f"\nRunning {num_frames} frames with motion playback...")

for frame in range(num_frames):
    # Apply motion data as positional offsets to each robot
    for name, cfg in robots_config.items():
        motion_data = motions[name]
        frame_data = motion_data[frame % motion_data.shape[0]]

        # Use first 3 joint values as XYZ translation offset (scaled down)
        # This simulates the robot moving based on retargeted joint data
        base_pos = cfg["pos"]
        dx = float(frame_data[0]) * 0.1
        dy = float(frame_data[1]) * 0.1 if len(frame_data) > 1 else 0.0
        dz = float(frame_data[2]) * 0.05 if len(frame_data) > 2 else 0.0

        new_pos = Gf.Vec3d(base_pos[0] + dx, base_pos[1] + dy, base_pos[2] + dz)

        # Update transform
        xform = robot_xforms[name]
        xform.GetPrim().GetAttribute("xformOp:translate").Set(new_pos)

    sim.update()

    if frame % 10 == 0 or frame == num_frames - 1:
        log(f"  Frame {frame}/{num_frames} - is_running={sim.is_running()}")

log(f"\nAll {num_frames} frames completed successfully!")
log(f"Motion '{motion_name}' applied to {len(robots_config)} robots.")

# --- Summary ---
log("\n=== Summary ===")
log(f"  Ground plane: 100x100m at z=0")
log(f"  Robots: {', '.join(robots_config.keys())}")
log(f"  Motion: {motion_name} ({motions['H1'].shape[0]} total frames, played {num_frames})")
log(f"  Status: PASSED")

log("\nClosing SimulationApp...")
sim.close()
