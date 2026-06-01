"""
Balance test for all supported robots in Isaac Sim.

Spawns each robot, commands default standing pose, runs 500 sim steps,
and checks that the robot's base height stays close to initial (i.e. it
isn't falling over or phasing through the floor).

Requires Isaac Sim + Isaac Lab to be installed.

Usage:
  python tests/test_balance_isaac.py                    # all robots, headless
  python tests/test_balance_isaac.py --robot unitree_h1 # single robot
  python tests/test_balance_isaac.py --gui              # with viewport
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("balance_test")

# ── Robot configs ──────────────────────────────────────────────

ROBOTS = {
    "unitree_g1": {
        "asset_module": "isaaclab_assets.robots.unitree",
        "asset_cfg": "G1_29DOF_CFG",
        "num_joints": 29,
        "expected_height_range": (0.6, 1.2),  # base z-height (m) when standing
    },
    "unitree_h1": {
        "asset_module": "isaaclab_assets.robots.unitree",
        "asset_cfg": "H1_CFG",
        "num_joints": 19,
        "expected_height_range": (0.7, 1.4),
    },
    "fourier_gr1t2": {
        "asset_module": "isaaclab_assets.robots.fourier",
        "asset_cfg": "GR1T2_CFG",
        "num_joints": 32,
        "expected_height_range": (0.7, 1.4),
    },
}


def run_balance_test(
    robot_name: str,
    headless: bool = True,
    num_steps: int = 500,
    dt: float = 0.02,
) -> dict:
    """Run a single-robot balance test in Isaac Sim.

    Returns dict with: passed, robot, steps, initial_height, final_height,
    max_drift, elapsed_sec
    """
    from isaacsim import SimulationApp

    robot_info = ROBOTS[robot_name]
    logger.info(f"{'='*60}")
    logger.info(f"  Balance Test: {robot_name}")
    logger.info(f"  Steps: {num_steps}, dt: {dt}s, headless: {headless}")
    logger.info(f"{'='*60}")

    # 1. Launch Sim
    sim_app = SimulationApp({"headless": headless})
    result = {
        "robot": robot_name,
        "passed": False,
        "steps": num_steps,
        "initial_height": 0.0,
        "final_height": 0.0,
        "max_drift": 0.0,
        "elapsed_sec": 0.0,
        "error": None,
    }

    try:
        import torch
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.sim import SimulationCfg, SimulationContext
        from omni.isaac.lab.assets import Articulation

        # 2. Load robot asset
        mod = importlib.import_module(robot_info["asset_module"])
        robot_cfg_cls = getattr(mod, robot_info["asset_cfg"])

        # 3. Build scene
        sim_cfg = SimulationCfg(dt=dt, render_interval=2, device="cuda:0")
        sim = SimulationContext(sim_cfg)
        sim.set_camera_view([2.5, 2.5, 2.0], [0.0, 0.0, 0.8])

        # Ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)

        # Spawn robot
        robot_cfg = robot_cfg_cls.replace(prim_path="/World/Robot")
        robot = Articulation(robot_cfg)

        # Reset
        sim.reset()
        robot.reset()

        actual_dof = robot.data.joint_pos.shape[1]
        num_joints = actual_dof
        logger.info(f"  Robot loaded: {actual_dof} DOF")

        # 4. Get initial state
        robot.update(dt)
        initial_pos = robot.data.root_pos_w[0].cpu().numpy()
        initial_height = float(initial_pos[2])
        result["initial_height"] = initial_height
        logger.info(f"  Initial base height: {initial_height:.3f}m")

        # Command default pose (zeros = standing)
        default_targets = torch.zeros(
            1, num_joints, dtype=torch.float32, device="cuda:0"
        )

        # 5. Run simulation
        heights = []
        t0 = time.time()
        for step in range(num_steps):
            robot.set_joint_position_target(default_targets)
            sim.step()
            robot.update(dt)

            base_pos = robot.data.root_pos_w[0].cpu().numpy()
            h = float(base_pos[2])
            heights.append(h)

            if step % 100 == 0:
                joint_pos = robot.data.joint_pos[0].cpu().numpy()
                logger.info(
                    f"  Step {step:4d}: height={h:.3f}m, "
                    f"joint_pos range=[{joint_pos.min():.3f}, {joint_pos.max():.3f}]"
                )

        elapsed = time.time() - t0
        result["elapsed_sec"] = elapsed

        # 6. Evaluate
        heights = np.array(heights)
        final_height = heights[-1]
        result["final_height"] = final_height

        height_drift = np.abs(heights - initial_height)
        result["max_drift"] = float(height_drift.max())

        h_min, h_max = robot_info["expected_height_range"]
        standing = h_min <= final_height <= h_max
        not_fallen = result["max_drift"] < 0.5  # less than 50cm drift

        result["passed"] = standing and not_fallen
        status = "PASS ✓" if result["passed"] else "FAIL ✗"

        logger.info(f"  Final height: {final_height:.3f}m (expected {h_min}-{h_max}m)")
        logger.info(f"  Max height drift: {result['max_drift']:.3f}m")
        logger.info(f"  Sim time: {elapsed:.1f}s for {num_steps} steps ({num_steps/elapsed:.0f} steps/s)")
        logger.info(f"  Result: {status}")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"  ERROR: {e}")
    finally:
        try:
            sim.close()
        except Exception:
            pass
        sim_app.close()

    return result


def main():
    parser = argparse.ArgumentParser(description="Balance test for robots in Isaac Sim")
    parser.add_argument(
        "--robot",
        choices=list(ROBOTS.keys()) + ["all"],
        default="all",
        help="Which robot to test (default: all)",
    )
    parser.add_argument("--gui", action="store_true", help="Show Isaac Sim viewport")
    parser.add_argument("--steps", type=int, default=500, help="Simulation steps per test")
    args = parser.parse_args()

    robots_to_test = list(ROBOTS.keys()) if args.robot == "all" else [args.robot]
    headless = not args.gui

    results = []
    for robot_name in robots_to_test:
        r = run_balance_test(robot_name, headless=headless, num_steps=args.steps)
        results.append(r)

    # Summary
    print("\n" + "=" * 60)
    print("  BALANCE TEST SUMMARY")
    print("=" * 60)
    all_pass = True
    for r in results:
        status = "PASS ✓" if r["passed"] else "FAIL ✗"
        if r["error"]:
            status = f"ERROR: {r['error'][:50]}"
            all_pass = False
        elif not r["passed"]:
            all_pass = False
        print(
            f"  {r['robot']:20s}  {status:12s}  "
            f"height={r['final_height']:.3f}m  drift={r['max_drift']:.3f}m  "
            f"time={r['elapsed_sec']:.1f}s"
        )
    print("=" * 60)
    print(f"  Overall: {'ALL PASSED ✓' if all_pass else 'SOME FAILED ✗'}")
    print("=" * 60)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
