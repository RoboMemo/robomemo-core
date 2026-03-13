# RoboForce — Solar Panel Screw Installation with Isaac Sim

End-to-end simulation, labeling, skill training, and validation pipeline for autonomous solar panel screw installation using the RoboForce tracked mobile manipulator.

## Overview

RoboForce is a tracked mobile manipulator designed for solar farm maintenance. This project provides:

1. **Isaac Sim Scene** (`roboforce_sim/`) — Full simulation environment with solar panel racks, screw/bolt objects, desert terrain, and weather variations (day/night/dusty).
2. **Auto-Labeling Pipeline** (`roboforce_labeling/`) — Synthetic data generation with domain randomization, producing COCO-format labels for screw detection and 6D pose estimation.
3. **Skill Training** (`roboforce_skills/`) — Demonstration collection in LeRobot V2 format, with configs for fine-tuning GR00T N1.6 and OpenPI (π₀) VLA models.
4. **Validation** (`roboforce_validation/`) — Evaluation across weather conditions, screw positions, and inference benchmarking.

## System Requirements

| Component | Version |
|---|---|
| IsaacLab | 2.3.2 (`/home/siyu/IsaacLab/`) |
| Python venv | `/home/siyu/isaac-sim-env/` |
| GPU | NVIDIA RTX 5090 (32GB) |
| OS | Ubuntu 22.04 |

## Quick Start

```bash
# Activate environment
source /home/siyu/isaac-sim-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the simulation environment
python -m roboforce_sim.envs.solar_panel_env

# Collect demonstration data
python -m roboforce_skills.data_collection --num_episodes 1000

# Generate labeled dataset
python -m roboforce_labeling.label_pipeline --num_images 10000

# Evaluate a trained policy
python -m roboforce_validation.sim_eval --policy_checkpoint /path/to/checkpoint
```

## Project Structure

```
roboforce_sim/          # Isaac Sim scene & environment
  envs/                 # ManagerBasedEnv environments
  assets/               # Procedural USD asset generators
  configs/              # YAML configurations
roboforce_labeling/     # Synthetic data & auto-labeling
roboforce_skills/       # Skill learning (GR00T, OpenPI)
roboforce_validation/   # Evaluation & benchmarking
```

## Robot Description

RoboForce features:
- **Tracked base** — Tank-drive for rough desert terrain
- **Dual 7-DOF arms** — Dexterous manipulation
- **Head-mounted sensors** — RGB + depth cameras
- **Screw-driving EE** — Automated rotary socket end-effector

## Task: Screw Driving

The robot approaches a solar panel rack, locates mounting screws, aligns the end-effector, and drives screws to target torque. The 8D action space covers:
- 6D EE delta pose (position + orientation)
- 1D screw rotation
- 1D gripper actuation

## References

- [IsaacLab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [GR00T N1.6](https://github.com/NVIDIA/Isaac-GR00T)
- [OpenPI (π₀)](https://github.com/Physical-Intelligence/openpi)
- [LeRobot](https://github.com/huggingface/lerobot)

## License

Proprietary — RoboForce Project
