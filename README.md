# RoboForce — Solar Panel Screw Installation with Isaac Sim

End-to-end simulation, labeling, skill training, and validation pipeline for autonomous solar panel screw installation using the RoboForce tracked mobile manipulator.

## Overview

RoboForce is a tracked mobile manipulator designed for solar farm maintenance. This project provides:

1. **Isaac Sim Scene** (`roboforce_sim/`) — Full simulation environment with solar panel racks, screw/bolt objects, desert terrain, and weather variations (day/night/dusty).
2. **Auto-Labeling Pipeline** (`roboforce_labeling/`) — Synthetic data generation with domain randomization, producing COCO-format labels for screw detection and 6D pose estimation.
3. **Skill Training** (`roboforce_skills/`) — Demonstration collection in LeRobot V2/V3 format, with configs for fine-tuning GR00T N1.6, OpenPI (π₀), and π₀.5 VLA models.
4. **Validation** (`roboforce_validation/`) — Evaluation across weather conditions, screw positions, and inference benchmarking.
5. **RoboMemo** (`roboforce_memory/`) — Episodic memory system for experience storage, retrieval, and summarization.

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
  sensors/              # Mock sensor suite (3 RGBD + 2 FT)
roboforce_labeling/     # Synthetic data & auto-labeling
roboforce_skills/       # Skill learning (GR00T, OpenPI, π₀.5)
roboforce_memory/       # RoboMemo episodic memory system
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

## π₀.5 SFT (Supervised Fine-Tuning)

Fine-tune Physical Intelligence's π₀.5 VLA model on RoboForce screw driving demonstrations.

**Key features:**
- **50-step action chunks** via flow matching (vs. 16-step for π₀)
- **Chain-of-thought** subtask decomposition (approach → align → insert → tighten → verify)
- **Heterogeneous co-training**: robot demos + web data + verbal instructions + bbox annotations
- **Force/torque as first-class observations**: wrist + EE tip F/T sensors (12D wrench)
- **3 cameras**: head_left, head_right, wrist at 640×480
- **Quantile normalization** for robust feature scaling
- Pretrained from `lerobot/pi05_base`

```bash
# Generate π₀.5 SFT config
python -m roboforce_skills.pi05_sft_config --generate_config --config_output configs/pi05_sft.json

# Convert roboforce_v2 dataset to LeRobot V3 format
python -m roboforce_skills.pi05_data_converter \
    --input_dir datasets/mock_demos_v2/roboforce_screw_3rgbd_2ft_v1 \
    --output_dir datasets/lerobot_v3/roboforce_pi05_v1

# Print training command
python -m roboforce_skills.pi05_sft_config --print_command

# Validate dataset compatibility
python -m roboforce_skills.pi05_sft_config --validate_dataset datasets/lerobot_v3/roboforce_pi05_v1
```

## RoboMemo — Episodic Memory System

Vector-indexed episodic memory for conditioning VLA policies on past experiences during inference.

**Components:**
- **Schema** (`schema.py`) — Pydantic `ScrewMemory` model with observation, 50-step action chunk, F/T feedback, screw properties, and outcome metadata
- **Store** (`store.py`) — Pluggable backend: Qdrant for production, numpy cosine similarity for development. CRUD + batch ingest + per-session delete (GDPR-ready). P99 < 200ms target
- **Retriever** (`retriever.py`) — Semantic similarity + exponential time-decay weighting, filter by success/screw_type/environment
- **Summarizer** (`summarizer.py`) — Auto-summarize trajectories (e.g. "M4 hex screw, 2.5Nm peak torque, success in 1.0s")
- **ROS 2 Bridge** (`ros2_bridge.py`) — Pub/sub stubs for live memory create/query events
- **CLI** (`cli.py`) — Command-line interface for ingest/search/stats/delete

```bash
# Ingest demonstration data into memory
python -m roboforce_memory.cli ingest --dataset datasets/mock_demos_v2/roboforce_screw_3rgbd_2ft_v1

# Search memories
python -m roboforce_memory.cli search --query "M4 screw alignment" --top_k 5

# View memory store statistics
python -m roboforce_memory.cli stats

# Delete all memories for a session (GDPR)
python -m roboforce_memory.cli delete --session <session_id> --force
```

## References

- [IsaacLab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [GR00T N1.6](https://github.com/NVIDIA/Isaac-GR00T)
- [OpenPI (π₀)](https://github.com/Physical-Intelligence/openpi)
- [π₀.5 Paper](https://arxiv.org/abs/2504.16054)
- [LeRobot](https://github.com/huggingface/lerobot)

## License

Proprietary — RoboForce Project
