# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Convert roboforce_v2 datasets to LeRobot V3 format for π₀.5 SFT.

Handles the following conversions:
- roboforce_v2 frame data → LeRobot V3 Parquet + metadata
- 8D single-step actions → 50-step action chunks (with interpolation/padding)
- Phase labels (APPROACH/ALIGN/INSERT/DRIVE/DONE) → subtask text sequences
- Quantile statistics computation for robust normalization
- Instruction diversity injection (multiple natural language task descriptions)

References:
    - LeRobot V3 format: https://github.com/huggingface/lerobot
    - Quantile stats: src/lerobot/datasets/v30/augment_dataset_quantile_stats.py

Usage:
    python -m roboforce_skills.pi05_data_converter \\
        --input_dir datasets/mock_demos_v2/roboforce_screw_3rgbd_2ft_v1 \\
        --output_dir datasets/lerobot_v3/roboforce_pi05_v1

    python -m roboforce_skills.pi05_data_converter \\
        --input_dir datasets/mock_demos_v2/roboforce_screw_3rgbd_2ft_v1 \\
        --output_dir datasets/lerobot_v3/roboforce_pi05_v1 \\
        --compute_quantile_stats
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_DIM = 8
ACTION_HORIZON_V2 = 1   # roboforce_v2: single-step actions
ACTION_HORIZON_PI05 = 50  # π₀.5: 50-step action chunks

PHASE_TO_SUBTASK: dict[str, str] = {
    "APPROACH": "approach_screw",
    "ALIGN": "align_driver",
    "INSERT": "insert_screw",
    "DRIVE": "tighten_clockwise",
    "DONE": "verify_torque",
}

SUBTASK_DESCRIPTIONS: dict[str, str] = {
    "approach_screw": "Move end-effector toward the screw location",
    "align_driver": "Align the screw driver axis with the screw head",
    "insert_screw": "Insert the screw tip into the mounting hole",
    "tighten_clockwise": "Rotate screw driver clockwise to tighten",
    "verify_torque": "Verify target torque reached; task complete",
}

INSTRUCTION_VARIANTS: list[str] = [
    "Pick up the screw and drive it into the solar panel mounting bracket",
    "Tighten the screw into the mounting bracket on the solar panel",
    "Install the screw into the bracket by driving it clockwise",
    "Fasten the bolt into the solar panel frame bracket",
    "Use the screw driver to install the fastener into the bracket",
    "Secure the solar panel by driving the screw into the mount",
    "Grab the M4 hex screw and fasten it into the mounting hole",
    "Align the screw driver with the screw head and tighten until secure",
]


# ---------------------------------------------------------------------------
# Action Chunk Construction
# ---------------------------------------------------------------------------

def build_action_chunks(
    actions: np.ndarray,
    horizon: int = ACTION_HORIZON_PI05,
) -> np.ndarray:
    """Convert single-step actions to fixed-horizon action chunks.

    For each timestep t, the action chunk is actions[t:t+horizon]. If fewer
    than ``horizon`` steps remain, the last action is repeated to pad.

    Args:
        actions: (T, action_dim) array of single-step actions.
        horizon: Target chunk length.

    Returns:
        (T, horizon, action_dim) array of action chunks.
    """
    T, D = actions.shape
    chunks = np.zeros((T, horizon, D), dtype=actions.dtype)

    for t in range(T):
        end = min(t + horizon, T)
        length = end - t
        chunks[t, :length] = actions[t:end]
        # Pad with last available action
        if length < horizon:
            chunks[t, length:] = actions[end - 1]

    return chunks


# ---------------------------------------------------------------------------
# Subtask Label Generation
# ---------------------------------------------------------------------------

def generate_subtask_labels(phases: list[str]) -> list[dict[str, Any]]:
    """Generate subtask labels and chain-of-thought text from phase labels.

    Each frame gets:
    - ``subtask``: Current subtask name (e.g. "tighten_clockwise")
    - ``subtask_description``: Human-readable description
    - ``cot_sequence``: Chain-of-thought text from current phase to DONE

    Args:
        phases: List of phase strings per frame (e.g. ["APPROACH", "ALIGN", ...]).

    Returns:
        List of dicts with subtask labels per frame.
    """
    ordered_subtasks = [
        "approach_screw", "align_driver", "insert_screw",
        "tighten_clockwise", "verify_torque",
    ]

    labels = []
    for phase in phases:
        subtask = PHASE_TO_SUBTASK.get(phase, "approach_screw")
        description = SUBTASK_DESCRIPTIONS.get(subtask, "")

        # Build chain-of-thought: remaining subtasks from current to end
        try:
            idx = ordered_subtasks.index(subtask)
        except ValueError:
            idx = 0
        remaining = ordered_subtasks[idx:]
        cot_text = " -> ".join(remaining)

        labels.append({
            "subtask": subtask,
            "subtask_description": description,
            "cot_sequence": cot_text,
        })

    return labels


# ---------------------------------------------------------------------------
# Quantile Statistics Computation
# ---------------------------------------------------------------------------

def compute_quantile_stats(
    values: np.ndarray,
    quantile_range: tuple[float, float] = (0.01, 0.99),
) -> dict[str, Any]:
    """Compute quantile statistics for robust normalization.

    Reference: ``src/lerobot/datasets/v30/augment_dataset_quantile_stats.py``

    Args:
        values: (N, D) array of feature values.
        quantile_range: Lower and upper quantile bounds.

    Returns:
        Dict with per-feature quantile stats.
    """
    q_low, q_high = quantile_range
    lower = np.quantile(values, q_low, axis=0).tolist()
    upper = np.quantile(values, q_high, axis=0).tolist()
    median = np.median(values, axis=0).tolist()
    mean = np.mean(values, axis=0).tolist()
    std = np.std(values, axis=0).tolist()

    return {
        "quantile_range": list(quantile_range),
        "lower": lower,
        "upper": upper,
        "median": median,
        "mean": mean,
        "std": std,
        "num_samples": int(values.shape[0]),
        "num_features": int(values.shape[1]),
    }


def compute_dataset_quantile_stats(
    all_states: np.ndarray,
    all_actions: np.ndarray,
    all_ft_wrist: np.ndarray,
    all_ft_ee: np.ndarray,
    quantile_range: tuple[float, float] = (0.01, 0.99),
) -> dict[str, Any]:
    """Compute quantile stats for all feature groups in the dataset.

    Args:
        all_states: (N, state_dim) proprioceptive states.
        all_actions: (N, action_dim) actions.
        all_ft_wrist: (N, 6) wrist F/T wrench.
        all_ft_ee: (N, 6) end-effector F/T wrench.
        quantile_range: Quantile bounds.

    Returns:
        Dict with stats per feature group.
    """
    return {
        "state": compute_quantile_stats(all_states, quantile_range),
        "action": compute_quantile_stats(all_actions, quantile_range),
        "ft_wrist": compute_quantile_stats(all_ft_wrist, quantile_range),
        "ft_ee": compute_quantile_stats(all_ft_ee, quantile_range),
    }


# ---------------------------------------------------------------------------
# V2 → V3 Converter
# ---------------------------------------------------------------------------

class RoboForceV2ToLeRobotV3Converter:
    """Convert roboforce_v2 datasets to LeRobot V3 format for π₀.5 SFT.

    Input: roboforce_v2 dataset (from data_collection_v2.py)
    Output: LeRobot V3 dataset with action chunks, subtask labels, and
            quantile stats.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        action_horizon: int = ACTION_HORIZON_PI05,
        quantile_range: tuple[float, float] = (0.01, 0.99),
        seed: int = 42,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.action_horizon = action_horizon
        self.quantile_range = quantile_range
        self.rng = np.random.default_rng(seed)

    def _load_v2_episode(self, ep_dir: Path) -> list[dict]:
        """Load frames from a roboforce_v2 episode directory."""
        obs_path = ep_dir / "observations.jsonl"
        if not obs_path.exists():
            logger.warning(f"Missing observations.jsonl in {ep_dir}")
            return []

        frames = []
        with open(obs_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    frames.append(json.loads(line))
        return frames

    def _convert_episode(
        self,
        frames: list[dict],
        ep_idx: int,
    ) -> dict[str, Any]:
        """Convert a single episode from V2 to V3 format.

        Returns:
            Dict with V3 episode data including action chunks and subtask labels.
        """
        T = len(frames)
        if T == 0:
            return {"frames": [], "num_frames": 0}

        # Extract arrays
        actions = np.array([f["action"] for f in frames], dtype=np.float32)
        states = np.array([f["observation.state"] for f in frames], dtype=np.float32)
        phases = [f.get("phase", "APPROACH") for f in frames]

        # F/T data
        ft_wrist = np.array([
            f.get("observation.ft.wrist_ft.wrench", [0.0] * 6)
            for f in frames
        ], dtype=np.float32)
        ft_ee = np.array([
            f.get("observation.ft.ee_tip_ft.wrench", [0.0] * 6)
            for f in frames
        ], dtype=np.float32)

        # Build 50-step action chunks
        action_chunks = build_action_chunks(actions, self.action_horizon)

        # Generate subtask labels
        subtask_labels = generate_subtask_labels(phases)

        # Select instruction variant (one per episode for consistency)
        instruction = self.rng.choice(INSTRUCTION_VARIANTS)

        # Build V3 frames
        v3_frames = []
        for t in range(T):
            v3_frame: dict[str, Any] = {
                "episode_index": ep_idx,
                "frame_index": t,
                "timestamp": frames[t].get("timestamp", t * 0.02),
                # Observation
                "observation.state": states[t].tolist(),
                "observation.ft.wrist_ft": ft_wrist[t].tolist(),
                "observation.ft.ee_tip_ft": ft_ee[t].tolist(),
                # Action chunk (50-step)
                "action": action_chunks[t].tolist(),
                # Language
                "task_instruction": instruction,
                # Subtask / chain-of-thought
                "subtask": subtask_labels[t]["subtask"],
                "subtask_description": subtask_labels[t]["subtask_description"],
                "cot_sequence": subtask_labels[t]["cot_sequence"],
                # Metadata
                "phase": phases[t],
                "reward": frames[t].get("reward", 0.0),
                "done": frames[t].get("done", False),
            }

            # Image paths (remap to V3 structure)
            for cam in ["head_left", "head_right", "wrist"]:
                rgb_key = f"observation.images.{cam}.rgb"
                depth_key = f"observation.images.{cam}.depth"
                if rgb_key in frames[t] and frames[t][rgb_key] is not None:
                    v3_frame[rgb_key] = frames[t][rgb_key]
                if depth_key in frames[t] and frames[t][depth_key] is not None:
                    v3_frame[depth_key] = frames[t][depth_key]

            v3_frames.append(v3_frame)

        return {
            "frames": v3_frames,
            "num_frames": T,
            "instruction": instruction,
        }

    def convert(self) -> str:
        """Run the full V2 → V3 conversion.

        Returns:
            Output dataset path.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Discover episodes
        ep_dirs = sorted(
            d for d in self.input_dir.iterdir()
            if d.is_dir() and d.name.startswith("ep_")
        )

        if not ep_dirs:
            # Try flat file fallback
            flat_file = self.input_dir / "all_episodes.jsonl"
            if flat_file.exists():
                return self._convert_flat(flat_file)
            raise FileNotFoundError(
                f"No episode directories found in {self.input_dir}"
            )

        logger.info(f"Found {len(ep_dirs)} episodes in {self.input_dir}")

        # Accumulators for quantile stats
        all_states: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        all_ft_wrist: list[np.ndarray] = []
        all_ft_ee: list[np.ndarray] = []

        episodes_meta: list[dict] = []
        total_frames = 0

        # Data output directory
        data_dir = self.output_dir / "data"
        data_dir.mkdir(exist_ok=True)

        for ep_idx, ep_dir in enumerate(ep_dirs):
            frames = self._load_v2_episode(ep_dir)
            if not frames:
                continue

            result = self._convert_episode(frames, ep_idx)
            v3_frames = result["frames"]

            if not v3_frames:
                continue

            # Write episode data as JSONL (V3 compatible)
            ep_out = data_dir / f"episode_{ep_idx:06d}.jsonl"
            with open(ep_out, "w") as f:
                for frame in v3_frames:
                    f.write(json.dumps(frame) + "\n")

            # Copy images if they exist
            src_images = ep_dir / "images"
            if src_images.exists():
                dst_images = self.output_dir / "images" / f"episode_{ep_idx:06d}"
                dst_images.mkdir(parents=True, exist_ok=True)
                self._copy_images(src_images, dst_images)

            # Accumulate for stats
            ep_actions = np.array([f["action"] for f in frames], dtype=np.float32)
            ep_states = np.array(
                [f["observation.state"] for f in frames], dtype=np.float32
            )
            ep_ft_wrist = np.array([
                f.get("observation.ft.wrist_ft.wrench", [0.0] * 6)
                for f in frames
            ], dtype=np.float32)
            ep_ft_ee = np.array([
                f.get("observation.ft.ee_tip_ft.wrench", [0.0] * 6)
                for f in frames
            ], dtype=np.float32)

            all_states.append(ep_states)
            all_actions.append(ep_actions)
            all_ft_wrist.append(ep_ft_wrist)
            all_ft_ee.append(ep_ft_ee)

            episodes_meta.append({
                "episode_index": ep_idx,
                "num_frames": result["num_frames"],
                "instruction": result["instruction"],
            })
            total_frames += result["num_frames"]

            if (ep_idx + 1) % 20 == 0:
                logger.info(
                    f"  Converted {ep_idx + 1}/{len(ep_dirs)} episodes "
                    f"({total_frames} frames)"
                )

        # Write metadata
        self._write_metadata(episodes_meta, total_frames)

        # Write episodes index
        with open(self.output_dir / "episodes.json", "w") as f:
            json.dump(episodes_meta, f, indent=2)

        # Compute and write quantile stats
        if all_states:
            stats = compute_dataset_quantile_stats(
                np.concatenate(all_states),
                np.concatenate(all_actions),
                np.concatenate(all_ft_wrist),
                np.concatenate(all_ft_ee),
                self.quantile_range,
            )
            with open(self.output_dir / "quantile_stats.json", "w") as f:
                json.dump(stats, f, indent=2)

        # Write subtask label reference
        self._write_subtask_labels()

        logger.info(
            f"Conversion complete: {len(episodes_meta)} episodes, "
            f"{total_frames} frames -> {self.output_dir}"
        )
        return str(self.output_dir)

    def _convert_flat(self, flat_file: Path) -> str:
        """Convert from the flat all_episodes.jsonl fallback format."""
        logger.info(f"Using flat file: {flat_file}")

        episodes: dict[int, list[dict]] = {}
        with open(flat_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                frame = json.loads(line)
                ep_idx = frame.get("episode_index", 0)
                episodes.setdefault(ep_idx, []).append(frame)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        data_dir = self.output_dir / "data"
        data_dir.mkdir(exist_ok=True)

        all_states: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        all_ft_wrist: list[np.ndarray] = []
        all_ft_ee: list[np.ndarray] = []

        episodes_meta: list[dict] = []
        total_frames = 0

        for ep_idx in sorted(episodes.keys()):
            frames = episodes[ep_idx]
            result = self._convert_episode(frames, ep_idx)
            v3_frames = result["frames"]

            if not v3_frames:
                continue

            ep_out = data_dir / f"episode_{ep_idx:06d}.jsonl"
            with open(ep_out, "w") as f:
                for frame in v3_frames:
                    f.write(json.dumps(frame) + "\n")

            ep_actions = np.array([f["action"] for f in frames], dtype=np.float32)
            ep_states = np.array(
                [f["observation.state"] for f in frames], dtype=np.float32
            )
            ep_ft_wrist = np.array([
                f.get("observation.ft.wrist_ft.wrench", [0.0] * 6)
                for f in frames
            ], dtype=np.float32)
            ep_ft_ee = np.array([
                f.get("observation.ft.ee_tip_ft.wrench", [0.0] * 6)
                for f in frames
            ], dtype=np.float32)

            all_states.append(ep_states)
            all_actions.append(ep_actions)
            all_ft_wrist.append(ep_ft_wrist)
            all_ft_ee.append(ep_ft_ee)

            episodes_meta.append({
                "episode_index": ep_idx,
                "num_frames": result["num_frames"],
                "instruction": result["instruction"],
            })
            total_frames += result["num_frames"]

        self._write_metadata(episodes_meta, total_frames)

        with open(self.output_dir / "episodes.json", "w") as f:
            json.dump(episodes_meta, f, indent=2)

        if all_states:
            stats = compute_dataset_quantile_stats(
                np.concatenate(all_states),
                np.concatenate(all_actions),
                np.concatenate(all_ft_wrist),
                np.concatenate(all_ft_ee),
                self.quantile_range,
            )
            with open(self.output_dir / "quantile_stats.json", "w") as f:
                json.dump(stats, f, indent=2)

        self._write_subtask_labels()

        logger.info(
            f"Conversion complete: {len(episodes_meta)} episodes, "
            f"{total_frames} frames -> {self.output_dir}"
        )
        return str(self.output_dir)

    def _write_metadata(
        self,
        episodes_meta: list[dict],
        total_frames: int,
    ) -> None:
        """Write LeRobot V3 metadata.json."""
        metadata = {
            "format": "lerobot_v3",
            "num_episodes": len(episodes_meta),
            "total_frames": total_frames,
            "robot": "RoboForce",
            "task": "screw_driving",
            "fps": 50,
            "action_horizon": self.action_horizon,
            "sensors": {
                "cameras": ["head_left", "head_right", "wrist"],
                "ft_sensors": ["wrist_ft", "ee_tip_ft"],
                "camera_resolution": [640, 480],
                "camera_channels": {"rgb": 3, "depth": 1},
            },
            "observation_spec": {
                "state_dim": "variable",
                "ft_wrist_dim": 6,
                "ft_ee_dim": 6,
                "image_shape": [480, 640, 3],
                "depth_shape": [480, 640],
                "action_dim": ACTION_DIM,
                "action_chunk_shape": [self.action_horizon, ACTION_DIM],
            },
            "features": {
                "action": {
                    "shape": [self.action_horizon, ACTION_DIM],
                    "dtype": "float32",
                },
                "observation.state": {"dtype": "float32"},
                "observation.ft.wrist_ft": {"shape": [6], "dtype": "float32"},
                "observation.ft.ee_tip_ft": {"shape": [6], "dtype": "float32"},
                "subtask": {"dtype": "string"},
                "cot_sequence": {"dtype": "string"},
                "task_instruction": {"dtype": "string"},
            },
            "source": {
                "original_format": "roboforce_v2",
                "converter": "pi05_data_converter",
            },
        }
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _write_subtask_labels(self) -> None:
        """Write subtask label reference file."""
        labels = {
            "phase_to_subtask": PHASE_TO_SUBTASK,
            "subtask_descriptions": SUBTASK_DESCRIPTIONS,
            "ordered_subtasks": [
                "approach_screw", "align_driver", "insert_screw",
                "tighten_clockwise", "verify_torque",
            ],
            "instruction_variants": INSTRUCTION_VARIANTS,
        }
        with open(self.output_dir / "subtask_labels.json", "w") as f:
            json.dump(labels, f, indent=2)

    @staticmethod
    def _copy_images(src: Path, dst: Path) -> None:
        """Copy image files from V2 episode to V3 structure."""
        import shutil

        for cam_dir in src.iterdir():
            if cam_dir.is_dir():
                dst_cam = dst / cam_dir.name
                dst_cam.mkdir(parents=True, exist_ok=True)
                for img_file in cam_dir.iterdir():
                    if img_file.suffix == ".npy":
                        shutil.copy2(img_file, dst_cam / img_file.name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RoboForce V2 → LeRobot V3 Data Converter (π₀.5 SFT)"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Path to roboforce_v2 dataset directory",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path for the LeRobot V3 dataset",
    )
    parser.add_argument(
        "--action_horizon", type=int, default=ACTION_HORIZON_PI05,
        help=f"Action chunk length (default: {ACTION_HORIZON_PI05})",
    )
    parser.add_argument(
        "--quantile_low", type=float, default=0.01,
        help="Lower quantile bound (default: 0.01)",
    )
    parser.add_argument(
        "--quantile_high", type=float, default=0.99,
        help="Upper quantile bound (default: 0.99)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for instruction selection",
    )
    parser.add_argument(
        "--compute_quantile_stats", action="store_true",
        help="Compute and save quantile stats (runs automatically during convert)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    converter = RoboForceV2ToLeRobotV3Converter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        action_horizon=args.action_horizon,
        quantile_range=(args.quantile_low, args.quantile_high),
        seed=args.seed,
    )

    output_path = converter.convert()
    print(f"\nDataset converted: {output_path}")


if __name__ == "__main__":
    main()
