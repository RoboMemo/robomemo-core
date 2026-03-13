"""
LeRobot Format Exporter
Converts auto-label pipeline JSONL output into LeRobot-compatible dataset format.

Usage:
    python lerobot_exporter.py <input.jsonl> --output-dir <dir> [--robot-type single_arm]
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any


LEROBOT_VERSION = "2.1"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL file and return list of episode dicts."""
    episodes = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping malformed line: {e}", file=sys.stderr)
    return episodes


def export_lerobot(
    episodes: List[Dict[str, Any]],
    output_dir: str,
    robot_type: str = "single_arm",
    fps: int = 30,
) -> Dict[str, Any]:
    """Export labeled episodes to LeRobot dataset format."""
    out = Path(output_dir)
    meta_dir = out / "meta"
    data_dir = out / "data"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Collect unique tasks
    tasks_map: Dict[str, int] = {}
    task_idx = 0
    for ep in episodes:
        summary = ep.get("task_summary", "unknown task")
        if summary not in tasks_map:
            tasks_map[summary] = task_idx
            task_idx += 1

    # 1. meta/info.json
    total_frames = sum(
        ep.get("video_info", {}).get("total_frames", 0) for ep in episodes
    )
    total_episodes = len(episodes)
    ep_fps = episodes[0].get("video_info", {}).get("fps", fps) if episodes else fps

    info = {
        "codebase_version": LEROBOT_VERSION,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "fps": ep_fps,
        "total_tasks": len(tasks_map),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/episode_{episode_index:06d}.mp4",
        "features": {
            "action_primitive": {
                "dtype": "string",
                "shape": [1],
                "description": "Predicted action primitive label",
            },
            "target_object": {
                "dtype": "string",
                "shape": [1],
                "description": "Target object of the action",
            },
            "gripper_state": {
                "dtype": "string",
                "shape": [1],
                "description": "Gripper state: open/closing/closed/opening",
            },
            "contact_type": {
                "dtype": "string",
                "shape": [1],
                "description": "Contact type: none/point/surface/edge/wrap",
            },
            "force_level": {
                "dtype": "string",
                "shape": [1],
                "description": "Force level: none/light/medium/strong",
            },
            "phase_name": {
                "dtype": "string",
                "shape": [1],
                "description": "Phase name from temporal segmentation",
            },
            "confidence": {
                "dtype": "float32",
                "shape": [1],
                "description": "VLM prediction confidence",
            },
        },
        "auto_label_model": episodes[0].get("model", "unknown") if episodes else "unknown",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # 2. meta/episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for i, ep in enumerate(episodes):
            vid_info = ep.get("video_info", {})
            episode_meta = {
                "episode_index": i,
                "episode_id": ep.get("episode_id", f"episode_{i}"),
                "tasks": [tasks_map.get(ep.get("task_summary", "unknown task"), 0)],
                "length": vid_info.get("total_frames", 0),
                "duration": vid_info.get("duration", 0),
                "video_path": ep.get("video_path", ""),
                "num_phases": len(ep.get("phases", [])),
                "success": ep.get("success", True),
            }
            f.write(json.dumps(episode_meta, ensure_ascii=False) + "\n")

    # 3. meta/tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task_desc, task_id in sorted(tasks_map.items(), key=lambda x: x[1]):
            # Collect skill labels from episodes with this task
            skill_labels = set()
            for ep in episodes:
                if ep.get("task_summary") == task_desc:
                    for phase in ep.get("phases", []):
                        skill_labels.add(phase.get("action_primitive", "unknown"))

            task_entry = {
                "task_index": task_id,
                "task": task_desc,
                "skill_labels": sorted(skill_labels),
            }
            f.write(json.dumps(task_entry, ensure_ascii=False) + "\n")

    # 4. data/ — action label chunks as parquet-like JSON (placeholder)
    # Real parquet would need pyarrow, so we write JSON with the same structure
    chunk_dir = data_dir / "chunk-000"
    chunk_dir.mkdir(exist_ok=True)

    for i, ep in enumerate(episodes):
        rows = []
        for phase in ep.get("phases", []):
            # One row per phase — in a real pipeline, each frame would get a row
            # with interpolated labels from the phase it belongs to
            row = {
                "episode_index": i,
                "phase_index": phase.get("phase_idx", 0),
                "start_frame": phase.get("start_frame", 0),
                "end_frame": phase.get("end_frame", 0),
                "start_time": phase.get("start_time", 0),
                "end_time": phase.get("end_time", 0),
                "action_primitive": phase.get("action_primitive", "wait"),
                "target_object": phase.get("target_object", "unknown"),
                "gripper_state": phase.get("gripper_state", "open"),
                "phase_name": phase.get("phase_name", ""),
                "confidence": phase.get("confidence", 0.0),
            }
            mechanics = phase.get("mechanics", {})
            row["contact_type"] = mechanics.get("contact_type", "none")
            row["force_level"] = mechanics.get("force_level", "none")
            row["contact_points"] = mechanics.get("contact_points", "")
            row["motion_direction"] = mechanics.get("motion_direction", "linear")
            rows.append(row)

        ep_file = chunk_dir / f"episode_{i:06d}.json"
        with open(ep_file, "w") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    result = {
        "output_dir": str(out),
        "total_episodes": total_episodes,
        "total_tasks": len(tasks_map),
        "files_created": [
            str(meta_dir / "info.json"),
            str(meta_dir / "episodes.jsonl"),
            str(meta_dir / "tasks.jsonl"),
        ] + [str(chunk_dir / f"episode_{i:06d}.json") for i in range(len(episodes))],
    }

    print(f"Exported {total_episodes} episodes, {len(tasks_map)} tasks to {out}", file=sys.stderr)
    return result


def main():
    parser = argparse.ArgumentParser(description="Convert auto-label JSONL to LeRobot format")
    parser.add_argument("input", help="Input JSONL file from auto_label_pipeline")
    parser.add_argument("--output-dir", required=True, help="Output directory for LeRobot dataset")
    parser.add_argument("--robot-type", default="single_arm", help="Robot type (default: single_arm)")
    parser.add_argument("--fps", type=int, default=30, help="Default FPS if not in data")

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    episodes = read_jsonl(args.input)
    if not episodes:
        print("ERROR: No valid episodes found in input file", file=sys.stderr)
        sys.exit(1)

    result = export_lerobot(
        episodes,
        args.output_dir,
        robot_type=args.robot_type,
        fps=args.fps,
    )

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
