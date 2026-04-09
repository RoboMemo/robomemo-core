"""
LeRobot Exporter
================
将VLM标注结果导出为LeRobot V2.1格式。

输出结构:
  lerobot/
  ├── meta/
  │   ├── info.json         # 数据集元信息
  │   ├── episodes.jsonl    # Episode索引
  │   └── tasks.jsonl       # 任务描述
  └── data/
      └── chunk-000/
          └── episode_000000.json
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any


LEROBOT_VERSION = "2.1"


def export_lerobot(
    episodes: List[dict],
    output_dir: str,
    robot_type: str = "single_arm",
    fps: int = 30,
) -> Dict[str, Any]:
    """
    将自动标注结果导出为LeRobot V2.1格式。
    
    Args:
        episodes: 标注结果列表（来自AutoLabelPipeline.run()）
        output_dir: 输出目录
        robot_type: 机器人类型
        fps: 帧率
        
    Returns:
        导出结果摘要
    """
    out = Path(output_dir)
    meta_dir = out / "meta"
    data_dir = out / "data"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLeRobot V2 Exporter → {out}", file=__import__('sys').stderr)

    # 收集任务
    tasks_map: Dict[str, int] = {}
    for ep in episodes:
        summary = ep.get("task_summary", "unknown task")
        if summary not in tasks_map:
            tasks_map[summary] = len(tasks_map)

    # ── meta/info.json ───────────────────────────────────────────────────────
    total_frames = sum(ep.get("video_info", {}).get("total_frames", 0) for ep in episodes)
    ep_fps = episodes[0].get("video_info", {}).get("fps", fps) if episodes else fps

    info = {
        "codebase_version": LEROBOT_VERSION,
        "robot_type": robot_type,
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "fps": ep_fps,
        "total_tasks": len(tasks_map),
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.json",
        "features": {
            "observation.task_description": {"dtype": "string", "shape": [1], "description": "Task instruction for VLA model"},
            "action_primitive": {"dtype": "string", "shape": [1], "description": "Action primitive from VLM"},
            "target_object": {"dtype": "string", "shape": [1], "description": "Target object"},
            "gripper_state": {"dtype": "string", "shape": [1], "description": "Gripper state"},
            "contact_type": {"dtype": "string", "shape": [1], "description": "Contact type"},
            "force_level": {"dtype": "string", "shape": [1], "description": "Force level"},
            "motion_direction": {"dtype": "string", "shape": [1], "description": "Motion direction"},
            "phase_name": {"dtype": "string", "shape": [1], "description": "Phase label"},
            "confidence": {"dtype": "float32", "shape": [1], "description": "VLM confidence"},
        },
        "vlm_backend": episodes[0].get("vlm_backend", "unknown") if episodes else "unknown",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    
    with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print("  ✓ meta/info.json", file=__import__('sys').stderr)

    # ── meta/episodes.jsonl ───────────────────────────────────────────────────
    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for i, ep in enumerate(episodes):
            vi = ep.get("video_info", {})
            f.write(json.dumps({
                "episode_index": i,
                "episode_id": ep.get("episode_id", f"episode_{i}"),
                "tasks": [tasks_map.get(ep.get("task_summary", ""), 0)],
                "length": vi.get("total_frames", 0),
                "duration": vi.get("duration", 0),
                "video_path": ep.get("video_path", ""),
                "num_phases": len(ep.get("phases", [])),
                "success": ep.get("success", True),
                "elapsed_sec": ep.get("elapsed_sec", 0),
            }, ensure_ascii=False) + "\n")
    print("  ✓ meta/episodes.jsonl", file=__import__('sys').stderr)

    # ── meta/tasks.jsonl ──────────────────────────────────────────────────────
    with open(meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        for task_desc, task_id in sorted(tasks_map.items(), key=lambda x: x[1]):
            skill_labels = set()
            for ep in episodes:
                if ep.get("task_summary") == task_desc:
                    for phase in ep.get("phases", []):
                        skill_labels.add(phase.get("action_primitive", "unknown"))
            f.write(json.dumps({
                "task_index": task_id,
                "task": task_desc,
                "skill_labels": sorted(skill_labels),
            }, ensure_ascii=False) + "\n")
    print("  ✓ meta/tasks.jsonl", file=__import__('sys').stderr)

    # ── data/chunk-000/episode_XXXXXX.json ────────────────────────────────────
    chunk_dir = data_dir / "chunk-000"
    chunk_dir.mkdir(exist_ok=True)

    for i, ep in enumerate(episodes):
        rows = []
        task_desc = ep.get("task_summary", "unknown task")
        for phase in ep.get("phases", []):
            mech = phase.get("mechanics", {})
            rows.append({
                "episode_index": i,
                "observation.task_description": task_desc,
                "phase_index": phase.get("phase_idx", 0),
                "phase_name": phase.get("phase_name", ""),
                "start_frame": phase.get("start_frame", 0),
                "end_frame": phase.get("end_frame", 0),
                "start_time": phase.get("start_time", 0),
                "end_time": phase.get("end_time", 0),
                "action_primitive": phase.get("action_primitive", "wait"),
                "target_object": phase.get("target_object", "unknown"),
                "gripper_state": phase.get("gripper_state", "open"),
                "confidence": phase.get("confidence", 0.0),
                "contact_type": mech.get("contact_type", "none"),
                "force_level": mech.get("force_level", "none"),
                "contact_points": mech.get("contact_points", ""),
                "motion_direction": mech.get("motion_direction", "linear"),
            })
        
        ep_file = chunk_dir / f"episode_{i:06d}.json"
        with open(ep_file, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"  ✓ data/chunk-000/ ({len(episodes)} episodes)", file=__import__('sys').stderr)

    return {
        "output_dir": str(out),
        "total_episodes": len(episodes),
        "total_tasks": len(tasks_map),
        "lerobot_version": LEROBOT_VERSION,
    }
