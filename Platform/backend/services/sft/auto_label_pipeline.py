"""
Auto Label Pipeline
===================
4阶段VLM自动标注流水线。
从 demo_screw_to_pi05_sft.py 提取核心逻辑。

Stage 1: 运动自适应帧采样 + 阶段分割
Stage 2: 动作原语标注
Stage 3: 接触力学估计
Stage 4: 任务摘要生成
"""

from __future__ import annotations

import base64
import json
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 可选依赖
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# 词表常量
# ═══════════════════════════════════════════════════════════════════════════════

ACTION_PRIMITIVES = [
    "approach", "align", "grasp", "lift", "move",
    "rotate_cw", "rotate_ccw", "insert", "push", "pull",
    "place", "release", "inspect", "wait", "retract",
]
GRIPPER_STATES = ["open", "closing", "closed", "opening"]
FORCE_LEVELS = ["none", "light", "medium", "strong"]
CONTACT_TYPES = ["none", "point", "surface", "edge", "wrap"]
MOTION_DIRS = ["linear", "rotational", "complex"]


# ═══════════════════════════════════════════════════════════════════════════════
# VLM 后端
# ═══════════════════════════════════════════════════════════════════════════════

class VLMBackend(ABC):
    """VLM 后端抽象基类"""

    @abstractmethod
    def query(self, prompt: str, images_b64: List[str]) -> str:
        pass

    def parse_json(self, text: str) -> Any:
        """从VLM输出中解析JSON"""
        text = text.replace("\\_", "_").replace("\\*", "*")
        for attempt in [
            lambda t: json.loads(t),
            lambda t: json.loads(re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", t).group(1)) if re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", t) else None,
            lambda t: json.loads(re.search(r"\[[\s\S]*\]", t).group(0)) if re.search(r"\[[\s\S]*\]", t) else None,
            lambda t: json.loads(re.search(r"\{[\s\S]*\}", t).group(0)) if re.search(r"\{[\s\S]*\}", t) else None,
        ]:
            try:
                result = attempt(text)
                if result is not None:
                    return result
            except Exception:
                pass
        return None


class OllamaBackend(VLMBackend):
    """Ollama 本地 VLM 后端"""

    def __init__(self, model: str, url: str = "http://localhost:11434"):
        self.model = model
        self.url = url.rstrip("/")

    def query(self, prompt: str, images_b64: List[str]) -> str:
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("pip install requests")
        msg = {"role": "user", "content": prompt}
        if images_b64:
            msg["images"] = images_b64
        try:
            resp = requests.post(
                f"{self.url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [msg],
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 4096, "num_ctx": 8192},
                },
                timeout=600,
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot reach Ollama at {self.url}")


class GeminiBackend(VLMBackend):
    """Google Gemini API 后端"""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model

    def query(self, prompt: str, images_b64: List[str]) -> str:
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("pip install requests")
        parts = []
        for img_b64 in images_b64:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})
        parts.append({"text": prompt})

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
        }
        resp = requests.post(url, json=body, timeout=120)
        resp.raise_for_status()
        candidates = resp.json().get("candidates", [])
        return candidates[0]["content"]["parts"][0].get("text", "") if candidates else ""


class MockVLMBackend(VLMBackend):
    """Mock VLM 用于测试（无需GPU/API）"""

    SCREW_PHASES = [
        {"phase_name": "approach_screw", "start_frame_index": 1, "end_frame_index": 3,
         "description": "Robot arm moves toward target screw, gripper open"},
        {"phase_name": "align_socket", "start_frame_index": 4, "end_frame_index": 5,
         "description": "Fine alignment of screw driver socket with screw head"},
        {"phase_name": "insert_socket", "start_frame_index": 6, "end_frame_index": 8,
         "description": "Socket lowered onto screw head, light contact"},
        {"phase_name": "drive_screw_cw", "start_frame_index": 9, "end_frame_index": 13,
         "description": "Screw driven clockwise to full torque"},
        {"phase_name": "retract", "start_frame_index": 14, "end_frame_index": 16,
         "description": "Robot retracts after successful screw installation"},
    ]

    _primitive_map = {
        "approach_screw": ("approach", "screw", "open", 0.95),
        "align_socket": ("align", "screw", "closing", 0.90),
        "insert_socket": ("insert", "screw", "closed", 0.92),
        "drive_screw_cw": ("rotate_cw", "screw", "closed", 0.97),
        "retract": ("retract", "none", "opening", 0.88),
    }

    _mechanics_map = {
        "approach_screw": ("none", "none", "", "linear"),
        "align_socket": ("point", "light", "screw head center", "linear"),
        "insert_socket": ("point", "medium", "screw head socket", "linear"),
        "drive_screw_cw": ("surface", "strong", "thread engagement", "rotational"),
        "retract": ("none", "none", "", "linear"),
    }

    def query(self, prompt: str, images_b64: List[str]) -> str:
        if "segment" in prompt.lower() or "temporal phases" in prompt.lower():
            return json.dumps(self.SCREW_PHASES)
        if "action primitive" in prompt.lower():
            for phase_name, (ap, obj, gs, conf) in self._primitive_map.items():
                if phase_name in prompt:
                    return json.dumps({"action_primitive": ap, "target_object": obj, "gripper_state": gs, "confidence": conf})
            return json.dumps({"action_primitive": "wait", "target_object": "unknown", "gripper_state": "open", "confidence": 0.5})
        if "contact_type" in prompt.lower() or "force" in prompt.lower():
            for phase_name, (ct, fl, cp, md) in self._mechanics_map.items():
                if phase_name in prompt:
                    return json.dumps({"contact_type": ct, "force_level": fl, "contact_points": cp, "motion_direction": md})
            return json.dumps({"contact_type": "none", "force_level": "none", "contact_points": "", "motion_direction": "linear"})
        if "summariz" in prompt.lower():
            return "Drive the screw clockwise into the mounting bracket."
        return "Mock response."


# ═══════════════════════════════════════════════════════════════════════════════
# 帧提取
# ═══════════════════════════════════════════════════════════════════════════════

def _encode_frame(frame_bgr, max_side: int = 512) -> str:
    """将BGR帧编码为base64"""
    h, w = frame_bgr.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode("ascii")


def extract_frames(
    video_path: str,
    num_frames: int,
    start_frame: int = 0,
    end_frame: int = -1,
) -> Tuple[List[str], List[dict], dict]:
    """均匀帧提取"""
    if not CV2_AVAILABLE or not NP_AVAILABLE:
        raise RuntimeError("pip install opencv-python numpy")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if end_frame < 0:
        end_frame = total - 1
    end_frame = min(end_frame, total - 1)

    indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    frames, metas = [], []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(_encode_frame(frame))
            metas.append({"frame_idx": int(idx), "timestamp": round(int(idx) / fps, 3)})
    cap.release()

    return frames, metas, {
        "duration": round(total / fps, 2),
        "fps": round(fps, 2),
        "resolution": [width, height],
        "total_frames": total,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4阶段自动标注流水线
# ═══════════════════════════════════════════════════════════════════════════════

class AutoLabelPipeline:
    """4阶段VLM自动标注流水线"""

    def __init__(
        self,
        vlm: VLMBackend,
        num_frames: int = 16,
        adaptive: bool = True,
        max_vlm_frames: int = 24,
        motion_threshold: float = 0.02,
    ):
        self.vlm = vlm
        self.num_frames = num_frames
        self.adaptive = adaptive
        self.max_vlm_frames = max_vlm_frames
        self.motion_threshold = motion_threshold

    def stage1_phase_segmentation(self, video_path: str, dry_run: bool = False) -> Tuple[List[dict], dict]:
        """Stage 1: 阶段分割"""
        print("[Stage 1/4] Phase segmentation...", file=sys.stderr)

        if dry_run or not CV2_AVAILABLE:
            frames = [""] * self.num_frames
            metas = [{"frame_idx": i * 5, "timestamp": round(i * 5 / 30, 3)} for i in range(self.num_frames)]
            video_info = {"duration": 10.0, "fps": 30.0, "resolution": [640, 480], "total_frames": 300}
        else:
            frames, metas, video_info = extract_frames(video_path, self.num_frames)

        # 逐帧描述
        frame_descs = []
        for i, (fr, meta) in enumerate(zip(frames, metas)):
            desc = self.vlm.query(
                f"Frame {i+1}/{len(frames)} ({meta['timestamp']:.1f}s). Describe robot arm action in one sentence.",
                [fr] if fr else []
            ).strip()
            frame_descs.append(f"Frame {i+1} ({meta['timestamp']:.1f}s): {desc}")

        # 分割
        seg_prompt = (
            f"Robot video: {video_info['total_frames']} frames at {video_info['fps']} FPS.\n\n"
            f"Per-frame descriptions:\n{chr(10).join(frame_descs)}\n\n"
            "Segment into 2-5 phases. Return JSON array: [{phase_name, start_frame_index, end_frame_index, description}]"
        )
        raw = self.vlm.query(seg_prompt, [])
        phases_raw = self.vlm.parse_json(raw) or [{

            "phase_name": "full_episode", "start_frame_index": 1, "end_frame_index": len(metas),
            "description": "Complete manipulation episode"
        }]

        phases = []
        for i, p in enumerate(phases_raw if isinstance(phases_raw, list) else [phases_raw]):
            si = max(0, min(int(p.get("start_frame_index", 1)) - 1, len(metas) - 1))
            ei = max(0, min(int(p.get("end_frame_index", len(metas))) - 1, len(metas) - 1))
            phases.append({
                "phase_idx": i,
                "phase_name": str(p.get("phase_name", f"phase_{i}")),
                "start_frame": metas[si]["frame_idx"],
                "end_frame": metas[ei]["frame_idx"],
                "start_time": metas[si]["timestamp"],
                "end_time": metas[ei]["timestamp"],
                "description": str(p.get("description", "")),
            })

        return phases, video_info

    def stage2_action_primitives(self, video_path: str, phases: List[dict], dry_run: bool = False) -> List[dict]:
        """Stage 2: 动作原语标注"""
        print(f"[Stage 2/4] Action primitives ({len(phases)} phases)...", file=sys.stderr)
        vocab = ", ".join(ACTION_PRIMITIVES)

        for phase in phases:
            if not dry_run and CV2_AVAILABLE:
                frames, _, _ = extract_frames(video_path, 4, phase["start_frame"], phase["end_frame"])
            else:
                frames = []

            prompt = (
                f"Time: {phase['start_time']:.1f}-{phase['end_time']:.1f}s. Phase: {phase['description']}\n"
                f"Choose action from: {vocab}\n"
                'Return JSON: {"action_primitive": "<one>", "target_object": "<object>", "gripper_state": "<open|closed>", "confidence": 0.9}'
            )
            raw = self.vlm.query(prompt, frames)
            data = self.vlm.parse_json(raw) or {}
            if isinstance(data, list):
                data = data[0] if data else {}

            phase["action_primitive"] = data.get("action_primitive", "wait")
            phase["target_object"] = data.get("target_object", "unknown")
            phase["gripper_state"] = data.get("gripper_state", "open")
            phase["confidence"] = float(data.get("confidence", 0.5))

        return phases

    def stage3_mechanics(self, video_path: str, phases: List[dict], dry_run: bool = False) -> List[dict]:
        """Stage 3: 接触力学估计"""
        print(f"[Stage 3/4] Contact mechanics ({len(phases)} phases)...", file=sys.stderr)

        for phase in phases:
            if not dry_run and CV2_AVAILABLE:
                frames, _, _ = extract_frames(video_path, 4, phase["start_frame"], phase["end_frame"])
            else:
                frames = []

            prompt = (
                f"Action: {phase.get('action_primitive', '')} on {phase.get('target_object', '')}\n"
                'Return JSON: {"contact_type": "<none|point|surface|edge|wrap>", "force_level": "<none|light|medium|strong>", '
                '"contact_points": "<location>", "motion_direction": "<linear|rotational|complex>"}'
            )
            raw = self.vlm.query(prompt, frames)
            data = self.vlm.parse_json(raw) or {}

            phase["mechanics"] = {
                "contact_type": data.get("contact_type", "none"),
                "force_level": data.get("force_level", "none"),
                "contact_points": data.get("contact_points", ""),
                "motion_direction": data.get("motion_direction", "linear"),
            }

        return phases

    def stage4_task_summary(self, video_path: str, phases: List[dict], dry_run: bool = False) -> str:
        """Stage 4: 任务摘要"""
        print("[Stage 4/4] Task summary...", file=sys.stderr)

        if not dry_run and CV2_AVAILABLE:
            frames, _, _ = extract_frames(video_path, 4)
        else:
            frames = []

        seq = " → ".join(f"{p['action_primitive']}({p.get('target_object', '?')})" for p in phases)
        prompt = f"Robot action sequence: {seq}\nSummarize the task in one sentence."
        summary = self.vlm.query(prompt, frames).strip().strip('"').strip("'")
        return summary or "Robot manipulation task"

    def run(self, video_path: str, dry_run: bool = False) -> dict:
        """运行完整流水线"""
        t0 = time.time()

        phases, video_info = self.stage1_phase_segmentation(video_path, dry_run)
        phases = self.stage2_action_primitives(video_path, phases, dry_run)
        phases = self.stage3_mechanics(video_path, phases, dry_run)
        task_summary = self.stage4_task_summary(video_path, phases, dry_run)

        return {
            "episode_id": Path(video_path).stem,
            "video_path": str(Path(video_path).resolve()),
            "video_info": video_info,
            "phases": phases,
            "task_summary": task_summary,
            "success": True,
            "vlm_backend": type(self.vlm).__name__,
            "labeled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_sec": round(time.time() - t0, 2),
        }
