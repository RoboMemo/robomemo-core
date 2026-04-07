#!/usr/bin/env python3
# Copyright (c) 2026, RoboMemo Project. All rights reserved.
"""
demo_screw_to_pi05_sft.py
=========================
End-to-end demo: Screw-tightening video  →  π₀.5 SFT-ready annotated dataset

Merged from three branches:
  • dev-VQApipeline-siyu   — AutoLabelPipeline (4-stage VLM annotation)
  • dev-VQApipeline-siyu   — LeRobotExporter   (LeRobot V2 format)
  • merge-local-features   — OpenPIFinetuneCfg (π₀ / π₀.5 SFT config)

Pipeline:
  Input video
    │
    ▼  Stage 1: Motion-adaptive frame sampling + Phase segmentation (VLM)
    ▼  Stage 2: Action primitive labeling     (VLM, per phase)
    ▼  Stage 3: Contact mechanics estimation  (VLM, per phase)
    ▼  Stage 4: Task summary generation       (VLM)
    │
    ▼  Export → LeRobot V2 format
    │           ├── meta/info.json
    │           ├── meta/episodes.jsonl
    │           ├── meta/tasks.jsonl
    │           └── data/chunk-000/episode_000000.json
    │
    ▼  Generate → π₀.5 SFT training config
                  └── configs/openpi_finetune.json

Usage:
    # Minimal (Ollama must be running locally):
    python demo_screw_to_pi05_sft.py --video screw_demo.mp4

    # With Gemini (recommended for quality):
    python demo_screw_to_pi05_sft.py --video screw_demo.mp4 --vlm gemini --gemini-key $GEMINI_API_KEY

    # Dry-run (mock VLM, no GPU/API needed — for pipeline testing):
    python demo_screw_to_pi05_sft.py --video screw_demo.mp4 --dry-run

    # Full options:
    python demo_screw_to_pi05_sft.py \\
        --video        screw_demo.mp4 \\
        --output-dir   ./sft_output \\
        --vlm          ollama \\
        --model        scomper/minicpm-v2.5:latest \\
        --ollama-url   http://localhost:11434 \\
        --robot-type   single_arm \\
        --pi05
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Optional heavy imports — graceful fallback
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary constants (shared across stages)
# ──────────────────────────────────────────────────────────────────────────────
ACTION_PRIMITIVES = [
    "approach", "align", "grasp", "lift", "move",
    "rotate_cw", "rotate_ccw", "insert", "push", "pull",
    "place", "release", "inspect", "wait", "retract",
]
GRIPPER_STATES  = ["open", "closing", "closed", "opening"]
FORCE_LEVELS    = ["none", "light", "medium", "strong"]
CONTACT_TYPES   = ["none", "point", "surface", "edge", "wrap"]
MOTION_DIRS     = ["linear", "rotational", "complex"]
LEROBOT_VERSION = "2.1"

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — VLM BACKENDS
# (from dev-VQApipeline-siyu :: auto_label_pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

class VLMBackend:
    """Abstract VLM backend."""

    def query(self, prompt: str, images_b64: List[str]) -> str:
        raise NotImplementedError

    def parse_json(self, text: str) -> Any:
        text = text.replace("\\_", "_").replace("\\*", "*")
        for attempt in [
            lambda t: json.loads(t),
            lambda t: json.loads(re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", t).group(1)),
            lambda t: json.loads(re.search(r"\[[\s\S]*\]", t).group(0)),
            lambda t: json.loads(re.search(r"\{[\s\S]*\}", t).group(0)),
        ]:
            try:
                return attempt(text)
            except Exception:
                pass
        _log(f"[WARN] Could not parse JSON: {text[:200]}")
        return None


class OllamaBackend(VLMBackend):
    """Ollama local VLM backend (minicpm-v, llava, qwen2-vl, etc.)."""

    def __init__(self, model: str, url: str):
        self.model = model
        self.url = url.rstrip("/")

    def query(self, prompt: str, images_b64: List[str]) -> str:
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("pip install requests")
        msg: dict = {"role": "user", "content": prompt}
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
            raise RuntimeError(
                f"Cannot reach Ollama at {self.url}. Start it with: ollama serve"
            )


class GeminiBackend(VLMBackend):
    """Google Gemini API backend (gemini-1.5-flash / gemini-2.0-flash)."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model

    def query(self, prompt: str, images_b64: List[str]) -> str:
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("pip install requests")
        parts: List[dict] = []
        for img_b64 in images_b64:
            parts.append({
                "inline_data": {"mime_type": "image/jpeg", "data": img_b64}
            })
        parts.append({"text": prompt})

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
        }
        resp = requests.post(url, json=body, timeout=120)
        resp.raise_for_status()
        candidates = resp.json().get("candidates", [])
        if not candidates:
            return ""
        return candidates[0]["content"]["parts"][0].get("text", "")


class HybridOllamaBackend(VLMBackend):
    """Hybrid backend: vision queries → vision_model, text-only → reasoning_model.

    Best combination: minicpm-v (vision) + qwen3:32b (reasoning)
    The reasoning model handles Stage 1b phase segmentation and Stage 4 summary
    where stronger language understanding matters more than vision.
    """

    def __init__(self, vision_model: str, reasoning_model: str, url: str):
        self.vision   = OllamaBackend(vision_model, url)
        self.reasoner = OllamaBackend(reasoning_model, url)
        self.vision_model    = vision_model
        self.reasoning_model = reasoning_model

    def query(self, prompt: str, images_b64: List[str]) -> str:
        if images_b64:
            return self.vision.query(prompt, images_b64)
        else:
            return self.reasoner.query(prompt, [])

    def parse_json(self, text: str) -> Any:
        return self.vision.parse_json(text)


class MockVLMBackend(VLMBackend):
    """Mock VLM for dry-run / CI testing — no GPU or API key needed."""

    SCREW_PHASES = [
        {"phase_name": "approach_screw",    "start_frame_index": 1,  "end_frame_index": 3,
         "description": "Robot arm moves toward target screw, gripper open"},
        {"phase_name": "align_socket",      "start_frame_index": 4,  "end_frame_index": 5,
         "description": "Fine alignment of screw driver socket with screw head"},
        {"phase_name": "insert_socket",     "start_frame_index": 6,  "end_frame_index": 8,
         "description": "Socket lowered onto screw head, light contact"},
        {"phase_name": "drive_screw_cw",    "start_frame_index": 9,  "end_frame_index": 13,
         "description": "Screw driven clockwise to full torque"},
        {"phase_name": "retract",           "start_frame_index": 14, "end_frame_index": 16,
         "description": "Robot retracts after successful screw installation"},
    ]

    _primitive_map = {
        "approach_screw":  ("approach",   "screw",   "open",    0.95),
        "align_socket":    ("align",      "screw",   "closing", 0.90),
        "insert_socket":   ("insert",     "screw",   "closed",  0.92),
        "drive_screw_cw":  ("rotate_cw",  "screw",   "closed",  0.97),
        "retract":         ("retract",    "none",    "opening", 0.88),
    }

    _mechanics_map = {
        "approach_screw":  ("none",    "none",   "",                  "linear"),
        "align_socket":    ("point",   "light",  "screw head center", "linear"),
        "insert_socket":   ("point",   "medium", "screw head socket", "linear"),
        "drive_screw_cw":  ("surface", "strong", "thread engagement", "rotational"),
        "retract":         ("none",    "none",   "",                  "linear"),
    }

    def query(self, prompt: str, images_b64: List[str]) -> str:
        # Stage 1: phase list
        if "segment" in prompt.lower() or "temporal phases" in prompt.lower():
            return json.dumps(self.SCREW_PHASES)
        # Stage 1a: per-frame description
        if "one sentence" in prompt.lower() and "robot arm" in prompt.lower():
            return "The robot arm is approaching the screw with the gripper open."
        # Stage 2: action primitives
        if "action primitive" in prompt.lower():
            for phase_name, (ap, obj, gs, conf) in self._primitive_map.items():
                if phase_name in prompt:
                    return json.dumps({
                        "action_primitive": ap, "target_object": obj,
                        "gripper_state": gs, "confidence": conf,
                    })
            return json.dumps({
                "action_primitive": "wait", "target_object": "unknown",
                "gripper_state": "open", "confidence": 0.5,
            })
        # Stage 3: mechanics
        if "contact_type" in prompt.lower() or "force" in prompt.lower():
            for phase_name, (ct, fl, cp, md) in self._mechanics_map.items():
                if phase_name in prompt:
                    return json.dumps({
                        "contact_type": ct, "force_level": fl,
                        "contact_points": cp, "motion_direction": md,
                    })
            return json.dumps({
                "contact_type": "none", "force_level": "none",
                "contact_points": "", "motion_direction": "linear",
            })
        # Stage 4: task summary
        if "summariz" in prompt.lower() or "overall manipulation" in prompt.lower():
            return "Drive the screw clockwise into the solar panel mounting bracket."
        return "Mock response."


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — AUTO-LABEL PIPELINE (4 stages)
# (from dev-VQApipeline-siyu :: auto_label_pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def _encode_frame(frame_bgr, max_side: int = 512) -> str:
    """Resize and JPEG-encode a BGR frame to base64."""
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
    """Uniform frame extraction. Returns (frames_b64, metas, video_info)."""
    if not CV2_AVAILABLE or not NP_AVAILABLE:
        raise RuntimeError("pip install opencv-python numpy")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if end_frame < 0:
        end_frame = total - 1
    end_frame = min(end_frame, total - 1)

    indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    frames, metas = [], []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(_encode_frame(frame))
        metas.append({"frame_idx": int(idx), "timestamp": round(int(idx) / fps, 3)})
    cap.release()

    video_info = {
        "duration": round(total / fps, 2), "fps": round(fps, 2),
        "resolution": [width, height], "total_frames": total,
    }
    return frames, metas, video_info


def extract_adaptive_frames(
    video_path: str,
    max_vlm_frames: int = 24,
    motion_threshold: float = 0.02,
) -> Tuple[List[str], List[dict], dict]:
    """Motion-adaptive frame extraction (from dev-autolabel_rf_siyu).

    2-pass algorithm:
      Pass 1 — compute per-frame motion scores via frame differencing.
      Pass 2 — dense-sample around detected action boundaries.
    """
    if not CV2_AVAILABLE or not NP_AVAILABLE:
        raise RuntimeError("pip install opencv-python numpy")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_info = {
        "duration": round(total / fps, 2), "fps": round(fps, 2),
        "resolution": [width, height], "total_frames": total,
    }

    # ── Pass 1: motion scores ─────────────────────────────────────────
    subsample = max(1, int(fps / 5))
    raw_scores: List[Tuple[int, float]] = []
    prev_gray = None
    for fi in range(0, total, subsample):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120))
        score = float(np.mean(cv2.absdiff(gray, prev_gray)) / 255.0) if prev_gray is not None else 0.0
        raw_scores.append((fi, score))
        prev_gray = gray

    if len(raw_scores) < 2:
        cap.release()
        return extract_frames(video_path, max_vlm_frames)

    idxs   = np.array([s[0] for s in raw_scores])
    scores = np.array([s[1] for s in raw_scores])
    k = min(5, len(scores))
    smooth = np.convolve(scores, np.ones(k) / k, mode="same")

    # Peaks + plateaus → key_indices
    key: List[int] = []
    for i in range(1, len(smooth) - 1):
        if smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1] and smooth[i] >= motion_threshold:
            key.append(i)
    in_p, p0 = False, 0
    for i, s in enumerate(smooth):
        if s >= motion_threshold and not in_p:
            in_p, p0 = True, i
        elif s < motion_threshold and in_p:
            in_p = False
            m = (p0 + i) // 2
            if m not in key:
                key.append(m)
    if in_p:
        m = (p0 + len(smooth) - 1) // 2
        if m not in key:
            key.append(m)

    if 0 not in key:
        key.insert(0, 0)
    if (len(smooth) - 1) not in key:
        key.append(len(smooth) - 1)
    key = sorted(set(key))

    _log(f"  [Adaptive] {len(key)} action boundaries detected in {total} frames")

    # ── Pass 2: dense sample around boundaries ────────────────────────
    ctx = 3
    cands: set = set()
    for ki in key:
        for off in range(-ctx, ctx + 1):
            ci = ki + off
            if 0 <= ci < len(idxs):
                cands.add(ci)
    cands_sorted = sorted(cands)

    if len(cands_sorted) > max_vlm_frames:
        scored = [(c, smooth[c]) for c in cands_sorted]
        first, last = scored[0], scored[-1]
        mid = sorted(scored[1:-1], key=lambda x: x[1], reverse=True)
        cands_sorted = sorted({first[0]} | {s[0] for s in mid[:max_vlm_frames - 2]} | {last[0]})

    sel_fi = [int(idxs[c]) for c in cands_sorted]
    sel_sc = [float(smooth[c]) for c in cands_sorted]

    frames_b64, metas = [], []
    for fi, sc in zip(sel_fi, sel_sc):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        frames_b64.append(_encode_frame(frame))
        metas.append({"frame_idx": fi, "timestamp": round(fi / fps, 3), "motion_score": round(sc, 4)})
    cap.release()

    _log(f"  [Adaptive] Selected {len(frames_b64)} frames for VLM")
    return frames_b64, metas, video_info


class AutoLabelPipeline:
    """
    4-stage VLM annotation pipeline.
    Merged from: dev-VQApipeline-siyu :: auto_label_pipeline.py
    """

    def __init__(
        self,
        vlm: VLMBackend,
        num_frames: int = 16,
        adaptive: bool = True,
        max_vlm_frames: int = 24,
        motion_threshold: float = 0.02,
    ):
        self.vlm               = vlm
        self.num_frames        = num_frames
        self.adaptive          = adaptive
        self.max_vlm_frames    = max_vlm_frames
        self.motion_threshold  = motion_threshold

    # ── Stage 1: Phase Segmentation ──────────────────────────────────

    def stage1_phase_segmentation(
        self, video_path: str, dry_run: bool = False
    ) -> Tuple[List[dict], dict]:
        _log("[Stage 1/4] Phase segmentation...")

        if dry_run or not CV2_AVAILABLE:
            # Mock frames for dry-run
            frames, metas = [], []
            for i in range(self.num_frames):
                frames.append("")  # empty b64
                metas.append({"frame_idx": i * 5, "timestamp": round(i * 5 / 30, 3)})
            video_info = {"duration": 10.0, "fps": 30.0,
                          "resolution": [640, 480], "total_frames": 300}
        elif self.adaptive:
            frames, metas, video_info = extract_adaptive_frames(
                video_path, self.max_vlm_frames, self.motion_threshold
            )
        else:
            frames, metas, video_info = extract_frames(video_path, self.num_frames)

        _log(f"  Extracted {len(frames)} frames ({video_info['duration']:.1f}s)")

        # Step 1a: describe each frame individually (prevents hallucination)
        frame_descs = []
        for i, (fr, meta) in enumerate(zip(frames, metas)):
            ms = meta.get("motion_score")
            motion_ctx = ""
            if ms is not None:
                if ms > 0.10:
                    motion_ctx = " (high motion — action transition)"
                elif ms > 0.03:
                    motion_ctx = " (moderate motion — active manipulation)"
                else:
                    motion_ctx = " (low motion — stable pose)"

            desc_prompt = (
                f"This is frame {i+1}/{len(frames)} from a robot manipulation video "
                f"(timestamp {meta['timestamp']:.1f}s){motion_ctx}. "
                "In one sentence, describe ONLY what the robot arm and gripper are doing RIGHT NOW. "
                "Mention arm position, gripper state (open/closing/closed), and any object contact."
            )
            desc = self.vlm.query(desc_prompt, [fr] if fr else []).strip()
            frame_descs.append(f"Frame {i+1} ({meta['timestamp']:.1f}s): {desc}")
            _log(f"    Frame {i+1:2d}: {desc[:70]}")

        # Step 1b: text-only segmentation
        combined = "\n".join(frame_descs)
        seg_prompt = (
            f"A robot manipulation video: {video_info['total_frames']} frames "
            f"at {video_info['fps']} FPS ({video_info['duration']:.1f}s total).\n\n"
            f"Per-frame descriptions:\n{combined}\n\n"
            "Segment into 2–5 distinct temporal phases. "
            "Return a JSON array where each object has:\n"
            "  phase_name (snake_case), start_frame_index (1-based), "
            "  end_frame_index (1-based), description (one sentence).\n"
            "Return ONLY the JSON array."
        )
        raw = self.vlm.query(seg_prompt, [])
        phases_raw = self.vlm.parse_json(raw)

        if not isinstance(phases_raw, list) or not phases_raw:
            _log("  [WARN] Phase segmentation fallback to single phase")
            phases_raw = [{
                "phase_name": "full_episode", "start_frame_index": 1,
                "end_frame_index": len(metas),
                "description": "Complete manipulation episode",
            }]

        phases = []
        for i, p in enumerate(phases_raw):
            si = max(0, min(int(p.get("start_frame_index", 1)) - 1, len(metas) - 1))
            ei = max(0, min(int(p.get("end_frame_index", len(metas))) - 1, len(metas) - 1))
            if ei < si:
                ei = si
            phases.append({
                "phase_idx": i,
                "phase_name": str(p.get("phase_name", f"phase_{i}")),
                "start_frame": metas[si]["frame_idx"],
                "end_frame":   metas[ei]["frame_idx"],
                "start_time":  metas[si]["timestamp"],
                "end_time":    metas[ei]["timestamp"],
                "description": str(p.get("description", "")),
            })

        _log(f"  → {len(phases)} phases identified")
        return phases, video_info

    # ── Stage 2: Action Primitive Labeling ───────────────────────────

    def stage2_action_primitives(
        self, video_path: str, phases: List[dict], dry_run: bool = False
    ) -> List[dict]:
        _log(f"[Stage 2/4] Action primitive labeling ({len(phases)} phases)...")
        vocab = ", ".join(ACTION_PRIMITIVES)

        for phase in phases:
            if not dry_run and CV2_AVAILABLE:
                frames, _, _ = extract_frames(
                    video_path, 4, phase["start_frame"], phase["end_frame"]
                )
            else:
                frames = []

            prompt = (
                f"Look at these robot arm images. Time: {phase['start_time']:.1f}–{phase['end_time']:.1f}s.\n"
                f"Phase description: {phase['description']}\n\n"
                f"Choose ONE action from this list:\n{vocab}\n\n"
                "Answer in JSON only, no explanation:\n"
                '{"action_primitive": "<one from list above>", '
                '"target_object": "<object being touched or near>", '
                '"gripper_state": "<open|closing|closed|opening>", '
                '"confidence": <0.0-1.0>}'
            )
            raw  = self.vlm.query(prompt, frames)
            parsed = self.vlm.parse_json(raw)
            # Normalise: VLM may return a list wrapping the dict
            if isinstance(parsed, list):
                data = parsed[0] if parsed and isinstance(parsed[0], dict) else {}
            elif isinstance(parsed, dict):
                data = parsed
            else:
                data = {}

            ap = str(data.get("action_primitive", "wait")).lower()
            if ap not in ACTION_PRIMITIVES:
                ap = "wait"
            gs = str(data.get("gripper_state", "open")).lower()
            if gs not in GRIPPER_STATES:
                gs = "open"

            phase["action_primitive"] = ap
            phase["target_object"]    = str(data.get("target_object", "unknown"))
            phase["gripper_state"]    = gs
            # confidence: VLM may return "low"/"high"/float — normalise robustly
            raw_conf = data.get("confidence", 0.5)
            conf_map = {"low": 0.3, "medium": 0.6, "high": 0.9, "very high": 0.95}
            if isinstance(raw_conf, str):
                try:
                    raw_conf = float(raw_conf)
                except ValueError:
                    raw_conf = conf_map.get(raw_conf.lower().strip(), 0.5)
            phase["confidence"] = float(raw_conf) if 0.0 <= float(raw_conf) <= 1.0 else 0.5

            _log(f"  Phase {phase['phase_idx']}: {ap:12s} │ obj={phase['target_object']:10s} │ grip={gs}")

        return phases

    # ── Stage 3: Contact Mechanics ────────────────────────────────────

    def stage3_mechanics(
        self, video_path: str, phases: List[dict], dry_run: bool = False
    ) -> List[dict]:
        _log(f"[Stage 3/4] Contact mechanics estimation ({len(phases)} phases)...")

        for phase in phases:
            if not dry_run and CV2_AVAILABLE:
                frames, _, _ = extract_frames(
                    video_path, 4, phase["start_frame"], phase["end_frame"]
                )
            else:
                frames = []

            prompt = (
                f"Look at these robot arm images. Time: {phase['start_time']:.1f}–{phase['end_time']:.1f}s.\n"
                f"Observed action: {phase['action_primitive']} on {phase['target_object']}\n\n"
                "Describe physical contact. Answer in JSON only:\n"
                '{"contact_type": "<none|point|surface|edge|wrap>", '
                '"force_level": "<none|light|medium|strong>", '
                '"contact_points": "<brief location>", '
                '"motion_direction": "<linear|rotational|complex>"}'
            )
            raw  = self.vlm.query(prompt, frames)
            parsed = self.vlm.parse_json(raw)
            if isinstance(parsed, list):
                data = parsed[0] if parsed and isinstance(parsed[0], dict) else {}
            elif isinstance(parsed, dict):
                data = parsed
            else:
                data = {}

            ct = str(data.get("contact_type",   "none")).lower()
            fl = str(data.get("force_level",     "none")).lower()
            md = str(data.get("motion_direction","linear")).lower()

            phase["mechanics"] = {
                "contact_type":    ct if ct in CONTACT_TYPES else "none",
                "force_level":     fl if fl in FORCE_LEVELS  else "none",
                "contact_points":  str(data.get("contact_points", "")),
                "motion_direction": md if md in MOTION_DIRS  else "linear",
            }

            _log(f"  Phase {phase['phase_idx']}: contact={phase['mechanics']['contact_type']:8s} │ "
                 f"force={phase['mechanics']['force_level']:6s} │ motion={phase['mechanics']['motion_direction']}")

        return phases

    # ── Stage 4: Task Summary ──────────────────────────────────────────

    def stage4_task_summary(
        self, video_path: str, phases: List[dict], dry_run: bool = False
    ) -> str:
        _log("[Stage 4/4] Task summary generation...")

        if not dry_run and CV2_AVAILABLE:
            frames, _, _ = extract_frames(video_path, 4)
        else:
            frames = []

        seq = " → ".join(
            f"{p['action_primitive']}({p.get('target_object', '?')})"
            for p in phases
        )
        prompt = (
            f"Robot manipulation video. Detected action sequence: {seq}\n\n"
            "Write a single sentence summarizing the overall task "
            "(e.g., 'Pick up the screw and drive it into the solar panel bracket.').\n"
            "Return ONLY the summary sentence."
        )
        summary = self.vlm.query(prompt, frames).strip().strip('"').strip("'")
        if not summary:
            summary = "Robot manipulation task"
        _log(f"  → {summary}")
        return summary

    # ── Full Pipeline ──────────────────────────────────────────────────

    def run(self, video_path: str, dry_run: bool = False) -> dict:
        _log(f"\n{'='*60}")
        _log(f"AutoLabelPipeline")
        _log(f"  video : {video_path}")
        _log(f"  backend: {type(self.vlm).__name__}")
        _log(f"  adaptive: {self.adaptive}  dry_run: {dry_run}")
        _log(f"{'='*60}")
        t0 = time.time()

        phases, video_info = self.stage1_phase_segmentation(video_path, dry_run)
        phases = self.stage2_action_primitives(video_path, phases, dry_run)
        phases = self.stage3_mechanics(video_path, phases, dry_run)
        task_summary = self.stage4_task_summary(video_path, phases, dry_run)

        result = {
            "episode_id":   Path(video_path).stem,
            "video_path":   str(Path(video_path).resolve()),
            "video_info":   video_info,
            "phases":       phases,
            "task_summary": task_summary,
            "success":      True,
            "vlm_backend":  (
                f"{self.vlm.vision_model} + {self.vlm.reasoning_model}"
                if isinstance(self.vlm, HybridOllamaBackend)
                else type(self.vlm).__name__
            ),
            "labeled_at":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_sec":  round(time.time() - t0, 2),
        }

        _log(f"\n  Pipeline complete in {result['elapsed_sec']:.1f}s")
        return result


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — LEROBOT V2 EXPORTER
# (from dev-VQApipeline-siyu :: lerobot_exporter.py)
# ══════════════════════════════════════════════════════════════════════════════

def export_lerobot(
    episodes: List[dict],
    output_dir: str,
    robot_type: str = "single_arm",
    fps: int = 30,
) -> dict:
    """
    Export auto-label results to LeRobot V2.1 format.
    Merged from: dev-VQApipeline-siyu :: lerobot_exporter.py
    """
    out      = Path(output_dir)
    meta_dir = out / "meta"
    data_dir = out / "data"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    _log(f"\n{'='*60}")
    _log(f"LeRobot V2 Exporter → {out}")
    _log(f"{'='*60}")

    # Collect unique tasks
    tasks_map: Dict[str, int] = {}
    for ep in episodes:
        summary = ep.get("task_summary", "unknown task")
        if summary not in tasks_map:
            tasks_map[summary] = len(tasks_map)

    # ── meta/info.json ────────────────────────────────────────────────
    total_frames   = sum(ep.get("video_info", {}).get("total_frames", 0) for ep in episodes)
    ep_fps         = episodes[0].get("video_info", {}).get("fps", fps) if episodes else fps

    info = {
        "codebase_version": LEROBOT_VERSION,
        "robot_type":       robot_type,
        "total_episodes":   len(episodes),
        "total_frames":     total_frames,
        "fps":              ep_fps,
        "total_tasks":      len(tasks_map),
        "splits":           {"train": f"0:{len(episodes)}"},
        "data_path":        "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.json",
        "features": {
            "observation.task_description": {
                "dtype": "string", "shape": [1],
                "description": "Natural language task instruction for VLA model",
            },
            "action_primitive": {
                "dtype": "string", "shape": [1],
                "description": "Predicted action primitive from VLM",
            },
            "target_object": {
                "dtype": "string", "shape": [1],
                "description": "Target object of the action",
            },
            "gripper_state": {
                "dtype": "string", "shape": [1],
                "description": "Gripper state: open/closing/closed/opening",
            },
            "contact_type": {
                "dtype": "string", "shape": [1],
                "description": "Contact type: none/point/surface/edge/wrap",
            },
            "force_level": {
                "dtype": "string", "shape": [1],
                "description": "Estimated force level: none/light/medium/strong",
            },
            "motion_direction": {
                "dtype": "string", "shape": [1],
                "description": "Motion direction: linear/rotational/complex",
            },
            "phase_name": {
                "dtype": "string", "shape": [1],
                "description": "Phase label from temporal segmentation",
            },
            "confidence": {
                "dtype": "float32", "shape": [1],
                "description": "VLM prediction confidence score 0–1",
            },
        },
        "vlm_backend":  episodes[0].get("vlm_backend", "unknown") if episodes else "unknown",
        "created_at":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    _log("  ✓ meta/info.json")

    # ── meta/episodes.jsonl ───────────────────────────────────────────
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for i, ep in enumerate(episodes):
            vi = ep.get("video_info", {})
            f.write(json.dumps({
                "episode_index":  i,
                "episode_id":     ep.get("episode_id", f"episode_{i}"),
                "tasks":          [tasks_map.get(ep.get("task_summary", ""), 0)],
                "length":         vi.get("total_frames", 0),
                "duration":       vi.get("duration", 0),
                "video_path":     ep.get("video_path", ""),
                "num_phases":     len(ep.get("phases", [])),
                "success":        ep.get("success", True),
                "elapsed_sec":    ep.get("elapsed_sec", 0),
            }, ensure_ascii=False) + "\n")
    _log("  ✓ meta/episodes.jsonl")

    # ── meta/tasks.jsonl ──────────────────────────────────────────────
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task_desc, task_id in sorted(tasks_map.items(), key=lambda x: x[1]):
            skill_labels: set = set()
            for ep in episodes:
                if ep.get("task_summary") == task_desc:
                    for phase in ep.get("phases", []):
                        skill_labels.add(phase.get("action_primitive", "unknown"))
            f.write(json.dumps({
                "task_index":   task_id,
                "task":         task_desc,
                "skill_labels": sorted(skill_labels),
            }, ensure_ascii=False) + "\n")
    _log("  ✓ meta/tasks.jsonl")

    # ── data/chunk-000/episode_XXXXXX.json ────────────────────────────
    chunk_dir = data_dir / "chunk-000"
    chunk_dir.mkdir(exist_ok=True)

    for i, ep in enumerate(episodes):
        rows = []
        task_desc = ep.get("task_summary", "unknown task")
        for phase in ep.get("phases", []):
            mech = phase.get("mechanics", {})
            rows.append({
                # LeRobot required fields
                "episode_index":               i,
                "observation.task_description": task_desc,
                # Phase annotation
                "phase_index":     phase.get("phase_idx", 0),
                "phase_name":      phase.get("phase_name", ""),
                "start_frame":     phase.get("start_frame", 0),
                "end_frame":       phase.get("end_frame", 0),
                "start_time":      phase.get("start_time", 0),
                "end_time":        phase.get("end_time", 0),
                # Action labels
                "action_primitive": phase.get("action_primitive", "wait"),
                "target_object":    phase.get("target_object", "unknown"),
                "gripper_state":    phase.get("gripper_state", "open"),
                "confidence":       phase.get("confidence", 0.0),
                # Mechanics labels
                "contact_type":    mech.get("contact_type", "none"),
                "force_level":     mech.get("force_level", "none"),
                "contact_points":  mech.get("contact_points", ""),
                "motion_direction": mech.get("motion_direction", "linear"),
            })
        ep_file = chunk_dir / f"episode_{i:06d}.json"
        with open(ep_file, "w") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    _log(f"  ✓ data/chunk-000/ ({len(episodes)} episode files)")
    _log(f"  → {out}")

    return {
        "output_dir":     str(out),
        "total_episodes": len(episodes),
        "total_tasks":    len(tasks_map),
        "lerobot_version": LEROBOT_VERSION,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — π₀ / π₀.5 SFT CONFIG GENERATOR
# (from merge-local-features :: roboforce_skills/openpi_finetune_config.py)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OpenPIModelCfg:
    model_name:              str           = "pi05"          # pi0 | pi0_fast | pi05
    pretrained_path:         str           = "s3://openpi-assets/checkpoints/pi0_base"
    vlm_backbone:            str           = "paligemma"
    freeze_vlm:              bool          = True
    unfreeze_vlm_after_steps: int          = 10_000
    image_size:              tuple         = (224, 224)
    num_cameras:             int           = 1
    state_dim:               int           = 32
    action_dim:              int           = 8               # 6D pose + screw_rot + gripper
    action_horizon:          int           = 16
    num_flow_steps:          int           = 10
    flow_schedule:           str           = "linear"

@dataclass
class OpenPILoraCfg:
    enabled:              bool       = True
    rank:                 int        = 32
    alpha:                float      = 64.0
    dropout:              float      = 0.05
    target_modules:       List[str]  = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    apply_to_vlm:         bool       = False
    apply_to_action_head: bool       = True
    merge_after_training: bool       = True

@dataclass
class OpenPITrainingCfg:
    learning_rate:               float = 5e-5
    weight_decay:                float = 0.01
    warmup_steps:                int   = 1_000
    max_steps:                   int   = 80_000
    batch_size:                  int   = 16
    gradient_accumulation_steps: int   = 4
    lr_scheduler:                str   = "cosine"
    bf16:                        bool  = True
    flow_loss_weight:            float = 1.0
    vlm_loss_weight:             float = 0.1
    gradient_clip_norm:          float = 1.0
    use_ema:                     bool  = True
    ema_decay:                   float = 0.9999
    save_steps:                  int   = 5_000
    eval_steps:                  int   = 2_000
    logging_steps:               int   = 100
    num_gpus:                    int   = 1

@dataclass
class OpenPIDataCfg:
    dataset_path:     str       = "sft_output/lerobot"
    dataset_format:   str       = "lerobot_v2"
    task_instruction: str       = (
        "Pick up the screw and drive it into the solar panel mounting bracket"
    )
    instruction_variants: List[str] = field(default_factory=lambda: [
        "Pick up the screw and drive it into the solar panel mounting bracket",
        "Tighten the screw into the mounting bracket on the solar panel",
        "Install the screw into the bracket by driving it clockwise",
        "Fasten the bolt into the solar panel frame bracket",
        "Secure the solar panel by driving the screw into the mount",
    ])
    image_augmentation:  bool  = True
    state_normalization: str   = "per_feature"
    action_normalization: str  = "per_feature"
    train_ratio:         float = 0.9
    embodiment_name:     str   = "roboforce"
    joint_names: List[str] = field(default_factory=lambda: [
        "base_x", "base_y", "base_yaw",
        "right_shoulder_pan", "right_shoulder_lift", "right_elbow",
        "right_wrist_1", "right_wrist_2", "right_wrist_3", "right_wrist_roll",
        "screw_driver_rotation", "head_pan", "head_tilt",
    ])
    camera_names:          List[str] = field(default_factory=lambda: ["head_rgb"])
    control_frequency_hz:  float     = 50.0

@dataclass
class OpenPIFinetuneCfg:
    model:    OpenPIModelCfg    = field(default_factory=OpenPIModelCfg)
    lora:     OpenPILoraCfg     = field(default_factory=OpenPILoraCfg)
    training: OpenPITrainingCfg = field(default_factory=OpenPITrainingCfg)
    data:     OpenPIDataCfg     = field(default_factory=OpenPIDataCfg)
    output_dir:       str = "checkpoints/openpi_screw_driving"
    experiment_name:  str = "roboforce_pi05_screw_v1"
    use_wandb:        bool = True
    wandb_project:    str = "roboforce-openpi"


def generate_openpi_config(cfg: OpenPIFinetuneCfg) -> dict:
    """Produce config dict for openpi.training.train."""
    return {
        "model": {
            "name":                       cfg.model.model_name,
            "pretrained_path":            cfg.model.pretrained_path,
            "vlm_backbone":               cfg.model.vlm_backbone,
            "freeze_vlm":                 cfg.model.freeze_vlm,
            "unfreeze_vlm_after_steps":   cfg.model.unfreeze_vlm_after_steps,
            "image_size":                 list(cfg.model.image_size),
            "num_cameras":                cfg.model.num_cameras,
            "state_dim":                  cfg.model.state_dim,
            "action_dim":                 cfg.model.action_dim,
            "action_horizon":             cfg.model.action_horizon,
            "flow_matching": {
                "num_flow_steps": cfg.model.num_flow_steps,
                "schedule":       cfg.model.flow_schedule,
            },
        },
        "lora": {
            "enabled":              cfg.lora.enabled,
            "rank":                 cfg.lora.rank,
            "alpha":                cfg.lora.alpha,
            "dropout":              cfg.lora.dropout,
            "target_modules":       cfg.lora.target_modules,
            "apply_to_vlm":         cfg.lora.apply_to_vlm,
            "apply_to_action_head": cfg.lora.apply_to_action_head,
            "merge_after_training": cfg.lora.merge_after_training,
        },
        "training": {
            "learning_rate":               cfg.training.learning_rate,
            "weight_decay":                cfg.training.weight_decay,
            "warmup_steps":                cfg.training.warmup_steps,
            "max_steps":                   cfg.training.max_steps,
            "batch_size":                  cfg.training.batch_size,
            "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
            "lr_scheduler":                cfg.training.lr_scheduler,
            "bf16":                        cfg.training.bf16,
            "flow_loss_weight":            cfg.training.flow_loss_weight,
            "vlm_loss_weight":             cfg.training.vlm_loss_weight,
            "gradient_clip_norm":          cfg.training.gradient_clip_norm,
            "use_ema":                     cfg.training.use_ema,
            "ema_decay":                   cfg.training.ema_decay,
            "save_steps":                  cfg.training.save_steps,
            "eval_steps":                  cfg.training.eval_steps,
            "logging_steps":               cfg.training.logging_steps,
            "num_gpus":                    cfg.training.num_gpus,
        },
        "data": {
            "dataset_path":        cfg.data.dataset_path,
            "dataset_format":      cfg.data.dataset_format,
            "task_instruction":    cfg.data.task_instruction,
            "instruction_variants": cfg.data.instruction_variants,
            "image_augmentation":  cfg.data.image_augmentation,
            "state_normalization": cfg.data.state_normalization,
            "action_normalization": cfg.data.action_normalization,
            "train_ratio":         cfg.data.train_ratio,
            "embodiment_name":     cfg.data.embodiment_name,
            "joint_names":         cfg.data.joint_names,
            "camera_names":        cfg.data.camera_names,
            "control_frequency_hz": cfg.data.control_frequency_hz,
        },
        "embodiment": {
            "name": cfg.data.embodiment_name,
            "modality": {
                "video": {
                    "cameras":    cfg.data.camera_names,
                    "resolution": list(cfg.model.image_size),
                    "fps":        cfg.data.control_frequency_hz,
                },
                "action": {
                    "type":       "delta_ee_pose_and_screw",
                    "components": [
                        "delta_x", "delta_y", "delta_z",
                        "delta_rx", "delta_ry", "delta_rz",
                        "screw_rotation", "gripper",
                    ],
                    "dim": cfg.model.action_dim,
                },
            },
        },
        "output_dir":      cfg.output_dir,
        "experiment_name": cfg.experiment_name,
        "wandb": {
            "enabled": cfg.use_wandb,
            "project": cfg.wandb_project,
        },
    }


def save_openpi_config(output_dir: str, cfg: OpenPIFinetuneCfg) -> str:
    """Generate and write configs/openpi_finetune.json."""
    _log(f"\n{'='*60}")
    _log("π₀.5 SFT Config Generator")
    _log(f"{'='*60}")

    config = generate_openpi_config(cfg)
    config_path = Path(output_dir) / "configs" / "openpi_finetune.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    _log(f"  ✓ {config_path}")

    # Training launch command
    cmd = (
        f"python -m openpi.training.train \\\n"
        f"  --config {config_path} \\\n"
        f"  --output_dir {cfg.output_dir} \\\n"
        f"  --experiment_name {cfg.experiment_name} \\\n"
        f"  --bf16 \\\n"
        f"  --lora_rank {cfg.lora.rank} \\\n"
        f"  --lora_alpha {cfg.lora.alpha}"
    )
    cmd_path = Path(output_dir) / "configs" / "launch_training.sh"
    with open(cmd_path, "w") as f:
        f.write("#!/bin/bash\n# Auto-generated by demo_screw_to_pi05_sft.py\n\n")
        f.write(cmd + "\n")
    cmd_path.chmod(0o755)
    _log(f"  ✓ {cmd_path}")

    return str(config_path)


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — SUMMARY PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(result: dict, lerobot_result: dict, config_path: str, output_dir: str):
    out = Path(output_dir)

    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        Screw-Tightening Video  →  π₀.5 SFT Data            ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    print(f"\n  Input video   : {result['video_path']}")
    print(f"  Duration      : {result['video_info'].get('duration', '?'):.1f}s "
          f"  ({result['video_info'].get('fps', '?'):.0f} FPS  "
          f"{result['video_info'].get('resolution', '?')})")
    print(f"  VLM backend   : {result['vlm_backend']}")
    if "+" in result['vlm_backend']:
        print(f"                  (vision queries → {result['vlm_backend'].split('+')[0].strip()})")
        print(f"                  (text reasoning → {result['vlm_backend'].split('+')[1].strip()})")
    print(f"  Pipeline time : {result['elapsed_sec']:.1f}s")

    print(f"\n  ─── Phase Annotations ({'%d phases' % len(result['phases'])}) ───")
    for ph in result["phases"]:
        mech = ph.get("mechanics", {})
        print(f"    [{ph['phase_idx']}] {ph['phase_name']:22s} "
              f"│ {ph['action_primitive']:12s} "
              f"│ grip={ph['gripper_state']:8s} "
              f"│ force={mech.get('force_level','?'):6s} "
              f"│ motion={mech.get('motion_direction','?'):11s} "
              f"│ conf={ph['confidence']:.2f}")

    print(f"\n  Task summary  : \"{result['task_summary']}\"")

    print(f"\n  ─── LeRobot V2 Output ({lerobot_result['lerobot_version']}) ───")
    print(f"    {out / 'meta' / 'info.json'}")
    print(f"    {out / 'meta' / 'episodes.jsonl'}    ({lerobot_result['total_episodes']} episodes)")
    print(f"    {out / 'meta' / 'tasks.jsonl'}       ({lerobot_result['total_tasks']} tasks)")
    print(f"    {out / 'data' / 'chunk-000' / 'episode_000000.json'}")

    print(f"\n  ─── π₀.5 SFT Config ───")
    print(f"    {config_path}")
    print(f"    {out / 'configs' / 'launch_training.sh'}")

    print(f"\n  ─── Next Steps ───")
    print(f"    1. Collect 50–500 more screw-tightening demonstrations")
    print(f"    2. pip install openpi-client")
    print(f"    3. bash {out / 'configs' / 'launch_training.sh'}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def build_vlm(args) -> VLMBackend:
    if args.dry_run:
        _log("  [DRY-RUN] Using MockVLMBackend — no GPU or API key needed")
        return MockVLMBackend()
    if args.vlm == "gemini":
        if not args.gemini_key:
            print("ERROR: --gemini-key required for Gemini backend", file=sys.stderr)
            sys.exit(1)
        return GeminiBackend(api_key=args.gemini_key, model=args.gemini_model)
    if args.vlm == "hybrid":
        _log(f"  [Hybrid] Vision: {args.model}  |  Reasoning: {args.reasoning_model}")
        return HybridOllamaBackend(
            vision_model=args.model,
            reasoning_model=args.reasoning_model,
            url=args.ollama_url,
        )
    # Default: Ollama single model
    return OllamaBackend(model=args.model, url=args.ollama_url)


def main():
    parser = argparse.ArgumentParser(
        description="Screw-tightening video → π₀.5 SFT annotated dataset",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Input
    parser.add_argument("--video",       required=True,
                        help="Path to the screw-tightening video (MP4, AVI, ...)")
    parser.add_argument("--output-dir",  default="./sft_output",
                        help="Root output directory (default: ./sft_output)")
    # VLM backend
    parser.add_argument("--vlm",         default="ollama",
                        choices=["ollama", "gemini", "hybrid", "mock"],
                        help="VLM backend: ollama | gemini | hybrid | mock")
    parser.add_argument("--model",       default="scomper/minicpm-v2.5:latest",
                        help="Ollama model name")
    parser.add_argument("--ollama-url",  default="http://localhost:11434",
                        help="Ollama server URL")
    parser.add_argument("--reasoning-model", default="qwen3:32b",
                        help="Text-only reasoning model for hybrid mode (default: qwen3:32b)")
    parser.add_argument("--gemini-key",  default=None,
                        help="Gemini API key (required when --vlm gemini)")
    parser.add_argument("--gemini-model", default="gemini-2.0-flash",
                        help="Gemini model name (default: gemini-2.0-flash)")
    # Frame sampling
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Disable motion-adaptive sampling (use uniform frames)")
    parser.add_argument("--num-frames",  type=int, default=16,
                        help="Frames for uniform sampling (ignored in adaptive mode)")
    parser.add_argument("--max-vlm-frames", type=int, default=24,
                        help="Max frames sent to VLM in adaptive mode (default: 24)")
    parser.add_argument("--motion-threshold", type=float, default=0.02,
                        help="Motion sensitivity 0–1 (default: 0.02)")
    # π₀.5 config
    parser.add_argument("--pi05",        action="store_true", default=True,
                        help="Use π₀.5 model variant (default: True)")
    parser.add_argument("--robot-type",  default="single_arm",
                        help="Robot type for LeRobot metadata (default: single_arm)")
    # Misc
    parser.add_argument("--dry-run",     action="store_true",
                        help="Use MockVLM — no GPU or API key needed (for testing)")
    parser.add_argument("--output-jsonl", default=None,
                        help="Save intermediate JSONL labels to this file")

    args = parser.parse_args()

    # Validate input
    if not args.dry_run:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"ERROR: Video not found: {args.video}", file=sys.stderr)
            sys.exit(1)
        if not CV2_AVAILABLE:
            print("ERROR: pip install opencv-python", file=sys.stderr)
            sys.exit(1)
        if not NP_AVAILABLE:
            print("ERROR: pip install numpy", file=sys.stderr)
            sys.exit(1)
    else:
        args.video = args.video if Path(args.video).exists() else "/mock/screw_demo.mp4"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage A: Auto-label pipeline ────────────────────────────────
    vlm      = build_vlm(args)
    pipeline = AutoLabelPipeline(
        vlm=vlm,
        num_frames=args.num_frames,
        adaptive=not args.no_adaptive,
        max_vlm_frames=args.max_vlm_frames,
        motion_threshold=args.motion_threshold,
    )
    result = pipeline.run(args.video, dry_run=args.dry_run)

    # Save JSONL (intermediate labels)
    jsonl_path = args.output_jsonl or str(out_dir / "labels.jsonl")
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    _log(f"\n  JSONL saved → {jsonl_path}")

    # ── Stage B: LeRobot V2 export ───────────────────────────────────
    lerobot_out = str(out_dir / "lerobot")
    lerobot_result = export_lerobot(
        episodes=[result],
        output_dir=lerobot_out,
        robot_type=args.robot_type,
    )

    # ── Stage C: π₀.5 SFT config ────────────────────────────────────
    pi_cfg = OpenPIFinetuneCfg()
    pi_cfg.model.model_name       = "pi05" if args.pi05 else "pi0"
    pi_cfg.data.dataset_path      = lerobot_out
    pi_cfg.data.task_instruction  = result.get("task_summary", pi_cfg.data.task_instruction)
    config_path = save_openpi_config(str(out_dir), pi_cfg)

    # ── Summary ───────────────────────────────────────────────────────
    print_summary(result, lerobot_result, config_path, str(out_dir))


if __name__ == "__main__":
    main()
