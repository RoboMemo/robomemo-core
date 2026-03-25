"""
Auto-Label Pipeline for Robot Manipulation Videos
Turns robot manipulation videos into structured action labels via local Ollama VLM.

Pipeline: Video → Phase Segmentation → Action Primitive Labeling → Mechanics Estimation → JSONL

Usage:
    python auto_label_pipeline.py <video_path> [--model MODEL] [--ollama-url URL] [--output OUTPUT] [--num-frames N]
                                               [--adaptive] [--no-adaptive] [--max-vlm-frames N] [--motion-threshold F]
"""

import sys
import json
import cv2
import re
import base64
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(1)


ACTION_PRIMITIVES = [
    "approach", "align", "grasp", "lift", "move",
    "rotate_cw", "rotate_ccw", "insert", "push", "pull",
    "place", "release", "inspect", "wait", "retract",
]

GRIPPER_STATES = ["open", "closing", "closed", "opening"]
FORCE_LEVELS = ["none", "light", "medium", "strong"]
CONTACT_TYPES = ["none", "point", "surface", "edge", "wrap"]
MOTION_DIRECTIONS = ["linear", "rotational", "complex"]


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


class AutoLabelPipeline:
    """Core auto-labeling pipeline using Ollama VLM."""

    def __init__(
        self,
        model: str = "scomper/minicpm-v2.5:latest",
        ollama_url: str = "http://localhost:11434",
        num_frames: int = 16,
        adaptive: bool = True,
        max_vlm_frames: int = 24,
        motion_threshold: float = 0.02,
    ):
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.num_frames = num_frames
        self.adaptive = adaptive
        self.max_vlm_frames = max_vlm_frames
        self.motion_threshold = motion_threshold

    # ── Frame Extraction ─────────────────────────────────────────────────

    def extract_frames(
        self, video_path: str, num_frames: int, start_frame: int = 0, end_frame: int = -1
    ) -> tuple:
        """Extract N frames uniformly from video (or a segment). Returns (frames_b64, video_info)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total / fps

        if end_frame < 0:
            end_frame = total - 1
        end_frame = min(end_frame, total - 1)

        indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
        frames = []
        frame_metas = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            # Resize to max 512px on longest side for efficiency
            h, w = frame.shape[:2]
            scale = min(512 / max(h, w), 1.0)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64 = base64.b64encode(buf).decode("ascii")
            frames.append(b64)
            frame_metas.append({
                "frame_idx": int(idx),
                "timestamp": round(int(idx) / fps, 3),
            })

        cap.release()

        video_info = {
            "duration": round(duration, 2),
            "fps": round(fps, 2),
            "resolution": [width, height],
            "total_frames": total,
        }
        return frames, frame_metas, video_info

    # ── Motion-Adaptive Frame Extraction ────────────────────────────────

    def extract_adaptive_frames(
        self, video_path: str, max_vlm_frames: int = 24, motion_threshold: float = 0.02
    ) -> tuple:
        """Extract frames using motion-adaptive sampling (2-pass).

        Pass 1: Compute per-frame motion scores via frame differencing, find peaks.
        Pass 2: Dense-sample around key frames, cap at max_vlm_frames.

        Returns (frames_b64, frame_metas, video_info) — same format as extract_frames(),
        but frame_metas entries include an additional "motion_score" field.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total / fps

        video_info = {
            "duration": round(duration, 2),
            "fps": round(fps, 2),
            "resolution": [width, height],
            "total_frames": total,
        }

        # ── Pass 1: compute motion scores ────────────────────────────
        # Read every frame as grayscale and compute absolute difference
        # For very long videos, subsample at ~5 fps to keep it fast
        subsample = max(1, int(fps / 5))
        motion_raw: List[tuple] = []  # (frame_idx, score)
        prev_gray = None

        for fi in range(0, total, subsample):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Downsample for speed
            gray = cv2.resize(gray, (160, 120))
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                score = float(np.mean(diff) / 255.0)  # normalise to 0-1
            else:
                score = 0.0
            motion_raw.append((fi, score))
            prev_gray = gray

        if len(motion_raw) < 2:
            cap.release()
            log("Motion analysis: too few frames, falling back to uniform sampling")
            return self.extract_frames(video_path, max_vlm_frames)

        indices_arr = np.array([m[0] for m in motion_raw])
        scores_arr = np.array([m[1] for m in motion_raw])

        # Smooth with sliding window (size 5)
        kernel_size = min(5, len(scores_arr))
        kernel = np.ones(kernel_size) / kernel_size
        scores_smooth = np.convolve(scores_arr, kernel, mode="same")

        # Find peaks: local maxima above threshold
        key_indices: List[int] = []
        for i in range(1, len(scores_smooth) - 1):
            if (scores_smooth[i] > scores_smooth[i - 1]
                    and scores_smooth[i] > scores_smooth[i + 1]
                    and scores_smooth[i] >= motion_threshold):
                key_indices.append(i)

        # Also detect motion plateaus: sustained above threshold
        in_plateau = False
        plateau_start = 0
        for i, s in enumerate(scores_smooth):
            if s >= motion_threshold and not in_plateau:
                in_plateau = True
                plateau_start = i
            elif s < motion_threshold and in_plateau:
                in_plateau = False
                mid = (plateau_start + i) // 2
                if mid not in key_indices:
                    key_indices.append(mid)
        # Handle plateau that runs to the end
        if in_plateau:
            mid = (plateau_start + len(scores_smooth) - 1) // 2
            if mid not in key_indices:
                key_indices.append(mid)

        # Always include first and last sampled frame
        if 0 not in key_indices:
            key_indices.insert(0, 0)
        last_idx = len(scores_smooth) - 1
        if last_idx not in key_indices:
            key_indices.append(last_idx)

        key_indices = sorted(set(key_indices))

        log(f"Motion analysis: {len(key_indices)} key frames detected from {total} total frames")

        # ── Pass 2: dense sample around key frames ───────────────────
        context_n = 3  # ±N frames around each key frame (in subsampled space)
        candidate_set: set = set()
        for ki in key_indices:
            for offset in range(-context_n, context_n + 1):
                ci = ki + offset
                if 0 <= ci < len(indices_arr):
                    candidate_set.add(ci)

        candidates = sorted(candidate_set)

        # If too many, keep highest-motion-score ones
        if len(candidates) > max_vlm_frames:
            scored = [(c, scores_smooth[c]) for c in candidates]
            # Always keep first and last
            first, last = scored[0], scored[-1]
            middle = sorted(scored[1:-1], key=lambda x: x[1], reverse=True)
            selected = [first] + middle[:max_vlm_frames - 2] + [last]
            candidates = sorted(set(s[0] for s in selected))

        # Map back to actual frame indices and read frames
        selected_frame_indices = [int(indices_arr[c]) for c in candidates]
        selected_scores = [float(scores_smooth[c]) for c in candidates]

        log(f"Motion analysis: selected {len(selected_frame_indices)} frames for VLM")

        frames_b64 = []
        frame_metas = []
        for fi, score in zip(selected_frame_indices, selected_scores):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue
            h, w = frame.shape[:2]
            scale = min(512 / max(h, w), 1.0)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64 = base64.b64encode(buf).decode("ascii")
            frames_b64.append(b64)
            frame_metas.append({
                "frame_idx": fi,
                "timestamp": round(fi / fps, 3),
                "motion_score": round(score, 4),
            })

        cap.release()
        return frames_b64, frame_metas, video_info

    # ── Ollama VLM Call ──────────────────────────────────────────────────

    def call_vlm(self, prompt: str, images: List[str], temperature: float = 0.2) -> str:
        """Send prompt + images to Ollama and return the text response."""
        msg: dict = {"role": "user", "content": prompt}
        if images:
            msg["images"] = images
        payload = {
            "model": self.model,
            "messages": [msg],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 4096,
                "num_ctx": 8192,
            },
        }
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.ollama_url}. Is 'ollama serve' running?"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def parse_json_response(self, text: str) -> Any:
        """Try to parse JSON from VLM response. Falls back to regex extraction."""
        # Clean common VLM artifacts: markdown escapes like \_ → _
        text = text.replace("\\_", "_").replace("\\*", "*")
        
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code blocks
        m = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON array or object
        for pattern in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
            m = re.search(pattern, text)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass

        log(f"WARNING: Could not parse JSON from VLM response. Raw text: {text[:500]}")
        return None

    # ── Stage 1: Phase Segmentation ──────────────────────────────────────

    def stage1_phase_segmentation(
        self, video_path: str
    ) -> tuple:
        """Segment video into temporal phases."""
        if self.adaptive:
            log(f"[Stage 1] Using motion-adaptive sampling (max {self.max_vlm_frames} frames, threshold={self.motion_threshold})...")
            frames, frame_metas, video_info = self.extract_adaptive_frames(
                video_path, self.max_vlm_frames, self.motion_threshold
            )
        else:
            log(f"[Stage 1] Extracting {self.num_frames} frames for phase segmentation...")
            frames, frame_metas, video_info = self.extract_frames(video_path, self.num_frames)
        log(f"[Stage 1] Got {len(frames)} frames, sending to VLM...")

        ts_context = ", ".join(
            f"Frame {i+1}: {m['timestamp']}s (frame #{m['frame_idx']})"
            for i, m in enumerate(frame_metas)
        )

        # Step 1a: describe each frame individually to avoid VLM hallucinating from examples
        frame_descs = []
        for idx, (fr, meta) in enumerate(zip(frames, frame_metas)):
            # Add motion context if available (adaptive mode)
            motion_ctx = ""
            ms = meta.get("motion_score")
            if ms is not None:
                if ms > 0.1:
                    motion_ctx = " (high motion — likely action transition)"
                elif ms > 0.03:
                    motion_ctx = " (moderate motion — active manipulation)"
                else:
                    motion_ctx = " (low motion — stable pose)"

            desc_prompt = (
                f"This is frame {idx+1} of {len(frames)} from a robot manipulation video "
                f"(timestamp {meta['timestamp']:.1f}s){motion_ctx}. "
                "In one sentence, describe ONLY what the robot arm and gripper are doing RIGHT NOW. "
                "Be specific: mention arm position, gripper state (open/closing/closed), and any object contact. "
                "Do NOT guess future actions."
            )
            desc = self.call_vlm(desc_prompt, [fr])
            frame_descs.append(f"Frame {idx+1} ({meta['timestamp']:.1f}s): {desc.strip()}")
            log(f"  [S1] frame {idx+1}: {desc.strip()[:80]}")

        combined_desc = "\n".join(frame_descs)

        # Step 1b: ask LLM to segment based on those descriptions (no images needed here)
        prompt = f"""A robot manipulation video has {video_info['total_frames']} frames at {video_info['fps']} FPS ({video_info['duration']:.1f}s total).

Frame timestamps for the sampled frames: {ts_context}

Here are per-frame descriptions from {len(frames)} sampled frames:
{combined_desc}

Based on THESE DESCRIPTIONS ONLY, segment the video into 2-5 distinct temporal phases.
Each phase = a coherent stage of the manipulation (e.g. idle, reach, contact, grasp, lift, move, place, retract).

Return a JSON array. Each object must have:
- "phase_name": short snake_case name derived from what you actually observed (NOT generic examples)
- "start_frame_index": 1-indexed frame number where this phase starts
- "end_frame_index": 1-indexed frame number where this phase ends
- "description": one sentence describing the observed robot behavior

Return ONLY the JSON array, no markdown, no explanation."""

        raw = self.call_vlm(prompt, [])  # text-only segmentation based on descriptions
        phases_raw = self.parse_json_response(raw)

        if not phases_raw or not isinstance(phases_raw, list):
            log("WARNING: Stage 1 failed to get valid phases, creating single default phase")
            phases_raw = [{
                "phase_name": "full_episode",
                "start_frame_index": 1,
                "end_frame_index": len(frames),
                "description": "Complete manipulation episode",
            }]

        # Convert frame indices to actual timestamps
        phases = []
        for i, p in enumerate(phases_raw):
            si = max(0, min(int(p.get("start_frame_index", 1)) - 1, len(frame_metas) - 1))
            ei = max(0, min(int(p.get("end_frame_index", 1)) - 1, len(frame_metas) - 1))
            if ei < si:
                ei = si

            phases.append({
                "phase_idx": i,
                "phase_name": str(p.get("phase_name", f"phase_{i}")),
                "start_frame": frame_metas[si]["frame_idx"],
                "end_frame": frame_metas[ei]["frame_idx"],
                "start_time": frame_metas[si]["timestamp"],
                "end_time": frame_metas[ei]["timestamp"],
                "description": str(p.get("description", "")),
            })

        log(f"[Stage 1] Segmented into {len(phases)} phases")
        return phases, video_info

    # ── Stage 2: Action Primitive Labeling ───────────────────────────────

    def stage2_action_primitives(
        self, video_path: str, phases: List[Dict]
    ) -> List[Dict]:
        """Label each phase with an action primitive."""
        log(f"[Stage 2] Labeling {len(phases)} phases with action primitives...")
        vocab_str = ", ".join(ACTION_PRIMITIVES)

        for phase in phases:
            frames, metas, _ = self.extract_frames(
                video_path, 4, phase["start_frame"], phase["end_frame"]
            )
            if not frames:
                phase["action_primitive"] = "wait"
                phase["target_object"] = "unknown"
                phase["gripper_state"] = "open"
                phase["confidence"] = 0.0
                continue

            prompt = f"""Look carefully at these {len(frames)} robot video frames (time: {phase['start_time']:.1f}s to {phase['end_time']:.1f}s).

Identify EXACTLY what the robot is doing based ONLY on what you see in the images.

Action primitive vocabulary (pick exactly one):
{vocab_str}

Gripper states: open, closing, closed, opening

Answer these by looking at the images:
1. What specific action primitive best describes what the robot arm is doing?
2. What object is the robot interacting with (describe color/shape if unsure of name)?
3. What is the gripper state right now?
4. How confident are you (0.0-1.0)?

Return ONLY a JSON object with keys: action_primitive, target_object, gripper_state, confidence
No examples, no markdown, just the JSON."""

            raw = self.call_vlm(prompt, frames)
            result = self.parse_json_response(raw)

            if result and isinstance(result, dict):
                ap = str(result.get("action_primitive", "wait")).lower()
                if ap not in ACTION_PRIMITIVES:
                    ap = "wait"
                gs = str(result.get("gripper_state", "open")).lower()
                if gs not in GRIPPER_STATES:
                    gs = "open"
                phase["action_primitive"] = ap
                phase["target_object"] = str(result.get("target_object", "unknown"))
                phase["gripper_state"] = gs
                phase["confidence"] = float(result.get("confidence", 0.5))
            else:
                phase["action_primitive"] = "wait"
                phase["target_object"] = "unknown"
                phase["gripper_state"] = "open"
                phase["confidence"] = 0.0

            log(f"  Phase {phase['phase_idx']}: {phase['action_primitive']} → {phase['target_object']}")

        return phases

    # ── Stage 3: Mechanics Estimation ────────────────────────────────────

    def stage3_mechanics(
        self, video_path: str, phases: List[Dict]
    ) -> List[Dict]:
        """Estimate contact mechanics for each phase."""
        log(f"[Stage 3] Estimating mechanics for {len(phases)} phases...")

        for phase in phases:
            frames, metas, _ = self.extract_frames(
                video_path, 4, phase["start_frame"], phase["end_frame"]
            )
            if not frames:
                phase["mechanics"] = {
                    "contact_type": "none",
                    "force_level": "none",
                    "contact_points": "",
                    "motion_direction": "linear",
                }
                continue

            prompt = f"""Look carefully at these {len(frames)} robot video frames.

Describe the physical contact and forces you can observe in the images:
- Is the robot touching any object? (contact_type: none / point / surface / edge / wrap)
- How much force appears to be applied? (force_level: none / light / medium / strong)
- Where exactly is contact occurring? (brief description of what you see)
- What direction is the robot moving? (motion_direction: linear / rotational / complex)

Base your answer ONLY on what you can visually observe in these frames.

Return ONLY a JSON object:
{{"contact_type": "...", "force_level": "...", "contact_points": "...", "motion_direction": "..."}}
No markdown, no extra text."""

            raw = self.call_vlm(prompt, frames)
            result = self.parse_json_response(raw)

            if result and isinstance(result, dict):
                ct = str(result.get("contact_type", "none")).lower()
                if ct not in CONTACT_TYPES:
                    ct = "none"
                fl = str(result.get("force_level", "none")).lower()
                if fl not in FORCE_LEVELS:
                    fl = "none"
                md = str(result.get("motion_direction", "linear")).lower()
                if md not in MOTION_DIRECTIONS:
                    md = "linear"
                phase["mechanics"] = {
                    "contact_type": ct,
                    "force_level": fl,
                    "contact_points": str(result.get("contact_points", "")),
                    "motion_direction": md,
                }
            else:
                phase["mechanics"] = {
                    "contact_type": "none",
                    "force_level": "none",
                    "contact_points": "",
                    "motion_direction": "linear",
                }

            log(f"  Phase {phase['phase_idx']}: {phase['mechanics']['contact_type']}, force={phase['mechanics']['force_level']}")

        return phases

    # ── Stage 4: Task Summary ────────────────────────────────────────────

    def stage4_task_summary(
        self, video_path: str, phases: List[Dict]
    ) -> str:
        """Generate a brief task summary from phase labels."""
        frames, _, _ = self.extract_frames(video_path, 4)
        if not frames:
            return "Unknown manipulation task"

        phase_desc = "; ".join(
            f"{p['action_primitive']} {p.get('target_object', '')}" for p in phases
        )

        prompt = f"""You are analyzing a robot manipulation video.
The detected action sequence is: {phase_desc}

I'm showing you 4 frames from this video.

Task: Write a single sentence summarizing the overall manipulation task (e.g., "Pick up the red cup and place it on the platform").

Return ONLY the summary sentence, no quotes or extra text."""

        raw = self.call_vlm(prompt, frames, temperature=0.3)
        summary = raw.strip().strip('"').strip("'")
        if not summary:
            summary = "Robot manipulation task"
        return summary

    # ── Full Pipeline ────────────────────────────────────────────────────

    def run(self, video_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Run the full auto-label pipeline on a video."""
        video_path = str(Path(video_path).resolve())
        log(f"=== Auto-Label Pipeline ===")
        log(f"Video: {video_path}")
        log(f"Model: {self.model}")
        log(f"Ollama: {self.ollama_url}")
        log(f"Adaptive: {self.adaptive}")
        if self.adaptive:
            log(f"Max VLM frames: {self.max_vlm_frames}, Motion threshold: {self.motion_threshold}")
        else:
            log(f"Frames: {self.num_frames}")

        # Stage 1
        phases, video_info = self.stage1_phase_segmentation(video_path)

        # Save intermediate
        self._save_intermediate(video_path, "stage1", {"phases": phases, "video_info": video_info})

        # Stage 2
        phases = self.stage2_action_primitives(video_path, phases)
        self._save_intermediate(video_path, "stage2", {"phases": phases})

        # Stage 3
        phases = self.stage3_mechanics(video_path, phases)
        self._save_intermediate(video_path, "stage3", {"phases": phases})

        # Stage 4: Task summary
        log("[Stage 4] Generating task summary...")
        task_summary = self.stage4_task_summary(video_path, phases)
        log(f"[Stage 4] Summary: {task_summary}")

        # Assemble final output
        episode_id = Path(video_path).stem
        result = {
            "episode_id": episode_id,
            "video_path": video_path,
            "video_info": video_info,
            "phases": phases,
            "task_summary": task_summary,
            "success": True,
            "model": self.model,
            "labeled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        # Output
        result_json = json.dumps(result, ensure_ascii=False)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "a") as f:
                f.write(result_json + "\n")
            log(f"Results appended to {output_path}")
        else:
            print(result_json)

        log("=== Pipeline Complete ===")
        return result

    def _save_intermediate(self, video_path: str, stage: str, data: Dict):
        """Save intermediate results for resumability."""
        cache_dir = Path(video_path).parent / ".auto_label_cache"
        cache_dir.mkdir(exist_ok=True)
        stem = Path(video_path).stem
        cache_file = cache_dir / f"{stem}_{stage}.json"
        with open(cache_file, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Auto-Label Pipeline for Robot Manipulation Videos")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--model", default="scomper/minicpm-v2.5:latest", help="Ollama model name")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama API URL")
    parser.add_argument("--output", default=None, help="Output JSONL file path (default: stdout)")
    parser.add_argument("--num-frames", type=int, default=16, help="Number of frames for segmentation (used when --no-adaptive)")
    parser.add_argument("--adaptive", action="store_true", default=True, help="Use motion-adaptive frame sampling (default: True)")
    parser.add_argument("--no-adaptive", action="store_true", default=False, help="Disable adaptive sampling, use uniform frames")
    parser.add_argument("--max-vlm-frames", type=int, default=24, help="Max frames sent to VLM in adaptive mode (default: 24)")
    parser.add_argument("--motion-threshold", type=float, default=0.02, help="Motion detection sensitivity 0.0-1.0 (default: 0.02)")

    args = parser.parse_args()

    # --no-adaptive overrides --adaptive
    adaptive = args.adaptive and not args.no_adaptive

    if not Path(args.video_path).exists():
        print(f"ERROR: Video not found: {args.video_path}", file=sys.stderr)
        sys.exit(1)

    pipeline = AutoLabelPipeline(
        model=args.model,
        ollama_url=args.ollama_url,
        num_frames=args.num_frames,
        adaptive=adaptive,
        max_vlm_frames=args.max_vlm_frames,
        motion_threshold=args.motion_threshold,
    )

    try:
        pipeline.run(args.video_path, args.output)
    except Exception as e:
        error_result = {
            "episode_id": Path(args.video_path).stem,
            "video_path": str(Path(args.video_path).resolve()),
            "error": str(e),
            "success": False,
            "model": args.model,
            "labeled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        print(json.dumps(error_result, ensure_ascii=False))
        log(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
