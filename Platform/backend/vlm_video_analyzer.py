"""
VLM Video Analyzer for Structured VQA Generation
Uses state-of-the-art Vision Language Models to decompose videos into 7 types of structured annotations:
1. Temporal - Time relationships
2. Spatial - Spatial relationships
3. Attribute - Object attributes
4. Mechanics - Force and contact information
5. Reasoning - Action reasoning
6. Summary - Scene summary
7. Trajectory - Motion trajectory

Each annotation is grounded in visual evidence with temporal consistency.
"""

import sys
import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Tuple
import base64
import io
from datetime import timedelta

# Support for multiple VLM backends
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from google import genai as google_genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class VideoFrameExtractor:
    """Extract frames from video with temporal information"""
    
    @staticmethod
    def extract_frames_with_timestamps(
        video_path: str, 
        num_frames: int = 32,
        include_temporal_context: bool = True
    ) -> List[Dict[str, Any]]:
        """Extract frames with timestamp and temporal context"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        # Select frame indices uniformly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames_data = []
        for idx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Calculate timestamp
            timestamp = frame_idx / fps if fps > 0 else 0
            
            frame_info = {
                'frame_idx': int(frame_idx),
                'sequence_idx': idx,
                'timestamp': timestamp,
                'timestamp_str': str(timedelta(seconds=timestamp)),
                'image': pil_image,
                'relative_position': idx / (num_frames - 1) if num_frames > 1 else 0
            }
            
            frames_data.append(frame_info)
        
        cap.release()
        
        # Add temporal context
        if include_temporal_context:
            for i, frame_data in enumerate(frames_data):
                frame_data['temporal_context'] = {
                    'is_first': i == 0,
                    'is_last': i == len(frames_data) - 1,
                    'prev_frame_idx': frames_data[i-1]['frame_idx'] if i > 0 else None,
                    'next_frame_idx': frames_data[i+1]['frame_idx'] if i < len(frames_data) - 1 else None
                }
        
        return frames_data, {'total_frames': total_frames, 'fps': fps, 'duration': duration}


class VLMAnalyzer:
    """Base class for VLM-based video analysis"""
    
    # ── Grounding helper doc injected into every prompt ──────────────────────
    _GROUNDING_SCHEMA = """
GROUNDING SCHEMA — required in every annotated item:
Every claim must include a "grounding" object with:
  - frame_indices: list of integer frame indices (from the sequence you were shown) that visually support this claim
  - timestamps: matching human-readable timestamps e.g. ["0:02", "0:03.5"]
  - bboxes: list of {"x": 0-1, "y": 0-1, "width": 0-1, "height": 0-1, "label": "string"} in normalised [0,1] coords (x/y = top-left corner). Include bounding boxes for every relevant object region. If unsure, omit or use approximate values.
  - description: one sentence describing what the visual evidence shows
  - confidence: float 0.0–1.0 reflecting how certain you are based on the visual evidence

Example grounding object:
{
  "frame_indices": [24, 36, 48],
  "timestamps": ["0:02", "0:03", "0:04"],
  "bboxes": [{"x": 0.30, "y": 0.40, "width": 0.15, "height": 0.20, "label": "gripper"}],
  "description": "Frames 24-48: gripper jaws visibly closing around cup edge",
  "confidence": 0.91
}
"""

    VQA_PROMPT_TEMPLATE = """
You are an expert robot manipulation video analyst. Decompose the video into 7 structured VQA categories.
EVERY annotated item MUST include a "grounding" field anchoring the claim to specific frames and bounding-box regions.

VIDEO CONTEXT:
- Total frames: {total_frames}
- Duration: {duration:.2f} seconds  
- FPS: {fps:.2f}
- Frames are labelled with [Frame N/M @ timestamp] markers in the sequence you received.

""" + _GROUNDING_SCHEMA.strip() + """

CATEGORIES TO ANALYSE:

1. TEMPORAL — chronological action sequence, what happens before/after what
2. SPATIAL — spatial relationships (left/right/above/below/near) between gripper, manipulated objects, and workspace
3. ATTRIBUTE — object properties: color, material, shape, size, state changes
4. MECHANICS — contact type, force level (light/medium/strong), contact points, force profile over time
5. REASONING — why each action is performed this way; strategy and constraints
6. SUMMARY — high-level task description, start state, end state, milestones, success/failure
7. TRAJECTORY — end-effector motion: path type (linear/curved/rotational), velocity (slow/medium/fast), waypoints

OUTPUT FORMAT — return ONLY valid JSON (no markdown):
{{
  "temporal": {{
    "action_sequence": [
      {{
        "action": "Approach cup",
        "timestamp": "0:02",
        "frame_range": [12, 36],
        "description": "End-effector moves toward cup from upper right",
        "grounding": {{
          "frame_indices": [12, 24, 36],
          "timestamps": ["0:02", "0:03", "0:04"],
          "bboxes": [{{"x": 0.55, "y": 0.30, "width": 0.12, "height": 0.18, "label": "end-effector"}}],
          "description": "Frames 12-36 show arm approaching left side of frame where cup sits",
          "confidence": 0.93
        }}
      }}
    ],
    "relationships": ["Approach BEFORE Pre-grasp", "Pre-grasp BEFORE Grasp"]
  }},
  "spatial": {{
    "key_relationships": [
      {{
        "timestamp": "0:03",
        "relationship": "Gripper is directly above cup",
        "details": "End-effector centroid is within 2 cm of cup opening center",
        "grounding": {{
          "frame_indices": [36, 42],
          "timestamps": ["0:04", "0:04.5"],
          "bboxes": [
            {{"x": 0.38, "y": 0.28, "width": 0.10, "height": 0.08, "label": "gripper"}},
            {{"x": 0.40, "y": 0.46, "width": 0.08, "height": 0.12, "label": "cup"}}
          ],
          "description": "Overhead view shows gripper aligned with cup top opening",
          "confidence": 0.90
        }}
      }}
    ],
    "trajectory_spatial": "Arm operates in 40x30 cm workspace. Cup transported left-to-right."
  }},
  "attribute": {{
    "objects": [
      {{
        "name": "Red plastic cup",
        "properties": {{"color": "red", "material": "plastic", "shape": "cylindrical", "size": "~9cm tall"}},
        "state_changes": ["Stationary → Grasped at t=0:03", "Grasped → Placed at t=0:07"],
        "grounding": {{
          "frame_indices": [0, 60, 180],
          "timestamps": ["0:00", "0:03", "0:07"],
          "bboxes": [{{"x": 0.30, "y": 0.46, "width": 0.09, "height": 0.14, "label": "cup"}}],
          "description": "Cup visible throughout. Color and shape consistent. Deformation at contact visible in frames 60-80.",
          "confidence": 0.96
        }}
      }}
    ]
  }},
  "mechanics": {{
    "contacts": [
      {{
        "timestamp": "0:03",
        "contact_type": "Parallel-jaw pinch grip",
        "force_level": "medium",
        "contact_points": "Both fingers contact cup outer wall at mid-height",
        "area": "Two ~2x1 cm patches on opposite sides",
        "grounding": {{
          "frame_indices": [60, 72, 84],
          "timestamps": ["0:03", "0:03.5", "0:04"],
          "bboxes": [
            {{"x": 0.36, "y": 0.44, "width": 0.05, "height": 0.08, "label": "contact-left"}},
            {{"x": 0.52, "y": 0.44, "width": 0.05, "height": 0.08, "label": "contact-right"}}
          ],
          "description": "Frames 60-84: slight cup wall deformation visible at gripper finger contact",
          "confidence": 0.88
        }}
      }}
    ],
    "force_profile": "0N at approach, increases to ~5N at grasp, maintained through transport, 0N at release"
  }},
  "reasoning": {{
    "action_justifications": [
      {{
        "action": "Approach from above at 45 degrees",
        "reason": "Top-down approach avoids arm self-occlusion and keeps camera view clear",
        "constraints": ["No collision with table edge", "Camera line-of-sight to cup"],
        "grounding": {{
          "frame_indices": [0, 24],
          "timestamps": ["0:00", "0:02"],
          "bboxes": [],
          "description": "Arm trajectory visible approaching from upper-right diagonal",
          "confidence": 0.85
        }}
      }}
    ],
    "overall_strategy": "Conservative pick-and-place: minimise spill risk with slow speed and sufficient lift height"
  }},
  "summary": {{
    "task_description": "Pick up red cup from left zone and place in right target zone",
    "start_state": "Cup upright on left table zone. Gripper at rest.",
    "end_state": "Cup in target zone. Gripper open and returned.",
    "success": true,
    "key_milestones": ["Grasp at 0:03", "Lift at 0:04", "Place at 0:07"],
    "duration": "{duration:.1f} seconds",
    "grounding_start": {{
      "frame_indices": [0, 6],
      "timestamps": ["0:00", "0:00.2"],
      "bboxes": [{{"x": 0.28, "y": 0.46, "width": 0.09, "height": 0.14, "label": "cup-start"}}],
      "description": "Frame 0: initial state with cup on left, gripper at rest upper right",
      "confidence": 0.98
    }},
    "grounding_end": {{
      "frame_indices": [175, 180],
      "timestamps": ["0:07.8", "0:08"],
      "bboxes": [{{"x": 0.65, "y": 0.46, "width": 0.09, "height": 0.14, "label": "cup-end"}}],
      "description": "Final frames: cup stable in target zone, gripper open",
      "confidence": 0.97
    }}
  }},
  "trajectory": {{
    "motion_segments": [
      {{
        "segment": "Approach arc",
        "time_range": "0:00-0:02",
        "motion_type": "curved",
        "velocity": "medium",
        "waypoints": ["Rest position", "Mid-arc above workspace", "Above cup"],
        "grounding": {{
          "frame_indices": [0, 18, 36],
          "timestamps": ["0:00", "0:01", "0:02"],
          "bboxes": [{{"x": 0.40, "y": 0.20, "width": 0.35, "height": 0.30, "label": "approach-arc"}}],
          "description": "Curved arm path from top-right to center-left visible across frames 0-36",
          "confidence": 0.91
        }}
      }}
    ],
    "overall_path": "J-shaped: arc approach → vertical descent → vertical lift → horizontal traverse → vertical descent to place"
  }},
  "visual_evidence": {{
    "key_frames": [
      {{"frame_idx": 0,  "timestamp": "0:00", "significance": "Initial state"}},
      {{"frame_idx": 36, "timestamp": "0:02", "significance": "Gripper aligned above cup"}},
      {{"frame_idx": 60, "timestamp": "0:03", "significance": "Grasp established"}},
      {{"frame_idx": 84, "timestamp": "0:04", "significance": "Cup lifted clear of table"}}
    ]
  }},
  "confidence_scores": {{
    "temporal":   0.0,
    "spatial":    0.0,
    "attribute":  0.0,
    "mechanics":  0.0,
    "reasoning":  0.0,
    "summary":    0.0,
    "trajectory": 0.0
  }}
}}

Replace all example values with your actual observations from the video frames provided.
Confidence scores should be your genuine assessment (0.0–1.0) for each category.
Return ONLY the JSON object. No markdown. No explanation outside the JSON.
"""

    def __init__(self, api_key: str, model_name: str = "auto"):
        """Initialize VLM analyzer with API key"""
        self.api_key = api_key
        self.model_name = model_name
        
    def analyze_video(self, video_path: str, num_frames: int = 32) -> Dict[str, Any]:
        """Analyze video and generate structured VQA annotations"""
        raise NotImplementedError("Subclass must implement analyze_video")
    
    @staticmethod
    def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


class GeminiAnalyzer(VLMAnalyzer):
    """Gemini 2.5 Pro Analyzer using new google-genai SDK"""

    MODEL = "gemini-2.5-pro-preview-05-06"

    def __init__(self, api_key: str):
        super().__init__(api_key, self.MODEL)
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai not installed. Install: pip install google-genai")
        self.client = google_genai.Client(api_key=api_key)

    def analyze_video(self, video_path: str, num_frames: int = 32) -> Dict[str, Any]:
        """Analyze video using Gemini 2.5 Pro"""
        print(f"Extracting {num_frames} frames from video...")
        frames_data, video_info = VideoFrameExtractor.extract_frames_with_timestamps(
            video_path, num_frames
        )

        print(f"Analyzing with {self.MODEL} ({len(frames_data)} frames)...")

        prompt = self.VQA_PROMPT_TEMPLATE.format(
            total_frames=video_info['total_frames'],
            duration=video_info['duration'],
            fps=video_info['fps']
        )

        # Build content parts: interleave timestamp labels + images
        contents = []
        for i, frame_data in enumerate(frames_data):
            # Inline image bytes
            buf = io.BytesIO()
            frame_data['image'].save(buf, format='JPEG', quality=85)
            contents.append(
                genai_types.Part.from_bytes(
                    data=buf.getvalue(),
                    mime_type='image/jpeg'
                )
            )
            # Add timestamp label every 8 frames
            if i % 8 == 0:
                contents.append(
                    genai_types.Part.from_text(
                        f"[Frame {i+1}/{len(frames_data)} @ {frame_data['timestamp_str']}]"
                    )
                )
        # Final prompt
        contents.append(genai_types.Part.from_text(prompt))

        try:
            response = self.client.models.generate_content(
                model=self.MODEL,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=8192,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=5000),
                ),
            )

            result_text = response.text

            # Strip markdown fences if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            analysis = json.loads(result_text.strip())
            analysis['metadata'] = {
                'video_path': video_path,
                'video_info': video_info,
                'num_frames_analyzed': len(frames_data),
                'model': self.MODEL,
                'frame_timestamps': [f['timestamp_str'] for f in frames_data]
            }
            return analysis

        except Exception as e:
            return {
                'error': str(e),
                'video_path': video_path,
                'model': self.MODEL
            }


class ClaudeAnalyzer(VLMAnalyzer):
    """Claude 3 Opus/Sonnet Vision Analyzer"""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        super().__init__(api_key, model)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic not installed. Install: pip install anthropic")
        
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def analyze_video(self, video_path: str, num_frames: int = 32) -> Dict[str, Any]:
        """Analyze video using Claude"""
        print(f"Extracting {num_frames} frames from video...")
        frames_data, video_info = VideoFrameExtractor.extract_frames_with_timestamps(
            video_path, num_frames
        )
        
        print(f"Analyzing with Claude ({len(frames_data)} frames)...")
        
        # Prepare prompt
        prompt = self.VQA_PROMPT_TEMPLATE.format(
            total_frames=video_info['total_frames'],
            duration=video_info['duration'],
            fps=video_info['fps']
        )
        
        # Prepare content with frames
        content = []
        for i, frame_data in enumerate(frames_data):
            base64_image = self.image_to_base64(frame_data['image'])
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_image
                }
            })
            # Add timestamp context every few frames
            if i % 8 == 0:
                content.append({
                    "type": "text",
                    "text": f"[Frame {i+1}/{len(frames_data)} at {frame_data['timestamp_str']}]"
                })
        
        content.append({
            "type": "text",
            "text": prompt
        })
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                temperature=0.2,
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )
            
            result_text = response.content[0].text
            
            # Parse JSON response
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            analysis = json.loads(result_text.strip())
            
            # Add metadata
            analysis['metadata'] = {
                'video_path': video_path,
                'video_info': video_info,
                'num_frames_analyzed': len(frames_data),
                'model': self.model_name,
                'frame_timestamps': [f['timestamp_str'] for f in frames_data]
            }
            
            return analysis
            
        except Exception as e:
            return {
                'error': str(e),
                'video_path': video_path,
                'model': self.model_name
            }


class GPT4VisionAnalyzer(VLMAnalyzer):
    """GPT-4 Vision Analyzer"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        super().__init__(api_key, model)
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not installed. Install: pip install openai")
        
        self.client = openai.OpenAI(api_key=api_key)
    
    def analyze_video(self, video_path: str, num_frames: int = 32) -> Dict[str, Any]:
        """Analyze video using GPT-4 Vision"""
        print(f"Extracting {num_frames} frames from video...")
        frames_data, video_info = VideoFrameExtractor.extract_frames_with_timestamps(
            video_path, num_frames
        )
        
        print(f"Analyzing with {self.model_name} ({len(frames_data)} frames)...")
        
        # Prepare prompt
        prompt = self.VQA_PROMPT_TEMPLATE.format(
            total_frames=video_info['total_frames'],
            duration=video_info['duration'],
            fps=video_info['fps']
        )
        
        # Prepare content with frames
        content = [{"type": "text", "text": prompt}]
        
        for i, frame_data in enumerate(frames_data):
            base64_image = self.image_to_base64(frame_data['image'])
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
            })
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": content
                }],
                max_tokens=4096,
                temperature=0.2
            )
            
            result_text = response.choices[0].message.content
            
            # Parse JSON response
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            analysis = json.loads(result_text.strip())
            
            # Add metadata
            analysis['metadata'] = {
                'video_path': video_path,
                'video_info': video_info,
                'num_frames_analyzed': len(frames_data),
                'model': self.model_name,
                'frame_timestamps': [f['timestamp_str'] for f in frames_data]
            }
            
            return analysis
            
        except Exception as e:
            return {
                'error': str(e),
                'video_path': video_path,
                'model': self.model_name
            }


class OllamaAnalyzer(VLMAnalyzer):
    """本地 Ollama VLM 分析器 — 无需 API Key，完全离线运行"""

    VISION_TAGS = ["llava", "llama3.2-vision", "minicpm-v", "bakllava", "moondream", "minicpm"]

    def __init__(self, model: str = "llama3.2-vision:latest",
                 ollama_url: str = "http://localhost:11434"):
        super().__init__(api_key="local", model_name=model)
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")

    @classmethod
    def list_vision_models(cls, ollama_url: str = "http://localhost:11434") -> list:
        """从 Ollama 获取已安装的视觉模型列表"""
        import requests as _req
        try:
            r = _req.get(f"{ollama_url}/api/tags", timeout=5)
            r.raise_for_status()
            all_models = r.json().get("models", [])
            vision = [
                {"name": m["name"], "size": m.get("size", 0),
                 "modified_at": m.get("modified_at", "")}
                for m in all_models
                if any(tag in m["name"].lower() for tag in cls.VISION_TAGS)
            ]
            # 如果没有识别到，返回全部
            return vision if vision else [
                {"name": m["name"], "size": m.get("size", 0)} for m in all_models
            ]
        except Exception:
            return []

    def analyze_video(self, video_path: str, num_frames: int = 16) -> Dict[str, Any]:
        """使用 Ollama 本地视觉模型分析视频"""
        import requests as _req

        # 本地模型建议 8–16 帧，避免超时与上下文溢出
        num_frames = min(num_frames, 16)

        print(f"提取 {num_frames} 帧...")
        frames_data, video_info = VideoFrameExtractor.extract_frames_with_timestamps(
            video_path, num_frames
        )
        print(f"使用 Ollama/{self.model} 分析 ({len(frames_data)} 帧)...")

        prompt = self.VQA_PROMPT_TEMPLATE.format(
            total_frames=video_info['total_frames'],
            duration=video_info['duration'],
            fps=video_info['fps']
        )
        # 把时间戳信息注入 prompt 开头（本地模型无法交替图像与文本）
        ts_ctx = " | ".join(
            f"Frame{i+1}={fd['timestamp_str']}" for i, fd in enumerate(frames_data)
        )
        full_prompt = f"帧时间戳: {ts_ctx}\n\n{prompt}"

        # base64 图像列表（纯 base64，不含 data URI 前缀）
        images = [
            self.image_to_base64(fd['image'], format='JPEG')
            for fd in frames_data
        ]

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": full_prompt, "images": images}
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 4096,
                "num_ctx": 8192,
            },
        }

        try:
            resp = _req.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=600,        # 本地模型最多等 10 分钟
            )
            resp.raise_for_status()
            result_text = resp.json()["message"]["content"]

            # 去掉 Markdown 代码块
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            # 提取 JSON 主体
            start = result_text.find("{")
            end = result_text.rfind("}") + 1
            if start >= 0 and end > start:
                result_text = result_text[start:end]

            analysis = json.loads(result_text.strip())
            analysis['metadata'] = {
                'video_path': video_path,
                'video_info': video_info,
                'num_frames_analyzed': len(frames_data),
                'model': f"ollama/{self.model}",
                'frame_timestamps': [f['timestamp_str'] for f in frames_data],
                'local': True,
            }
            return analysis

        except json.JSONDecodeError as e:
            # 本地模型有时 JSON 不完整，保存原始文本供调试
            return {
                'error': f'JSON 解析失败: {e}',
                'raw_response': result_text[:3000],
                'video_path': video_path,
                'model': f"ollama/{self.model}",
                'metadata': {'model': f"ollama/{self.model}", 'local': True},
            }
        except Exception as e:
            return {
                'error': str(e),
                'video_path': video_path,
                'model': f"ollama/{self.model}",
            }


def create_analyzer(provider: str = "gemini", api_key: str = None, model: str = None) -> VLMAnalyzer:
    """Factory function to create appropriate analyzer"""
    if provider == "gemini":
        return GeminiAnalyzer(api_key)
    elif provider == "claude":
        model = model or "claude-3-5-sonnet-20241022"
        return ClaudeAnalyzer(api_key, model)
    elif provider == "openai":
        model = model or "gpt-4o"
        return GPT4VisionAnalyzer(api_key, model)
    elif provider in ("local", "ollama"):
        model = model or "llama3.2-vision:latest"
        return OllamaAnalyzer(model)
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose from: gemini, claude, openai, local")


def main():
    """Main entry point for CLI usage"""
    if len(sys.argv) < 4:
        print(json.dumps({
            "error": "Usage: python vlm_video_analyzer.py <provider> <api_key> <video_path> [num_frames] [model]",
            "providers": ["gemini", "claude", "openai", "local"]
        }))
        return
    
    provider = sys.argv[1]
    api_key = sys.argv[2]
    video_path = sys.argv[3]
    num_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    model = sys.argv[5] if len(sys.argv) > 5 else None
    
    try:
        # Create analyzer
        analyzer = create_analyzer(provider, api_key, model)
        
        # Analyze video
        result = analyzer.analyze_video(video_path, num_frames)
        
        # Print result
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(json.dumps({
            "error": str(e),
            "traceback": str(e.__traceback__)
        }))


if __name__ == "__main__":
    main()
