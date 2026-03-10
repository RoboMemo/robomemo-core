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
    
    VQA_PROMPT_TEMPLATE = """
You are analyzing a robot manipulation video. Provide detailed, structured answers to the following 7 types of questions. Each answer must be grounded in specific visual evidence from the frames.

VIDEO CONTEXT:
- Total frames: {total_frames}
- Duration: {duration:.2f} seconds
- FPS: {fps:.2f}

ANALYZE THE VIDEO AND PROVIDE STRUCTURED ANSWERS FOR EACH CATEGORY:

1. TEMPORAL (Time Relationships):
   - Identify key actions in chronological order
   - Specify which actions happen before/after others
   - Provide exact timestamps for each action
   - Example: "At 0:02, gripper opens BEFORE grasping object at 0:04"

2. SPATIAL (Spatial Relationships):
   - Describe positions of hands/grippers relative to objects
   - Use precise spatial terms (left/right, above/below, near/far)
   - Track how spatial relationships change over time
   - Example: "Hand approaches from left side of cup at 45-degree angle"

3. ATTRIBUTE (Object Properties):
   - Identify all objects and their properties
   - Describe colors, materials, shapes, sizes
   - Note any changes in object states
   - Example: "Red plastic cup, cylindrical, ~10cm tall, initially empty"

4. MECHANICS (Force and Contact):
   - Identify contact points between gripper/hand and objects
   - Estimate force magnitude (light/medium/strong)
   - Describe grip type and contact area
   - Track force changes during manipulation
   - Example: "Pinch grip with 2 fingers, light force at 0:03, increases to medium at 0:05"

5. REASONING (Why This Way):
   - Explain the purpose of each action
   - Justify the approach and strategy
   - Identify constraints being satisfied
   - Example: "Approach from side to avoid blocking camera view and maintain stable grip"

6. SUMMARY (Overall Task):
   - Provide high-level description of complete task
   - Identify start and end states
   - List key milestones
   - Example: "Task: Pick and place red cup from table to shelf. Success: Yes"

7. TRAJECTORY (Motion Path):
   - Describe end-effector/hand motion paths
   - Specify movement types (linear, curved, rotational)
   - Note velocity changes (slow/fast)
   - Provide waypoints with timestamps
   - Example: "Linear approach 0:00-0:02 (slow), curved grasp 0:02-0:04, linear lift 0:04-0:06 (fast)"

IMPORTANT REQUIREMENTS:
- Ground every claim in visible evidence from specific frames
- Use exact timestamps (MM:SS format)
- Be precise with measurements and directions
- Ensure temporal consistency across all answers
- Identify uncertainties explicitly

Provide your analysis in JSON format with this structure:
{{
  "temporal": {{
    "action_sequence": [
      {{"action": "action_name", "timestamp": "MM:SS", "frame_range": [start, end], "description": "detailed description"}}
    ],
    "relationships": ["action1 BEFORE action2", ...]
  }},
  "spatial": {{
    "key_relationships": [
      {{"timestamp": "MM:SS", "relationship": "object1 position relative to object2", "details": "precise spatial description"}}
    ],
    "trajectory_spatial": "overall spatial strategy"
  }},
  "attribute": {{
    "objects": [
      {{"name": "object_name", "properties": {{"color": "", "material": "", "shape": "", "size": ""}}, "state_changes": []}}
    ]
  }},
  "mechanics": {{
    "contacts": [
      {{"timestamp": "MM:SS", "contact_type": "grip_type", "force_level": "light/medium/strong", "contact_points": "description", "area": "description"}}
    ],
    "force_profile": "how force changes over time"
  }},
  "reasoning": {{
    "action_justifications": [
      {{"action": "action_name", "reason": "why this approach", "constraints": ["constraint1", ...]}}
    ],
    "overall_strategy": "high-level reasoning"
  }},
  "summary": {{
    "task_description": "what task is being performed",
    "start_state": "initial configuration",
    "end_state": "final configuration",
    "success": true/false,
    "key_milestones": ["milestone1", ...],
    "duration": "total time"
  }},
  "trajectory": {{
    "motion_segments": [
      {{"segment": "segment_name", "time_range": "start-end", "motion_type": "linear/curved/rotational", "velocity": "slow/medium/fast", "waypoints": []}}
    ],
    "overall_path": "description of complete trajectory"
  }},
  "visual_evidence": {{
    "key_frames": [
      {{"frame_idx": 0, "timestamp": "MM:SS", "significance": "why this frame is important"}}
    ]
  }},
  "confidence_scores": {{
    "temporal": 0.0-1.0,
    "spatial": 0.0-1.0,
    "attribute": 0.0-1.0,
    "mechanics": 0.0-1.0,
    "reasoning": 0.0-1.0,
    "summary": 0.0-1.0,
    "trajectory": 0.0-1.0
  }}
}}

Return ONLY the JSON, no additional text.
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
