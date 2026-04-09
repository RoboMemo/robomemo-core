"""
SFT Pipeline
============
完整的一站式SFT流水线：视频 → LeRobot V2 → π₀.5配置
整合B站搜索、下载、VLM标注、导出全流程。
"""

import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加父目录到 Python 路径，支持相对导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用绝对导入（当作为模块运行时）或相对导入
try:
    from .auto_label_pipeline import (
        AutoLabelPipeline,
        VLMBackend,
        OllamaBackend,
        GeminiBackend,
        MockVLMBackend,
    )
    from .lerobot_exporter import export_lerobot
    from .openpi_config_generator import OpenPIFinetuneCfg, save_openpi_config
except ImportError:
    # 当作为 CLI 运行时，使用绝对导入
    from auto_label_pipeline import (
        AutoLabelPipeline,
        VLMBackend,
        OllamaBackend,
        GeminiBackend,
        MockVLMBackend,
    )
    from lerobot_exporter import export_lerobot
    from openpi_config_generator import OpenPIFinetuneCfg, save_openpi_config


class SFTPipeline:
    """
    完整的SFT流水线。
    
    Usage:
        pipeline = SFTPipeline(vlm_backend='gemini', api_key='xxx')
        result = pipeline.run(['video1.mp4', 'video2.mp4'])
    """

    def __init__(
        self,
        vlm_backend: str = 'mock',
        api_key: Optional[str] = None,
        model: str = 'gemini-2.0-flash',
        ollama_url: str = 'http://localhost:11434',
        output_dir: str = './sft_output',
        robot_type: str = 'single_arm',
    ):
        self.output_dir = output_dir
        self.robot_type = robot_type
        self.vlm = self._init_vlm(vlm_backend, api_key, model, ollama_url)

    def _init_vlm(
        self,
        backend: str,
        api_key: Optional[str],
        model: str,
        ollama_url: str,
    ) -> VLMBackend:
        """初始化 VLM 后端"""
        if backend == 'gemini':
            if not api_key:
                api_key = os.environ.get('GEMINI_API_KEY')
            return GeminiBackend(api_key=api_key, model=model)
        elif backend == 'ollama':
            return OllamaBackend(base_url=ollama_url, model=model)
        else:
            return MockVLMBackend()

    def run(
        self,
        video_paths: List[str],
        task_name: str = 'demo_task',
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        运行完整 SFT 流水线。
        
        Args:
            video_paths: 视频文件路径列表
            task_name: 任务名称
            dry_run: 是否只模拟运行（不实际调用 VLM）
            
        Returns:
            {
                "success": True,
                "output_dir": "...",
                "episodes": [...],
                "stats": {...}
            }
        """
        # 创建输出目录
        output_path = Path(self.output_dir) / task_name / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. VLM 自动标注 (传入 vlm 实例，而不是 vlm_backend)
        labeler = AutoLabelPipeline(vlm=self.vlm, num_frames=16)
        episodes = []
        
        for i, video_path in enumerate(video_paths):
            print(f"[SFT] Processing video {i+1}/{len(video_paths)}: {video_path}")
            
            try:
                if dry_run:
                    # 模拟运行，使用 mock 数据
                    episode = self._create_mock_episode(video_path, i)
                else:
                    episode = labeler.run(video_path, dry_run=False)
                episodes.append(episode)
            except Exception as e:
                print(f"[SFT] Error processing {video_path}: {e}")
                continue
        
        # 2. 导出 LeRobot 格式
        lerobot_output = output_path / 'lerobot'
        export_result = export_lerobot(episodes, output_dir=str(lerobot_output))
        
        # 3. 生成 π₀.5 训练配置
        config = OpenPIFinetuneCfg(
            output_dir=str(output_path / 'checkpoints'),
            experiment_name=f"{task_name}_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        )
        # 设置数据目录
        config.data.data_dir = str(lerobot_output)
        config_path = save_openpi_config(str(output_path), config)
        
        return {
            "success": True,
            "output_dir": str(output_path),
            "episodes_count": len(episodes),
            "lerobot_dir": str(lerobot_output),
            "config_path": str(config_path),
            "stats": {
                "total_videos": len(video_paths),
                "successful": len(episodes),
                "failed": len(video_paths) - len(episodes),
            }
        }

    def _create_mock_episode(self, video_path: str, episode_id: int) -> Dict[str, Any]:
        """创建模拟 episode 数据"""
        return {
            "episode_id": episode_id,
            "video_path": video_path,
            "frames": [
                {
                    "frame_id": i,
                    "timestamp": i * 0.5,
                    "action": [0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "observation": {
                        "joint_positions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        "gripper_state": 0.5,
                    },
                    "description": f"Frame {i}: mock action",
                }
                for i in range(10)
            ],
            "metadata": {
                "task": "demo_task",
                "robot_type": self.robot_type,
                "duration": 5.0,
            }
        }


def run_sft_pipeline(
    video_paths: List[str],
    vlm_backend: str = 'mock',
    output_dir: str = './sft_output',
    task_name: str = 'demo_task',
    dry_run: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    便捷函数：运行 SFT 流水线。
    """
    pipeline = SFTPipeline(
        vlm_backend=vlm_backend,
        output_dir=output_dir,
        **kwargs,
    )
    return pipeline.run(video_paths, task_name=task_name, dry_run=dry_run)


# CLI 接口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SFT Pipeline")
    parser.add_argument("videos", nargs="*", help="视频文件路径")
    parser.add_argument("--task", default="demo_task", help="任务名称")
    parser.add_argument("--vlm", default="mock", choices=["mock", "gemini", "ollama"], help="VLM后端")
    parser.add_argument("--output", default="./sft_output", help="输出目录")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行")
    parser.add_argument("--json", action="store_true", help="输出JSON格式")
    
    args = parser.parse_args()
    
    if not args.videos:
        # 如果没有提供视频，创建测试数据
        args.videos = ["test_video.mp4"]
        args.dry_run = True
        print("[SFT] No videos provided, using mock data for testing")
    
    result = run_sft_pipeline(
        video_paths=args.videos,
        vlm_backend=args.vlm,
        output_dir=args.output,
        task_name=args.task,
        dry_run=args.dry_run,
    )
    
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"SFT Pipeline Complete!")
        print(f"  Output: {result['output_dir']}")
        print(f"  Episodes: {result['episodes_count']}")
        print(f"  Stats: {result['stats']}")
        print(f"{'='*60}")
