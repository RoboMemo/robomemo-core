"""
Video Downloader Service
========================
使用 yt-dlp 下载B站、YouTube等平台的视频。
从 openclaw_backup_20260408/workspace/skills/video-downloader/scripts/download-video.py 迁移。

Usage:
    from services.bili.video_downloader import download_video, download_batch
    
    # 下载单个视频
    result = download_video("BV1zMqsYNERc", output_dir="./downloads")
    
    # 批量下载
    results = download_batch(["BV1zMqsYNERc", "BV1LH4y1c756"])
"""

import os
import sys
import subprocess
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import urlparse


# ═══════════════════════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_DOWNLOADS_DIR = Path(__file__).parent.parent.parent / "uploads" / "videos"
MAX_RETRIES = 3
DEFAULT_FORMAT = "best[height<=1080]"


@dataclass
class DownloadResult:
    """下载结果"""
    success: bool
    bvid: str
    url: str
    platform: str
    filename: Optional[str] = None
    filepath: Optional[str] = None
    error: Optional[str] = None
    duration: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# 核心功能
# ═══════════════════════════════════════════════════════════════════════════════

def detect_platform(url: str) -> str:
    """检测平台类型"""
    domain = urlparse(url).netloc.lower()
    platform_map = {
        "bilibili.com": "bilibili",
        "b23.tv": "bilibili",
        "youtube.com": "youtube",
        "youtu.be": "youtube",
        "vimeo.com": "vimeo",
    }
    for domain_prefix, platform in platform_map.items():
        if domain_prefix in domain:
            return platform
    return "generic"


def bvid_to_url(bvid: str) -> str:
    """将BV号转换为完整URL"""
    if bvid.startswith("BV"):
        return f"https://www.bilibili.com/video/{bvid}"
    return bvid  # 假设已经是URL


def extract_bvid(text: str) -> Optional[str]:
    """从文本中提取BV号"""
    match = re.search(r'(BV[a-zA-Z0-9]{10})', text)
    return match.group(1) if match else None


def build_yt_dlp_args(
    url: str,
    output_dir: str,
    format_spec: str = DEFAULT_FORMAT,
    embed_subtitles: bool = False
) -> List[str]:
    """
    构建 yt-dlp 命令参数。
    
    Args:
        url: 视频URL
        output_dir: 输出目录
        format_spec: 格式选择器
        embed_subtitles: 是否嵌入字幕
        
    Returns:
        命令参数列表
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    platform = detect_platform(url)
    
    args = [
        "yt-dlp",
        "-f", format_spec,
        "--no-playlist",
        "--no-warnings",
        "--newline",  # 实时输出进度
    ]
    
    if platform == "bilibili":
        args.extend([
            "--write-subs",
            "--embed-subs",
            "-o", str(output_path / "%(title)s_%(id)s.%(ext)s"),
        ])
    elif platform == "youtube":
        args.extend([
            "--write-auto-subs",
            "--sub-lang", "zh-Hans,zh,en",
            "--embed-subs",
            "-o", str(output_path / "%(title)s_%(id)s.%(ext)s"),
        ])
    else:
        args.extend([
            "-o", str(output_path / "%(title)s_%(id)s.%(ext)s"),
        ])
    
    args.append(url)
    return args


def download_video(
    bvid_or_url: str,
    output_dir: Optional[str] = None,
    format_spec: str = DEFAULT_FORMAT,
    timeout: int = 3600
) -> DownloadResult:
    """
    下载单个视频。
    
    Args:
        bvid_or_url: BV号或视频URL
        output_dir: 输出目录（默认: backend/uploads/videos）
        format_spec: yt-dlp 格式选择器
        timeout: 超时时间（秒）
        
    Returns:
        DownloadResult 对象
    """
    # 处理输入
    if bvid_or_url.startswith("BV"):
        url = bvid_to_url(bvid_or_url)
        bvid = bvid_or_url
    else:
        url = bvid_or_url
        bvid = extract_bvid(url) or url
    
    if output_dir is None:
        output_dir = str(DEFAULT_DOWNLOADS_DIR)
    
    platform = detect_platform(url)
    
    # 构建 yt-dlp 命令
    args = build_yt_dlp_args(url, output_dir, format_spec)
    
    for attempt in range(MAX_RETRIES):
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=output_dir
            )
            
            if result.returncode == 0:
                # 解析输出文件名
                output_path = Path(output_dir)
                video_files = list(output_path.glob("*.mp4")) + list(output_path.glob("*.webm"))
                
                if video_files:
                    # 找到最新下载的文件
                    latest_file = sorted(video_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                    
                    return DownloadResult(
                        success=True,
                        bvid=bvid,
                        url=url,
                        platform=platform,
                        filename=latest_file.name,
                        filepath=str(latest_file),
                        duration=_extract_duration(result.stderr)
                    )
            
            # 失败重试
            error_msg = f"yt-dlp error (attempt {attempt + 1}/{MAX_RETRIES})"
            if "HTTP Error" in result.stderr:
                error_msg += " - HTTP Error, try updating yt-dlp"
            
        except subprocess.TimeoutExpired:
            error_msg = f"Download timeout after {timeout}s (attempt {attempt + 1}/{MAX_RETRIES})"
        except FileNotFoundError:
            return DownloadResult(
                success=False,
                bvid=bvid,
                url=url,
                platform=platform,
                error="yt-dlp not found. Install with: pip install yt-dlp"
            )
        except Exception as e:
            error_msg = f"Download failed: {str(e)}"
    
    return DownloadResult(
        success=False,
        bvid=bvid,
        url=url,
        platform=platform,
        error=error_msg
    )


def download_batch(
    bvids: List[str],
    output_dir: Optional[str] = None,
    format_spec: str = DEFAULT_FORMAT
) -> Dict[str, Any]:
    """
    批量下载视频。
    
    Args:
        bvids: BV号列表
        output_dir: 输出目录
        format_spec: 格式选择器
        
    Returns:
        {
            "total": 10,
            "success": 8,
            "failed": 2,
            "results": [DownloadResult, ...]
        }
    """
    results = []
    success_count = 0
    failed_count = 0
    
    for i, bvid in enumerate(bvids, 1):
        print(f"[{i}/{len(bvids)}] Downloading: {bvid}")
        
        result = download_video(bvid, output_dir, format_spec)
        results.append(result)
        
        if result.success:
            success_count += 1
            print(f"  ✅ Success: {result.filename}")
        else:
            failed_count += 1
            print(f"  ❌ Failed: {result.error}")
    
    return {
        "total": len(bvids),
        "success": success_count,
        "failed": failed_count,
        "results": [{
            "success": r.success,
            "bvid": r.bvid,
            "url": r.url,
            "platform": r.platform,
            "filename": r.filename,
            "filepath": r.filepath,
            "error": r.error
        } for r in results]
    }


def get_video_info(url: str) -> Dict[str, Any]:
    """
    获取视频信息（不下载）。
    
    Args:
        url: 视频URL或BV号
        
    Returns:
        视频信息字典
    """
    if url.startswith("BV"):
        url = bvid_to_url(url)
    
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--dump-json",
                "--no-playlist",
                url
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            info = json.loads(result.stdout)
            return {
                "success": True,
                "title": info.get("title", ""),
                "duration": info.get("duration", 0),
                "uploader": info.get("uploader", ""),
                "view_count": info.get("view_count", 0),
                "platform": detect_platform(url),
                "url": url
            }
        else:
            return {
                "success": False,
                "error": f"Failed to get info: {result.stderr[:200]}"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def _extract_duration(stderr: str) -> Optional[str]:
    """从 yt-dlp 输出中提取下载时长"""
    match = re.search(r'elapsed[^\d]*(\d+:\d+:\d+|\d+:\d+)', stderr)
    return match.group(1) if match else None


# ═══════════════════════════════════════════════════════════════════════════════
# CLI 接口
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python video_downloader.py <bvid|url> [--batch|--info]")
        print("\nCommands:")
        print("  <bvid|url>       Download single video")
        print("  --batch          Read BV IDs from stdin")
        print("  --info           Get video info without downloading")
        print("\nExample:")
        print("  python video_downloader.py BV1zMqsYNERc")
        print("  python video_downloader.py BV1zMqsYNERc --info")
        sys.exit(1)
    
    bvid_or_url = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else None
    
    if mode == "--info":
        result = get_video_info(bvid_or_url)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    elif mode == "--batch":
        # 从标准输入读取
        bvids = [line.strip() for line in sys.stdin if line.strip()]
        result = download_batch(bvids)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    else:
        result = download_video(bvid_or_url)
        if result.success:
            print(f"✅ Download completed:")
            print(f"   File: {result.filename}")
            print(f"   Path: {result.filepath}")
        else:
            print(f"❌ Download failed: {result.error}")
