"""
Bilibili Video Intelligence Service
====================================
搜索B站视频并提取BV号、标题、简介等信息。
从 openclaw_backup_20260408/workspace/skills/bili_hunter/scripts/bili_intel.py 迁移。

Usage:
    from services.bili.bili_intel import search_bilibili, get_video_info
    
    # 搜索视频
    results = search_bilibili("拧螺丝", page_size=20)
    
    # 获取视频详情
    info = get_video_info("BV1zMqsYNERc")
"""

import json
import re
import sys
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# 使用 requests 库处理 gzip 和更好的请求头
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    import urllib.parse
    HAS_REQUESTS = False


@dataclass
class BiliVideo:
    """B站视频信息结构"""
    bvid: str
    title: str
    url: str
    snippet: str
    duration: str
    views: int
    author: str
    cover: str
    pubdate: int


# B站请求头
BILI_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.bilibili.com/',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Origin': 'https://www.bilibili.com',
    'Connection': 'keep-alive',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-site',
    'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"Windows"',
}


def _fetch(url: str, timeout: int = 20) -> dict:
    """发送HTTP请求获取JSON数据"""
    if HAS_REQUESTS:
        resp = requests.get(url, headers=BILI_HEADERS, timeout=timeout)
        return resp.json()
    else:
        req = urllib.request.Request(url, headers=BILI_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))


def search_bilibili(
    keyword: str,
    page: int = 1,
    page_size: int = 20,
    order: str = "totalrank"  # totalrank: 综合排序, pubdate: 发布日期, click: 播放量
) -> Dict[str, Any]:
    """
    搜索B站视频，返回视频列表。
    
    Args:
        keyword: 搜索关键词
        page: 页码
        page_size: 每页数量 (最大50)
        order: 排序方式
        
    Returns:
        {
            "success": True,
            "total": 100,
            "videos": [BiliVideo, ...]
        }
    """
    search_url = "https://api.bilibili.com/x/web-interface/search/type"
    
    try:
        params = {
            "search_type": "video",
            "keyword": keyword,
            "page": page,
            "page_size": min(page_size, 50),
            "order": order
        }
        
        if HAS_REQUESTS:
            resp = requests.get(search_url, params=params, headers=BILI_HEADERS, timeout=20)
            data = resp.json()
        else:
            url = f"{search_url}?{urllib.parse.urlencode(params)}"
            data = _fetch(url)
        
        if data.get('code') != 0:
            return {
                "success": False,
                "error": f"B站API错误：{data.get('message', '未知错误')}",
                "videos": []
            }
        
        videos = []
        for item in data.get('data', {}).get('result', []):
            # 清理标题中的HTML标签
            title = re.sub(r'<[^>]+>', '', item.get('title', ''))
            
            video = BiliVideo(
                bvid=item.get('bvid', ''),
                title=title,
                url=f"https://www.bilibili.com/video/{item.get('bvid', '')}",
                snippet=item.get('description', '')[:200],
                duration=_format_duration(item.get('duration', 0)),
                views=item.get('play', 0),
                author=item.get('author', ''),
                cover=item.get('pic', ''),
                pubdate=item.get('pubdate', 0)
            )
            videos.append(video.__dict__)
        
        return {
            "success": True,
            "total": data.get('data', {}).get('numResults', 0),
            "page": page,
            "page_size": page_size,
            "videos": videos
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"搜索失败：{str(e)}",
            "videos": []
        }


def get_video_info(bvid: str) -> Dict[str, Any]:
    """
    获取单个视频的详细信息。
    
    Args:
        bvid: 视频BV号
        
    Returns:
        视频详细信息字典
    """
    api_url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    
    try:
        data = _fetch(api_url)
        
        if data.get('code') != 0:
            return {
                "success": False,
                "error": f"获取视频信息失败：{data.get('message', '未知错误')}"
            }
        
        info = data.get('data', {})
        stat = info.get('stat', {})
        
        return {
            "success": True,
            "bvid": bvid,
            "title": info.get('title', ''),
            "description": info.get('desc', ''),
            "duration": info.get('duration', 0),
            "duration_formatted": _format_duration(info.get('duration', 0)),
            "views": stat.get('view', 0),
            "likes": stat.get('like', 0),
            "coins": stat.get('coin', 0),
            "favorites": stat.get('favorite', 0),
            "shares": stat.get('share', 0),
            "danmaku": stat.get('danmaku', 0),
            "author": info.get('owner', {}).get('name', ''),
            "author_mid": info.get('owner', {}).get('mid', 0),
            "cover": info.get('pic', ''),
            "pubdate": info.get('pubdate', 0),
            "tid": info.get('tid', 0),  # 分区ID
            "tname": info.get('tname', ''),  # 分区名称
            "dimension": info.get('dimension', {}),
            "url": f"https://www.bilibili.com/video/{bvid}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"获取视频信息失败：{str(e)}"
        }


def get_video_tags(bvid: str) -> List[str]:
    """
    获取视频标签。
    
    Args:
        bvid: 视频BV号
        
    Returns:
        标签列表
    """
    api_url = f"https://api.bilibili.com/x/tag/archive/tags?bvid={bvid}"
    
    try:
        data = _fetch(api_url)
        
        if data.get('code') != 0:
            return []
        
        return [t.get('tag_name', '') for t in data.get('data', [])]
        
    except Exception:
        return []


def _format_duration(seconds: int) -> str:
    """格式化时长为 HH:MM:SS 格式"""
    if isinstance(seconds, str):
        return seconds
    
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def extract_bvid(text: str) -> Optional[str]:
    """
    从文本中提取BV号。
    
    Args:
        text: 可能包含BV号或URL的文本
        
    Returns:
        BV号或None
    """
    match = re.search(r'(BV[a-zA-Z0-9]{10})', text)
    return match.group(1) if match else None


# CLI 接口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bilibili Video Intelligence")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # search 命令
    search_parser = subparsers.add_parser("search", help="Search videos")
    search_parser.add_argument("keyword", nargs="?", default="拧螺丝", help="Search keyword")
    search_parser.add_argument("--page", type=int, default=1, help="Page number")
    search_parser.add_argument("--page-size", type=int, default=20, help="Page size")
    search_parser.add_argument("--order", default="totalrank", help="Sort order")
    
    # info 命令
    info_parser = subparsers.add_parser("info", help="Get video info")
    info_parser.add_argument("bvid", nargs="?", default="BV1zMqsYNERc", help="Video BV ID")
    
    args = parser.parse_args()
    
    if args.command == "search":
        result = search_bilibili(args.keyword, page=args.page, page_size=args.page_size, order=args.order)
        print(json.dumps(result, ensure_ascii=False))
    
    elif args.command == "info":
        result = get_video_info(args.bvid)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    else:
        parser.print_help()
        sys.exit(1)
