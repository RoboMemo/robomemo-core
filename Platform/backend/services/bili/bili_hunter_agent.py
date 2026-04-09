"""
Bilibili Hunter Agent
=====================
自动化BV号收集Agent - 从 OpenClaw 迁移并增强

这个 Agent 可以：
1. 根据用户意图自动生成搜索关键词
2. 多轮搜索并收集BV号
3. 智能过滤不相关内容（游戏、动画、广告等）
4. 预筛选视频质量
5. 输出干净的BV号列表

Usage:
    from services.bili.bili_hunter_agent import BilibiliHunterAgent
    
    agent = BilibiliHunterAgent()
    result = agent.hunt("寻找拧螺丝的第一人称视角视频")
"""

import json
import re
import sys
from typing import List, Dict, Any, Optional
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
class HuntResult:
    """搜索结果"""
    bvid: str
    title: str
    url: str
    snippet: str
    duration: str
    views: int
    author: str
    score: int = 0
    passed_filter: bool = True
    filter_reason: str = ""


class BilibiliHunterAgent:
    """
    B站视频情报分析师 Agent
    
    自动化搜索、过滤、收集BV号
    """
    
    # 负面关键词（游戏、动画、广告等）
    NEGATIVE_KEYWORDS = [
        # 游戏相关
        '游戏', 'game', 'gaming', '实况', '通关', '攻略游戏', '我的世界', 'Minecraft',
        '原神', '崩坏', '王者', 'LOL', 'CSGO', '吃鸡', '绝地求生',
        # 动画相关
        '动画', 'anime', '番剧', 'MAD', 'AMV', '二次元', '动漫',
        # 广告相关
        '广告', '推广', '带货', '恰饭', '赞助', '合作',
        # 其他
        'Vlog', '日常', '搞笑', '整活', '鬼畜', '混剪', '集锦',
    ]
    
    # 目标关键词（机器人操作相关）
    TARGET_KEYWORDS = [
        # 中文
        'DIY', '拆解', '手工', '教程', '维修', '评测', '开箱', '木工',
        '烹饪', '组装', '制作', '安装', '焊接', '修复', '改装', '工具',
        '螺丝', '电钻', '电路', '电子', '机械', '3D打印', '缝纫',
        '第一视角', 'POV', '操作', '实操', '机器人', '机械臂', '抓取',
        '拧螺丝', '拧紧', '螺丝刀', '电动螺丝刀',
        # 英文
        'teardown', 'repair', 'build', 'assembly', 'workshop', 'tutorial',
        'how to', 'first person', 'manipulation', 'robot', 'robotic',
        'screw', 'drill', 'tighten', 'fasten',
    ]
    
    # B站搜索请求头
    HEADERS = {
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
    
    def __init__(self, max_results: int = 50, auto_filter: bool = True):
        self.max_results = max_results
        self.auto_filter = auto_filter
    
    def search(self, keyword: str, page_size: int = 20) -> List[HuntResult]:
        """
        搜索B站视频
        
        Args:
            keyword: 搜索关键词
            page_size: 每页数量
            
        Returns:
            HuntResult 列表
        """
        search_url = "https://api.bilibili.com/x/web-interface/search/type"
        
        try:
            params = {
                "search_type": "video",
                "keyword": keyword,
                "page": 1,
                "page_size": min(page_size, 50),
                "order": "totalrank"
            }
            
            if HAS_REQUESTS:
                # 使用 requests 库
                resp = requests.get(
                    search_url,
                    params=params,
                    headers=self.HEADERS,
                    timeout=20
                )
                data = resp.json()
            else:
                # 回退到 urllib
                url = f"{search_url}?{urllib.parse.urlencode(params)}"
                req = urllib.request.Request(url, headers=self.HEADERS)
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
            
            if data.get('code') != 0:
                print(f"B站API错误: {data.get('message', '未知')}", file=sys.stderr)
                return []
            
            results = []
            for item in data.get('data', {}).get('result', []):
                title = re.sub(r'<[^>]+>', '', item.get('title', ''))
                
                result = HuntResult(
                    bvid=item.get('bvid', ''),
                    title=title,
                    url=f"https://www.bilibili.com/video/{item.get('bvid', '')}",
                    snippet=item.get('description', '')[:200],
                    duration=self._format_duration(item.get('duration', 0)),
                    views=item.get('play', 0),
                    author=item.get('author', ''),
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Search error: {e}", file=sys.stderr)
            return []
    
    def filter_result(self, result: HuntResult) -> HuntResult:
        """
        过滤单个结果
        
        Returns:
            更新后的 HuntResult（包含 score 和 filter 信息）
        """
        text = f"{result.title} {result.snippet}".lower()
        score = 0
        
        # 检查负面关键词
        for neg_kw in self.NEGATIVE_KEYWORDS:
            if neg_kw.lower() in text:
                score -= 30
                result.filter_reason = f"包含负面关键词: {neg_kw}"
        
        # 检查目标关键词
        for target_kw in self.TARGET_KEYWORDS:
            if target_kw.lower() in text:
                score += 15
        
        # 时长评分（3-15分钟最佳）
        duration = result.duration
        if isinstance(duration, str):
            parts = duration.split(':')
            if len(parts) == 2:
                mins = int(parts[0])
                if 3 <= mins <= 15:
                    score += 10
                elif 1 <= mins < 3 or 15 < mins <= 30:
                    score += 5
        
        result.score = max(0, min(100, score + 50))  # 基础分50
        result.passed_filter = result.score >= 40
        
        return result
    
    def hunt(
        self,
        intent: str,
        keywords: Optional[List[str]] = None,
        auto_expand: bool = True,
        min_score: int = 40,
    ) -> Dict[str, Any]:
        """
        根据用户意图自动收集BV号
        
        Args:
            intent: 用户意图描述（如 "寻找拧螺丝的第一人称视角视频"）
            keywords: 手动指定的关键词列表（可选）
            auto_expand: 是否自动扩展搜索关键词
            min_score: 最低分数阈值
            
        Returns:
            {
                "intent": str,
                "keywords_used": List[str],
                "total_found": int,
                "passed_filter": int,
                "results": List[HuntResult],
                "bvids": List[str],  # 干净的BV号列表
            }
        """
        # 如果没有提供关键词，从意图中提取
        if not keywords:
            keywords = self._extract_keywords(intent)
        
        # 扩展关键词
        if auto_expand:
            keywords = self._expand_keywords(keywords)
        
        all_results: Dict[str, HuntResult] = {}  # 用 dict 去重
        
        # 多轮搜索
        for kw in keywords:
            results = self.search(kw, page_size=20)
            for r in results:
                if r.bvid not in all_results:
                    # 过滤
                    if self.auto_filter:
                        r = self.filter_result(r)
                    all_results[r.bvid] = r
        
        # 排序并筛选
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x.score,
            reverse=True
        )
        
        passed_results = [r for r in sorted_results if r.passed_filter and r.score >= min_score]
        passed_results = passed_results[:self.max_results]
        
        return {
            "intent": intent,
            "keywords_used": keywords,
            "total_found": len(all_results),
            "passed_filter": len(passed_results),
            "results": [{
                "bvid": r.bvid,
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet,
                "duration": r.duration,
                "views": r.views,
                "author": r.author,
                "score": r.score,
                "passed_filter": r.passed_filter,
                "filter_reason": r.filter_reason,
            } for r in passed_results],
            "bvids": [r.bvid for r in passed_results],
        }
    
    def _extract_keywords(self, intent: str) -> List[str]:
        """从意图中提取关键词"""
        # 简单的关键词提取（实际可以用 LLM 增强）
        keywords = []
        
        # 常见的关键词映射
        intent_keywords = {
            '拧螺丝': ['拧螺丝', '螺丝刀', '电动螺丝刀', '螺丝紧固'],
            '抓取': ['机器人抓取', '机械臂抓取', '物体抓取'],
            '第一视角': ['第一视角', 'POV', '第一人称'],
            '维修': ['维修', '修理', '拆解'],
            '组装': ['组装', '安装', '装配'],
        }
        
        for key, kws in intent_keywords.items():
            if key in intent:
                keywords.extend(kws)
        
        # 如果没有匹配，直接使用意图作为关键词
        if not keywords:
            keywords = [intent]
        
        return list(set(keywords))
    
    def _expand_keywords(self, keywords: List[str]) -> List[str]:
        """扩展关键词"""
        expanded = list(keywords)
        
        # 添加一些常见的高质量搜索组合
        suffixes = [' 教程', ' 实操', ' DIY', ' 第一视角']
        for kw in keywords[:3]:  # 只扩展前3个
            for suffix in suffixes:
                expanded.append(f"{kw}{suffix}")
        
        return list(set(expanded))[:10]  # 最多10个关键词
    
    def _format_duration(self, seconds) -> str:
        """格式化时长"""
        if isinstance(seconds, str):
            return seconds
        if isinstance(seconds, int):
            mins = seconds // 60
            secs = seconds % 60
            return f"{mins}:{secs:02d}"
        return "0:00"


# 便捷函数
def hunt_bvids(intent: str, min_score: int = 40, max_results: int = 20) -> List[str]:
    """
    便捷函数：根据意图快速获取BV号列表
    
    Args:
        intent: 用户意图
        min_score: 最低分数
        max_results: 最大结果数
        
    Returns:
        BV号列表
    """
    agent = BilibiliHunterAgent(max_results=max_results)
    result = agent.hunt(intent, min_score=min_score)
    return result["bvids"]


# CLI 接口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bilibili Hunter Agent")
    parser.add_argument("intent", help="搜索意图，如：寻找拧螺丝的第一人称视角视频")
    parser.add_argument("--min-score", type=int, default=40, help="最低分数阈值")
    parser.add_argument("--max-results", type=int, default=20, help="最大结果数")
    parser.add_argument("--json", action="store_true", help="输出JSON格式")
    
    args = parser.parse_args()
    
    agent = BilibiliHunterAgent(max_results=args.max_results)
    result = agent.hunt(args.intent, min_score=args.min_score)
    
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"搜索意图: {args.intent}")
        print(f"使用关键词: {', '.join(result['keywords_used'])}")
        print(f"找到 {result['total_found']} 个视频，{result['passed_filter']} 个通过筛选")
        print(f"{'='*60}\n")
        
        for i, r in enumerate(result['results'], 1):
            print(f"{i}. [{r['score']}分] {r['title'][:50]}")
            print(f"   {r['bvid']} | {r['duration']} | {r['views']}播放 | {r['author']}")
            if r['filter_reason']:
                print(f"   ⚠️ {r['filter_reason']}")
            print()
        
        print(f"\n干净的BV号列表：")
        print(", ".join(result['bvids']))
