"""
Bilibili Video Prescreen Service
================================
评估B站视频质量，筛选适合机器人学习训练的视频。
从 DataCapture/Data_Preprocessing/bilibili_prescreen.py 迁移并增强。

Usage:
    from services.bili.prescreen import evaluate_video, batch_prescreen
    
    # 评估单个视频
    result = evaluate_video("BV1zMqsYNERc")
    
    # 批量评估
    results = batch_prescreen(["BV1zMqsYNERc", "BV1LH4y1c756"])
"""

import urllib.request
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# 评分配置
# ═══════════════════════════════════════════════════════════════════════════════

# 目标关键词（匹配加分）
TARGET_KEYWORDS = [
    # 中文
    'DIY', '拆解', '手工', '教程', '维修', '评测', '开箱', '木工',
    '烹饪', '组装', '制作', '安装', '焊接', '修复', '改装', '工具',
    '螺丝', '电钻', '电路', '电子', '机械', '3D打印', '缝纫',
    '第一视角', 'POV', '操作', '实操', '机器人', '机械臂',
    # 英文
    'teardown', 'repair', 'build', 'assembly', 'workshop', 'tutorial',
    'how to', 'first person', 'manipulation', 'robot', 'robotic'
]

# 负面关键词（匹配减分）
NEGATIVE_KEYWORDS = [
    'Vlog', '日常', '搞笑', '混剪', '集锦', '解说', '聊天', '直播',
    '舞蹈', '唱歌', '剧情', '动漫', '游戏', '电影', '综艺', '访谈',
    '广告', '带货', '推广', '游戏实况'
]

# B站分区 tid 参考（科技/生活/知识相关 = 高质量分区）
GOOD_TIDS = {
    95: "数码", 230: "极客DIY", 231: "电脑装机", 232: "硬件评测",
    21: "日常", 76: "美食制作", 138: "手工", 161: "生活技巧",
    162: "家居房产", 163: "家电维修", 36: "知识科普", 201: "科学科普",
    122: "技能学习", 124: "科普", 188: "科技", 189: "电脑技术"
}


# ═══════════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PrescreenResult:
    """预筛选结果"""
    bvid: str
    title: str
    score: int
    verdict: str
    reasons: List[str]
    details: Dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════════
# API 调用
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch(url: str, timeout: int = 10) -> dict:
    """发送HTTP请求获取JSON数据"""
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def get_video_info(bvid: str) -> Optional[dict]:
    """获取视频基本信息"""
    try:
        data = _fetch(f'https://api.bilibili.com/x/web-interface/view?bvid={bvid}')
        return data['data'] if data['code'] == 0 else None
    except Exception:
        return None


def get_video_tags(bvid: str) -> List[str]:
    """获取视频标签"""
    try:
        data = _fetch(f'https://api.bilibili.com/x/tag/archive/tags?bvid={bvid}')
        if data['code'] == 0:
            return [t['tag_name'] for t in data.get('data', [])]
    except Exception:
        pass
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# 评分引擎
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_video(bvid: str, verbose: bool = False) -> PrescreenResult:
    """
    评估单个视频质量。
    
    评分维度：
    1. 关键词匹配 (0-40分)
    2. 视频分区 (0-20分)
    3. 视频时长 (0-15分)
    4. 分辨率 (0-15分)
    5. 互动率 (0-10分)
    
    Args:
        bvid: 视频BV号
        verbose: 是否打印详细信息
        
    Returns:
        PrescreenResult 对象
    """
    info = get_video_info(bvid)
    if not info:
        return PrescreenResult(
            bvid=bvid,
            title="未知",
            score=0,
            verdict="❌ 无法获取视频信息",
            reasons=["视频不存在或API访问失败"],
            details={}
        )
    
    tags = get_video_tags(bvid)
    text_corpus = (info['title'] + " " + info.get('desc', '') + " " + " ".join(tags)).lower()
    
    score = 0
    reasons = []
    
    # ── 维度1：关键词匹配 (满分 40) ─────────────────────────────────
    kw_score = 0
    hit_positive = []
    hit_negative = []
    
    for kw in TARGET_KEYWORDS:
        if kw.lower() in text_corpus:
            kw_score += 10
            hit_positive.append(kw)
    
    for kw in NEGATIVE_KEYWORDS:
        if kw.lower() in text_corpus:
            kw_score -= 15
            hit_negative.append(kw)
    
    kw_score = min(40, max(0, kw_score))
    score += kw_score
    
    if hit_positive:
        reasons.append(f"[+{kw_score}] 命中目标关键词: {', '.join(hit_positive[:5])}")
    if hit_negative:
        reasons.append(f"[-] 命中负面关键词: {', '.join(hit_negative)}")
    if not hit_positive and not hit_negative:
        reasons.append("[+0] 未命中关键词")
    
    # ── 维度2：视频分区 (满分 20) ─────────────────────────────────────
    tid = info.get('tid', 0)
    if tid in GOOD_TIDS:
        score += 20
        reasons.append(f"[+20] 分区优质: {GOOD_TIDS[tid]}({tid})")
    else:
        score += 8
        reasons.append(f"[+8] 分区一般 (tid={tid})")
    
    # ── 维度3：视频时长 (满分 15) ─────────────────────────────────────
    duration = info['duration']
    if 180 <= duration <= 900:  # 3-15分钟
        score += 15
        reasons.append(f"[+15] 时长适中: {duration//60}分{duration%60}秒")
    elif 60 <= duration < 180 or 900 < duration <= 1800:
        score += 7
        reasons.append(f"[+7] 时长尚可: {duration//60}分{duration%60}秒")
    else:
        reasons.append(f"[+0] 时长异常: {duration//60}分{duration%60}秒")
    
    # ── 维度4：分辨率 (满分 15) ───────────────────────────────────────
    w = info.get('dimension', {}).get('width', 0)
    h = info.get('dimension', {}).get('height', 0)
    if w >= 1920 or h >= 1080:
        score += 15
        reasons.append(f"[+15] 分辨率达标: {w}x{h} (≥1080P)")
    elif w >= 1280 or h >= 720:
        score += 7
        reasons.append(f"[+7] 分辨率一般: {w}x{h} (720P)")
    else:
        reasons.append(f"[+0] 分辨率过低: {w}x{h}")
    
    # ── 维度5：互动率 (满分 10) ───────────────────────────────────────
    view = info['stat']['view']
    if view > 0:
        like = info['stat']['like']
        fav = info['stat']['favorite']
        coin = info['stat']['coin']
        ir = (like + fav + coin) / view
        if ir > 0.08:
            score += 10
            reasons.append(f"[+10] 互动率极高: {ir:.1%}")
        elif ir > 0.04:
            score += 5
            reasons.append(f"[+5] 互动率良好: {ir:.1%}")
        else:
            reasons.append(f"[+0] 互动率偏低: {ir:.1%}")
    else:
        reasons.append("[+0] 播放量为0")
    
    # ── 综合判定 ─────────────────────────────────────────────────────
    if score >= 70:
        verdict = "✅ 强烈推荐下载！极有可能包含高质量操作演示。"
    elif score >= 50:
        verdict = "⚠️ 可以下载，建议人工复核封面和简介。"
    elif score >= 30:
        verdict = "❓ 质量存疑，建议仔细查看后再决定。"
    else:
        verdict = "❌ 不建议下载，大概率不符合训练要求。"
    
    details = {
        "title": info['title'],
        "author": info.get('owner', {}).get('name', ''),
        "duration": duration,
        "duration_formatted": f"{duration//60}:{duration%60:02d}",
        "views": view,
        "likes": info['stat']['like'],
        "coins": info['stat']['coin'],
        "favorites": info['stat']['favorite'],
        "resolution": f"{w}x{h}",
        "tid": tid,
        "tname": info.get('tname', ''),
        "tags": tags,
        "cover": info.get('pic', ''),
        "pubdate": info.get('pubdate', 0)
    }
    
    if verbose:
        print(f"\n{'═'*50}")
        print(f"  {info['title']}")
        print(f"  BVID: {bvid}  |  时长: {duration//60}:{duration%60:02d}  |  分辨率: {w}x{h}")
        print(f"  评分: {score}/100  |  {verdict}")
        for r in reasons:
            print(f"    {r}")
        print(f"{'═'*50}")
    
    return PrescreenResult(
        bvid=bvid,
        title=info['title'],
        score=score,
        verdict=verdict,
        reasons=reasons,
        details=details
    )


def batch_prescreen(
    bvids: List[str],
    min_score: int = 50,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    批量预筛选视频。
    
    Args:
        bvids: BV号列表
        min_score: 最低分数阈值
        verbose: 是否打印详细信息
        
    Returns:
        {
            "total": 10,
            "passed": 5,
            "results": [PrescreenResult, ...],
            "recommended": [bvid, ...]  # 通过筛选的BV号
        }
    """
    results = []
    
    for bvid in bvids:
        result = evaluate_video(bvid, verbose=verbose)
        results.append(result)
    
    # 按分数排序
    results.sort(key=lambda x: x.score, reverse=True)
    
    # 筛选通过的视频
    passed = [r for r in results if r.score >= min_score]
    
    return {
        "total": len(bvids),
        "passed": len(passed),
        "min_score": min_score,
        "results": [{
            "bvid": r.bvid,
            "title": r.title,
            "score": r.score,
            "verdict": r.verdict,
            "reasons": r.reasons,
            "details": r.details
        } for r in results],
        "recommended": [r.bvid for r in passed]
    }


def extract_bvid(text: str) -> Optional[str]:
    """从文本中提取BV号"""
    match = re.search(r'(BV[a-zA-Z0-9]{10})', text)
    return match.group(1) if match else None


# ═══════════════════════════════════════════════════════════════════════════════
# CLI 接口
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Bilibili Video Prescreen")
    parser.add_argument("bvids", nargs="*", help="BV号列表")
    parser.add_argument("--min-score", type=int, default=50, help="最低分数阈值")
    parser.add_argument("--json", action="store_true", help="输出JSON格式")
    
    args = parser.parse_args()
    
    # 提取BV号
    bvids = []
    for arg in args.bvids:
        bvid = extract_bvid(arg)
        if bvid:
            bvids.append(bvid)
    
    if not bvids:
        result = {"error": "未找到有效的BV号", "results": []}
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)
    
    # 执行批量筛选
    result = batch_prescreen(bvids, min_score=args.min_score, verbose=not args.json)
    
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"\n{'═'*50}")
        print(f"  批量筛选汇总: {result['passed']}/{result['total']} 通过")
        print(f"{'═'*50}")
