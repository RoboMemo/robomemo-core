"""
RynnVLA-001 B站视频质量预筛选工具
用法: python bilibili_prescreen.py <URL或BVID> [<URL或BVID> ...]
示例: python bilibili_prescreen.py BV1zMqsYNERc BV1LH4y1c756
"""

import urllib.request
import json
import re
import argparse
import sys

# ─────────────────────────────────────────────
# 评分配置（可根据需要调整）
# ─────────────────────────────────────────────
TARGET_KEYWORDS = [
    'DIY', '拆解', '手工', '教程', '维修', '评测', '开箱', '木工',
    '烹饪', '组装', '制作', '安装', '焊接', '修复', '改装', '工具',
    '螺丝', '电钻', '电路', '电子', '机械', '3D打印', '缝纫',
    'teardown', 'repair', 'build', 'assembly', 'workshop'
]
NEGATIVE_KEYWORDS = [
    'Vlog', '日常', '搞笑', '混剪', '集锦', '解说', '聊天', '直播',
    '舞蹈', '唱歌', '剧情', '动漫', '游戏', '电影', '综艺', '访谈'
]
# B站分区 tid 参考（科技/生活/知识相关）
GOOD_TIDS = {
    95: "数码", 230: "极客DIY", 231: "电脑装机", 232: "硬件评测",
    21: "日常", 76: "美食制作", 138: "手工", 161: "生活技巧",
    162: "家居房产", 163: "家电维修", 36: "知识科普", 201: "科学科普"
}

# ─────────────────────────────────────────────
# API 调用
# ─────────────────────────────────────────────
def _fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())

def get_video_info(bvid):
    data = _fetch(f'https://api.bilibili.com/x/web-interface/view?bvid={bvid}')
    return data['data'] if data['code'] == 0 else None

def get_video_tags(bvid):
    try:
        data = _fetch(f'https://api.bilibili.com/x/tag/archive/tags?bvid={bvid}')
        if data['code'] == 0:
            return [t['tag_name'] for t in data.get('data', [])]
    except:
        pass
    return []

# ─────────────────────────────────────────────
# 评分引擎
# ─────────────────────────────────────────────
def evaluate_video(bvid):
    print(f"\n正在查询 {bvid}...")
    info = get_video_info(bvid)
    if not info:
        print(f"  ❌ 无法获取视频信息，跳过。")
        return None

    tags = get_video_tags(bvid)
    text_corpus = (info['title'] + " " + info['desc'] + " " + " ".join(tags)).lower()

    score = 0
    reasons = []

    # ── 维度1：关键词匹配 (满分 40) ──────────────
    kw_score = 0
    hit_positive = []
    hit_negative = []
    for kw in TARGET_KEYWORDS:
        if kw.lower() in text_corpus:
            kw_score += 12
            hit_positive.append(kw)
    for kw in NEGATIVE_KEYWORDS:
        if kw.lower() in text_corpus:
            kw_score -= 20
            hit_negative.append(kw)
    kw_score = min(40, max(0, kw_score))
    score += kw_score
    if hit_positive:
        reasons.append(f"[+{kw_score}] 命中目标关键词: {', '.join(hit_positive[:5])}")
    if hit_negative:
        reasons.append(f"[-] 命中负面关键词: {', '.join(hit_negative)}")

    # ── 维度2：视频分区 (满分 20) ─────────────────
    tid = info.get('tid', 0)
    if tid in GOOD_TIDS:
        score += 20
        reasons.append(f"[+20] 分区优质: {GOOD_TIDS[tid]}({tid})")
    else:
        score += 8
        reasons.append(f"[+8]  分区未知或一般 (tid={tid})")

    # ── 维度3：视频时长 (满分 15) ─────────────────
    duration = info['duration']
    if 180 <= duration <= 900:
        score += 15
        reasons.append(f"[+15] 时长适中: {duration//60}分{duration%60}秒 (3-15分钟最佳)")
    elif 60 <= duration < 180 or 900 < duration <= 1800:
        score += 7
        reasons.append(f"[+7]  时长尚可: {duration//60}分{duration%60}秒")
    else:
        reasons.append(f"[+0]  时长异常: {duration//60}分{duration%60}秒 (过短或过长)")

    # ── 维度4：分辨率 (满分 15) ───────────────────
    w, h = info['dimension']['width'], info['dimension']['height']
    if w >= 1920 or h >= 1080:
        score += 15
        reasons.append(f"[+15] 分辨率达标: {w}x{h} (≥1080P)")
    elif w >= 1280 or h >= 720:
        score += 7
        reasons.append(f"[+7]  分辨率一般: {w}x{h} (720P)")
    else:
        reasons.append(f"[+0]  分辨率过低: {w}x{h}，关键点检测效果差")

    # ── 维度5：互动率 (满分 10) ───────────────────
    view = info['stat']['view']
    if view > 0:
        like = info['stat']['like']
        fav  = info['stat']['favorite']
        coin = info['stat']['coin']
        ir = (like + fav + coin) / view
        if ir > 0.08:
            score += 10
            reasons.append(f"[+10] 互动率极高: {ir:.1%} (内容硬核)")
        elif ir > 0.04:
            score += 5
            reasons.append(f"[+5]  互动率良好: {ir:.1%}")
        else:
            reasons.append(f"[+0]  互动率偏低: {ir:.1%}")
    else:
        reasons.append("[+0]  播放量为0，无法计算互动率")

    # ── 输出报告 ──────────────────────────────────
    bar = "█" * (score // 5) + "░" * (20 - score // 5)
    print(f"\n{'═'*50}")
    print(f"  {info['title']}")
    print(f"  BVID: {bvid}  |  时长: {duration//60}:{duration%60:02d}  |  分辨率: {w}x{h}")
    print(f"  标签: {', '.join(tags[:6])}")
    print(f"{'─'*50}")
    print(f"  综合评分: [{bar}] {score}/100")
    print(f"{'─'*50}")
    for r in reasons:
        print(f"  {r}")
    print(f"{'─'*50}")
    if score >= 70:
        verdict = "✅  强烈建议下载！极有可能包含高质量第一视角操作。"
    elif score >= 50:
        verdict = "⚠️  可以下载，建议结合封面和简介人工复核。"
    else:
        verdict = "❌  不建议下载，大概率不符合 RynnVLA-001 训练要求。"
    print(f"  {verdict}")
    print(f"{'═'*50}")

    return {"bvid": bvid, "title": info['title'], "score": score, "verdict": verdict}


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
def extract_bvid(s):
    m = re.search(r'(BV[a-zA-Z0-9]{10})', s)
    return m.group(1) if m else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RynnVLA-001 B站视频质量预筛选工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python bilibili_prescreen.py BV1zMqsYNERc\n  python bilibili_prescreen.py https://www.bilibili.com/video/BV1zMqsYNERc BV1LH4y1c756"
    )
    parser.add_argument("videos", nargs="+", help="B站视频URL或BVID，支持批量输入")
    args = parser.parse_args()

    results = []
    for v in args.videos:
        bvid = extract_bvid(v)
        if bvid:
            r = evaluate_video(bvid)
            if r:
                results.append(r)
        else:
            print(f"无法识别 BVID: {v}")

    if len(results) > 1:
        print(f"\n{'═'*50}")
        print(f"  批量筛选汇总 ({len(results)} 个视频)")
        print(f"{'─'*50}")
        results.sort(key=lambda x: x['score'], reverse=True)
        for r in results:
            bar = "█" * (r['score'] // 10)
            print(f"  {r['score']:3d}/100 [{bar:<10}] {r['bvid']}  {r['title'][:25]}")
        print(f"{'═'*50}")
