#!/usr/bin/env python3
"""
Rokoko Vision 测试脚本
Rokoko Vision 是浏览器端工具，没有公开的 REST API 来直接上传视频处理。
这个脚本通过 Rokoko Studio Command API 与本地运行的 Rokoko Studio 交互。

测试流程:
1. 手动: 去 vision.rokoko.com 上传视频获取动捕数据
2. 自动: 如果本地有 Rokoko Studio，通过 Command API 控制

需要: Rokoko Studio 运行在本机
"""
import requests
import json
import sys

# Rokoko Studio Command API defaults
STUDIO_IP = "127.0.0.1"
STUDIO_PORT = 14053
API_KEY = "1234"
BASE_URL = f"http://{STUDIO_IP}:{STUDIO_PORT}/v1/{API_KEY}"


def check_studio():
    """Check if Rokoko Studio is running"""
    try:
        resp = requests.post(f"{BASE_URL}/info", timeout=3)
        if resp.status_code == 200:
            info = resp.json()
            print("Rokoko Studio is running!")
            print(json.dumps(info, indent=2))
            return True
    except requests.exceptions.ConnectionError:
        print("Rokoko Studio is not running on this machine.")
        print("To test Rokoko Vision:")
        print("  1. Go to https://vision.rokoko.com")
        print("  2. Create account / login")
        print("  3. Upload test_ego_video.mp4")
        print("  4. Process and download FBX/BVH result")
        return False


def start_recording(filename="ego_test"):
    """Start a recording in Rokoko Studio"""
    resp = requests.post(f"{BASE_URL}/recording/start", json={"filename": filename})
    print(f"Start recording: {resp.status_code} - {resp.text}")


def stop_recording():
    """Stop recording"""
    resp = requests.post(f"{BASE_URL}/recording/stop", json={"back_to_live": True})
    print(f"Stop recording: {resp.status_code} - {resp.text}")


def get_scene_info():
    """Get current scene info"""
    resp = requests.post(f"{BASE_URL}/info")
    print(f"Scene info: {json.dumps(resp.json(), indent=2)}")


def main():
    print("=== Rokoko Vision / Studio Test ===\n")
    
    if check_studio():
        print("\n--- Scene Info ---")
        get_scene_info()
    else:
        print("\n--- Manual Test Instructions ---")
        print("""
Rokoko Vision 手动测试步骤:

1. 打开 https://vision.rokoko.com
2. 注册/登录 Rokoko ID
3. 选择 "Upload a video"
4. 上传: comparison/test_ego_video.mp4
5. 点击 "Animate" 处理
6. 在 Rokoko Studio (免费下载) 中查看结果
7. 导出 FBX/BVH 文件到 comparison/rokoko_output/

注意事项:
- 免费版限制 15 秒 (测试视频 32 秒，需要裁剪)
- 需要全身在画面中 (ego 视角可能不适合)
- 需要良好的光照和清晰的背景

⚠️ 关键发现: Rokoko Vision 和 Move.ai 都需要第三人称全身视角，
   而 Gen-EgoData 是第一人称（头部挂载）视角。
   对于第一人称视角的人类操作数据，这两个方案可能不直接适用。
   它们更适合：
   - 将第三人称视频转换为全身骨骼动画
   - 配合外部摄像头在数采现场同时录制第三人称视角
""")


if __name__ == "__main__":
    main()
