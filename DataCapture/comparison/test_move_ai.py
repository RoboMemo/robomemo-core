#!/usr/bin/env python3
"""
Move.ai API 测试脚本
使用 Move One (单摄像头) API 处理 ego-centric 视频
需要: MOVE_API_KEY 环境变量
"""
import os
import sys
import requests
from move_ugc import MoveUgc
from move_ugc.schemas.sources import SourceIn

API_KEY = os.environ.get("MOVE_API_KEY")
if not API_KEY:
    print("Error: Set MOVE_API_KEY environment variable")
    print("Get your API key at: https://platform.move.ai")
    sys.exit(1)

ugc = MoveUgc(api_key=API_KEY)

VIDEO_PATH = os.path.join(os.path.dirname(__file__), "test_ego_video.mp4")

def test_move_ai():
    print("=== Move.ai Single-cam Test ===")
    
    # Step 1: Create a file entry
    print("1. Creating file entry...")
    video_file = ugc.files.create(file_type="mp4")
    print(f"   File ID: {video_file.id}")
    print(f"   Presigned URL: {video_file.presigned_url[:80]}...")
    
    # Step 2: Upload video
    print("2. Uploading video...")
    with open(VIDEO_PATH, 'rb') as f:
        resp = requests.put(video_file.presigned_url, data=f.read())
        print(f"   Upload status: {resp.status_code}")
    
    # Step 3: Create a take
    print("3. Creating take...")
    take = ugc.takes.create_singlecam(
        sources=[
            SourceIn(
                device_label="ego-camera",
                file_id=video_file.id,
                format=video_file.type
            )
        ]
    )
    print(f"   Take ID: {take.id}")
    
    # Step 4: Create a job (trigger processing)
    print("4. Creating job (processing)...")
    job = ugc.jobs.create_singlecam(take_id=take.id)
    print(f"   Job ID: {job.id}")
    print(f"   Status: {job.state}")
    
    # Step 5: Poll for results
    print("5. Waiting for results...")
    import time
    while True:
        job = ugc.jobs.retrieve(job_id=job.id)
        print(f"   Status: {job.state}")
        if job.state in ("FINISHED", "FAILED"):
            break
        time.sleep(10)
    
    if job.state == "FINISHED":
        print("6. Downloading results...")
        outputs = job.outputs
        for output in outputs:
            print(f"   Output: {output.type} -> {output.file.presigned_url[:80]}...")
            # Download output
            resp = requests.get(output.file.presigned_url)
            out_path = f"move_ai_output.{output.type}"
            with open(out_path, 'wb') as f:
                f.write(resp.content)
            print(f"   Saved to: {out_path}")
    else:
        print(f"Job failed: {job}")

if __name__ == "__main__":
    test_move_ai()
