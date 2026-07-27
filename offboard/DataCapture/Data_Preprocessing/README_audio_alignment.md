# Audio Alignment Script

将多个视频文件的音频轨道对齐到参考轨道。使用能量包络和归一化互相关。

## 使用方法

### 方法 1：仅进行音频对齐（生成 JSON）

```bash
cd /Users/siyuwang/Downloads/Project/Project_Robot/RoboMemo/offboard/SelfLabelingDemo/EgoView_V0
python3 align_audio.py
```

默认会读取：
- `H.MOV` (头部摄像头)
- `L.MP4` (左手摄像头)
- `R.MP4` (右手摄像头)

输出：`audio_alignment.json`

### 方法 2：音频对齐 + 视频同步（推荐！）

```bash
python3 align_audio.py --sync-video
```

这会：
1. 进行音频对齐分析
2. 根据延迟自动生成对齐后的新视频文件
3. 输出：
   - `audio_alignment.json` (对齐参数)
   - `L_aligned.MP4` (对齐后的左手视频)
   - `R_aligned.MP4` (对齐后的右手视频)

### 自定义参数

```bash
# 指定参考摄像头（默认为 H）
python3 align_audio.py --reference L --sync-video

# 指定输出文件名后缀
python3 align_audio.py --sync-video --output-suffix "_sync"
# 输出：L_sync.MP4, R_sync.MP4

# 指定对齐 JSON 输出文件
python3 align_audio.py --output my_alignment.json --sync-video

# 指定采样率（默认 16000 Hz）
python3 align_audio.py --sr 16000 --sync-video

# 使用自定义视频映射文件
python3 align_audio.py --videos video_mapping.json --sync-video
```

### 自定义视频映射

创建 `video_mapping.json`：
```json
{
  "camera1": "path/to/video1.mp4",
  "camera2": "path/to/video2.mp4",
  "camera3": "path/to/video3.mp4"
}
```

然后运行：
```bash
python3 align_audio.py --videos video_mapping.json --reference camera1
```

## 输出格式

生成的 `audio_alignment.json`：

```json
{
  "H": {
    "delay_samples": 0,
    "delay_seconds": 0.0,
    "method": "normalized_xcorr",
    "hop_length": 512
  },
  "L": {
    "delay_frames": -53,
    "delay_samples": -27136,
    "delay_seconds": -1.696,
    "correlation": 712.81
  },
  "R": {
    "delay_frames": -86,
    "delay_samples": -44032,
    "delay_seconds": -2.752,
    "correlation": 609.19
  }
}
```

### 字段说明

- **delay_seconds**: 延迟时间（秒）
  - 负值表示该摄像头比参考摄像头早
  - 正值表示该摄像头比参考摄像头晚
- **delay_samples**: 延迟采样数（16kHz 采样率）
- **delay_frames**: 延迟帧数（mel-spectrogram 帧）
- **correlation**: 归一化互相关峰值（越高越好）

## 依赖

```bash
pip install librosa scipy numpy
```

需要系统安装 ffmpeg：
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg

# Windows
# 从 https://ffmpeg.org/download.html 下载
```

## 原理

### 音频对齐
1. 从每个视频提取音频
2. 计算 mel-spectrogram 能量包络
3. 对包络进行归一化
4. 计算与参考摄像头的归一化互相关
5. 找到相关性峰值对应的时间延迟

### 视频同步（--sync-video）
1. 读取音频对齐的延迟信息
2. 对于比参考摄像头**早**的视频（负延迟）：
   - 从延迟时间点开始截取视频
   - 确保长度与参考视频相同
3. 对于比参考摄像头**晚**的视频（正延迟）：
   - 在视频开头添加黑色帧/静音
   - 填充到延迟时间长度
   - 确保长度与参考视频相同
4. 统一分辨率为参考视频的分辨率
5. 生成对齐后的新视频文件

## 精度

- 能量包络帧数：512 个样本 (16kHz) ≈ 32ms
- 理论时间精度：±32ms
- 实际精度取决于音频内容和相关性得分

## 在代码中使用结果

```python
import json

with open('audio_alignment.json') as f:
    alignment = json.load(f)

# 获取延迟
delay_L = alignment['L']['delay_seconds']  # -1.696
delay_R = alignment['R']['delay_seconds']  # -2.752

# 同步帧时间
t_H = 1.5  # 头部摄像头的时间戳（秒）
t_L = t_H + delay_L  # 左手摄像头的对应时间
t_R = t_H + delay_R  # 右手摄像头的对应时间
```

或按帧号同步（假设30fps）：
```python
fps = 30
frame_offset_L = int(abs(delay_L) * fps)  # ~51 frames ahead
frame_offset_R = int(abs(delay_R) * fps)  # ~83 frames ahead
```

## 在代码中调用视频同步函数

```python
from align_audio import sync_video_to_reference

# 调用视频同步函数
video_dict = {
    'H': 'H.MOV',
    'L': 'L.MP4',
    'R': 'R.MP4'
}

output_videos = sync_video_to_reference(
    alignment_json='audio_alignment.json',
    video_dict=video_dict,
    reference='H',
    output_suffix='_aligned'
)

# 使用对齐后的视频
print(f"Aligned videos: {output_videos}")
# Output:
# {
#   'H': 'H.MOV',
#   'L': 'L_aligned.MP4',
#   'R': 'R_aligned.MP4'
# }
```

## 完整工作流示例

```bash
# Step 1: 进行音频对齐和视频同步
python3 align_audio.py --sync-video

# Step 2: 检查输出文件
ls -lh *.MP4 *.MOV audio_alignment.json
# 应该看到：
# - H.MOV (原始，不变)
# - L_aligned.MP4 (新生成)
# - R_aligned.MP4 (新生成)
# - audio_alignment.json (对齐参数)

# Step 3: 验证对齐质量（使用 ffprobe）
ffprobe -v error -select_streams v:0 -show_entries stream=duration -of default=noprint_wrappers=1:nokey=1 H.MOV
ffprobe -v error -select_streams v:0 -show_entries stream=duration -of default=noprint_wrappers=1:nokey=1 L_aligned.MP4
ffprobe -v error -select_streams v:0 -show_entries stream=duration -of default=noprint_wrappers=1:nokey=1 R_aligned.MP4
# 这些文件的时长应该基本相同
```

## 常见问题

### Q: 视频转换很慢，如何加速？
A: 编辑脚本，在 ffmpeg 命令中修改 preset：
- `-preset ultrafast` 最快但质量最低
- `-preset fast` 快速，质量中等（推荐）
- `-preset medium` 默认
- `-preset slow` 慢但质量最好

### Q: 可以不生成对齐视频，只要 JSON 吗？
A: 可以，直接运行 `python3 align_audio.py`（不加 `--sync-video`）

### Q: 分辨率不一致怎么办？
A: 脚本会自动缩放所有视频到参考视频的分辨率

### Q: 如何改变参考摄像头？
A: 使用 `--reference` 参数，例如 `python3 align_audio.py --reference L --sync-video`

