# RoboMemo 数据流水线 (Data Pipeline) 完全指南

## 概述 | Overview

RoboMemo **数据流水线** (DataPipeline) 是一个端到端的视频处理与数据集生成工具，支持：
- ✅ **视角智能筛选** — OpenCV 人脸/手腕检测，自动过滤非第一人称视角视频
- ✅ **VLM 自动标注** — 支持 Gemini、Claude、GPT-4o、本地 Ollama，提取动作语义、接触机制、任务摘要
- ✅ **多格式导出** — LeRobot V2/V3、π₀.5 SFT JSONL、Web3 NFT、完整数据包

**核心特性**：
- 🎬 批量处理多个视频文件
- 🤖 可配置的 VLM 提供商 (API Key 认证)
- 📊 实时进度跟踪与日志审计
- 🔐 数据加密存储与访问控制
- 🌐 分布式导出支持 (本地 / S3)

---

## 架构图 | Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RoboMemo Data Pipeline                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   Frontend UI     │
            (React 19 + Vite + Radix)
                    │                   │
        Step 1: 视频输入        Step 2: 视角筛选
        (Video Upload)   →   (View Filter)
                    │                   │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┴────────────────────────┐
        │                                              │
    Step 3: AutoLabel 标注                      Step 4: 导出
    (VLM Annotation)                         (Export)
        │                                              │
        ├─ Gemini Vision API                      ├─ LeRobot V2 JSON
        ├─ Claude Vision API                      ├─ π₀.5 SFT JSONL
        ├─ GPT-4o Vision API                      ├─ LeRobot V3 Parquet
        └─ Ollama Local Model                     └─ Web3 NFT (IPFS)
        │                                              │
        └─────────────────────┬────────────────────────┘
                              │
        ┌─────────────────────┴────────────────────────┐
        │     Backend (Express + Python)              │
        └─────────────────────┬────────────────────────┘
                              │
        ┌─────────────────────┴────────────────────────┐
        │                                              │
    Python Services                          Output Storage
        │                                              │
    ├─ pipeline_filter.py                     ├─ JSON / JSONL
    │   (OpenCV 人脸+手腕检测)                  ├─ Parquet
    │                                         ├─ HDF5
    ├─ vlm_video_analyzer.py                 ├─ Local FS
    │   (Video → VLM 提取标注)                 └─ S3 (可选)
    │
    └─ lerobot_v3_exporter.py
       (Parquet 导出)
```

---

## 快速开始 | Quick Start

### 前置条件 | Prerequisites

**系统要求：**
- Node.js 18+ (后端)
- Python 3.10+ (视角筛选、VLM 分析)
- OpenCV 4.5+ (人脸/手腕检测)
- macOS / Linux / Windows (WSL2)

**API Keys (至少选择一个)：**
- Google Gemini API Key: https://ai.google.dev
- OpenAI GPT-4o Vision Key: https://platform.openai.com/api-keys
- Anthropic Claude Vision Key: https://console.anthropic.com
- 或使用本地 Ollama (无需 API Key)

### 安装 | Installation

#### 1️⃣ 后端安装 (Backend)

```bash
cd RoboMemo/Platform/backend
npm install

# 复制环境配置
cp .env.example .env

# 编辑 .env，填入你的 VLM API Key
# GEMINI_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
```

#### 2️⃣ Python 环境设置 (Python Environment)

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  (Windows)

# 安装依赖
pip install -r requirements.txt

# 安装 OpenCV（包含 Haar Cascade 人脸检测）
pip install opencv-python

# 检查 Haar Cascade 文件
python -c "import cv2; print(cv2.data.haarcascades)"
```

#### 3️⃣ 前端安装 (Frontend)

```bash
cd RoboMemo/Platform/app
npm install
```

---

## 启动应用 | Running the Application

### 开发模式 | Development Mode

**终端 1 — 启动后端 (Terminal 1 — Backend)**

```bash
cd RoboMemo/Platform/backend
npm run dev
# 输出: Server running on http://localhost:3001
```

**终端 2 — 启动前端 (Terminal 2 — Frontend)**

```bash
cd RoboMemo/Platform/app
npm run dev
# 输出: Local: http://localhost:5173
```

打开浏览器访问 **http://localhost:5173** ✅

### 生产模式 | Production Mode

```bash
# 后端
cd backend && npm start

# 前端
cd app && npm run build && npm start
```

---

## UI 使用指南 | UI Usage Guide

### 🎬 Step 1: 视频输入 | Video Input

**界面位置：** Dashboard → 数据流水线 Tab → "Step 1: 视频输入"

**输入字段：**
- **bvid** — 如果来自 Bilibili，输入视频 ID（可选）
  ```
  示例: BV1Xx4y1k7aY
  ```
- **标题** — 数据集名称
  ```
  示例: "机器人夹取物体演示 v2"
  ```
- **视频文件** — 选择本地视频（支持批量）
  ```
  支持格式: .mp4, .mov, .avi, .webm
  最大单个文件: 2GB
  ```

**操作：**
1. 填写上述字段
2. 点击"下一步 (Next)" → 进入 Step 2
3. **导入的视频会自动上传到 `/uploads` 目录**

---

### 🔍 Step 2: 视角筛选 | View Filtering

**界面位置：** Step 2 Tab

**工作原理：**
- 后端调用 `pipeline_filter.py`（基于 OpenCV Haar Cascade）
- 对每个视频逐帧扫描：
  - **检测人脸** — 计算人脸占比 (face_ratio)
  - **检测手腕** — 计算手腕占比 (wrist_ratio)

**筛选规则：**
| 条件 | 结果 | 说明 |
|------|------|------|
| face_ratio > 15% | ❌ 拒绝 | 人脸过大 → 非第一人称 |
| wrist_ratio < 30% | ❌ 拒绝 | 手腕过小 → 视角不清晰 |
| 其他 | ✅ 接受 | 符合第一人称视角 |

**配置参数（可在 `.env` 中修改）：**
```bash
FACE_RATIO_THRESHOLD=0.15         # 人脸占比阈值
WRIST_RATIO_THRESHOLD=0.30        # 手腕占比阈值
SAMPLE_EVERY_N_FRAMES=10          # 每 N 帧采样一次（加速处理）
```

**UI 操作：**
1. 点击"开始筛选 (Filter)"
2. 进度条显示实时处理进度
3. 完成后显示：
   - ✅ 通过视频列表
   - ❌ 被拒绝视频列表及原因

**下一步：** 点击"下一步" → Step 3

---

### 🤖 Step 3: AutoLabel 标注 | VLM Auto-Annotation

**界面位置：** Step 3 Tab

**VLM 配置（需要先选择）：**

| 提供商 | 优势 | 成本 | 配置 |
|--------|------|------|------|
| **Gemini** | 免费额度充足 | 100 req/day free | `GEMINI_API_KEY` |
| **GPT-4o** | 精度最高 | $0.01/1K tokens | `OPENAI_API_KEY` |
| **Claude 3.5** | 推理能力强 | $0.003-$0.03/1K | `ANTHROPIC_API_KEY` |
| **Ollama** (本地) | 完全免费 | 0 | 需要本地 Ollama 服务 |

**选择 VLM 提供商：**
1. 点击"选择 VLM" 下拉菜单
2. 选择提供商（如 Gemini）
3. 输入对应 API Key
4. (可选) 选择模型版本（如 `gemini-2.0-flash`）

**标注字段（自动提取）：**
```json
{
  "action_primitives": ["picking", "placing", "rotating"],
  "contact_mechanics": ["grasp", "slide", "tap"],
  "task_summary": "用夹爪从箱子中取出红色立方体，放在目标位置上"
}
```

**UI 操作：**
1. 选择 VLM 提供商
2. 点击"开始标注 (Annotate)"
3. 实时日志显示：
   - 正在处理的视频名称
   - VLM 提取进度
   - Token 消耗统计
4. 完成后显示标注结果摘要

**高级选项：**
- ✅ 启用缓存 — 相同视频不重复调用 VLM
- ✅ 批量大小 — 调整 VLM 并发数（默认 5）
- ✅ 超时设置 — 单个视频超时时间（默认 60s）

**下一步：** 点击"下一步" → Step 4

---

### 💾 Step 4: 导出 | Export

**界面位置：** Step 4 Tab

**导出格式选项：**

#### 4.1 LeRobot V2 JSON
**用途：** LeRobot 框架训练、开源数据集兼容性
**输出文件：** `robomemo_pipeline_lerobot_v2.json`
**结构：**
```json
{
  "meta": {
    "codebase_version": "2.0",
    "format_version": "1.0",
    "name": "机器人夹取演示 v2",
    "resolution": [1280, 720],
    "fps": 30
  },
  "episodes": [
    {
      "episode_index": 0,
      "tasks": [
        {
          "task_index": 0,
          "summary": "用夹爪从箱子中取出红色立方体",
          "steps": [
            {
              "step_index": 0,
              "type": "observation",
              "frame_index": 0,
              "image_path": "episode_0/frame_0000.jpg"
            },
            {
              "step_index": 1,
              "type": "action",
              "action_primitives": ["picking"],
              "contact_mechanics": ["grasp"]
            }
          ]
        }
      ]
    }
  ]
}
```

#### 4.2 π₀.5 SFT JSONL
**用途：** π₀.5 模型微调、监督学习
**输出文件：** `robomemo_pipeline_pi05_sft.jsonl`
**每行格式：**
```json
{"episode": 0, "frame": 0, "image": "base64_encoded_jpg", "action": "picking", "description": "用夹爪从箱子中取出红色立方体", "contact": "grasp"}
```
**特点：** 每行一条训练数据，支持流式处理

#### 4.3 LeRobot V3 Parquet (高性能)
**用途：** 大规模训练、分布式处理
**输出位置：** `outputs/lerobot_v3/episode_XXXXXX.parquet` + 元数据
**结构：**
```
outputs/
├── episode_000000.parquet    # 第 1 个 Episode
├── episode_000001.parquet    # 第 2 个 Episode
├── meta/
│   ├── info.json             # 元数据 (format version, fps, resolution)
│   └── stats.json            # 统计 (总 frame 数、action 分布)
```

**Parquet 列结构：**
| 列名 | 类型 | 说明 |
|------|------|------|
| frame_index | int64 | 帧索引 |
| timestamp | float64 | 时间戳 |
| image | binary | JPEG 二进制 |
| action | string | 动作原始语义 |
| action_embedding | binary | VLM 嵌入向量 |
| contact | string | 接触机制 |
| task_description | string | 任务摘要 |

#### 4.4 Web3 NFT (区块链)
**用途：** 数据集上链、知识产权保护
**跳转至：** Web3 Marketplace Tab
**输出：**
- 上传至 IPFS
- 铸造 NFT (链上证书)
- 支持转让与许可管理

**UI 操作：**
1. 勾选需要导出的格式（可多选）
   ```
   ☑ LeRobot V2 JSON
   ☑ π₀.5 SFT JSONL
   ☑ LeRobot V3 Parquet
   ☑ Web3 NFT
   ```
2. (可选) 选择存储位置：
   - 本地文件系统 (默认)
   - AWS S3 (需要配置 `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
3. 点击"导出 (Export)" 按钮
4. 实时进度显示：
   - 格式转换进度
   - 文件写入大小
   - 预计剩余时间
5. 完成后：
   - 下载按钮链接
   - 元数据摘要（总文件大小、episode 数、frame 数）

**导出后的目录结构：**
```
robomemo_exports/
├── pipeline_output_2024-12-15_12-34-56/
│   ├── robomemo_pipeline_lerobot_v2.json
│   ├── robomemo_pipeline_pi05_sft.jsonl
│   ├── lerobot_v3/
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   ├── meta/
│   │   │   ├── info.json
│   │   │   └── stats.json
│   ├── metadata.json (整体元数据)
│   └── logs.txt (处理日志)
```

---

## CLI 直接使用 | CLI Usage

无需 UI，直接在命令行调用 Python 脚本处理数据。

### 🔍 视角筛选 | View Filtering (CLI)

**脚本：** `backend/pipeline_filter.py`

**基础用法：**
```bash
python backend/pipeline_filter.py <video_path> [sample_every_n] [output_file]
```

**参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `video_path` | 输入视频路径 | (必须) |
| `sample_every_n` | 每 N 帧采样一次 | 10 |
| `output_file` | 输出 JSON 结果 | `filter_result.json` |

**示例 1 — 简单用法：**
```bash
python backend/pipeline_filter.py ~/videos/demo.mp4
# 输出: filter_result.json
```

**示例 2 — 加速处理（每 30 帧采样）：**
```bash
python backend/pipeline_filter.py ~/videos/demo.mp4 30 result.json
# 输出: result.json (处理速度快 3x，精度略低)
```

**输出格式：**
```json
{
  "video_path": "/Users/user/videos/demo.mp4",
  "total_frames": 900,
  "sampled_frames": 90,
  "result": "PASS",
  "face_ratio": 0.08,
  "wrist_ratio": 0.42,
  "reason": "符合第一人称视角",
  "frames_analyzed": [
    {
      "frame_index": 0,
      "face_detected": false,
      "wrist_detected": true,
      "face_ratio": 0.05,
      "wrist_ratio": 0.45
    },
    ...
  ]
}
```

**判定标准：**
```
IF face_ratio > 0.15:
    result = "REJECT" (reason: "人脸占比过大")
ELIF wrist_ratio < 0.30:
    result = "REJECT" (reason: "手腕占比过小")
ELSE:
    result = "PASS"
```

---

### 🤖 VLM 视频分析 | VLM Video Analysis (CLI)

**脚本：** `backend/vlm_video_analyzer.py`

**基础用法：**
```bash
python backend/vlm_video_analyzer.py \
  --video <path> \
  --provider <provider> \
  --api-key <key> \
  --model <model> \
  --output <output_file>
```

**参数：**
| 参数 | 说明 | 示例 |
|------|------|------|
| `--video` | 输入视频路径 | `~/videos/demo.mp4` |
| `--provider` | VLM 提供商 | `gemini` / `openai` / `anthropic` / `ollama` |
| `--api-key` | API 密钥 | `xxx-yyy-zzz` (ollama 可省略) |
| `--model` | 模型名 | `gemini-2.0-flash` / `gpt-4o-vision` / `claude-3-5-sonnet` |
| `--output` | 输出 JSON 文件 | `annotation_result.json` |
| `--sample-every-n` | 采样间隔 | 5 (默认) |

**示例 1 — 使用 Gemini：**
```bash
export GEMINI_API_KEY="your-key-here"

python backend/vlm_video_analyzer.py \
  --video ~/videos/demo.mp4 \
  --provider gemini \
  --api-key $GEMINI_API_KEY \
  --model gemini-2.0-flash \
  --output annotation.json
```

**示例 2 — 使用本地 Ollama (免费)：**
```bash
# 先启动本地 Ollama 服务
ollama serve

# 另一个终端
python backend/vlm_video_analyzer.py \
  --video ~/videos/demo.mp4 \
  --provider ollama \
  --model llava:latest \
  --output annotation.json
```

**示例 3 — 使用 GPT-4o Vision：**
```bash
export OPENAI_API_KEY="sk-..."

python backend/vlm_video_analyzer.py \
  --video ~/videos/demo.mp4 \
  --provider openai \
  --api-key $OPENAI_API_KEY \
  --model gpt-4o-vision \
  --output annotation.json
```

**输出格式：**
```json
{
  "video_path": "~/videos/demo.mp4",
  "provider": "gemini",
  "model": "gemini-2.0-flash",
  "total_frames": 900,
  "sampled_frames": 180,
  "annotations": [
    {
      "frame_index": 0,
      "action_primitives": ["observing"],
      "contact_mechanics": [],
      "task_summary": "观察工作台上的物体布置"
    },
    {
      "frame_index": 50,
      "action_primitives": ["reaching", "grasping"],
      "contact_mechanics": ["grasp"],
      "task_summary": "用夹爪靠近并夹取红色立方体"
    }
  ],
  "token_usage": {
    "input_tokens": 45000,
    "output_tokens": 3200,
    "total_tokens": 48200,
    "estimated_cost": "$0.48"
  }
}
```

---

### 💾 LeRobot V3 Parquet 导出 | LeRobot V3 Exporter (CLI)

**脚本：** `backend/lerobot_v3_exporter.py`

**基础用法：**
```bash
python backend/lerobot_v3_exporter.py \
  --manifest <manifest_json> \
  --h5-dir <h5_directory> \
  --output-dir <output_directory>
```

**参数：**
| 参数 | 说明 | 示例 |
|------|------|------|
| `--manifest` | manifest 文件路径 (LeRobot 格式) | `data/manifest.json` |
| `--h5-dir` | HDF5 文件目录 | `data/hdf5/` |
| `--output-dir` | 输出 Parquet 目录 | `outputs/lerobot_v3/` |
| `--compression` | 压缩算法 | `snappy` (默认) / `gzip` / `brotli` |
| `--batch-size` | 单批处理 episode 数 | 10 (默认) |

**示例：**
```bash
python backend/lerobot_v3_exporter.py \
  --manifest data/lerobot_manifest.json \
  --h5-dir data/hdf5_episodes/ \
  --output-dir outputs/lerobot_v3/ \
  --compression snappy \
  --batch-size 10
```

**输出：**
```
outputs/lerobot_v3/
├── episode_000000.parquet
├── episode_000001.parquet
├── ...
├── meta/
│   ├── info.json
│   └── stats.json
└── logs.txt
```

---

## 输出格式详解 | Output Format Specifications

### LeRobot V2 JSON Schema

```json
{
  "meta": {
    "codebase_version": "2.0",
    "format_version": "1.0",
    "name": "数据集名称",
    "description": "简短描述",
    "resolution": [1280, 720],
    "fps": 30,
    "total_episodes": 5,
    "total_frames": 4500
  },
  "episodes": [
    {
      "episode_index": 0,
      "duration_seconds": 30,
      "tasks": [
        {
          "task_index": 0,
          "summary": "任务摘要",
          "action_primitives": ["picking", "placing"],
          "contact_mechanics": ["grasp"],
          "steps": [
            {
              "step_index": 0,
              "type": "observation",
              "frame_index": 0,
              "image_path": "episode_0/frame_0000.jpg",
              "timestamp": 0.0
            },
            {
              "step_index": 1,
              "type": "action",
              "frame_index": 1,
              "action_primitives": ["picking"],
              "contact_mechanics": ["grasp"],
              "timestamp": 0.033
            }
          ]
        }
      ]
    }
  ]
}
```

### π₀.5 SFT JSONL Schema

**格式：** 每行一条 JSON 对象（JSONL = JSON Lines）

```jsonl
{"episode": 0, "frame": 0, "image": "base64_...", "action": "observing", "description": "观察物体", "contact": "none"}
{"episode": 0, "frame": 50, "image": "base64_...", "action": "reaching", "description": "伸手靠近立方体", "contact": "none"}
{"episode": 0, "frame": 100, "image": "base64_...", "action": "grasping", "description": "夹取立方体", "contact": "grasp"}
{"episode": 0, "frame": 150, "image": "base64_...", "action": "placing", "description": "放置目标位置", "contact": "grasp"}
```

**字段说明：**
| 字段 | 类型 | 说明 |
|------|------|------|
| episode | int | Episode 索引 |
| frame | int | 帧索引 |
| image | string | JPEG Base64 编码 |
| action | string | 动作原始语义 |
| description | string | 任务摘要 |
| contact | string | 接触机制 (none / grasp / slide / tap / ...) |

### LeRobot V3 Parquet Schema

**文件结构：**
```
episode_XXXXXX.parquet  (Apache Parquet 格式)
meta/info.json          (元数据)
meta/stats.json         (统计)
```

**Parquet 列：**
```
Frame Index (int64) ─┐
Timestamp (float64)  ├─ 帧级数据
Image (binary)       ├─ JPEG 二进制，Row Group 压缩
──────────────────────┤
Action (string)      ├─ 标注数据
Action Embedding     ├─
Contact (string)     │
Task Description     │
──────────────────────┘
```

**Parquet 优势：**
- 🚀 列式存储，查询快 10-50x
- 💾 压缩率高 (snappy/gzip/brotli)
- 🔄 支持分布式处理 (Spark / Dask)
- 🎯 支持部分列读取（只读需要的列）

**meta/info.json：**
```json
{
  "format_version": "v3.0",
  "fps": 30,
  "image_width": 1280,
  "image_height": 720,
  "compression": "snappy",
  "total_episodes": 5,
  "total_frames": 4500
}
```

**meta/stats.json：**
```json
{
  "total_frames": 4500,
  "episodes_per_action": {
    "picking": 450,
    "placing": 380,
    "rotating": 200
  },
  "contact_distribution": {
    "grasp": 630,
    "none": 3870
  },
  "avg_episode_length": 900,
  "image_size_bytes": 1048576
}
```

---

## 配置参考 | Configuration Reference

### 环境变量 | Environment Variables

**创建 `.env` 文件于 `backend/` 目录：**

```bash
# VLM API Keys
GEMINI_API_KEY=xxx
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=xxx

# 视角筛选参数
FACE_RATIO_THRESHOLD=0.15
WRIST_RATIO_THRESHOLD=0.30
SAMPLE_EVERY_N_FRAMES=10

# 后端服务
NODE_ENV=development
PORT=3001
LOG_LEVEL=debug

# 数据存储
UPLOAD_DIR=./uploads
EXPORT_DIR=./exports

# S3 配置（可选，用于云存储导出）
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
AWS_S3_BUCKET=robomemo-exports

# VLM 通用参数
VLM_TIMEOUT=60
VLM_BATCH_SIZE=5
VLM_CACHE_ENABLED=true
```

### Python 依赖 | Python Requirements

**`backend/requirements.txt`：**
```
opencv-python==4.8.0
numpy==1.24.3
pillow==10.0.0
requests==2.31.0
google-generativeai==0.3.0
openai==1.3.0
anthropic==0.7.0
pydantic==2.0.0
python-dotenv==1.0.0
pyarrow==12.0.0
```

### 视角筛选参数 | View Filter Tuning

如果筛选结果不佳，可调整 `.env` 中的阈值：

**问题 1：太多视频被拒绝（人脸占比过大）**
```bash
# 增大人脸阈值 (默认 0.15)
FACE_RATIO_THRESHOLD=0.25
```

**问题 2：接受了非第一人称视角**
```bash
# 降低人脸阈值 (默认 0.15)
FACE_RATIO_THRESHOLD=0.10
```

**问题 3：处理速度慢**
```bash
# 增大采样间隔 (默认 10，即每 10 帧采样 1 次)
SAMPLE_EVERY_N_FRAMES=30  # 3x 加速，精度略低
```

**问题 4：手腕检测不准**
```bash
# 降低手腕阈值 (默认 0.30)
WRIST_RATIO_THRESHOLD=0.20
```

---

## 常见问题 | FAQ

### Q1: 启动后端时出现 "Port 3001 already in use"

**解决方案：**
```bash
# macOS/Linux
lsof -i :3001  # 查看占用进程
kill -9 <PID>  # 杀死进程

# Windows
netstat -ano | findstr :3001
taskkill /PID <PID> /F
```

或在 `.env` 中改变端口：
```bash
PORT=3002
```

---

### Q2: Python 找不到 OpenCV Haar Cascade 文件

**症状：**
```
cv2.error: (-5:Bad argument) in function 'detectMultiScale'
```

**解决方案：**
```bash
# 检查 OpenCV 数据目录
python -c "import cv2; print(cv2.data.haarcascades)"

# 如果找不到，手动指定路径
export OPENCV_HAAR_CASCADES_PATH="/path/to/cascades"
```

---

### Q3: VLM API 调用频繁超时

**症状：**
```
Request timeout: 60s exceeded
```

**解决方案：**

1. **增加超时时间：**
   ```bash
   VLM_TIMEOUT=120  # 改为 120 秒
   ```

2. **降低批处理大小：**
   ```bash
   VLM_BATCH_SIZE=2  # 默认 5，改为 2 降低并发
   ```

3. **检查网络连接：**
   ```bash
   curl https://generativelanguage.googleapis.com  # 测试 Gemini
   curl https://api.openai.com  # 测试 OpenAI
   ```

4. **使用本地 Ollama（无网络依赖）：**
   ```bash
   ollama pull llava:latest
   python backend/vlm_video_analyzer.py \
     --video ~/video.mp4 \
     --provider ollama \
     --model llava:latest \
     --output result.json
   ```

---

### Q4: 导出的 JSON 文件过大（超过 1GB）

**症状：**
- LeRobot V2 JSON 包含所有图像，文件极大
- 浏览器打开卡顿

**解决方案：**

1. **使用 LeRobot V3 Parquet（推荐）：**
   - 自动分多个 episode 文件
   - 支持流式处理
   - 压缩率 10-50x

2. **或者使用 π₀.5 SFT JSONL：**
   - 每行一条数据，支持流式
   - 图像 Base64 编码，便于传输

3. **减少 episode 数量：**
   - 选择性导出（在 UI Step 4 中取消部分 episode）

---

### Q5: 某个 VLM 提供商的标注结果为空

**症状：**
```json
{
  "action_primitives": [],
  "contact_mechanics": [],
  "task_summary": ""
}
```

**原因 & 解决方案：**

| 原因 | 解决 |
|------|------|
| API Key 无效 | 检查 `.env`，重新复制 API Key |
| 余额不足 | 登录 API 控制台充值 |
| 模型不支持视频 | 切换到 `gemini-2.0-flash` / `gpt-4o-vision` |
| 网络错误 | 检查 VPN / 梯子，或改用 Ollama |

---

### Q6: 如何批量处理多个视频不用 UI？

**直接使用 CLI 脚本：**

```bash
#!/bin/bash
# process_batch.sh

VIDEOS_DIR="./videos"
OUTPUT_DIR="./results"

mkdir -p $OUTPUT_DIR

for video in $VIDEOS_DIR/*.mp4; do
    echo "Processing: $video"

    # Step 1: 视角筛选
    python backend/pipeline_filter.py "$video" 10 "$OUTPUT_DIR/$(basename $video .mp4)_filter.json"

    # Step 2: VLM 标注（Gemini 免费额度）
    python backend/vlm_video_analyzer.py \
      --video "$video" \
      --provider gemini \
      --api-key $GEMINI_API_KEY \
      --model gemini-2.0-flash \
      --output "$OUTPUT_DIR/$(basename $video .mp4)_annotation.json"
done

echo "Done!"
```

运行：
```bash
chmod +x process_batch.sh
./process_batch.sh
```

---

### Q7: 支持哪些视频格式？

**支持列表：**
| 格式 | 编码 | 支持 |
|------|------|------|
| MP4 | H.264 / H.265 | ✅ |
| MOV | H.264 / ProRes | ✅ |
| AVI | MPEG-4 / DivX | ✅ |
| WebM | VP8 / VP9 | ✅ |
| MKV | 多种 | ✅ |
| FLV | H.264 | ⚠️ (需要 ffmpeg) |

**不支持：** `.gif`, `.webp` (可转码为 MP4)

---

### Q8: 数据隐私与安全

**关键点：**
- ✅ 所有上传数据存储于本地 `/uploads` 目录（不上传云端）
- ✅ API 调用时仅传输视频 **帧的描述**，不传送原始视频（Gemini / GPT-4o 除外，需要二进制数据）
- ✅ 支持启用文件加密（`.env` 中配置 `ENCRYPT_UPLOADS=true`）
- ⚠️ 如使用云 VLM（Gemini / GPT-4o），需同意其隐私政策

**建议：**
- 敏感数据：使用本地 Ollama（完全离线）
- 公开数据：使用 Gemini（免费额度）
- 企业级：部署私有 VLM 或使用 Anthropic Claude（企业合同）

---

## 故障排除 | Troubleshooting

**通用调试步骤：**

1. **检查日志：**
   ```bash
   # 后端日志
   tail -f backend/logs/app.log

   # 前端浏览器控制台
   F12 → Console 标签页
   ```

2. **运行诊断脚本：**
   ```bash
   python backend/diagnose.py
   # 检查：OpenCV、VLM API、网络连接
   ```

3. **重启服务：**
   ```bash
   # 杀死现有进程
   pkill -f "npm run dev"

   # 清缓存重启
   rm -rf node_modules/.cache
   npm run dev
   ```

---

## 更新日志 | Changelog

### v1.2.0 (2024-12-15)
- ✨ 新增 LeRobot V3 Parquet 导出
- 🚀 支持本地 Ollama VLM（免费离线）
- 🐛 修复视角筛选手腕检测误报
- 📊 改进导出性能 (10-50x 加快)

### v1.1.0 (2024-11-20)
- 🤖 新增 Claude Vision 支持
- 🎨 UI 优化（4 步 Stepper）
- 📝 完善文档和示例

### v1.0.0 (2024-10-01)
- 🎉 初始发布
- ✅ Gemini / GPT-4o Vision 支持
- 💾 LeRobot V2 JSON / π₀.5 SFT JSONL 导出

---

## 支持与反馈 | Support & Feedback

**有问题？**
- 📧 邮件：robomemo-support@example.com
- 💬 讨论区：GitHub Discussions
- 🐛 报告 Bug：GitHub Issues

**想贡献？**
- 提交 Pull Request
- 改进文档
- 报告性能瓶颈

---

## 许可证 | License

RoboMemo Data Pipeline © 2024 RoboMemo Team. All rights reserved.

**使用条款：**
- ✅ 学术研究（引用本项目）
- ✅ 非商业用途（机器人实验室、高校）
- ⚠️ 商业用途（需要授权）

---

**Happy data collection! 🤖📊**

*Last updated: 2024-12-15 | 文档版本: 1.2.0*
