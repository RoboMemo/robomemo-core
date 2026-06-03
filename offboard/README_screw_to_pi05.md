# 拧螺丝视频 → π₀.5 SFT 标注数据集

> `demo_screw_to_pi05_sft.py` — 一键将拧螺丝操作视频转换为 Physical Intelligence π₀.5 所需的全套 SFT 监督微调数据

---

## 概览

```
输入：一段拧螺丝的机器人操作视频（MP4 / AVI / MOV）
                    │
        ┌───────────▼───────────┐
        │   Stage 1: 阶段分割    │  运动自适应帧采样 + VLM 时序分割
        │   Stage 2: 动作标注    │  VLM 逐阶段判断动作原语 / 夹爪状态
        │   Stage 3: 力学估计    │  VLM 估计接触类型 / 力度 / 运动方向
        │   Stage 4: 任务摘要    │  VLM 生成自然语言任务描述
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │   LeRobot V2.1 导出    │  meta/info.json · episodes.jsonl
        │                       │  tasks.jsonl · data/*.json
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │  π₀.5 SFT 训练配置     │  openpi_finetune.json
        │                       │  launch_training.sh
        └───────────────────────┘

输出：可直接送入 openpi.training.train 的完整 SFT 数据集
```

---

## 来源说明

本脚本合并自三个开发分支：

| 模块 | 来源分支 | 功能 |
|------|---------|------|
| `AutoLabelPipeline` (4阶段) | `dev-VQApipeline-siyu` | 视频 → VLM 结构化标注 |
| `extract_adaptive_frames` | `dev-autolabel_rf_siyu` | 运动边界自适应帧采样 |
| `export_lerobot` | `dev-VQApipeline-siyu` | LeRobot V2.1 格式导出 |
| `OpenPIFinetuneCfg` | `merge-local-features` | π₀ / π₀.5 SFT 训练配置 |

---

## 环境安装

```bash
# 必须
pip install opencv-python numpy requests

# 可选：使用 Gemini（推荐，标注质量最高）
# 在 Google AI Studio 申请 API Key: https://aistudio.google.com/apikey

# 可选：使用本地 Ollama
brew install ollama          # macOS
ollama pull scomper/minicpm-v2.5  # 下载多模态模型
ollama serve                 # 启动服务
```

---

## 快速开始

### 方式一：Gemini API（推荐）

```bash
python3 demo_screw_to_pi05_sft.py \
    --video        /path/to/screw_tightening.mp4 \
    --vlm          gemini \
    --gemini-key   YOUR_GEMINI_API_KEY \
    --output-dir   ./sft_output
```

### 方式二：本地 Ollama

```bash
# 先启动 Ollama 服务
ollama serve

python3 demo_screw_to_pi05_sft.py \
    --video      /path/to/screw_tightening.mp4 \
    --vlm        ollama \
    --model      scomper/minicpm-v2.5:latest \
    --output-dir ./sft_output
```

### 方式三：Dry-run（无需 GPU / API，用于测试流程）

```bash
python3 demo_screw_to_pi05_sft.py \
    --video    any_file.mp4 \
    --dry-run
```

---

## 完整参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video` | *(必填)* | 输入视频路径 |
| `--output-dir` | `./sft_output` | 输出根目录 |
| `--vlm` | `ollama` | VLM 后端：`ollama` \| `gemini` |
| `--model` | `scomper/minicpm-v2.5:latest` | Ollama 模型名 |
| `--ollama-url` | `http://localhost:11434` | Ollama 服务地址 |
| `--gemini-key` | — | Gemini API Key |
| `--gemini-model` | `gemini-2.0-flash` | Gemini 模型名 |
| `--no-adaptive` | — | 关闭运动自适应采样，改用均匀采样 |
| `--num-frames` | `16` | 均匀采样帧数（仅 `--no-adaptive` 时生效）|
| `--max-vlm-frames` | `24` | 自适应模式下发给 VLM 的最大帧数 |
| `--motion-threshold` | `0.02` | 运动检测灵敏度（0~1，越小越灵敏）|
| `--robot-type` | `single_arm` | LeRobot 元数据中的机器人类型 |
| `--pi05` | `True` | 使用 π₀.5 模型（否则用 π₀）|
| `--dry-run` | — | 使用 MockVLM，无需任何 API / GPU |

---

## 输出文件结构

```
sft_output/
│
├── labels.jsonl                          # 中间结果：VLM 原始标注 (JSONL)
│
├── lerobot/                              # LeRobot V2.1 格式数据集
│   ├── meta/
│   │   ├── info.json                     # 数据集元信息（机器人类型、FPS、特征定义）
│   │   ├── episodes.jsonl                # 每条轨迹的索引与元数据
│   │   └── tasks.jsonl                   # 任务描述 + skill label 列表
│   └── data/
│       └── chunk-000/
│           └── episode_000000.json       # 逐阶段 SFT 标注数据
│
└── configs/
    ├── openpi_finetune.json              # π₀.5 完整训练配置
    └── launch_training.sh               # 一键启动训练的 Shell 脚本
```

### `episode_000000.json` 字段说明

每条记录对应视频中的一个**操作阶段**，包含：

| 字段 | 示例值 | 说明 |
|------|--------|------|
| `observation.task_description` | `"Drive the screw clockwise..."` | π₀.5 语言输入 |
| `phase_name` | `"drive_screw_cw"` | 阶段名称 |
| `action_primitive` | `"rotate_cw"` | 动作原语（15类词表） |
| `target_object` | `"screw"` | 操作对象 |
| `gripper_state` | `"closed"` | 夹爪状态 |
| `contact_type` | `"surface"` | 接触类型 |
| `force_level` | `"strong"` | 力度估计 |
| `motion_direction` | `"rotational"` | 运动方向 |
| `confidence` | `0.97` | VLM 置信度 |

---

## 动作原语词表（15类）

```
approach   align      grasp    lift      move
rotate_cw  rotate_ccw insert   push      pull
place      release    inspect  wait      retract
```

---

## 4 阶段 VLM 标注流程

```
Stage 1 — 时序阶段分割
  ├─ Pass 1: 运动自适应采样（帧差分 → 动作边界峰值检测）
  ├─ Pass 2: 对每帧单独描述（防止 VLM 幻觉）
  └─ 文本聚合 → VLM 输出 2~5 个阶段 JSON

Stage 2 — 动作原语标注
  └─ 每个阶段取 4 帧 → VLM 输出 {action_primitive, target_object, gripper_state, confidence}

Stage 3 — 接触力学估计
  └─ 每个阶段取 4 帧 → VLM 输出 {contact_type, force_level, contact_points, motion_direction}

Stage 4 — 任务摘要生成
  └─ 整体 4 帧 + 阶段序列 → VLM 输出单句自然语言任务描述
```

---

## π₀.5 SFT 训练配置关键参数

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 模型 | `pi05` | π₀.5（flow-matching VLA）|
| 骨干网络 | PaliGemma | VLM 视觉-语言编码器 |
| 动作空间 | 8D | 6D EE delta pose + screw_rotation + gripper |
| 动作预测步长 | 16 | action horizon（每次预测未来16步）|
| Flow 步数 | 10 | 推理时去噪迭代次数 |
| LoRA rank | 32 | 参数高效微调 |
| 学习率 | 5e-5 | AdamW + cosine scheduler |
| 最大训练步数 | 80,000 | 约需 50~500 条演示 |
| 精度 | bf16 | 推荐 RTX 5090 / A100 |

生成训练配置后，直接运行：

```bash
bash sft_output/configs/launch_training.sh
```

---

## 演示效果（Dry-run 输出）

```
╔══════════════════════════════════════════════════════════════╗
║        Screw-Tightening Video  →  π₀.5 SFT Data            ║
╚══════════════════════════════════════════════════════════════╝

  ─── Phase Annotations (5 phases) ───
    [0] approach_screw         │ approach     │ grip=open     │ force=none   │ motion=linear
    [1] align_socket           │ align        │ grip=closing  │ force=light  │ motion=linear
    [2] insert_socket          │ insert       │ grip=closed   │ force=medium │ motion=linear
    [3] drive_screw_cw         │ rotate_cw    │ grip=closed   │ force=strong │ motion=rotational
    [4] retract                │ retract      │ grip=opening  │ force=none   │ motion=linear

  Task summary: "Drive the screw clockwise into the solar panel mounting bracket."
```

---

## 后续步骤

1. **采集更多数据**：运行脚本处理 50~500 条拧螺丝演示视频
2. **安装 OpenPI**：`pip install openpi-client`
3. **启动微调**：`bash sft_output/configs/launch_training.sh`
4. **验证策略**：使用 `SelfLabel/Atom_Skills/screw_fasten/roboforce_validation/sim_eval.py`

---

## 相关模块

| 路径 | 说明 |
|------|------|
| `Platform/backend/auto_label_pipeline.py` | 原始 VLM 标注pipeline（`dev-VQApipeline-siyu`）|
| `Platform/backend/lerobot_exporter.py` | 原始 LeRobot 导出器（`dev-VQApipeline-siyu`）|
| `SelfLabel/Atom_Skills/screw_fasten/roboforce_skills/openpi_finetune_config.py` | 原始 π₀ 配置（`merge-local-features`）|
| `SelfLabel/Atom_Skills/screw_fasten/roboforce_sim/` | IsaacLab 螺丝安装仿真环境 |
| `SelfLabel/Atom_Skills/screw_fasten/roboforce_skills/data_collection.py` | 专家策略 + LeRobot V2 演示采集 |
