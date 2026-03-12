# Cross-Embodiment Retarget Demo

基于 NVIDIA SONIC Latent Retarget 的跨本体动作迁移 MVP Demo。

将人类全身动作（头部 + 双手 + 双脚/腰部）通过 SONIC 的 Shared Latent Space 实时迁移到仿真中的 Unitree G1 人形机器人。

## 系统架构

```
┌─────────────────────────────────────────────┐
│         Input Sources (4 modes)             │
│  ┌─────────┐ ┌───────┐ ┌─────┐ ┌────────┐  │
│  │  Mock   │ │Webcam │ │PICO │ │ Xreal  │  │
│  │(testing)│ │MediaPi│ │4 Ult│ │Air2 Ult│  │
│  └────┬────┘ └───┬───┘ └──┬──┘ └───┬────┘  │
│       └──────────┴────────┴────────┘        │
│                    │ BodyPose (Z-up)        │
│                    ▼                        │
│       ┌────────────────────────┐            │
│       │  SONIC Retarget Engine │            │
│       │  Hybrid Encoder → FSQ  │            │
│       │  → Robot Decoder (29J) │            │
│       └───────────┬────────────┘            │
│                   │ Joint Targets (50Hz)    │
│                   ▼                         │
│       ┌────────────────────────┐            │
│       │   Simulation Backend   │            │
│       │  MockPhysics / IsaacLab│            │
│       └───────────┬────────────┘            │
│                   │                         │
│                   ▼                         │
│       ┌────────────────────────┐            │
│       │  3D Visualiser (MPL)   │            │
│       └────────────────────────┘            │
└─────────────────────────────────────────────┘
```

## 支持的输入模式

| 模式 | 设备 | 追踪能力 | 用途 |
|------|------|---------|------|
| `mock` | 无（合成数据） | 全身（含腿/腰） | 开发测试 |
| `webcam` | USB 摄像头 | 全身（MediaPipe） | 快速验证 |
| `pico` | PICO 4 Ultra + Motion Trackers | 头+手+腿+腰（6DoF） | **生产级全身遥操** |
| `xreal` | Xreal Air 2 Ultra + Beam Pro | 头+手（6DoF），腿自动生成 | 上半身遥操 |

## 快速开始

### 1. 安装依赖

```bash
# 创建 conda 环境
conda create -n retarget python=3.10 -y
conda activate retarget

# 一键安装
bash setup/install_dependencies.sh

# 或手动安装
pip install numpy pyyaml pyzmq matplotlib websockets
pip install mediapipe opencv-python  # 可选：webcam 模式
pip install onnxruntime-gpu          # 可选：真实 SONIC 模型
```

### 2. 运行测试

```bash
cd cross_embodiment_retarget_demo
conda activate retarget
python -m pytest tests/test_retarget.py -v
```

### 3. 运行 Demo

```bash
# Mock 模式（无需任何硬件）
python -m src.demo_runner --input mock --motion walk

# 带 3D 可视化
python -m src.demo_runner --input mock --motion wave

# PICO 模式（需要 PICO 4 Ultra + XRoboToolkit）
# 终端 1: 启动 demo
python -m src.demo_runner --input pico
# 终端 2: 启动模拟 PICO 数据（测试用）
python -m tests.mock_pico_sender --motion_type walk --port 5555

# 禁用可视化
python -m src.demo_runner --input mock --motion squat --no-viz
```

## 项目结构

```
cross_embodiment_retarget_demo/
├── configs/
│   └── demo_config.yaml       # 统一配置（IP、端口、机器人参数等）
├── src/
│   ├── body_types.py          # 共享数据类型 & 坐标系转换
│   ├── pico_receiver.py       # PICO 4 Ultra ZMQ 接收器
│   ├── xreal_receiver.py      # Xreal Air 2 Ultra WebSocket 接收器
│   ├── webcam_capture.py      # MediaPipe 全身追踪
│   ├── mock_motion.py         # 合成动作生成器
│   ├── sonic_retarget.py      # SONIC Retarget 引擎（Mock + ONNX 后端）
│   ├── isaac_env.py           # 仿真环境（MockPhysics + Isaac Lab）
│   ├── visualiser.py          # 3D 骨架可视化
│   └── demo_runner.py         # 主程序（协调所有模块）
├── tests/
│   ├── test_retarget.py       # 单元测试 & 集成测试（16 tests）
│   └── mock_pico_sender.py    # ZMQ 模拟 PICO 数据发送器
├── setup/
│   ├── install_dependencies.sh    # 一键安装脚本
│   └── download_checkpoints.py    # 下载 SONIC 模型权重
└── README.md
```

## 硬件配置

### 已验证环境
- **GPU**: NVIDIA RTX 5090 (32GB VRAM) — CUDA 13.0
- **OS**: Ubuntu 22.04
- **Python**: 3.10 (conda)

### PICO 4 Ultra 配置
1. 安装 XRoboToolkit PC Service
2. 配对 PICO Motion Trackers（2x 脚踝 + 1x 腰部）
3. 在 `demo_config.yaml` 中设置 `input.source: pico`
4. 确保 PICO 和工作站在同一 Wi-Fi 网络

### Xreal Air 2 Ultra + Beam Pro 配置
1. Beam Pro 通过 USB-C 连接 Xreal Air 2 Ultra
2. 安装 Beam Pro streaming app
3. 在 `demo_config.yaml` 中设置 `input.source: xreal` 和 Beam Pro IP

## 升级到真实 SONIC 模型

```bash
# 1. 下载 SONIC ONNX 模型
python setup/download_checkpoints.py

# 2. 或者手动克隆 GR00T-WholeBodyControl
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
python download_from_hf.py

# 3. 更新配置指向模型路径
# demo_config.yaml → sonic.model_dir

# 4. 系统会自动检测 ONNX 模型并切换到真实推理后端
```

## 升级到 Isaac Lab 仿真

```bash
# 1. 安装 Isaac Sim 4.5+ 和 Isaac Lab 2.3+
# https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

# 2. 切换仿真后端
# demo_config.yaml → simulation.backend: isaac_lab

# 3. 运行（需要 Isaac Sim 环境激活）
python -m src.demo_runner --sim isaac_lab
```

## 性能指标

| 指标 | Mock 后端 | 目标（ONNX + Isaac Lab） |
|------|----------|------------------------|
| 控制频率 | 49.8 Hz | 45-50 Hz |
| 推理延迟 | 0.5 ms | < 20 ms |
| 关节限幅 | ✅ [-3.14, 3.14] rad | ✅ |
| 断连恢复 | ✅ 2s 回归站立 | ✅ |

## License

Apache 2.0
