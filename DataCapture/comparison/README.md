# DataCapture 方案对比测试

## 目标
评估适用于 RoboMemo 无感数采的动捕/姿态估计方案，用于采集人类操作技能数据。

## 候选方案

### 1. Rokoko Vision
- **类型**: 基于 AI 的无标记视频动捕（云端处理）
- **输入**: 单/双摄像头视频
- **输出**: FBX/BVH 骨骼动画
- **价格**: 免费（≤15秒片段），Pro 需订阅 Rokoko Studio
- **特点**: 
  - 浏览器端直接使用 (vision.rokoko.com)
  - 全身骨骼追踪
  - 支持双摄像头提升精度
  - 有 Studio Command API 可编程控制录制
  - 插件生态: Blender/Unity/Unreal/C4D
- **局限**:
  - 云端处理，有延迟
  - 免费版限制 15 秒
  - 主要面向动画/影视，不专注机器人操作
  - 不追踪手指细节
- **API**: Rokoko Studio Command API (本地 HTTP，控制录制/校准/回放)
- **SDK**: [github.com/Rokoko](https://github.com/Rokoko)

### 2. Move.ai
- **类型**: AI 无标记动捕（云端/本地处理）
- **输入**: 单/多摄像头视频（iPhone/GoPro/任意）
- **输出**: FBX/BVH/USDC/USDZ/GLB/C3D/JSON/CSV
- **价格**: 
  - Move One Personal: $15/月 (375秒)
  - Move One Standard: $50/月 (1250秒)
  - Move Pro 试用: $995/月 (1000秒多摄像头)
  - Enterprise: 定制
- **特点**:
  - 支持手指追踪 (Move Pro)
  - 多人追踪 (Enterprise)
  - 有 Python SDK (`move-ugc-python`)
  - REST API 可编程
  - 专门提到 Robot Imitation Learning 用例
- **局限**:
  - 收费
  - 云端处理依赖网络
  - 实时动捕 (Move Live) 需要 Enterprise
- **API**: GraphQL API + Python SDK
- **SDK**: [github.com/move-ai/move-ugc-python](https://github.com/move-ai/move-ugc-python)

### 3. Gen-EgoData DAS (已有基线)
- **类型**: 专用可穿戴数采设备
- **输入**: 6 相机 + IMU + 触觉传感器 + 磁编码器
- **输出**: MCAP 格式，含 VIO 位姿 + 6DoF 轨迹
- **特点**: 
  - 真正的第一人称视角
  - SLAM 高精度定位
  - 完整的操作数据 (视觉+力觉+位姿)
  - 开源数据格式和工具
- **局限**: 需要专用硬件

## 对比维度

| 维度 | Rokoko Vision | Move.ai | DAS (基线) |
|------|-------------|---------|-----------|
| 手指追踪 | ❌ | ✅ (Pro) | 🔜 后续版本 |
| 第一人称视角 | ❌ | ❌ | ✅ |
| 实时性 | ❌ 云端延迟 | ❌/✅ (Live版) | ✅ |
| 6DoF 位姿 | ✅ 全身骨骼 | ✅ 全身骨骼 | ✅ 末端执行器 |
| 力觉数据 | ❌ | ❌ | ✅ 触觉传感 |
| 部署成本 | 免费/低 | 中等 | 高(专用硬件) |
| 可编程性 | ✅ Studio API | ✅ Python SDK | ✅ 开源工具 |
| 输出格式 | FBX/BVH | FBX/BVH/JSON/CSV | MCAP/H5 |
| 机器人适配 | 需要二次开发 | 有 robotics 用例 | 原生适配 |

## 测试计划

1. **Rokoko Vision**: 上传 test_ego_video.mp4 到 vision.rokoko.com，评估骨骼追踪质量
2. **Move.ai**: 通过 Move One API 上传视频，获取动捕数据
3. **对比**: 与 DAS 设备的 VIO 位姿数据做精度对比

## 文件说明
- `test_ego_video.mp4` - 从 Gen-EgoData 提取的测试视频（camera0，折叠衣服任务）
- `test_frames/` - 8 帧关键帧截图
- `rokoko-api-examples/` - Rokoko Studio Command API 示例
- `move-ugc-python/` - Move.ai Python SDK
