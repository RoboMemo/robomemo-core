# Egocentric 3-相机 → 人体 76 关节姿态 Pipeline 设计方案

> 状态：**方案稿（待用户确认）**。先于代码，锁定架构与技术选型，确认后再写代码骨架。
> 作者：PoseEstimation 模块 | 日期：2026-07-16（v3：手部三路径递进 + OpenXR 26 关节锁定）
> 关联：与 **cc2（COLMAP 相机轨迹恢复）** 协同——cc2 产物（相机外参 + 尺度标定）直接喂本 pipeline（cc3）。

---

## 0. 目标

从 **3 路 egocentric 视频**（H1 头戴 / L1 左腕 / R1 右腕，已时间对齐），逐帧输出 **76 个关节**：

```
24 身体关节 + 52 手部关节(OpenXR 26/手 × 左右) = 76
每关节 = position(x,y,z) + rotation(四元数) + 置信度 + 数据来源
```

- **手部**：检测端用 MediaPipe Hands(21点，成熟轻量)，输出端统一补到 **OpenXR 26 点**（PICO/Quest 终态对齐），三条递进恢复路径（§3）。
- **身体**：正视 ego 可观测性硬伤——~5 实测、~17 走先验并留接口等 IMU/第三人称覆盖。
- 全程为「之后带 IMU 的相机」与「Nori 双目升级」预留接口。

---

## 1. 硬约束：egocentric 可观测性

### 1.1 身体 24 关节（历史实测佐证）

`comparison/test_ego_pose.py` 实测（`pose_results/test_report.json`）：`pose_detection_rate=0.25`、`hand_detection_rate=0.81`。→ **3 路 ego 视频无法恢复身体全身**。

| 档位 | 关节 (~24) | 来源 |
|---|---|---|
| **A. 直接可观 (~5)** | `Head`(H1 `org_quat`)、`L/R_Hand`、`L/R_Wrist` | 手部模块 + 相机外参(cc2) |
| **B. 半可观 (~2)** | `Neck`、`L/R_Elbow` | 部分视频 + 先验 |
| **C. 不可观 (~17)** | `Pelvis, Spine1/2/3, L/R_Hip, L/R_Knee, L/R_Ankle, L/R_Foot, L/R_Collar, L/R_Shoulder` | **先验(EgoPoser) + 留接口等 IMU/第三人称覆盖** |

> C 档诚实标 `source=prior, overridable=true`，绝不假装是测出来的。

### 1.2 手部关节 —— 利好，但分三路径

L1/R1 腕戴相机清晰拍得到自己的手（检出率 0.81）。**手部是本 pipeline 优先落地部分。** 能否复刻 PICO 三步法取决于「同一只手有没有两个视角」——H1 头戴恰好可作为左右手**共享的第二视角**（§3 路径2）。

---

## 2. 现状盘点（避免重复造轮子）

### 2.1 已有数据（`dataset/`）

| 文件 | 角色 |
|---|---|
| `H1.MP4`/`L1_aligned.MP4`/`R1_aligned.MP4` | 已对齐 3 路 ego 视频，**主输入** |
| `*_metadata.json` | DJI 内参 `fisheye_params` + 逐帧 IMU 四元数 `Quaternion.Data`(1000Hz) + `sensor_frame_rate` → Head 朝向现成 + IMU 数据源 |
| `L1_metadata_gyroflow.json.csv` | GyroFlow：`frame,timestamp_ms,org_acc_xyz,pitch/yaw/roll` → 相机姿态先验，校验 cc2 |
| `audio_alignment.json` | `normalized_xcorr` 已算好 delay → **时间对齐直接消费** |
| `H_cali/L_cali/R_cali.MP4` | 8×8 棋盘格（**单目**内参标定，非双目对；可作尺度锚点候选，见 §3.4） |

内参：H/L `770,960,720@1920×1440`(输出裁剪1080)；R `770,960,540@1920×1080`。

### 2.2 已有代码（`Data_Preprocessing/`）

- `rynn_vla_data_pipeline.py` — YOLOv8-Pose + MediaPipe HandLandmarker(21点) + YOLOv8-World；**复用 MediaPipe 调用约定 + `hand_landmarker.task`**。
- `align_audio.py` — 音频对齐产出，**直接消费**。

### 2.3 对标与升级硬件

- 第三方动捕（Rokoko/Move.ai）= 第三人称，实测 ego 差，**排除**。
- Gen-EgoData DAS（6相机+IMU+触觉→MCAP+VIO）= **对标架构**。
- **`v1/02/时间戳IMU解码/`（cc2 发现）**：Nori Xvision。SDK 实锤 `4000x1200_*.bmp`(双目拼接) + `icm42688_decode.*`(ICM-42688 IMU) → 「真双目 + IMU」一体，路径3 硬件。

---

## 3. 手部模块：三条递进路径（核心）

**统一目标**：每手 26 个 6-DoF 关节 `(R_i, t_i) ∈ SE(3)`。三路径共用 **模块A（NN 2D 检测）** 与 **模块C（运动学约束）**，差别在 **模块B（3D 恢复）**。

### 3.1 三路径总览

| | 路径1 单目 fallback | **路径2 跨相机三角化 ★当前最优** | 路径3 Nori 真双目 |
|---|---|---|---|
| 视角 | L1/R1 各自单目 | **H1+L1→左手，H1+R1→右手**（H 共享第二视角） | Nori 左右目（同机双目） |
| 模块B | MANO 拟合 / MediaPipe 3D | **DLT 三角化**（两视角 2D→3D） | DLT 三角化（双目） |
| 尺度 | ❌ 不可靠（手大小先验） | ⚠️ COLMAP 相对单位，**需尺度锚点**(§3.4) | ✅ 真实 mm（出厂标定） |
| 旋转 | MANO 关节角 | 三角化关节点 + IK 构关节系 | 同路径2 |
| 依赖 | 仅视频 | **cc2 轨迹外参 [R\|t]_{HL/HR}** + 尺度锚点 + **H 视角手检测**(§3.5) | Nori 硬件 + 双目标定 |
| 当前可用性 | ✅ 可用（弱） | ✅ **可用，DJI 数据下首选** | ❌ 等硬件 |

> **路径2 是本次重点**：用现有 3 路 DJI 数据即可拿到带（待标定）尺度的手部 3D，是当前阶段最优产出。可行性取决于 §3.5（H 视角手检测率）。

### 3.2 模块A — NN 2D 关键点检测（三路径共用）`hand/detector_2d.py`

- **现在**：MediaPipe Hands → 21 点；按 §4 映射补到 OpenXR 26（21 检测 + 5 补全）。
- **升级**：自训/换 26 点检测器对齐 PICO 输出头。
- 输出：每手每帧 26 点 2D `{(u,v,conf)}`。双腕相机天然左右分离，不需 handedness 消歧。

### 3.3 模块B — 3D 恢复（三路径分叉）`hand/recovery_3d.py`

- **路径1 `MonoFallback`**：MediaPipe metric 3D（弱尺度）或 MANO 拟合（HaMeR 思路：2D→MANO 参数→26 关节，尺度来自手大小先验）。输出标 `scale_unreliable=true`。
- **路径2 `CrossCamTriangulator`**：对同一只手的 H 视角 2D + 腕戴视角 2D，用 cc2 外参 DLT 线性最小二乘三角化出 3D（COLMAP 世界系）。
  - 输入：左手 = {H 帧 26×2D, L 帧 26×2D, `K_H,K_L,dist`, `[R|t]_{HL}`}；右手同理用 R。
  - **前提**：该关节在 H 与腕戴视角**同时被检出**（见 §3.5 风险）。
- **路径3 `StereoTriangulator`**：Nori 左右目 DLT，真实 mm。需 `StereoCalibrationProvider{K_L,K_R,dist,R_LR,t_LR,baseline}`。

### 3.4 路径2 的三条关键依赖（决定能否跑通）

**(1) 外参 H↔L、H↔R = cc2 COLMAP 轨迹（cc2→cc3 协同点）**
- cc2 恢复同一 COLMAP 世界系 W 下的逐帧轨迹 `T_H(t),T_L(t),T_R(t) ∈ SE(3)`。
- 路径2 所需相对外参由轨迹直接合成：`[R|t]_{HL}(t) = T_H(t)⁻¹·T_L(t)`，`[R|t]_{HR}(t)=T_H(t)⁻¹·T_R(t)`。
- **接口契约**（本框架 `external/trajectory_provider.py`）：
  ```
  TrajectoryProvider.get_pose(cam, t) -> SE(3)            # cam ∈ {H,L,R}, 世界系 W
  TrajectoryProvider.get_relative(a, b, t) -> SE(3)       # T_a⁻¹·T_b，三角化直接用
  ```
  cc2 产出 `trajectories_{H,L,R}.json`（逐帧 R|t + 时间戳）即可。

**(2) 尺度问题（与 cc2 同源，须保持一致）**
- COLMAP 单目 SfM 无绝对 mm 尺度 → H+L 三角化出的手部 3D 也是无尺度的（COLMAP 相对单位）。
- **尺度锚点候选**（按可靠性排序，建议主用 ②，辅以 ③ 校验）：

| 锚点 | 可靠性 | 适用 | 备注 |
|---|---|---|---|
| ② 头↔腕人体测量先验 | **中高**（±几 cm，若知穿戴者尺寸） | **操作帧**（相机在身上，cc2 已恢复 H/腕轨迹） | **推荐主用**；与 cc2 尺度标定须一致 |
| ③ MANO 手大小先验 | 中（±10%，手有个体差） | 任何检出手的帧 | 局部校验/兜底，不作主锚 |
| ① 8×8 棋盘格（已知方格尺寸） | 高（若同帧） | 仅 `*_cali.MP4` 标定帧 | 标定帧与操作帧**不同录制**，不能直接套到操作帧；用于内参 + 出厂级尺度参考 |

> ⚠️ **一致性要求**：cc2 在做"轨迹 mm 尺度标定"面对的是同一问题。本 pipeline 的手部尺度**必须复用 cc2 选定的同一个尺度因子 s**（三角化点 × s 即 mm），不能各标各的。需与 cc2 对齐锚点选型（建议 ask_human 一并确认，或由 cc2 主导、本框架消费）。

**(3) 风险点：H 视角里手部关键点的检测率**（见 §3.5，**路径2 可行性的决定性不确定项**）

### 3.5 路径2 可行性探针（关键，待权威验证）

**问题**：H1 头戴广角里，左右手可能偏小或被身体遮挡。若 H 里检不出与腕戴视角对应的手部关节，则无 2D 对应 → 无法三角化 → 路径2 降级为路径1。

**本会话已尝试**：从 H1(22s,1920×1440@59.94fps) 抽 7 帧目视。
- 环境限制：**mediapipe 未安装**（`pip3 show` 空、`import` 失败，疑 numpy 2.x 冲突），无法跑检测器；MCP 视觉工具本会话不稳定（4 次 analyze 中 2 次格式错误、2 次返回"办公场景无手"）。
- **结论：样本(2/1300 帧)+工具不稳，不能下定论**。2 帧返回"无手"是**弱提示**（H 视角手可见性可能偏低），但绝非判定。

**权威探针（下一步明确动作，需用户点头即跑）**——`hand/probe_h_visibility.py`：
1. 在隔离 venv 装 mediapipe（不动用户 numpy 2.x 环境）。
2. 对 H1 均匀抽 ~60 帧，跑 MediaPipe Hands，统计：
   - (a) H 里手**检出率**（handedness 左/右）；
   - (b) 检出手在 H 中的 **bbox 占比**（判断够不够大做关节检测）；
   - (c) 与 L1/R1 **同时间戳帧**的**关节级共可见率**（左手关节在 H 与 L 同帧都被检出 → 才能三角化）；
   - (d) 抽样可视化叠加。
3. 判据：若 (c) 共可见率 > ~40% → 路径2 可行（首选）；若 < ~15% → 降级路径1，等 Nori。

> 这一探针是路径2 落地前的硬门槛。**建议在写代码骨架前先跑**，结果回填本节。

---

## 4. 手部 26 关节定义（OpenXR 标准集，已锁定）

```
0:PALM  1:WRIST
THUMB(4,无INTERMEDIATE):  2 METACARPAL / 3 PROXIMAL / 4 DISTAL / 5 TIP
INDEX(5):  6 METACARPAL / 7 PROXIMAL / 8 INTERMEDIATE / 9 DISTAL / 10 TIP
MIDDLE(5): 11 METACARPAL / 12 PROXIMAL / 13 INTERMEDIATE / 14 DISTAL / 15 TIP
RING(5):   16 METACARPAL / 17 PROXIMAL / 18 INTERMEDIATE / 19 DISTAL / 20 TIP
LITTLE(5): 21 METACARPAL / 22 PROXIMAL / 23 INTERMEDIATE / 24 DISTAL / 25 TIP
= 26/手，左右各一份 = 52
```
（拇指解剖正确只两节指骨，故 4 关节、无 INTERMEDIATE。）

### 4.1 MediaPipe 21 → OpenXR 26 映射表

| OpenXR | 关节 | 来源 | MediaPipe | OpenXR | 关节 | 来源 | MediaPipe |
|---|---|---|---|---|---|---|---|
| 0 | PALM | **补全** | — | 13 | MIDDLE_INTERMEDIATE | **补全** | — |
| 1 | WRIST | 直接 | 0 | 14 | MIDDLE_DISTAL | 直接 | 11 (DIP) |
| 2 | THUMB_METACARPAL | 直接 | 1 (CMC) | 15 | MIDDLE_TIP | 直接 | 12 |
| 3 | THUMB_PROXIMAL | 直接 | 2 (MCP) | 16 | RING_METACARPAL | 直接 | 13 (MCP) |
| 4 | THUMB_DISTAL | 直接 | 3 (IP) | 17 | RING_PROXIMAL | 直接 | 14 (PIP) |
| 5 | THUMB_TIP | 直接 | 4 | 18 | RING_INTERMEDIATE | **补全** | — |
| 6 | INDEX_METACARPAL | 直接 | 5 (MCP) | 19 | RING_DISTAL | 直接 | 15 (DIP) |
| 7 | INDEX_PROXIMAL | 直接 | 6 (PIP) | 20 | RING_TIP | 直接 | 16 |
| 8 | INDEX_INTERMEDIATE | **补全** | — | 21 | LITTLE_METACARPAL | 直接 | 17 (MCP) |
| 9 | INDEX_DISTAL | 直接 | 7 (DIP) | 22 | LITTLE_PROXIMAL | 直接 | 18 (PIP) |
| 10 | INDEX_TIP | 直接 | 8 | 23 | LITTLE_INTERMEDIATE | **补全** | — |
| 11 | MIDDLE_METACARPAL | 直接 | 9 (MCP) | 24 | LITTLE_DISTAL | 直接 | 19 (DIP) |
| 12 | MIDDLE_PROXIMAL | 直接 | 10 (PIP) | 25 | LITTLE_TIP | 直接 | 20 |

**补全 5 点的方法与误差预期**：
- `*_INTERMEDIATE`(食/中/无名/小指，共4)：在 `PROXIMAL–DISTAL` 间按指骨长度比插值（近:远 ≈ 2:1，可标定）。误差：亚像素~1px（2D），3D 下视三角化精度而定。
- `PALM`：取 {WRIST + 4×METACARPAL} 加权重心，或 MediaPipe hand center。误差：掌心位置 ±数 mm。
- 旋转：补全点继承父关节旋转（INTERMEDIATE 继承 PROXIMAL），或由相邻骨头方向重算。

> ⚠️ **映射约定**：MediaPipe 官方把 "MCP" 标注为掌指关节(knuckle)，严格解剖上更贴近 OpenXR **PROXIMAL** 而非 METACARPAL。本表按"MP MCP → OpenXR METACARPAL"约定，以使补全恰为 `PALM + 4 INTERMEDIATE`（与已锁定决策一致）。若后续精度要求高，可改 `MP MCP → PROXIMAL` 并改为补 4 个 METACARPAL（数学等价，解剖更贴）。**当前按锁定约定实现。**

### 4.2 模块C — 运动学约束（三路径共用）`hand/kinematic_constraint.py`

- **路径1**：MANO shape prior β 隐式保证骨长一致，不需单独 QP。
- **路径2/3**：**显式 QP/IK**（骨长固定 + 关节角范围）作用在三角化点上——我们没有 PICO 把约束烤进 NPU NN 的定制模型，显式优化更现实。旋转 R_i 由相邻三角化关节的骨头方向经 IK 构造关节坐标系。
- PICO 路线（CanonPose 式骨长归一化烤进 NN 输出层，NPU<5ms）= 理想态，列未来工作（需自训 NN）。

---

## 5. 框架数据流

```
dataset/ 3路对齐视频 ──▶ [I/O时间对齐层] ──▶ 帧三元组(H_t,L_t,R_t)+全局时间轴+逐帧IMU
                              │
        ┌─────────────────────┼─────────────────────────────────────────┐
        ▼                     ▼                                           ▼
  [手部子模块]            H1 metadata org_quat                        (预留)IMU/外部
   模块A: MP21→26          → Head 朝向                                  → 身体C档覆盖
   模块B: 路径1/2/3分叉         │                              ◀──── cc2 轨迹外参 [R|t]_{HL/HR}
   模块C: 运动学约束            ▼                              ◀──── cc2 尺度因子 s (须一致)
        │                  [身体子模块]
        │                   A实测(Head,L/R_Hand,L/R_Wrist←手部+外参×s)
        │                   B半可观(Neck,L/R_Elbow)
        │                   C先验(EgoPoser: head+hand→SMPL,可被IMU覆盖)
        └─────────────────────┬─────────────────────────────────────────┘
                              ▼
                  [序列化层] 逐帧 76 关节 pos+rot+conf+source
                              ▼ poses.json / poses.npz (+可视化)
```

---

## 6. 输出 Schema（逐帧）

```jsonc
{
  "version":"0.3","fps":30,"timeline_master":"H1",
  "hand_backend":"crosscam_triangulation",  // mono_fallback | crosscam_triangulation | stereo
  "scale_source":"anthropometric",           // 尺度锚点; 若路径1则为 "hand_prior(不可靠)"
  "frames":[{
    "t":12.345,"frame_idx":370,
    "joints":{
      // 身体24
      "Head":{"pos":[..],"rot":[w,x,y,z],"conf":0.95,"source":"imu","observable":true},
      "Pelvis":{"pos":..,"rot":..,"conf":0.3,"source":"prior","observable":false,"overridable":true},
      // 左手26(OpenXR名)
      "L_PALM":{"pos":..,"rot":..,"conf":0.8,"source":"derived","observable":true},
      "L_WRIST":{"pos":..,"rot":..,"conf":0.9,"source":"video","observable":true},
      "L_INDEX_TIP":{"pos":..,"rot":..,"conf":0.85,"source":"video","observable":true},
      // 右手26 ...
      // 共 24 + 26 + 26 = 76
    }
  }]
}
```
`source`∈{video,imu,prior,derived,external}；`observable`=真观测与否；`overridable`=身体C档；路径1手部带 `scale_unreliable:true`。

---

## 7. 待用户确认的决策（→ ask_human）

1. **路径2 探针**：[推荐] 我现在就在隔离 venv 装 mediapipe 跑 H1 检测探针（出真实检测率 + 与 L1/R1 的关节共可见率），再定路径2 是否首选 / 还是先把探针脚本写好等你跑？
2. **手部当前首选路径**：[推荐] 路径2（H+L/H+R 跨相机三角化，吃 cc2 外参）/ 或先路径1 单目 fallback 跑通？
3. **尺度锚点**：[推荐] 主用「头↔腕人体测量先验」+ MANO 兜底，且**尺度因子与 cc2 共用同一标定** / 或另选？
4. **模块C 约束**：[推荐] 路径1 用 MANO prior(隐式)、路径2/3 用显式 QP / 或统一显式 QP？
5. **身体 C 档(17关节)**：[推荐] EgoPoser 先验补全 + overridable 占位 / 或留空等外部？
6. **L/R_Hand、Wrist 的 world 坐标**：[推荐] 先相机局部系交付，world 拼接依赖 cc2 外参留接口 / 或等 cc2？
7. **目标 fps**：[推荐] 30Hz / 其他？
8. **落地优先级**：[推荐] 手部路径2(或探针后定的首选)先跑通，身体先验骨架并行。

---

## 8. 预案：目录与代码骨架（确认后落地）

```
PoseEstimation/
├── DESIGN.md
├── README.md                      # 用法（确认后补）
├── requirements.txt               # mediapipe / manopth / hamer / scipy(QP) / numpy ...
├── config.yaml                    # 数据路径、fps、hand_backend、尺度锚点
├── pose_pipeline.py               # 主编排：DataLoader→[手|身体]→Writer
├── io/
│   ├── data_loader.py             # 读 metadata/视频/对齐
│   ├── time_aligner.py            # master 时间轴 + 帧三元组
│   └── pose_writer.py             # json/npz 序列化
├── hand/
│   ├── hand_estimator.py          # 编排 A→B→C
│   ├── detector_2d.py             # 模块A：MediaPipe21→26
│   ├── recovery_3d.py             # 模块B：MonoFallback / CrossCamTriangulator / StereoTriangulator
│   ├── kinematic_constraint.py    # 模块C：显式QP / MANO prior
│   ├── hand_model_mapping.py      # §4.1 MediaPipe21→OpenXR26 + PALM/INTERMEDIATE 补全
│   └── probe_h_visibility.py      # 【§3.5 探针】H 视角手检测率+共可见率
├── body/
│   ├── body_estimator.py          # A实测/B半/C先验
│   ├── smpl_prior.py              # EgoPoser 思路占位
│   └── joints.py                  # 24 关节定义 + 可观测性表
├── external/
│   ├── base_provider.py           # ExternalPoseProvider 抽象
│   ├── trajectory_provider.py     # 【cc2→cc3 接口】get_pose/get_relative
│   ├── scale_calibrator.py        # 【§3.4】尺度锚点（人体测量/MANO/棋盘格），与cc2一致
│   ├── dji_imu_provider.py        # 现有 metadata Quaternion.Data
│   ├── nori_imu_provider.py       # 未来 Nori ICM-42688
│   ├── colmap_extrinsics.py       # 消费 cc2 相机外参
│   └── stereo_calibration.py      # 【路径3 缺口】K_L/K_R + [R|t]_LR
└── viz/
    └── skeleton_viewer.py         # 调试骨架可视化
```
每个文件先放 **接口签名 + docstring + 占位(NotImplementedError)**，不闷头写实现。
```
