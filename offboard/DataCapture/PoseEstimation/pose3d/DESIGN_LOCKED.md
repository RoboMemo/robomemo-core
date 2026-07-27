# 3D 全身 Pose Pipeline —— 锁定方案（里程碑 1，待用户确认）

> 状态：**待确认**。用户（贾维斯）确认前不写实现代码。
> 本文件已按【最终锁定信息】覆盖旧版（旧版两处错误已改：HaMeR 由必选→可选；多视角由 docs 扩展→主线）。
> 目标平台：**Linux + NVIDIA CUDA**；本 Mac（M4 Pro 无 CUDA）只写代码 + 可选 CPU 对齐冒烟测试，**不跑任何 3D 重推理**。
> 交付：**自包含、可移植**的 pipeline，拷到 Linux+CUDA 机器直接跑出逐帧 3D `poses.json`。

---

## 0. 核心结论（一句话）

**单用 SMPLer-X-H32 出全身（身+手 SMPL-X 拓扑）→ 棋盘格标定 3 相机 → 身体关节跨视角 DLT 三角化拿真实米制全局 3D → 手太小留单视角输出 → 融合成 76 关节 `poses.json`。** HaMeR 仅作可选增强；SMPL-X body model 已就位，主线**零额外 body model 下载**。

---

## 1. 相机配置（事实基线，作废旧 cc7 假设）

- 3 个相机**都在头上**，自上而下三视角拍全身 → **身体关节可观**。
- ⚠️ cc7 旧 `DESIGN.md` 的 "H 头戴 + L/R 腕戴看手 → 18 关节不可观" 假设**作废**。本方案**只继承其 §4.1 手部 21→26 映射表**，其余可观测性结论一律不用。
- 头戴三相机近似**刚性同体**（固连在头具上）→ 相对外参 `[R|t]_{HL}, [R|t]_{HR}` 在时间上恒定，**标定一次即可**（§5）。

---

## 2. SOTA 选型（已查证）

### 2.1 主模型：SMPLer-X-H32（身+手一次出）

| 项 | 锁定值 |
|---|---|
| 角色 | **主模型**，单模型一次性回归身体 + 双手 + 脸（SMPL-X 拓扑） |
| 仓库 | `https://github.com/caizhongang/SMPLer-X`（同 `MotrixLab/SMPLer-X`） |
| 论文 | *SMPLer-X: Scaling Up Expressive Human Pose and Shape Estimation* (CVPR'24 / TPAMI) |
| 检查点 | **`SMPLer-X-H32*`**（2024-03 camera-fix 版），HF `smpler_x_h32_correct.pth.tar`（`caizhongang/SMPLer-X`） |
| 备选档位 | HF 实有 `smpler_x_{s32,b32,l32,h32}.pth.tar`（无 H64；H64 为早期设想，上游未发布） |
| 消费 | **已就位**的 `dataset/models/smplx/SMPLX_NEUTRAL.npz`（无需下载） |
| 产出 | SMPL-X 参数（上游 `Demoer.model` 输出 `smplx_root_pose/body_pose/lhand/rhand/shape/cam_trans`，**透视 `cam_trans`，非弱透视 pred_cam**）→ 跑 SMPL-X 前向 → 3D 身体关节（≥22）+ 每手 SMPL-X 手部关节（每手 ~15） |

### 2.2 可选增强：HaMeR（默认不启用）

| 项 | 锁定值 |
|---|---|
| 角色 | **可选**：仅当 SMPLer-X 手部精度不够、要极致精细手指时启用 |
| 仓库 | `https://github.com/geopavlakos/hamer`（CVPR'24） |
| Body model | `MANO_RIGHT.pkl`（用户**未下**，主线不上 HaMeR） |
| `download_models.sh` | HaMeR/MANO 段标 **"可选 / 暂不启用"**，给获取指引但不自动下 |

### 2.3 为什么不再用 HaMeR 当主线

SMPLer-X 本身就是 SMPL-X 全身估计器，一次出身体+手；只要手部精度够用就不必再引 MANO/HaMeR，少一套 body model、少一套环境依赖。主线**零额外 body model 下载**。

---

## 3. 输出 schema（逐帧 76 关节，3D）

- **身体 24**（SMPL 运动学树名，精确不变）：
  `Pelvis, Left_Hip, Right_Hip, Spine1, Left_Knee, Right_Knee, Spine2, Left_Ankle, Right_Ankle, Spine3, Left_Foot, Right_Foot, Neck, Left_Collar, Right_Collar, Head, Left_Shoulder, Right_Shoulder, Left_Elbow, Right_Elbow, Left_Wrist, Right_Wrist, Left_Hand, Right_Hand`
- **每只手 26**（名字沿用 DESIGN.md；左手 `L_`、右手 `R_` 前缀）：`Wrist` + `{Thumb,Index,Middle,Ring,Pinky}_{CMC/MCP, MCP/PIP, IP/PIP, Tip, Extra}` + `Palm_Center`（详见 DESIGN.md §4.1 映射；拇指用 `CMC/MCP/IP/Tip/Extra`）。
- 每关节字段：`{"xyz":[x,y,z], "conf":float, "source":str}`，坐标 **3D、米制、H 相机参考系**。
- `source ∈ {triangulated, singleview, derived, hamer(可选), missing}`；`*_Extra` / `Palm_Center` 标 `source=derived`。
- ⚠️ **手部命名数待定**：你给的精确名字清单实为 **27/手（78 总）**，比 "76" 多 2（拇指 `Thumb_Extra` 是多出的第 5 项）。→ 见 §9 决策①。

输出 `poses.json` 样例（每帧）：
```jsonc
{
  "version":"1.0", "fps":59.94, "timeline_master":"H",
  "scale":"metric_meters", "board_square_size_m":0.02,
  "frames":[{
    "t":12.345, "frame_idx":740, "primary_view":"H",
    "joints":{
      "Pelvis":     {"xyz":[0.12,-0.03,2.41],"conf":0.92,"source":"triangulated"},
      "Head":       {"xyz":[0.10,-0.45,2.38],"conf":0.89,"source":"triangulated"},
      "Left_Wrist": {"xyz":[-0.31,-0.20,2.20],"conf":0.84,"source":"triangulated"},
      // ...共 24 身体
      "L_Wrist":        {"xyz":[-0.31,-0.20,2.20],"conf":0.84,"source":"singleview"},
      "L_Thumb_CMC":    {"xyz":[-0.33,-0.22,2.19],"conf":0.80,"source":"singleview"},
      "L_Index_Extra":  {"xyz":[-0.35,-0.24,2.18],"conf":0.60,"source":"derived"},
      "L_Palm_Center":  {"xyz":[-0.34,-0.23,2.18],"conf":0.70,"source":"derived"}
      // ...左手 26、右手 26
    }
  }]
}
```

---

## 4. 目录结构（全部新写在 `PoseEstimation/pose3d/` 下）

```
PoseEstimation/pose3d/
├── README.md                      # Linux+CUDA 完整跑通步骤（里程碑 2）
├── DESIGN_LOCKED.md               # 本文件
├── environment.yml                # torch+cu118, pytorch3d, mmpose/mmdet/mmcv-full, numpy<2, opencv, scipy
├── requirements.txt               # pip 钉版（与 environment.yml 互补）
├── download_models.sh             # 拉 SMPLer-X-H32(HF)；SMPL-X 标"已就位"；HaMeR/MANO 标"可选/暂不启用"
├── config.yaml                    # 路径、board_square_size_m=0.02、棋盘格(11,8)、模型档位、融合策略
├── run_pipeline.py                # 主编排：对齐→逐视角SMPLer-X→标定→DLT三角化→融合→poses.json
├── third_party/
│   └── SMPLer-X/                  # body（git clone --depth 1，不改上游）
├── pose3d/                        # 我们的包
│   ├── schema.py                  # 24 身体名 + 26/手名 + source 枚举 + 映射表
│   ├── body/
│   │   ├── smplerx_wrapper.py     # 调 SMPLer-X：SMPL-X→24 身体 3D + 投影 2D（供三角化）+ conf
│   │   └── body_mapping.py        # SMPL-X 关节索引→24 名（Left_Hand/Right_Hand←wrist 根）
│   ├── hand/
│   │   ├── smplx_hand_mapping.py  # SMPL-X 手部(15/手)→26 关节(§4.1) + Extra/Palm_Center derived
│   │   └── hamer_wrapper.py       # 【可选增强】MANO→26；默认不启用
│   ├── calib/
│   │   ├── checkerboard.py        # findChessboardCorners(11,8)，失败 fallback(10,7)
│   │   └── multicam_calib.py      # 单目 K/dist + stereoCalibrate(H,L)/(H,R)→刚性相对外参(米)
│   ├── triangulate/
│   │   └── dlt.py                 # 跨视角 DLT 线性最小二乘三角化（SVD）→米制全局 3D（H 系）
│   ├── fuse/
│   │   ├── view_selector.py       # 逐帧/逐关节选视角；三角化失败 fallback 单视角
│   │   └── hand_attach.py         # 手挂到三角化腕（米制）+ 手尺度来自 β
│   ├── io/
│   │   ├── time_align.py          # 包 Data_Preprocessing/align_audio.py
│   │   ├── video_reader.py        # ffmpeg/cv2 抽帧（操作录制 + 标定录制）
│   │   └── pose_writer.py         # poses.json 序列化（精确 schema）
│   └── viz/
│       └── skeleton_viewer.py     # 抽帧可视化（CUDA 跑完后自检）
└── docs/
    └── BODY_HAND_MAP.md           # 身体/手部映射表 + 投影/三角化约定
```

---

## 5. `run_pipeline.py` 主流程（标定 + 三角化进主线）

```
[1] 时间对齐（对操作录制，复用 Data_Preprocessing/align_audio.py）
    python align_audio.py \
      --videos mapping.json      # {"H":"H.MP4","L":"L.MP4","R":"R.MP4"}
      --reference H --sync-video --output audio_alignment.json
    → H(原样,ref, delay=0) / L_aligned.MP4 / R_aligned.MP4 + audio_alignment.json

[2] 多视角标定（用 0721-cali 标定录制，CPU/GPU 均可，一次性）
    a. 3 路标定视频抽帧 → 检测棋盘格内角点 cv2.findChessboardCorners((11,8))，
       失败 fallback (10,7)；cornerSubRefine。
    b. cv2.calibrateCamera ×3 → K_H/K_L/K_R + dist（内参，假设同机同焦段恒定）。
    c. cv2.stereoCalibrate(H,L) 与 (H,R)，传 board_square_size_m=0.02 →
       R_{HL},t_{HL} / R_{HR},t_{HR}（**刚性相对外参，米**，时间恒定）。
    → 落盘 calibration.json（K_*, dist_*, R/t_{HL,HR}, board_square_size_m）

[3] 逐视角全身推理（对操作录制 0721-1 对齐帧，CUDA）
    对 H / L_aligned / R_aligned 各跑 SMPLer-X-H32：
      → SMPL-X 参数 → 24 身体 3D 关节（各自相机系）+ 投影 2D 像素（供三角化）
      + 双手 SMPL-X 手部关节（单视角，相机系）

[4] 多视角三角化（身体拿真实米制全局 3D）
    对每个身体关节：跨 H/L/R 视角的 2D 检测（来自[3]投影）+ 标定{K, 刚性外参}
      → DLT 线性最小二乘三角化（SVD）→ 米制 3D 点（H 相机参考系，米）
    ⚠️ 仅在视角重叠区有效；非重叠区 → fallback 单视角投影回投（source=singleview）
       或留空（source=missing），逐帧统计覆盖率写入 poses.json 头部。
    手太小、三角化精度差 → 手不进三角化，仍用[3]单视角 SMPLer-X 输出。

[5] 手部映射补全（每手 15→26）
    SMPLer-X 手部关节 → 26 关节 schema（DESIGN.md §4.1）；
    *_Extra / Palm_Center 按 §4.1 插值/重心补全，source=derived。
    手挂到[4]的三角化腕关节，手尺度由身体 shape β 推得（米）→ 与身体同一米制系。

[6] 融合 + 序列化
    逐帧：米制身体 3D（三角化）+ 米制手（单视角挂腕）→ 76 关节骨架 → poses.json
```

**已知 trade-off（README 点明）**：头戴三视角若 FOV 重叠小（分区拍摄），三角化只能覆盖重叠区 → 米制 3D 可能有空洞；此时身体局部退化为单视角（非米制/降级），逐关节 `source` 如实标注。

---

## 6. 环境方案（`environment.yml`，Linux+CUDA 单 env）

SMPLer-X 依赖 `pytorch3d + mmpose/mmdet/mmcv-full`，版本互锁易冲突 → 选**已知最稳栈**：

| 包 | 版本 | 说明 |
|---|---|---|
| python | 3.10 | SMPLer-X 推荐 |
| torch | **2.0.1 + cu118** | pytorch3d/mmcv-full 有预编译轮子；cu121 备选 |
| torchvision | 0.15.2+cu118 | 匹配 torch |
| pytorch3d | 0.7.5 | 预编译对应 torch2.0/cu118 |
| mmcv-full | 1.7.1 | 与 mmdet/mmpose 互锁 |
| mmdet | 2.28.2 | SMPLer-X 检测器依赖 |
| mmpose | 0.29.0 | ViTDet/ViTPose 配置 |
| numpy | **<2（钉 1.26.4）** | 避旧包冲突（DESIGN.md 已知坑） |
| opencv-python | 4.9.x | 棋盘格标定 |
| scipy / pyyaml / tqdm / einops | 最新兼容 | 三角化 SVD / 配置 |
| smplx | 0.1.28 | SMPL-X 前向 |
| huggingface_hub / git-lfs | 最新 | 拉检查点 |
| ~~manopth/chumpy~~ | （可选） | **仅启用 HaMeR 时**装，默认跳过 |

**降级预案**：若装不拢，拆双 env（`pose3d` 主 + 可选 `pose3d-hamer`）；默认单 env。

---

## 7. SMPL-X body model —— 已就位，无需下载

| 文件 | 路径（已确认存在） |
|---|---|
| `SMPLX_NEUTRAL.npz` | `dataset/models/smplx/`（+ `.pkl` 双格式） |
| `SMPLX_MALE.npz` / `SMPLX_FEMALE.npz` | 同上（+ `.pkl`） |
| `smplx_npz.zip` / `version.txt` | 同上 |

- `download_models.sh` 对 SMPL-X 段标 **"已提供，无需下载"**，仅校验存在性。
- `config.yaml` 默认 `smplx_model_dir: dataset/models/smplx/`，SMPLer-X 经包装层指过去（或软链到 `third_party/SMPLer-X/common/utils/human_model_files/smplx/`）。

---

## 8. 用户在 Linux 上**必须手动**做的事（已大幅减少）

1. `conda env create -f environment.yml && conda activate pose3d`。
2. 确保 NVIDIA 驱动支持 CUDA 11.8（或所选 cu 版本）；装 `git-lfs`、`huggingface-cli`（HF token/登录见 README）。
3. **棋盘格标定视频**：已提供 `dataset/Ego4WholeBody/0721-cali/{H,L,R}.MP4`（板为 11×8、方格 20mm）。若换板子/换尺寸，改 `config.yaml: board_square_size_m` 与棋盘格规格。
4. `bash download_models.sh`：自动拉 SMPLer-X-H32 权重；校验 SMPL-X 已就位；HaMeR/MANO 段打印"可选、暂不启用"。
5. 数据布局：`dataset/Ego4WholeBody/<录制名>/{H,L,R}.MP4`（如 `0721-1`）；用 `--recording 0721-1`（或 `--dataset-dir`）指定，pipeline 对任意录制目录通用。
6. 运行：`python run_pipeline.py --recording 0721-1`（README 给完整命令）。

> body model **不用下**（已就位）；MANO **不用下**（HaMeR 默认关）。

---

## 9. README 大纲（里程碑 2 交付）

1. 概述：3 路头戴 MP4 → 棋盘格标定 → 米制 3D 全身 76 关节 `poses.json`。
2. 前置：CUDA 驱动、conda、git-lfs、huggingface-cli。
3. 一键安装：`conda env create -f environment.yml` + `bash download_models.sh`（SMPL-X 校验）。
4. 数据/标定布局 + 棋盘格参数（11×8、20mm、`findChessboardCorners` 内角点坑 `(11,8)→(10,7)` fallback）。
5. 运行：`python run_pipeline.py --recording 0721-1 [--body-model h32]`。
6. 输出说明：`poses.json` 字段、坐标系（米制、H 参考系）、source 语义、覆盖率统计。
7. 已知坑：mmcv/torch/pytorch3d 版本冲突、numpy 2.x、棋盘格内角点、FOV 重叠空洞、双 env fallback。
8. 可选：HaMeR 启用步骤（下 MANO、装 manopth/chumpy、`--hand-backend hamer`）。

---

## 10. ⚠️ 待用户拍板（见 ask_human）

**① 手部 schema：27/手(78 总) vs 26/手(76 总)**
你给的精确名字清单实为 27/手（5 指 × {4 解剖 + 1 Extra} + Wrist + Palm_Center），比 "76" 多 2。
- (A) **保留全部 27 名/手 = 78 关节**（忠于"精确名称，不要改"，schema 对称干净）【推荐】
- (B) 去掉 `Thumb_Extra`（拇指解剖无第 5 关节）凑 **76**。

**② 视角融合/三角化策略**（手部不进三角化，仅身体）
- (A) **三角化优先 + 单视角 fallback**：身体关节多视角重叠→DLT 米制三角化；非重叠→best 单视角回投；逐关节标 source（推荐，出米制全局 3D，容忍空洞）
- (B) **一律单视角**：逐帧选最优视角，不做三角化（简单，但非米制、非全局一致）
- 子项：手部米制放置 = 挂到三角化腕 + β 推尺度（默认）；若你想要手部留 SMPLer-X 相机系另算，请说。

> 其余（repo 检查点、环境版本、目录结构、标定/三角化进主线）均已按最终信息锁定，不再追问。
