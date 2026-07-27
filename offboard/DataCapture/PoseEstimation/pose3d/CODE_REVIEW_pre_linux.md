# Pose3D Pipeline — Pre-Linux Code Review (read-only)

审阅对象：`PoseEstimation/pose3d/`（19 个 Python 文件 ~1709 行 + 配置/脚本/文档）
目标平台：Linux + NVIDIA CUDA（Mac 写、Mac 测不了 SMPLer-X 真推理）
审阅方式：**只读**，对照真实上游仓库源码核验关键假设（caizhongang/SMPLer-X、vchoutas/smplx、HuggingFace）。

严重度图例：🔴 上 Linux 必炸/出根本错结果 ｜ 🟡 需注意/有风险 ｜ 🟢 建议/已核实 OK

---

## TL;DR（先看这三条，都是 🔴）

1. **SMPLer-X 适配层（`body/smplerx_wrapper.py`）整套 API 假设是错的**——真实 `caizhongang/SMPLer-X` 根本没有 `apps/SMPLerX.py` / `SMPLerX` 类 / `inference(img, body_bbox)` 方法 / `pred_cam` 输出。clone 下来直接 `ModuleNotFoundError`，pipeline 在加载模型那一步（`run_pipeline.py:155`）就炸，一帧都跑不了。
2. **三角化的投影矩阵外参符号反了**（`calib/multicam_calib.py:build_projection_matrices`）——stereoCalibrate 返回的是 other→H，三角化需要 H→other。结果：身体 3D 要么全 `missing`，要么是垃圾点。
3. **`environment.yml` 没装 `librosa`**，但 `Data_Preprocessing/align_audio.py` 顶层 `import librosa`——第 1 步时间对齐就 `ModuleNotFoundError`。

好消息：78 关节 schema / SMPL-X 身体+手索引映射 / MANO 手指顺序 **核验正确**（详见 🟢-13）。

---

## 🔴 必炸 / 根本性错误

### 🔴-1  SMPLer-X 适配层写在一个不存在的上游 API 上（最高优先）
**文件**：`pose3d/body/smplerx_wrapper.py:54-89`（`_import_regressor`）、`:95-135`（`_raw_inference`）、`:180-207`（`_project_to_pixels`）；牵连 `config.yaml:57-58`、`download_models.sh:40,60`。

**核验依据（已拉真实上游源码）**：
- `https://raw.githubusercontent.com/caizhongang/SMPLer-X/main/apps/SMPLerX.py` → **404**（main / master / v0.1 / release / original / 2023.12 / 2024.03 全部 404）。`apps/` 目录在该仓库不存在。
- `core/path_config.py` → **404**。
- 真实入口 `main/inference.py`（200）里：`from config import cfg`（yacs）、`from base import Demoer`、`demoer._make_model()`、`out = demoer.model(inputs, targets, meta_info, 'test')`。
- 真实模型 forward 输出 key：`smplx_mesh_cam / smplx_root_pose / smplx_body_pose / smplx_lhand_pose / smplx_rhand_pose / smplx_shape(betas) / smplx_expr / cam_trans`（**透视平移，不是 `pred_cam` 弱透视**）。
- HuggingFace `caizhongang/SMPLer-X` 文件列表：`smpler_x_s32/b32/l32/h32.pth.tar` + `smpler_x_h32_correct.pth.tar`（后者即 DESIGN 要的 2024-03 camera-fix 版）。全是 `.pth.tar`。
- 真实 config 文件是 `main/config/config_smpler_x_h32.py`（yacs），不是 `exp/configs/smplx/h32.yaml`。

**当前代码站不住的假设**：
| 代码 | 行 | 真实情况 |
|---|---|---|
| `from apps.SMPLerX import SMPLerX` | 74 | 无此模块/类 → `ModuleNotFoundError` |
| `from core import path_config` | 65 | 无此文件（包在 try/except，只打 note） |
| `from mmcv import Config; Config.fromfile(...)` | 73,83 | 真实 config 是 yacs `.py`，且 `SMPLerX` 类不存在 |
| `SMPLerX(mmcfg, device=self.device)` | 89 | 无此类/构造 |
| `regressor.inference(rgb, body_bbox=None, hand_bbox_list=None)` | 99 | 无此方法；真实是 `demoer.model(...)` |
| 输出取 `pred_cam / global_orient / betas / transl / det_score` | 125-135 | 真实是 `cam_trans / smplx_root_pose / smplx_shape / ...`；无 `det_score`（检测分在 mmdet 侧） |
| `_project_to_pixels`（pred_cam 弱透视→像素） | 180-207 | 真实模型不输出 `pred_cam`；投影应按 `main/inference.py` 的 `focal = cfg.focal/input_body_shape*bbox`、`princpt = ...` 来做 |
| `config: exp/configs/smplx/h32.yaml` | config.yaml:57 | 404，真实 `main/config/config_smpler_x_h32.py` |
| `checkpoint: models/h32.pth` | config.yaml:58 | 真实 `pretrained_models/smpler_x_h32_correct.pth.tar` |
| `git clone --depth 1`（未 pin commit） | download_models.sh:40 | 拉到当前 HEAD(064baef) 的 Demoer 代码库，更不可能匹配 |
| `H32_FILE="smpler_x_h32.pth"`（下载） | download_models.sh:60 | 扩展名错（`.pth.tar`）+ camera-fix 版文件名是 `smpler_x_h32_correct.pth.tar` → HF 404 |

**为什么炸**：`SMPLerXWrapper.__init__` → `_import_regressor` 第 74 行 `from apps.SMPLerX import SMPLerX` 抛 `ModuleNotFoundError`，向上冒泡，`run_pipeline.py:155 wrapper = SMPLerXWrapper(...)` 直接挂掉，整条 pipeline 一帧都进不了。

**建议修法**（隔离层思路是对的，只需重写两个适配点 + 路径）：
- `_import_regressor` 改写成：`sys.path` 插入 `<repo>/main`、`<repo>/common`、`<repo>/data`；`from config import cfg`；`cfg.get_config_fromfile(<repo>/main/config/config_smpler_x_h32.py)`；`cfg.update_test_config(..., pretrained_model_path=<ckpt>)`；`from base import Demoer; d=Demoer(); d._make_model(); d.model.eval()`。内部还要 `init_detector` 起 mmdet（ViTDet/faster_rcnn）做人体检测。
- `_raw_inference` 改写成：mmdet 检测 → 取最大 body bbox → `process_bbox` + `generate_patch_image` 裁 224 → `demoer.model({'img':...},{},{},'test')` → 从 `out` 取 `smplx_root_pose/body_pose/lhand_pose/rhand_pose/shape/cam_trans`（归一化成统一 dict）。
- 投影：用 `cam_trans` + 按 bbox 缩放的 `focal/princpt`（照搬 `main/inference.py` 的 `render_mesh` 算法），**不要再用 pred_cam 弱透视那套**。
- config.yaml 改真实路径；download_models.sh 用 `smpler_x_h32_correct.pth.tar`、pin 一个 commit、去掉已废弃的 `--local-dir-use-symlinks`。
- 这块务必上 Linux 后先用一张样例帧把 `out` 的 key/shape 打印出来再定型（Mac 无法验证）。

---

### 🔴-2  三角化投影矩阵外参方向反了
**文件**：`pose3d/calib/multicam_calib.py:115-120`（stereoCalibrate 返回值）+ `:136-144`（`build_projection_matrices`）。

**问题**：`cv2.stereoCalibrate(obj, pts_ref(H), pts_other, K_ref, dist_ref, K_other, dist_other, ...)` 返回的 `R,t` 按 OpenCV 约定是 **把点从第二相机系搬到第一相机系**：`X_H = R·X_other + t`（即 other→H）。而 DLT 里 `P_H = K_H[I|0]` 把 3D 点钉在 H 系，因此非参考相机的 `P_other` 必须是 **H→other** 的投影，即应用逆变换 `(Rᵀ, -Rᵀ·t)`。

当前代码（`:143`）：`Ps[v] = K @ hstack([R, t])` ——直接把 other→H 的 `R,t` 当成 H→other 用。

**为什么炸**：DLT 最小化的重投影残差物理上无意义。后果二选一：各视角重投影误差巨大 → `dlt.triangulate_joint` 的离群剔除把视角全丢光 → 关节标 `missing`/`too_few`；或侥幸 2 视角收敛到一个**镜像/平移错的垃圾点**。即便 🔴-1 修好，身体米制 3D 仍是错的，并污染下游 single-view/手的 Procrustes 对齐（它们锚在三角化身体上）。

**核验**（举例）：L 相机相对 H 仅 x 方向平移 +1m。`X_H=(0,0,5)` 在 L 系应为 `(-1,0,5)`，正确投影 `x_L = fx·(-1/5)`；当前 `P_L=K_L[I|(1,0,0)]` 算出 `x_L = fx·(+1/5)`——符号反。

**建议修法**：
```python
# build_projection_matrices 里，非参考视角取逆
R_inv = R.T
t_inv = (-R.T @ t.reshape(3,1)).reshape(3)
Ps[v] = K @ np.hstack([R_inv, t_inv.reshape(3,1)])
```
或在标定时就存 H→other 方向。修后用 `dlt.reproj_errors` 的中位数 < ~3-5px 自检。

---

### 🔴-3  依赖缺 `librosa`，第 1 步对齐就炸
**文件**：`Data_Preprocessing/align_audio.py:17`（`import librosa`，模块顶层）；缺失于 `environment.yml:26-52` 与 `requirements.txt`。

**为什么炸**：`time_align.align_recording` 用 `subprocess.run(["python", align_audio.py, ...], check=True)`（`time_align.py:28,35`）调起对齐脚本。子进程顶层 `import librosa` → `ModuleNotFoundError` → `CalledProcessError` → `run_pipeline.step_align`（默认 `align.enabled=true`）抛错挂掉。`step_calibrate` 对 cali 视频也会强制再跑一遍对齐（`run_pipeline.py:98-99`，`run=True`），所以**即便跳过操作录制对齐，cali 对齐照样炸**。

**建议修法**：`environment.yml`/`requirements.txt` 加 `librosa`（带 `numba`/`soundfile`/`audioread`）；README 注明 `ffmpeg`/`ffprobe` 必须在 PATH（`align_audio.py` 大量 shell 调 ffmpeg/ffprobe）；子进程改用 `sys.executable` 而非裸 `"python"`（见 🟡-4）。

---

## 🟡 需注意 / 有风险

### 🟡-4  `align_recording` 位置参数错位，config 的 `sr` 被静默丢弃；子进程用裸 `python`
**文件**：`pose3d/io/time_align.py:15-17`（签名）、`:28`（子进程）；调用方 `run_pipeline.py:84-85` 与 `:98-99`。

签名是 `(recording_dir, views, reference, video_ext, script_path, videos_map, sr=16000, run=True)`，但两个调用方都把 `a["sr"]`（采样率 int）传进了 `videos_map` 槽位，`sr` 走默认 16000。结果：`cfg.align.sr`（配置里的 sr）完全失效、恒为 16000；`videos_map` 参数也形同虚设（map 在函数内按 `views+ext` 重建，`:23` 注释自承认）。

另外 `subprocess` 用裸 `"python"`，可能不是当前 conda 解释器（即便 env 装了 librosa 也可能 import 不到）。

**建议**：调用方改关键字 `sr=a["sr"]`；删/修无用的 `videos_map` 形参；子进程首参数改 `[sys.executable, ...]`。

### 🟡-5  cali 标定无条件重跑音频对齐，强依赖 cali 有音轨
**文件**：`run_pipeline.py:88-110`（`step_calibrate`，`:98-99` 恒 `run=True`）。

DESIGN 自己提示「cali 视频可能没有音频，需同步录制或手动对齐」，但代码对 cali 总是跑音频对齐。`align_audio.extract_audio`（`-q:a 9`）与 `sync_video_to_reference` 的 `-map 0:a:0`（`align_audio.py:418`）遇到无音轨视频会失败 → 对齐挂掉或不出 `*_aligned.MP4` → `run_pipeline.py:144 assert` 挂。

**建议**：给 cali 一个「硬件同步/手动对齐、直接按帧配对」的旁路（音频可选）；或 cali 视频无音频时跳过对齐、直接 `findChessboardCorners` 配对。

### 🟡-6  立体标定的角点跨视角方向一致性不保证
**文件**：`pose3d/calib/checkerboard.py:18-29`（`cv2.findChessboardCorners`）+ `multicam_calib.py:104-120`。

`findChessboardCorners` 不保证跨视角从同一个物理角点开始返回——头戴三视角看同一块板，板子相对某相机可能呈旋转姿态，导致 `corner[k]` 在 H/L/R 对应到不同物理点。object_points 一致也无法挽救，`stereoCalibrate` 会给出错误的 `R,t`（即便 🔴-2 修了）。当前没有任何角点方向归一化。

**建议**：改用 `cv2.findChessboardCornersSB`（方向一致）或 ChArUco 板；打印每对 stereo RMS 并设阈值（如 < 1.0px），超阈值就拒绝/重录。

### 🟡-7  `download_models.sh` clone 未 pin commit + checkpoint 名/扩展名错 + HF flag 废弃
**文件**：`download_models.sh:40`（clone）、`:60`（`H32_FILE="smpler_x_h32.pth"`）、`:63`（`--local-dir-use-symlinks False`）。

- `git clone --depth 1` 未 pin commit → 上游一旦重构（已发生过）就把已写好的（本就错的）适配假设再打破，且不可复现。
- checkpoint 真实文件是 `smpler_x_h32_correct.pth.tar`（camera-fix，DESIGN 目标）/ `smpler_x_h32.pth.tar`，脚本猜的 `smpler_x_h32.pth` 扩展名和名字都错 → HF 404（脚本能优雅退到手动指引，但有摩擦）。
- `--local-dir-use-symlinks False` 在新版 `huggingface_hub` 已废弃，可能报 unrecognized argument。

**建议**：pin commit；用正确 HF 文件名；删废弃 flag。

### 🟡-8  环境文件本身装不齐 mmcv-full / pytorch3d；torch 栈与官方 SMPLer-X 不一致
**文件**：`environment.yml:26-52`、`requirements.txt`。

- `environment.yml` 的 pip 段把 `mmcv-full`、`pytorch3d` 只写成注释（要走 wheel-index 手动装）。光 `conda env create -f` 不会装它们 → `from mmcv import Config`（`smplerx_wrapper.py:73`）和 SMPLer-X 的 pytorch3d/mmpose 导入全挂。README 必须把这两步做成「不可错过」或落进 post-create 脚本。
- 官方 SMPLer-X 是 `python3.8 / torch1.12 / cu113 / mmcv-full1.7.1(cu113/torch1.12)`、mmpose 来自 `main/transformer_utils`（`pip install -v -e .`）；本 pipeline 选 `torch2.0.1/cu118` + `pip install mmpose==0.29.0`。pytorch3d/mmcv-full 有 cu118/torch2.0 预编译轮（合理取舍），但 `mmpose==0.29.0`（pip）可能和 SMPLer-X 自带的 transformer_utils mmpose 冲突（`force=True` 注册那段 README 已预警）。

**建议**：把 wheel-index 安装固化进脚本；上机先验证 `python -c "from mmcv import Config; import mmpose, mmdet"`；决定 mmpose 用哪一版，避免双注册冲突。

### 🟡-9  pred_cam 投影逻辑（即便 🔴-1 修好后）是未验证的猜测
**文件**：`pose3d/body/smplerx_wrapper.py:180-207`（`_project_to_pixels`）。

`pred_cam` 弱透视→像素这段数学本身是标准 PIXIE 约定（`z=2·focal/(crop·scale)` + bbox 回投），**但前提是存在 `pred_cam`**——真实输出是 `cam_trans`（透视），不存在 `pred_cam`，所以这整套要按真实 API 重写（见 🔴-1）。此外 crop→image 的 `center=bbox_center, side=max(w,h)·bbox_scale` 假设必须和 SMPLer-X 内部 crop 完全一致，否则投影系统性偏移；作者已在 docstring 自标「必上机核对」，这是三角化输入质量的命门，务必用 `skeleton_viewer.overlay_body2d` 跑样例帧确认关节落在人身上。

### 🟡-10  `n_frames` 不可靠时可能写出上亿空帧；HEVC 随机seek不准
**文件**：`pose3d/io/video_reader.py:25-27`、`run_pipeline.py:162`（`min(... or 10**9)`）、`run_pipeline.py:213-224`（viz 随机 `read(idx)`）。

- 对齐脚本把视频重编码成 `libx265`（`align_audio.py:420`）。OpenCV 的 `CAP_PROP_FRAME_COUNT` / `CAP_PROP_POS_FRAMES` 在 HEVC 上常返回 0 或 seek 不准。若 `n_frames` 返回 0，`n=min(... or 10**9)=10**9`，三个生成器相继 StopIteration 后外层 `for i in range(10**9)` 仍会持续往 `frames_out` 追加空帧——内存/时间炸弹。
- viz 用 `read(idx)` 随机读 HEVC 帧，可能取到错位/重复帧。

**建议**：外层循环加「同帧三视角全 StopIteration 即 break」；中间视频避免 HEVC（改用帧精确、易 seek 的编码，或一次解码到内存/抽帧到图）。

### 🟡-11  NaN 在离群重试后未复检，Procrustes 锚点可能含 NaN
**文件**：`pose3d/triangulate/dlt.py:82-96`（重试后未复检 finite）、`fuse/view_selector.py:136-141`、`fuse/hand_attach.py:33-58`（Procrustes 锚点取 `xyz is not None`，但 NaN≠None）。

DLT 丢一个视角重算后 `X` 可能 NaN，但 `status` 仍 `"triangulated"`、`X is not None`（NaN）→ `fuse_body_frame` 把 NaN xyz 存进 `out`；`view_selector`/`hand_attach` 的 `fused={... if xyz is not None}` 会把 NaN 当有效点喂进 Procrustes → NaN 扩散到手。

**建议**：`dlt` 重试后再 `if not np.all(np.isfinite(X))`；融合/挂手处把 NaN 当无效（`xyz is not None and np.all(np.isfinite(xyz))`）。

### 🟡-12  `--body-model h64` 路径不存在（DESIGN 的 H64 备选是虚构的）
**文件**：`run_pipeline.py:120,128-129`；`DESIGN_LOCKED.md` §2.1。

DESIGN 称可切 `SMPLer-X-H64`，但 HF/README 模型表只有 S32/B32/L32/H32（无 H64）。`--body-model h64` 会设 `config=exp/configs/smplx/h64.yaml`、`checkpoint=models/h64.pth`，文件都不存在 → 挂。

**建议**：去掉 h64 选项或改成真实档位；更新 DESIGN。

---

## 🟢 建议 / 已核实 OK

### 🟢-13  SMPL-X 身体+手关节索引映射 **核验正确**
对照 `vchoutas/smplx` 源码（`body_models.py:511-513`：`NUM_BODY_JOINTS=21, NUM_HAND_JOINTS=15`；`vertex_joint_selector.py`：tip 顺序 thumb/index/middle/ring/pinky 追加在运动学关节之后）确认：
- 身体 0-21（`Pelvis=0…Right_Wrist=21`）、左/右手分别在 22-36 / 37-51 —— `schema.py:36-46` 与 `body_mapping.py` 正确。
- 手内 15 关节顺序 `index, middle, pinky, ring, thumb`（pinky 先于 ring）——与 MANO/SMPL-X 运动学一致，`schema.py:89-126` 正确。
- `_hand_smplx_layout` 的 `base=22(L)/37(R)`、`wrist=20/21` 正确。

唯一 nit：`body_mapping.py:6` 与 `docs/BODY_HAND_MAP.md:7` 写「`(B,127,3)`」——真实默认（`use_face_contour=False`）关节数不是 127（127 仅开 face contour 时），但 0-51 恒有效，不影响。上机确认 `out.joints.shape[1] >= 52` 即可。

### 🟢-14  DLT 线性方程构造本身正确
`dlt.py:29-31` 的 `u·P[2]-P[0]` / `v·P[2]-P[1]` 是标准 DLT；SVD 取末右奇异向量、`X[:3]/X[3]` 正确。问题只在输入 `P` 矩阵（见 🔴-2）。

### 🟢-15  其他小项
- `skeleton_viewer.py:96` 三元两分支完全相同（无操作 ternary），纯 cosmetic。
- `checkerboard.py:36` `np.mgrid[0:ch,0:cw]` 与 OpenCV 教程 `np.indices((cw,ch))` 轴序相反——但因生成器全链路一致，对内参/外参无影响（仅困惑），建议对齐成标准写法。
- `pose_writer.py:23` 默认 `indent=1`，长录制会显著增大 json；大数据集可考虑紧凑序列化。
- `hand/hamer_wrapper.py:61` `infer_hands` 抛 `NotImplementedError`——默认路径不 import 它（OK），但启用 hamer 需先补这块。
- `calib/checkerboard.py:24` `flags=None`：cv2 通常把 None 当 0，低风险，建议显式 `flags=0` 或 `CALIB_CB_ADAPTIVE_THRESH`。

---

## 建议的上 Linux 顺序（最小可跑路径）

1. 先修 🔴-3（加 librosa + ffmpeg + `sys.executable`）和 🟡-4，让对齐能跑。
2. 修 🔴-1（重写两个适配点 + config/ckpt 路径 + pin commit）——这是最大工作量；先用单帧把 `demoer.model(...)` 的 `out` 打印定型，再接 smplx forward（🟢-13 已确认索引可用）。
3. 修 🔴-2（投影矩阵取逆），用 `reproj_errors` 中位数自检；顺手考虑 🟡-6（ChArUco/SB）。
4. 跑通单视角 → 跑通 2 视角三角化 → 用 `skeleton_viewer.overlay_body2d` 验证 🟡-9 投影落点 → 全量。
5. 处理 🟡-8（环境固化）、🟡-10（HEVC/空帧）、🟡-11（NaN）等稳健性问题。

---

### 核验信息源
- 上游代码：`raw.githubusercontent.com/caizhongang/SMPLer-X/main/{main/inference.py, main/config.py, README.md}`（`apps/SMPLerX.py`、`core/path_config.py` 全 ref 404）
- 模型权重：`huggingface.co/api/models/caizhongang/SMPLer-X`（siblings：`smpler_x_{s,b,l,h32}.pth.tar`、`smpler_x_h32_correct.pth.tar`）
- SMPL-X 关节布局：`raw.githubusercontent.com/vchoutas/smplx/main/smplx/{body_models.py, vertex_joint_selector.py}`
