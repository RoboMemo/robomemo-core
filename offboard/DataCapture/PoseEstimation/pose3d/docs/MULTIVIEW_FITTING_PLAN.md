# Multi-View Global SMPL-X Fitting — Scope & Plan

> 状态：**scope 稿（等用户确认 + 等 cali 重录 + VPoser 到位后再全量实现）**。
> 方向：fusion 从「每视角独立 + 三角化」升级为「定义一个全局 SMPL-X(θ,β)，同时投影到 3 相机，最小化多视角重投影 + 先验」。
> 现三角化 pipeline **保留不动**，作为 baseline / 对比。
>
> 目标方程：
> ```
> min_{θ,β}  Σ_{v=1..3} ‖Proj_v(SMPLX(θ,β)) − Kp2d_v‖²
>          + λ1·VPoser(θ_body)  + λ2·JointLimit(θ)
>          + λ3·TemporalSmooth  + λ4·Penetration
> ```
> 其中 `Proj_v = K_v [R_v | t_v]`（来自 calibration.json，H 参考系）。

---

## 1. 基座：改 SMPLify-X，不从零写

基座 = **`vchoutas/smplify-x`**（github.com/vchoutas/smplify-x），SMPLify-X 的官方实现。已核对结构：

| 文件 | 作用 | 改动 |
|---|---|---|
| `smplifyx/camera.py` | 单目弱透视相机 `estimate_camera`（scale,tx,ty） | **改成多视角**：不再估单相机，改用 calibration.json 的 `K_v, [R_v\|t_v]` 投影到 3 相机 |
| `smplifyx/fit_single_frame.py` | 单帧 fitting 主入口（stage 循环、loss 组装） | **多视角 loss**：reproj 项 Σ over views；warm-start 从 SMPLer-X 初始化 |
| `smplifyx/fitting.py` | `Fitting.run_fitting`（LBFGS + closure + `guess_init`） | 保留优化器骨架；`guess_init` 改用 SMPLer-X 输出而非 bbox |
| `smplifyx/prior.py` | `BodyPrior`(VPoser)、`AnglePrior`(关节 box 限位) | 直接复用；VPoser 走 `create_prior('vposer')` |
| `smplifyx/optimizers/lbfgs_ls.py` | LBFGS + line search | 直接复用 |
| `smplifyx/losses.py` | `J2DCameraLoss`(重投影)、`ShapePrior`、`CollisionLoss`(穿透) | `J2DCameraLoss` 改多视角；其余复用 |
| `smplifyx/keypoints.py` | OpenPose JSON 加载 | **替换数据源**：改成消费我们 SMPLer-X per-view 2D |

**策略**：不 fork 大改上游，而是**在我们的 `pose3d/fit/multiview_fitter.py` 里 import smplify-x 的 `prior` / `optimizers` / `body_model` / `losses` 组件**，自己写多视角 closure + reproj + warm-start。只有当上游组件签名不适配时才薄薄包一层 adapter。这样升级 smplify-x 时冲击面最小。

---

## 2. Loss 各项 + 权重（针对头戴三视角）

| 项 | 实现 | 权重建议 | 头戴相机重要性 |
|---|---|---|---|
| **重投影-身体** | Σ_v ‖Proj_v(J_body) − Kp2d_body_v‖²（22→24 关节映射） | **1.0**（主项） | 最高：3 视角重投影是全局一致性的唯一硬约束 |
| **重投影-手** | Σ_v ‖Proj_v(J_hand) − Kp2d_hand_v‖²（每手 SMPL-X 15 关节） | **0.5~1.0**（手小、检测噪声大，按 confidence 加权） | 高：**手部天然获得多视角约束**（本架构相对三角化的最大增益点） |
| **VPoser(θ_body)** | `BodyPrior` 负 log-likelihood（身体姿态先验，**仅身体，不管手**） | λ1 ≈ **4.0**（smplify-x 默认 4.68） | 中：身体姿态合理化，对遮挡视角补全有用 |
| **JointLimit(θ)** | `AnglePrior` box 限位；**重点：手指 MCP/PIP/DIP** 防超生理范围 | λ2 ≈ **2.0**（身体）/ 手指可加大到 **5~10** | 高：手在头戴视角易被误检→翻转/超伸，限位兜底（**手靠它，不靠 VPoser**） |
| **Shape prior(β)** | ‖β‖² | ~5.0（smplify-x 默认） | 低（米制由标定给，β 仅补形状） |
| **TemporalSmooth** | ‖θ_t − θ_{t-1}‖²（或 velocity 一致） | λ3 ≈ **1~5**（sequence 模式才开） | 中：降抖动；per-frame 模式关 |
| **Penetration** | `CollisionLoss`（pytorch3d/bvh 自交检测） | λ4 ≈ **0~1**（先关，稳定后开） | 低：单人无交互时几乎不触发；开了防手臂穿躯干 |

**头戴三视角最重**：身体 + 手的重投影（多视角约束）、手指 joint limit（手噪声大）。VPoser/穿透为辅。

---

## 3. 优化器与 warm-start

- **优化器**：per-frame 用 **LBFGS + line search**（smplify-x 自带 `lbfgs_ls`），warm-start 下收敛快；Adam 作为备选（sequence 时序模式更稳）。
- **Warm-start（关键，省算力）**：
  - θ_body、θ_hand、β 用**现 SMPLer-X 的 per-view 输出初始化**——取 3 视角中 `det_score` 最高者，或 3 视角均值（axis-angle 均值用 quaternion 中值更稳）。
  - 全局平移（H 系）用三角化 pelvis 或 3 视角 cam_trans 中值初始化（避免 cold-start 在米制尺度上漂）。
  - 这样 fitting 从近最优解微调，而非从 VPoser 均值姿态 cold-start → 迭代数大降（~50→~15）。
- **Stage schedule**：沿用 smplify-x 多 stage（先松后紧重投影权重、逐步开先验），但 stage 数减半（warm-start 不需要长退火）。
- **per-frame vs sequence**：v1 先 **per-frame**（无时序）；v2 加时序项开 **sequence（滑动窗口）**。

---

## 4. VPoser 集成点

- VPoser = **身体姿态先验**（VAE，输入 body_pose latent → 负 log-likelihood）。**只约束身体 21 关节，不管手**。
- 集成：`from smplifyx.prior import create_prior; vposer = create_prior('vposer', data_dir=<vposer_dir>)`；在 closure 里对 body_pose 算 `BodyPrior`。
- **手部不进 VPoser**——手靠 `AnglePrior` 的手指 box 限位（MCP/PIP/DIP 角度范围）约束。
- 依赖：用户需下 VPoser（SMPL-X 官网 / `vchoutas/vposer`），放 `dataset/models/vposer/`，config 指过去。

---

## 5. 手部 2D keypoint → reproj 映射

我们 schema 是 **27/手**；SMPL-X 手是 **15 关节 + wrist**（smplx forward joints[22..36]/[37..51]）。映射（与 `hand/smplx_hand_mapping.py` 的逆）：

| 我们 27 名（去 L_/R_） | SMPL-X 源 | 进 reproj? |
|---|---|---|
| Wrist | wrist(20/21) | ✓ |
| {Thumb}_CMC/MCP/IP | thumb 1/2/3 | ✓ |
| {Index,Middle,Ring,Pinky}_MCP/PIP/DIP | finger 1/2/3 | ✓ |
| *_Tip | tip（mesh 顶点 / 外推） | ✓（可降权，因 derived） |
| *_Extra / Palm_Center | derived（插值/重心） | ✗（不进 reproj，fit 后补全） |

→ fitting 阶段 reproj 用 **16 个直出关节**（wrist + 15）；tip/extra/palm 在 fit 完后由 `smplx_hand_mapping` 补全（保持现 schema 输出一致）。

**2D 手 keypoint 来源**（待定，二选一，TODO）：
- (a) 把 SMPLer-X 的 SMPL-X 手部 3D 投影到 2D（投影约定同 body）——免费但带 SMPLer-X 偏差；
- (b) 上一个独立 2D 手检测器（如 MediaPipe Hand 21）——更独立但加依赖。**倾向 (a) 先跑通**（与 body 同源、零额外模型），效果不够再 (b)。

---

## 6. 计算量预估

- per-frame：warm-start LBFGS ~15 iter × (smplx forward + 3-view reproj + VPoser + collision) ≈ **5~15 秒/帧**（5080，估）。
- 2506 帧 × ~10s ≈ **~7 小时/录制**（per-frame，无时序）。
- **必做加速**：
  - 降采样：先在 1/3 帧（~835 帧）fit，其余线性插值/最近邻 → ~2.3 小时。
  - 批处理：多帧 batch 过 smplx/VPoser（需等长 stage，实现复杂，v2）。
  - 关 collision（λ4=0）省 bvh 开销。
- 建议先 per-frame + 降采样跑通全量，确认质量后再上 sequence/批处理。

---

## 7. 跟现 pipeline 的接法（不破坏三角化）

- 新模块：`pose3d/fit/`（`__init__.py` + `multiview_fitter.py`）。
- `run_pipeline.py` 加 `--fusion {triangulate, global_fit}`：
  - `triangulate`（**默认**，不动）：现 DLT + 单视角 fallback 路径。
  - `global_fit`：调 `MultiViewFitter.fit_sequence(per_view_keypoints, calib)` → 全局 θ,β → 跑 smplx forward 出 78 关节 → 同样写 `poses.json`（schema 不变，`source=fit`）。
- 共用：对齐、标定、SMPLer-X per-view 推理（提供 warm-start + 2D keypoint）、poses.json writer、viz。
- 输出区分：`poses.json`（triangulate）vs `poses_fit.json`（global_fit），便于对比。

---

## 8. 依赖（新增）

- `smplify-x`（vendored：`third_party/smplify-x`，`pip install -e`）。
- `vposer`（用户下，`dataset/models/vposer/`）。
- `pytorch3d`（collision/bvh；本就列在环境，Blackwell 需源码编译）。
- 其余（torch、smplx、numpy<2）现环境已有。

---

## 9. 里程碑（待用户确认后）

1. **scope 确认**（本步）→ ask_human review。
2. 等用户：① 重录 cali（板覆盖更全/更稳）；② 下 VPoser；③ smplify-x vendor。
3. 实现 `MultiViewFitter`：warm-start + 多视角 reproj + VPoser + joint limit。
4. 单帧跑通 → 降采样全量 → 对比三角化 baseline（reproj 残差 + 可视化）。
5. （v2）时序平滑、collision、批处理。
