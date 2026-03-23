# RoboMemo Data Sample Report
## Demo Dataset: FurnitureBench Assembly (Franka Robot)

**场景说明**：本报告展示 RoboMemo 对机械臂组装/紧固操作数据的自动标注能力，
对标 RoboForce 螺丝套 EE 操作场景（插销定位、紧固旋转、力控操作均有覆盖）。

---

## 处理概况

| 指标 | 数值 |
|------|------|
| 数据集 | FurnitureBench LeRobot (Franka) |
| Episodes 数量 | 10 |
| 任务类型 | assemble one_leg (6), assemble stool (3), assemble lamp (1) |
| 视频总时长 | 627.0 秒 |
| 平均每 episode 时长 | 62.7 秒 |
| 平均每 episode 标注 phases | 4.7 个 |
| 总 phases 数 | 47 个 |
| 自动标注耗时 | 平均 ~40 秒/episode |
| 模型 | scomper/minicpm-v2.5:latest (Ollama) |
| Pipeline 阶段 | 4 阶段：Phase Segmentation → Action Primitives → Mechanics → Task Summary |

---

## 各 Episode 处理明细

| episode_id | task | 时长(s) | phases数 | 耗时(s) | 状态 |
|------------|------|---------|---------|--------|------|
| episode_000000 | assemble one_leg | 43.8 | 10 | ~210 | ✓ |
| episode_000001 | assemble stool | 113.1 | 3 | 52 | ✓ |
| episode_000002 | assemble one_leg | 47.8 | 3 | 33 | ✓ |
| episode_000003 | assemble one_leg | 44.6 | 3 | 34 | ✓ |
| episode_000004 | assemble stool | 91.4 | 5 | 42 | ✓ |
| episode_000005 | assemble one_leg | 37.3 | 5 | 42 | ✓ |
| episode_000006 | assemble stool | 114.0 | 5 | 44 | ✓ |
| episode_000007 | assemble lamp | 53.9 | 3 | 33 | ✓ |
| episode_000008 | assemble one_leg | 42.6 | 5 | 44 | ✓ |
| episode_000009 | assemble one_leg | 38.5 | 5 | 44 | ✓ |

---

## 输出示例（3个完整 episode 标注）

### Episode 000000 — assemble one_leg (43.8s, 10 phases)

```json
{
  "episode_id": "episode_000000",
  "video_info": { "duration": 43.8, "fps": 10.0, "resolution": [224, 224], "total_frames": 438 },
  "phases": [
    {
      "phase_idx": 0, "phase_name": "reach",
      "start_frame": 0, "end_frame": 39, "start_time": 0.0, "end_time": 3.9,
      "description": "The robot arm is in a neutral position with the gripper open and not in contact with any objects.",
      "action_primitive": "approach", "target_object": "unknown",
      "gripper_state": "open", "confidence": 0.5,
      "mechanics": { "contact_type": "none", "force_level": "none", "contact_points": "", "motion_direction": "linear" }
    },
    {
      "phase_idx": 1, "phase_name": "move_towards_object",
      "start_frame": 39, "end_frame": 79, "start_time": 3.9, "end_time": 7.9,
      "description": "The robot arm is in the process of moving its gripper towards a nearby object, with the gripper currently in an open state.",
      "action_primitive": "grasp", "target_object": "a red ball",
      "gripper_state": "closing", "confidence": 0.9,
      "mechanics": { "contact_type": "surface", "force_level": "medium", "contact_points": "edge of table", "motion_direction": "linear" }
    },
    {
      "phase_idx": 2, "phase_name": "grasp_object",
      "start_frame": 79, "end_frame": 119, "start_time": 7.9, "end_time": 11.9,
      "description": "The robot arm is holding a small object with its gripper in a closed state.",
      "action_primitive": "grasp", "target_object": "a red and white object",
      "gripper_state": "closing", "confidence": 0.9,
      "mechanics": { "contact_type": "none", "force_level": "none", "contact_points": "", "motion_direction": "linear" }
    },
    {
      "phase_idx": 3, "phase_name": "close_gripper_around_object",
      "start_frame": 119, "end_frame": 158, "start_time": 11.9, "end_time": 15.8,
      "description": "The robot arm is in a horizontal position with the gripper in the process of closing around a small object.",
      "action_primitive": "grasp", "target_object": "a red and white box",
      "gripper_state": "closing", "confidence": 0.9,
      "mechanics": { "contact_type": "none", "force_level": "none", "contact_points": "", "motion_direction": "linear" }
    },
    {
      "phase_idx": 4, "phase_name": "open_gripper",
      "start_frame": 158, "end_frame": 238, "start_time": 15.8, "end_time": 23.8,
      "description": "The robot arm is in the process of opening its gripper.",
      "action_primitive": "grasp", "target_object": "a red and white object",
      "gripper_state": "closing", "confidence": 0.9,
      "mechanics": { "contact_type": "none", "force_level": "none", "contact_points": "", "motion_direction": "linear" }
    },
    "... (10 phases total) ..."
  ],
  "task_summary": "The robot's manipulation task involves picking up a red and white object and repeatedly grasping it.",
  "success": true, "model": "scomper/minicpm-v2.5:latest"
}
```

### Episode 000004 — assemble stool (91.4s, 5 phases)

```json
{
  "episode_id": "episode_000004",
  "video_info": { "duration": 91.4, "fps": 10.0, "resolution": [224, 224], "total_frames": 914 },
  "phases": [
    {
      "phase_idx": 0, "phase_name": "idle",
      "start_frame": 0, "end_frame": 166, "start_time": 0.0, "end_time": 16.6,
      "description": "The robot arm is positioned with its gripper open and not in contact with any objects.",
      "action_primitive": "approach", "target_object": "a red and black object",
      "gripper_state": "open", "confidence": 0.8,
      "mechanics": { "contact_type": "none", "force_level": "none", "contact_points": "none", "motion_direction": "linear" }
    },
    {
      "phase_idx": 1, "phase_name": "reach",
      "start_frame": 249, "end_frame": 332, "start_time": 24.9, "end_time": 33.2,
      "description": "The robot arm is in a vertical position with the gripper in the process of opening.",
      "action_primitive": "grasp", "target_object": "a red and white object",
      "gripper_state": "closing", "confidence": 0.9,
      "mechanics": { "contact_type": "none", "force_level": "none", "contact_points": "none", "motion_direction": "linear" }
    },
    {
      "phase_idx": 2, "phase_name": "contact",
      "start_frame": 415, "end_frame": 498, "start_time": 41.5, "end_time": 49.8,
      "description": "The robot arm is in the process of moving its gripper towards an object, with the gripper currently in an open state.",
      "action_primitive": "grasp", "target_object": "a red and black object",
      "gripper_state": "closing", "confidence": 0.9,
      "mechanics": { "contact_type": "none", "force_level": "none", "contact_points": "", "motion_direction": "linear" }
    },
    {
      "phase_idx": 3, "phase_name": "grasp",
      "start_frame": 581, "end_frame": 664, "start_time": 58.1, "end_time": 66.4,
      "description": "The robot arm is in the process of picking up a small object with its gripper, which is currently in the closing position.",
      "action_primitive": "grasp", "target_object": "a red and black object",
      "gripper_state": "closing", "confidence": 0.8,
      "mechanics": { "contact_type": "surface", "force_level": "medium", "contact_points": "edge of table", "motion_direction": "linear" }
    },
    {
      "phase_idx": 4, "phase_name": "lift",
      "start_frame": 747, "end_frame": 913, "start_time": 74.7, "end_time": 91.3,
      "description": "The robot arm is holding a small object with its gripper, in a closed state.",
      "action_primitive": "grasp", "target_object": "a red and black object",
      "gripper_state": "closing", "confidence": 0.9,
      "mechanics": { "contact_type": "none", "force_level": "none", "contact_points": "none", "motion_direction": "linear" }
    }
  ],
  "task_summary": "The robot is grasping and manipulating a red and black object.",
  "success": true, "model": "scomper/minicpm-v2.5:latest"
}
```

### Episode 000007 — assemble lamp (53.9s, 3 phases)

```json
{
  "episode_id": "episode_000007",
  "video_info": { "duration": 53.9, "fps": 10.0, "resolution": [224, 224], "total_frames": 539 },
  "phases": [
    {
      "phase_idx": 0, "phase_name": "reach",
      "start_frame": 0, "end_frame": 97, "start_time": 0.0, "end_time": 9.7,
      "description": "The robot arm is in the process of moving its gripper towards a nearby object, with the gripper currently in an open state.",
      "action_primitive": "approach", "target_object": "a red ball",
      "gripper_state": "open", "confidence": 0.8,
      "mechanics": { "contact_type": "point", "force_level": "light", "contact_points": "edge of table", "motion_direction": "linear" }
    },
    {
      "phase_idx": 1, "phase_name": "grasp",
      "start_frame": 146, "end_frame": 195, "start_time": 14.6, "end_time": 19.5,
      "description": "The robot arm is holding a small object with its gripper, which is in a closed state.",
      "action_primitive": "grasp", "target_object": "a red and white object",
      "gripper_state": "closing", "confidence": 0.8,
      "mechanics": { "contact_type": "surface", "force_level": "medium", "contact_points": "edge of table", "motion_direction": "linear" }
    },
    {
      "phase_idx": 2, "phase_name": "move",
      "start_frame": 293, "end_frame": 538, "start_time": 29.3, "end_time": 53.8,
      "description": "The robot arm is holding a small object with its gripper in a closed state, and the arm is positioned to move the object to a new location.",
      "action_primitive": "grasp", "target_object": "a red and white object",
      "gripper_state": "closing", "confidence": 0.9,
      "mechanics": { "contact_type": "none", "force_level": "none", "contact_points": "none", "motion_direction": "linear" }
    }
  ],
  "task_summary": "The robot is picking up a red and white object and placing it on a platform.",
  "success": true, "model": "scomper/minicpm-v2.5:latest"
}
```

---

## 人工核查结果（10条抽样）

从 47 个 phases 中随机抽取 10 条（seed=42）进行核查：

| # | episode | phase_name | action_primitive | target_object | gripper | confidence | description (摘要) | mechanics | 核查结论 |
|---|---------|-----------|-----------------|--------------|---------|-----------|-------------------|-----------|---------|
| 1 | ep_000008 | move | grasp | red plastic bottle | closing | 0.9 | 机械臂在打开夹爪，处于开放状态 | none/none/linear | ⚠️ action与描述矛盾（描述说open但ap是grasp） |
| 2 | ep_000000 | hold_object | grasp | red and white object | closing | 0.9 | 机械臂持握红色物体，夹爪闭合 | none/none/linear | ✓ 语义一致，但应为"hold/move"更准确 |
| 3 | ep_000000 | move_towards_object | grasp | red ball | closing | 0.9 | 机械臂移向物体，夹爪开启中 | surface/medium/linear | ✓ phase名称准确，但AP用grasp覆盖了approach |
| 4 | ep_000003 | hold | grasp | red and black object | closing | 0.9 | 机械臂持握小物体，夹爪闭合 | surface/light/linear | ✓ 描述准确，mechanics有surface接触记录 |
| 5 | ep_000002 | move | grasp | red and black object | closing | 0.9 | 机械臂持握绿色物体（描述颜色错误） | surface/medium/linear | ⚠️ 颜色识别不一致（target说red and black, desc说green） |
| 6 | ep_000002 | grasp | grasp | red object | closing | 0.9 | 机械臂持握小物体，夹爪闭合，移动中 | none/none/linear | ✓ phase和AP一致，描述准确 |
| 7 | ep_000000 | hold_object | grasp | red and white object | closing | 0.9 | 机械臂持握小物体，夹爪闭合，准备移动 | none/none/linear | ✓ 合理，但phase diversity不足（多phases=grasp） |
| 8 | ep_000000 | raise_arm_with_open_gripper | grasp | red and white object | closing | 0.9 | 机械臂抬起，夹爪正在闭合 | none/none/linear | ⚠️ phase_name说raise+open，但AP=grasp，gripper=closing，矛盾 |
| 9 | ep_000007 | reach | approach | red ball | open | 0.8 | 机械臂向物体移动，夹爪开启 | point/light/linear | ✓ 最佳样本：AP=approach, gripper=open, mechanics有point接触 |
| 10 | ep_000000 | move_towards_object_with_open_gripper | grasp | red and white object | closing | 0.9 | 机械臂向物体移动，夹爪开启中 | none/none/linear | ⚠️ phase_name说open_gripper但AP=grasp/gripper=closing，内部矛盾 |

**核查汇总**：
- ✓ 正常（语义一致）：5/10
- ⚠️ 有问题（内部矛盾或颜色错误）：4/10
- ✗ 失败：0/10

**主要问题**：
1. **action_primitive 多样性不足**：47个phases中约80%标注为"grasp"，缺少insert/rotate_cw/push等更丰富的primitive
2. **内部字段矛盾**：phase_name与action_primitive、description之间有时不一致（尤其在phase_name由S1文字描述推断，AP由S2图像推断时）
3. **对象颜色描述不稳定**：同一物体在不同调用中被描述为"red ball"、"red and white object"、"green object"等
4. **mechanics检测偏保守**：约60%的phases的contact_type=none，可能低估了实际接触
5. **分割偏粗粒度**：多episodes只有3个phases（可能更细的insert/align/tighten被合并到单一grasp phase）

---

## 标注格式说明

### JSONL Schema

每行一个 JSON 对象，表示一个完整 episode 的标注结果：

```
{
  "episode_id": "episode_000000",         // 视频文件名（不含扩展名）
  "video_path": "/abs/path/to/video.mp4", // 视频绝对路径
  "video_info": {
    "duration": 43.8,                     // 总时长（秒）
    "fps": 10.0,                          // 帧率
    "resolution": [224, 224],             // 分辨率 [width, height]
    "total_frames": 438                   // 总帧数
  },
  "phases": [                             // 时序分段列表
    {
      "phase_idx": 0,                     // 分段序号（0-indexed）
      "phase_name": "reach",              // 分段名称（snake_case，由VLM描述推断）
      "start_frame": 0,                   // 起始帧号
      "end_frame": 39,                    // 结束帧号
      "start_time": 0.0,                  // 起始时间（秒）
      "end_time": 3.9,                    // 结束时间（秒）
      "description": "...",               // 该分段的自然语言描述
      "action_primitive": "approach",     // 动作原语（枚举值，见下方词汇表）
      "target_object": "unknown",         // 目标物体描述
      "gripper_state": "open",            // 夹爪状态：open/closing/closed/opening
      "confidence": 0.5,                  // VLM置信度（0.0-1.0）
      "mechanics": {
        "contact_type": "none",           // 接触类型：none/point/surface/edge/wrap
        "force_level": "none",            // 力量级别：none/light/medium/strong
        "contact_points": "",             // 接触点描述
        "motion_direction": "linear"      // 运动方向：linear/rotational/complex
      }
    }
    // ... more phases
  ],
  "task_summary": "...",                  // 任务总结（单句自然语言）
  "success": true,                        // pipeline是否成功
  "model": "scomper/minicpm-v2.5:latest", // 使用的VLM模型
  "labeled_at": "2026-03-18T07:00:00Z"   // 标注时间（UTC）
}
```

### Action Primitive 词汇表

| primitive | 含义 | 对应机械臂动作 |
|-----------|------|-------------|
| approach | 接近目标 | 末端执行器向目标移动 |
| align | 对齐 | 调整角度/位置使组件对准插孔 |
| grasp | 抓取 | 夹爪闭合抓住物体 |
| lift | 抬起 | 垂直向上提起物体 |
| move | 移动 | 携物移动到目标位置 |
| rotate_cw | 顺时针旋转 | 手腕/物体顺时针旋转（拧紧方向） |
| rotate_ccw | 逆时针旋转 | 手腕/物体逆时针旋转（拧松方向） |
| insert | 插入 | 将部件插入孔位 |
| push | 推压 | 向下或水平施加力 |
| pull | 拉取 | 向外拉动物体 |
| place | 放置 | 将物体放到目标位置 |
| release | 释放 | 夹爪打开释放物体 |
| inspect | 检查 | 观察位置/对准状态 |
| wait | 等待 | 静止或等待状态 |
| retract | 收回 | 末端执行器退出接触区域 |

---

## 与 RoboForce 场景映射

RoboForce 螺丝套 EE（End Effector）操作场景与 FurnitureBench 家具组装任务高度相似：

### 操作子动作映射表

| RoboForce 螺丝套操作 | FurnitureBench 对应动作 | RoboMemo 标注字段 |
|---------------------|----------------------|----------------|
| 🔍 **寻孔定位** — 末端执行器搜索螺纹孔位置 | `approach` → 接近家具组件连接点 | `action_primitive: approach` |
| 🎯 **插销对准** — 套筒对准螺帽/螺栓头 | `align` → 腿部榫头对准桌面榫眼 | `action_primitive: align`, `contact_type: point` |
| ⬇️ **施压接触** — 套筒接触螺帽施加轴向力 | `insert` → 榫头压入榫眼 | `action_primitive: insert`, `force_level: medium/strong` |
| 🔩 **拧紧旋转** — 顺时针旋转套筒紧固螺帽 | `rotate_cw` → 旋转紧固腿部螺帽 | `action_primitive: rotate_cw`, `motion_direction: rotational` |
| 📏 **力控验证** — 检测扭矩达到目标值后停止 | `wait`/`inspect` → 验证组装到位 | `action_primitive: inspect`, `contact_type: wrap` |
| ↩️ **脱离收回** — 套筒退出工件区域 | `retract` → 末端执行器离开组件 | `action_primitive: retract` |

### 核心相似性分析

1. **力控操作覆盖**：FurnitureBench 包含需要精确力控的插入（insert leg into hole）和旋转（tighten screw）操作，与螺丝套的插入+旋转紧固动作完全对应

2. **旋转轨迹相似**：家具组装中的螺帽拧紧轨迹（`rotate_cw`）与螺丝套 EE 的末端执行器旋转轨迹在关节空间高度相似

3. **接触状态转换**：两类任务均涉及 `none → point → surface/wrap` 的接触状态转换，RoboMemo 的 `mechanics.contact_type` 字段可直接捕获这一转换

4. **对准挑战**：FurnitureBench 中腿部/桌面组件的插销对准（millimeter-level precision）与螺丝套套头对准螺帽的精度要求相当

5. **数据域差距**：
   - FurnitureBench: 视觉+本体感知（无力传感器数据）
   - RoboForce 实际场景: 需要额外的 F/T 传感器数据和扭矩反馈
   - → RoboMemo 可扩展 `mechanics` 字段纳入实际传感器读数

---

## 质量评估与 Pipeline Bug 分析

### 标注质量评分（主观 1-5 分）

| 维度 | 得分 | 说明 |
|------|------|------|
| Phase segmentation 合理性 | 3/5 | 时间覆盖完整，但粒度偏粗，insert/rotate等细节phase未被分出 |
| Action primitive 准确性 | 2/5 | 约80%为grasp，diversity严重不足；但approach/grasp/move语义基本正确 |
| Language description 可读性 | 4/5 | 句子结构清晰，可读性好；但有颜色识别不稳定问题 |
| JSON 格式完整性 | 5/5 | 所有必填字段均存在，格式规范 |
| Mechanics 估计质量 | 2/5 | 偏保守（大量none），但surface/point接触在关键phase有正确标注 |
| **综合** | **3.2/5** | 可用于演示，不适合直接训练 |

### 发现的 Pipeline 问题

#### Bug #1: Stage 1 → Stage 2 action_primitive 不一致
**现象**：Stage 1 phase_name 由帧描述文字推断（如"move_towards_object_with_open_gripper"），Stage 2 AP 由图像重新推断为"grasp"，两者经常矛盾。
**原因**：两阶段使用不同的信息源（文字描述 vs 图像），且Stage 2 prompt缺乏对phase_name的上下文。
**建议**：Stage 2 prompt 应传入 `phase_name` 和 Stage 1 的 `description`，让VLM在此上下文下选择AP，减少歧义。

#### Bug #2: Action Primitive 多样性不足（grasp 过度主导）
**现象**：47个phases中约38/47（81%）标注为grasp。
**原因**：FurnitureBench视频分辨率低（224×224），且是全局相机视角，VLM难以区分approach/insert/rotate等细节动作；同时Stage 2 prompt未给予足够的区分引导。
**建议**：(a) 增加wrist camera视频的标注；(b) Stage 2 prompt加入per-primitive的判断准则；(c) 使用更强的视觉模型。

#### Bug #3: 对象颜色描述不稳定
**现象**：同一对象在不同调用中描述为"red ball"、"red and white object"、"green object"，颜色不一致。
**原因**：VLM对小分辨率（224×224）图像的颜色感知不稳定；每次调用独立无上下文共享。
**建议**：Stage 1 完成后锁定 `target_object` 描述，后续 Stage 2/3 传入此信息作为锚点。

#### Bug #4: 帧索引边界有时跳空
**现象**：episode_000004中phase 0结束于frame 166，phase 1开始于frame 249，中间frame 167-248未被任何phase覆盖（约8.3秒空白）。
**原因**：Stage 1 使用12帧均匀采样，将frame_index映射回实际帧号时，VLM输出的phase边界只能对齐到采样帧，不连续间隙不可避免。
**建议**：后处理步骤：自动填充phase间的空白区间（合并到前/后phase，或创建"transition"phase）。

#### 改进建议总结
1. **传入 wrist camera 视频** — 手腕相机更接近接触点，VLM更容易识别insert/rotate
2. **Two-pass AP labeling** — 先用全视角定phase，再用wrist视角精化AP
3. **增加对象track** — 在phases间传递 target_object 描述，保持一致性
4. **后处理覆盖率检查** — 确保phases连续覆盖完整视频时长
5. **引入更强视觉模型** — llava-13b 或 llama3.2-vision 在细粒度动作识别上有优势

---

## 下一步

### 短期（接入 RoboForce 数据）
- [ ] 接入 RoboForce 真实传感器数据格式（F/T sensor + joint torque）
- [ ] 在 `mechanics` 字段中扩展 `torque_nm`、`force_n` 等实际读数
- [ ] 针对螺丝套场景补充 `rotate_cw/rotate_ccw/insert` 的专项 prompt

### 中期（提升标注质量）
- [ ] 训练 domain-specific auto-label 模型（基于 FurnitureBench + 人工校正标注）
- [ ] 引入 wrist camera 双视角融合标注
- [ ] 建立人工校正工具（RLHF-style 标注修正界面）

### 长期（数据飞轮）
- [ ] 用已标注数据训练 action recognition 模型，反过来提升 pipeline 精度
- [ ] 支持多机器人平台（UR5, xArm, Unitree等）
- [ ] 输出对接 LeRobot / OpenVLA 训练格式

---

*报告生成时间：2026-03-18 | RoboMemo Platform v0.1 | 模型：scomper/minicpm-v2.5:latest*
