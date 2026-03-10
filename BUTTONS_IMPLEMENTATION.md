# 按钮功能实现总结

本文档总结了为 RoboMemo 平台添加的所有按钮功能实现。

## 已实现的功能

### 1. 平台设置按钮 (App.tsx)
**位置**: 左侧边栏底部

**功能**:
- 打开设置模态框
- 支持的设置选项:
  - 🌙 深色模式切换
  - 🔊 声音效果开关
  - 🔔 通知开关
  - 💾 自动保存开关
  - 🔌 API 端点配置
  - 🔄 恢复默认设置按钮

**技术实现**:
- 使用 React Dialog 组件
- 设置数据存储在 localStorage 中
- 支持实时保存所有设置

### 2. 数据集管理按钮 (DatasetManager.tsx)
**位置**: 每个数据集卡片的右上角菜单

**功能**:

#### Open 按钮
- 打开数据集的详细视图
- 显示成功提示

#### Edit 按钮
- 打开编辑模态框
- 可编辑内容:
  - 数据集名称
  - 描述信息
  - 数据格式 (LeRobot, RT-X, RLDS, Open X-Embodiment)
  - 机器人类型 (单臂、双臂、人型等)
  - 任务描述
- 支持保存和取消操作

#### Export 按钮
- 导出数据集元数据为 JSON 文件
- 文件名格式: `{dataset-name}-metadata.json`
- 自动下载到用户设备

#### Delete 按钮
- 删除数据集前显示确认对话框
- 删除后自动刷新数据集列表

**技术实现**:
- 使用 DropdownMenu 组件组织菜单项
- Dialog 组件用于编辑操作
- Blob API 用于文件导出

### 3. 机器人连接按钮 (DataCollection.tsx)
**位置**: 真实机器人标签页

**功能**:
- 显示当前连接状态 (已连接/断开连接/正在连接)
- 允许输入机器人 IP 地址
- Reconnect 按钮:
  - 尝试连接到指定 IP 的机器人
  - 显示连接状态动画
  - 支持重试连接

**技术实现**:
- 状态管理: connectionStatus (disconnected | connecting | connected)
- isReconnecting 标志用于按钮禁用和动画
- IP 地址可配置和可编辑

### 4. 数据标注增强功能 (DataAnnotation.tsx)
**位置**: 帧播放控制区域

**新增按钮**:

#### Open 按钮 (文件夹图标)
- 打开当前帧的详细视图
- 用于全屏标注操作

#### Edit 按钮 (编辑图标)
- 打开帧编辑模态框
- 功能:
  - 添加帧注释和说明
  - 保存帧级别的元数据
  - 追踪最后修改时间

#### Export 按钮 (下载图标)
- 导出当前帧的所有标注
- 导出格式:
  ```json
  {
    "frameIndex": number,
    "episodeId": string,
    "datasetId": string,
    "annotations": Annotation[],
    "exportedAt": ISO8601时间戳
  }
  ```
- 文件名格式: `annotations-frame-{frameIndex}.json`

**技术实现**:
- Dialog 组件用于帧编辑
- 集成的 API 调用用于数据持久化
- Blob API 用于文件导出

## 技术栈

- **UI 组件库**: shadcn/ui (Button, Dialog, DropdownMenu, Badge, etc.)
- **状态管理**: React useState/useEffect
- **数据持久化**: localStorage 和后端 API
- **文件导出**: Blob 和 File API

## 文件修改清单

1. `/src/App.tsx` - 添加设置功能
2. `/src/sections/DatasetManager.tsx` - 添加数据集编辑和导出
3. `/src/sections/DataCollection.tsx` - 添加机器人重连功能
4. `/src/sections/DataAnnotation.tsx` - 添加帧级别操作和导出

## 使用指南

### 配置平台设置
1. 点击左侧边栏的 "Settings" 按钮
2. 在模态框中调整所需设置
3. 设置自动保存到 localStorage

### 管理数据集
1. 在 "Datasets" 页面查看所有数据集
2. 点击数据集卡片右上角的菜单
3. 选择 Open、Edit、Export 或 Delete

### 连接真实机器人
1. 切换到 "Collection" 页面的 "Real Robot" 标签
2. 输入机器人 IP 地址 (默认: 192.168.1.100)
3. 点击 "Reconnect" 按钮进行连接

### 标注数据
1. 在 "Annotation" 页面选择数据集和任务
2. 使用新的按钮:
   - Open: 打开帧详情
   - Edit: 添加帧注释
   - Export: 导出标注数据

## 未来改进

- [ ] 添加批量导出功能
- [ ] 支持更多导出格式 (CSV, COCO JSON)
- [ ] 添加撤销/重做功能
- [ ] 实现标注历史追踪
- [ ] 添加协作编辑支持
