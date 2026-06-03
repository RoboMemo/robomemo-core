# Odin 通信协议规范

本文档描述 Odin 雷达与上位机之间的通信协议。

- **设备发现**: UDP 广播 (0x01 DeviceQuery)
- **心跳连接**: TCP 独立通道 (0x07 Heartbeat)
- **命令控制**: TCP 连接 (VersionQuery / GetSensorCapability / SetMode / SetLiDARMode / ChannelConfig)
- **数据推送**: UDP 或 TCP（每通道独立配置）

[TOC]

## 1. 控制协议

| 端口 | 协议 | 用途 |
| --- | --- | --- |
| 60001 | UDP | 设备广播发现 (DeviceQuery) |
| 60001 | TCP | 命令控制通道 |
| 60002 | TCP | 心跳通道 |

- **UDP 60001**: 仅处理 `0x01 DeviceQuery` 广播发现。设备已被心跳占用时静默忽略
- **TCP 60001**: 命令控制（单客户端策略，断连自动停流）
- **TCP 60002**: 心跳维持（单客户端策略，超时触发全重置）

### 1.1 帧格式

| 字段 | 长度 | 描述 |
| --- | --- | --- |
| sof | 1 | 起始字节，固定为 0xAE |
| version | 1 | 协议版本 |
| length | 2 | 数据帧长度（从 sof 到 data 段结束） |
| seq_num | 4 | 序列号，REQ 递增，ACK 与 REQ 相同 |
| cmd_id | 2 | 指令 ID |
| cmd_type | 1 | 0x00: REQ（请求），0x01: ACK（响应） |
| send_type | 1 | 0: 上位机发送，1: 设备发送 |
| crc16 | 2 | 包头校验码（sof 到 send_type 共 12 字节） |
| crc32 | 4 | data 段校验码 |
| data | n | 负载数据（最大 1454 字节） |

### 1.2 返回码

ACK 消息的 data 段首字节为返回码：

| ret_code | 说明 |
| --- | --- |
| 0 | 成功 |
| 1 | 失败 |
| 2 | 参数错误 |

### 1.3 指令列表

#### 0x01 设备查询

**REQ** (Host → Device, 广播)

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| host_ip | uint8_t[4] | 上位机 IP 地址 |

**ACK** (Device → Host, 广播)

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| ret_code | uint8_t | 返回码 |
| sn | uint8_t[16] | 设备序列号 |
| device_ip | uint8_t[4] | 设备 IP 地址 |
| model | uint8_t[32] | 设备型号 |

#### 0x02 版本查询 (VersionQuery)（TCP）

**REQ** (Host → Device, TCP)

无数据

**ACK** (Device → Host, TCP)

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| ret_code | uint8_t | 返回码 |
| firmware_version_major | uint8_t | 主版本号 |
| firmware_version_minor | uint8_t | 次版本号 |
| firmware_version_patch | uint8_t | 修订号 |
| release_notes | char[64] | 版本说明 |

#### 0x03 传感器能力查询 (GetSensorCapability)（TCP）

查询设备传感器能力表（分辨率、帧率、数据格式等）。

**REQ** (Host → Device, TCP)

无数据（查询所有通道），或 `data_type` 列表（查询指定通道）。

**ACK** (Device → Host, TCP)

Payload 为 **JSON 文本**（UTF-8 编码，格式化输出便于调试）：

```json
{
  "channels": [
    {
      "data_type": 0,
      "format": 16,
      "default": [0, 0],
      "resolutions": [
        { "width": 256, "height": 192, "fps": [10] }
      ]
    },
    {
      "data_type": 1,
      "format": 32,
      "default": [0, 0],
      "resolutions": [
        { "width": 256, "height": 192, "fps": [10] }
      ]
    },
    {
      "data_type": 16,
      "format": 0,
      "default": [0, 0],
      "resolutions": [
        { "width": 1280, "height": 1088, "fps": [30] },
        { "width": 640, "height": 544, "fps": [30] }
      ]
    },
    {
      "data_type": 17,
      "format": 0,
      "default": [0, 0],
      "resolutions": [
        { "width": 1280, "height": 1088, "fps": [30] },
        { "width": 640, "height": 544, "fps": [30] }
      ]
    },
    {
      "data_type": 32,
      "format": 49,
      "default": [0, 0],
      "resolutions": [
        { "width": 0, "height": 0, "fps": [400] }
      ]
    },
    {
      "data_type": 48,
      "format": 64,
      "default": [0, 0],
      "resolutions": [
        { "width": 0, "height": 0, "fps": [10] }
      ]
    },
    {
      "data_type": 49,
      "format": 64,
      "default": [0, 0],
      "resolutions": [
        { "width": 0, "height": 0, "fps": [400] }
      ]
    }
  ]
}
```

**字段说明**：
- `data_type`: 通道类型 (0x00/0x01/0x10/0x11/0x20/0x30/0x31)
- `format`: OdinDataFormat 值（见下表）
- `default`: [resolution_id, fps_id] — 设备建议的默认配置
- `resolutions`: 分辨率/帧率能力列表
  - `width`, `height`: 分辨率尺寸（无分辨率特征的通道如 IMU/Odom 使用 0）
  - `fps`: 该分辨率支持的帧率数组，索引即 fps_id

所有通道均包含 `default` 和 `resolutions` 字段，结构统一。

**OdinDataFormat 统一定义**:

| format 值 | 含义 | 适用通道 |
| --- | --- | --- |
| 0x00 | MJPEG | Image |
| 0x01 | YUYV | Image |
| 0x02 | NV12 | Image |
| 0x03 | NV21 | Image |
| 0x04 | RGB24 | Image |
| 0x10 | XYZIC (8B/点) | RawPoint |
| 0x11 | XYZ (6B/点, 精简) | RawPoint |
| 0x20 | XYZRGBA (10B/点) | SlamPoint |
| 0x21 | XYZ (6B/点, 精简) | SlamPoint |
| 0x30 | 6-axis (Gyro + Acc) | IMU |
| 0x31 | 9-axis (Gyro + Acc + Mag) | IMU |
| 0x40 | Standard | Odom |

能力表从设备 XML 配置文件 `<sensors>` 节点加载。LiDAR 模式通过独立的 `0x05 SetLiDARMode` 命令配置。

#### 0x04 模式配置 (SetMode)（TCP）

**REQ** (Host → Device, TCP)

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| target_mode | uint8_t | 0x00: 待机，0x01: 正常工作 |

- `Normal`: 设备进入工作状态（门控开启）。若已有通道订阅配置，立即按已有配置恢复推流；若无配置则等待 `0x06 ChannelConfig` 订阅
- `Standby`: 暂停所有通道推流，但保持 TCP 数据连接不断开，保留全部通道订阅配置。再次切回 Normal 时零延迟恢复

> SetMode 仅作为全局推流门控，不影响通道订阅状态。TCP 命令连接断开时才完全重置所有配置。

**ACK** (Device → Host, TCP)

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| ret_code | uint8_t | 返回码 |
| current_mode | uint8_t | 当前模式 |

#### 0x05 LiDAR 模式设置 (SetLiDARMode)（TCP）

**REQ** (Host → Device, TCP)

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| sensor_id | uint8_t | 传感器序号（保留） |
| mode | uint8_t | 0: HDR, 1: High Peak, 2: Low Peak, 3: Adaptive |

**ACK** (Device → Host, TCP)

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| ret_code | uint8_t | 返回码 |

#### 0x07 心跳 (Heartbeat)（TCP 60002）

Host 周期性发送心跳，设备回传当前状态。详细设计见 `heartbeat-design.md`。

**REQ** (Host → Device, TCP 60002)

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| heartbeat_interval_ms | uint16_t | Host 心跳发送间隔(ms)，建议值 1000 |

**ACK** (Device → Host, TCP 60002)

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| ret_code | uint8_t | 返回码 (0=成功) |
| uptime_s | uint32_t | 设备运行时间(秒) |

**超时策略**：设备端若连续 `3 × heartbeat_interval_ms`（默认 3000ms）未收到心跳 → 停流 + 断开命令 TCP + 重置所有配置 + 关闭心跳 TCP + 恢复响应 DeviceQuery。

**设备独占**：心跳 TCP 连接建立后，设备静默忽略 UDP DeviceQuery 广播。心跳断开后恢复可发现。

---

#### 0x06 通道配置 (ChannelConfig)（TCP）

按通道订阅/取消数据推送，支持选择传输模式 (UDP/TCP)。

**REQ** (Host → Device, TCP)

```
[data_type: uint8_t]  ← 分段编号 (0x00/0x01/0x10/0x11/0x20/0x30)
[TLV 0] [type: uint8_t] [length: uint8_t] [value: N bytes]
[TLV 1] ...
```

`data_type` 按传感器类别分段编号，每段预留 16 个位置：

| data_type | 数据内容 | 内部索引 | 端口由 ChannelConfig 指定 |
| --- | --- | --- | --- |
| 0x00 | 原始点云 | 0 | 由上位机配置 |
| 0x01 | SLAM 点云 | 1 | 由上位机配置 |
| 0x10 | JPEG 图像（左） | 2 | 由上位机配置 |
| 0x11 | JPEG 图像（右） | 3 | 由上位机配置 |
| 0x20 | IMU 数据 | 4 | 由上位机配置 |
| 0x30 | Odom 数据 | 5 | 由上位机配置 |

**已定义 TLV Key**:

| type | length | value | 描述 |
| --- | --- | --- | --- |
| `0x01` | 2 | uint16_t (BE) | `dst_port`: 目标端口，0 = 取消订阅 |
| `0x02` | 1 | uint8_t | `transport_mode`: 0=UDP（默认），1=TCP |
| `0x03` | 1 | uint8_t | `resolution_id`: 能力表分辨率索引（仅 Image 通道，0xFF=未指定） |
| `0x04` | 1 | uint8_t | `fps_id`: 所选 resolution_id 下的帧率子列表索引（仅 Image 通道，0xFF=未指定） |
| `0x05` | 1 | uint8_t | `format`: OdinDataFormat 值（仅 Image 通道，0xFF=未指定） |

**行为逻辑**:
- `dst_port > 0` + `transport_mode=0(UDP)`: 设备通过 UDP sendto 推送数据到 host_ip:dst_port
- `dst_port > 0` + `transport_mode=1(TCP)`: 设备主动 TCP connect 到 host_ip:dst_port
- `dst_port = 0`: 取消订阅该通道，若有 TCP 数据连接则关闭
- `resolution_id` / `fps_id`: 必须在能力表有效索引范围内，否则 ACK ret_code=1

**模式交互**:
- **Normal 模式下**：ChannelConfig 立即生效，配置后立即开始/停止推流
- **Standby 模式下**：ChannelConfig 仅保存配置（含建立 TCP 数据连接），但不推送数据。切回 Normal 后按已有配置自动恢复推流
- **TCP 命令连接断开**：完全重置所有通道配置 + 关闭所有 TCP 数据连接

设备收到未知 TLV type 时跳过 (`offset += 2 + length`)，保证向前兼容。

**ACK** (Device → Host, TCP)

| 名称 | 类型 | 描述 |
| --- | --- | --- |
| ret_code | uint8_t | 0=成功, 1=参数无效, 2=通道不支持 |
| active_mask | uint8_t | 当前活跃通道位掩码 |

### 1.4 连接生命周期

完整的连接建立顺序：

1. Host 广播 UDP DeviceQuery (60001) 发现设备
2. Host TCP 连接心跳端口 (60002)
3. Host 发送首次 Heartbeat REQ，设备回 ACK，设备进入"已占用"状态
4. Host TCP 连接命令端口 (60001)
5. Host 发送 SetMode / ChannelConfig 等业务指令
6. Host 持续周期发送 Heartbeat REQ 维持连接

断开场景：
- **主动断开**: Host 关闭心跳 TCP → Device 全重置
- **心跳超时**: Device 连续 3×interval 未收到心跳 → 全重置
- **命令 TCP 断开但心跳在**: 停流+重置通道配置，保持"已占用"状态

全重置 = 停止推流 + 关闭 TCP 数据连接 + 关闭命令 TCP + 重置通道配置 + 关闭心跳 TCP + 恢复 DeviceQuery 响应。

详细设计及时序图见 `heartbeat-design.md`。

---

## 2. 数据协议

### 2.1 帧格式

| 字段 | 长度 | 描述 |
| --- | --- | --- |
| version | 1 | 协议版本（当前为 0） |
| length | 2 | UDP 数据段长度 |
| dot_num | 2 | 当前包有效点数/像素数 |
| udp_cnt | 2 | 当前帧 UDP 包计数（每帧从 0 开始） |
| frame_cnt | 4 | 帧编号 |
| data_type | 1 | 数据类型 |
| time_type | 1 | 时间戳类型 |
| resv | 2 | 保留 |
| crc32 | 4 | timestamp + data 校验码 |
| timestamp | 8 | 时间戳 |
| data | -- | 负载数据（最大 1445 字节） |

### 2.3 数据结构

#### 2.3.1、原始点云 (data_type=0)

每帧 49152 点 (256×192)，274 个 UDP 包

| 字段 | 类型 | 单位 |
| --- | --- | --- |
| x | uint16_t | mm |
| y | uint16_t | mm（需 -30000 偏移） |
| z | uint16_t | mm（需 -30000 偏移） |
| intensity | uint8_t | - |
| confidence | uint8_t | - |

> 每点 8 字节，每包 180 点

#### 2.3.2、SLAM 点云 (data_type=1)

每帧 49152 点，342 个 UDP 包 *(ceil(49152/144))*

| 字段 | 类型 | 单位 |
| --- | --- | --- |
| x | uint16_t | mm |
| y | uint16_t | mm（需 -30000 偏移） |
| z | uint16_t | mm（需 -30000 偏移） |
| r | uint8_t | - |
| g | uint8_t | - |
| b | uint8_t | - |
| a | uint8_t | - |

> 每点 10 字节，每包 144 点

#### 2.3.3、IMU 数据 (data_type=0x20)

每帧 1 个 IMU 数据包

| 字段 | 类型 | 单位 |
| --- | --- | --- |
| gyro_x | float | rad/s |
| gyro_y | float | rad/s |
| gyro_z | float | rad/s |
| acc_x | float | m/s² |
| acc_y | float | m/s² |
| acc_z | float | m/s² |

> 每个 IMU 24 字节

#### 2.3.4、Odom 数据 (data_type=0x30)

每帧 1 个 Odom 数据包

| 字段 | 类型 | 单位 |
| --- | --- | --- |
| pos | int64_t[3] | μm |
| orient | int64_t[4] | ×10⁻⁶ |
| linear_velocity | int64_t[3] | - |
| angular_velocity | int64_t[3] | - |
| cov | int64_t[18] | - |
| type | uint8_t | - |

> 每个 Odom 249 字节

---

## 3. HTTP 文件传输与 OTA

设备的文件收发与固件升级（OTA）已迁移至 HTTP 服务器（端口 8080），详见 `docs/http-ota-design.md`。

