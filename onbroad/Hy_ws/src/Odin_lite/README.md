# Odin ROS Driver

基于 Odin SDK 的 ROS 驱动程序，支持 ROS1 和 ROS2 多版本。

## 功能特性

- **多平台支持**：ROS1 Noetic、ROS2 Humble/Iron/Jazzy
- **热插拔检测**：自动发现和连接设备，支持设备断开重连
- **心跳监控**：实时监测设备连接状态
- **多数据流**：点云、图像、IMU、里程计同步输出
- **后处理**：图像去畸变、点云着色、立体校正

## 功能模块

| 节点 | 说明 |
| --- | --- |
| `odin_ros_driver_node` | 主驱动节点：连接设备、发布原始数据、提供标定服务 |
| `post_process_node` | 后处理节点：图像去畸变、点云着色、立体校正 |

## 系统要求

| Ubuntu 版本 | ROS 版本 |
| --- | --- |
| 20.04 | ROS1 Noetic, ROS2 Foxy |
| 22.04 | ROS2 Humble, ROS2 Iron |
| 24.04 | ROS2 Jazzy |

## 依赖安装

### 编译工具（必须）

```bash
sudo apt install build-essential cmake git
```

### ROS1 (Noetic)

```bash
# ROS1 基础
sudo apt install ros-noetic-desktop-full

# ROS1 功能包依赖
sudo apt install ros-noetic-ddynamic-reconfigure
sudo apt install ros-noetic-cv-bridge ros-noetic-image-transport
sudo apt install ros-noetic-pcl-ros ros-noetic-stereo-msgs

# 系统库依赖
sudo apt install libyaml-cpp-dev libopencv-dev libeigen3-dev libpcl-dev libssl-dev
```

### ROS2 (Humble/Iron/Jazzy)

```bash
# ROS2 基础（选择对应版本）
sudo apt install ros-humble-desktop  # Ubuntu 22.04
# 或
sudo apt install ros-iron-desktop    # Ubuntu 22.04
# 或
sudo apt install ros-jazzy-desktop   # Ubuntu 24.04

# ROS2 功能包依赖（将 humble 替换为对应版本：humble/iron/jazzy）
sudo apt install ros-humble-cv-bridge ros-humble-image-transport
sudo apt install ros-humble-pcl-ros ros-humble-stereo-msgs

# 系统库依赖
sudo apt install libyaml-cpp-dev libopencv-dev libeigen3-dev libpcl-dev libssl-dev
```



## 编译

SDK 会在编译时自动构建，无需单独安装。

### ROS1

```bash
source /opt/ros/noetic/setup.bash
cd your_ros_workspace
./src/odin_ros_driver/script/build_ros1.sh
```

### ROS2

```bash
source /opt/ros/humble/setup.bash
cd your_ros_workspace
./src/odin_ros_driver/script/build_ros2.sh
```

## 运行

### ROS1

```bash
source devel/setup.bash
# 默认配置（自动发现设备，使用设备默认分辨率）
roslaunch odin_ros_driver_rev1 driver.launch

# 指定图像分辨率和帧率
roslaunch odin_ros_driver_rev1 driver.launch image_width:=640 image_height:=544 image_fps:=30

# 不启动 RViz
roslaunch odin_ros_driver_rev1 driver.launch start_rviz:=false
```

### ROS2

```bash
source install/setup.bash
# 默认配置（自动发现设备，使用设备默认分辨率）
ros2 launch odin_ros_driver_rev1 driver.launch.py

# 指定图像分辨率和帧率
ros2 launch odin_ros_driver_rev1 driver.launch.py image_width:=640 image_height:=544 image_fps:=30

# 不启动 RViz
ros2 launch odin_ros_driver_rev1 driver.launch.py start_rviz:=false
```

### Launch 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `start_rviz` | `true` | 是否启动 RViz |
| `image_width` | `0` | 图像宽度，0 = 使用设备默认 |
| `image_height` | `0` | 图像高度，0 = 使用设备默认 |
| `image_fps` | `0` | 图像帧率，0 = 使用设备默认 |
| `image_format` | `mjpeg` | 图像格式：mjpeg/yuyv/nv12/nv21/rgb24 |

## 发布话题

### odin_ros_driver_node（主驱动节点）

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `odin/cloud_raw` | `sensor_msgs/PointCloud2` | 原始点云（256×192） |
| `odin/cloud_slam` | `sensor_msgs/PointCloud2` | SLAM 彩色点云 |
| `odin/image/compressed` | `sensor_msgs/CompressedImage` | 左相机 JPEG 压缩图像 |
| `odin/image_raw` | `sensor_msgs/Image` | 左相机解码后 RGB 图像 |
| `odin/image2/compressed` | `sensor_msgs/CompressedImage` | 右相机 JPEG 压缩图像 |
| `odin/image2_raw` | `sensor_msgs/Image` | 右相机解码后 RGB 图像 |
| `odin/imu` | `sensor_msgs/Imu` | IMU 数据（加速度、角速度） |
| `odin/odometry` | `nav_msgs/Odometry` | 里程计数据 |
| `odin/gray_image` | `sensor_msgs/Image` | 灰度图像（mono8） |

### post_process_node（后处理节点）

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `odin/image_undistort` | `sensor_msgs/Image` | 左相机去畸变图像 |
| `odin/image2_undistort` | `sensor_msgs/Image` | 右相机去畸变图像 |
| `odin/cloud_color` | `sensor_msgs/PointCloud2` | 着色点云（点云+图像融合） |
| `odin/left_rect` | `sensor_msgs/Image` | 立体校正后左图像 |
| `odin/right_rect` | `sensor_msgs/Image` | 立体校正后右图像 |
| `odin/left_camera_info` | `sensor_msgs/CameraInfo` | 左相机校正后内参 |
| `odin/right_camera_info` | `sensor_msgs/CameraInfo` | 右相机校正后内参 |
| `odin/depth_pointcloud` | `sensor_msgs/PointCloud2` | 视差转点云输出 |

## 服务

| 服务 | 类型 | 说明 |
| --- | --- | --- |
| `odin/get_calibration` | `GetCalibration` | 获取相机标定 YAML 内容 |

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `host_ip` | `0.0.0.0` | 本机网卡 IP |
| `device_ip` | `""` | 雷达 IP，留空启用广播发现 |
| `device_sn` | `""` | 指定设备序列号 |
| `auto_discover` | `true` | 是否广播发现设备 |
| `operating_mode` | `normal` | 工作模式（normal/standby） |
| `point_data_port` | `59000` | 原始点云端口 |
| `slam_data_port` | `59001` | SLAM 点云端口 |
| `jpeg_data_port` | `59002` | 左相机图像端口 |
| `imu_data_port` | `59003` | IMU 数据端口 |
| `odom_data_port` | `59004` | 里程计数据端口 |
| `jpeg_data_port2` | `59005` | 右相机图像端口 |

## 配置文件

| 文件 | 说明 |
| --- | --- |
| `config/camera_calib.yaml` | 相机标定文件（从设备自动获取） |
| `launch/driver.launch` | ROS1 启动配置 |
| `launch/driver.launch.py` | ROS2 启动配置 |
| `rviz/odin.rviz` | RViz 可视化配置 |

## 启动说明

1. `device_ip` 留空时自动广播发现设备
2. 连接后自动设置为 `normal` 模式开始数据推送
3. 默认启动 RViz，可通过 `start_rviz:=false` 禁用
4. Ctrl+C 退出时自动将设备设为 `standby` 模式

## 目录结构

```
ros_driver/
├── config/                    # 配置文件
│   └── control_command.yaml   # 数据流开关配置
├── include/utility/           # 工具类头文件
├── launch/                    # 启动文件
├── module/sdk_api/            # Odin SDK（自动编译）
├── msg/                       # 自定义消息
├── rviz/                      # RViz 配置
├── script/                    # 编译脚本
├── src/                       # 源代码
│   ├── odin_ros_driver_node.cpp  # 主驱动节点（ROS1/ROS2 统一）
│   └── post_process_node.cpp     # 后处理节点（ROS1/ROS2 统一）
└── srv/                       # 自定义服务
```

## 数据流配置

通过 `config/control_command.yaml` 可配置启用/禁用各数据流：

```yaml
register_keys:
  raw_point: true      # 原始点云
  slam_point: true     # SLAM 点云
  image0: true         # 左相机图像
  image1: true         # 右相机图像
  imu: true            # IMU 数据
  odom: true           # 里程计数据
```



## Q&A

### 1. 多台电脑话题数据干扰

**问题**：多台电脑通过 WiFi/局域网连接，启动 ROS Driver 时话题相同导致数据干扰。

**ROS2 解决方法**：
```bash
# 查看当前 DOMAIN_ID（空白或 0 为共享模式）
echo $ROS_DOMAIN_ID

# 设置独立的 DOMAIN_ID（0-100，0 为共享）
export ROS_DOMAIN_ID=1
```

**ROS1 解决方法**：
```bash
# 方法一（推荐）
export ROS_LOCALHOST_ONLY=1

# 方法二：使用不同端口
roscore -p 11312
export ROS_MASTER_URI=http://localhost:11312
```

> 以上命令只对当前终端生效。永久设置请添加到 `~/.bashrc` 或 `/etc/profile`。

### 2. 设备连接失败

**问题**：无法发现或连接设备。

**解决方法**：
1. 确保设备与电脑在同一网段
2. 检查防火墙是否阻止 UDP 广播（端口 60000-60010）
3. 尝试指定设备 IP：`roslaunch odin_ros_driver_rev1 driver.launch device_ip:=192.168.1.251`

### 3. 图像分辨率不生效（ROS1）

**问题**：ROS1 下设置 `image_width`/`image_height` 参数无效。

**解决方法**：升级到 v0.9.0 或更高版本，已修复此问题。

## 版本历史

详见 [CHANGELOG.md](CHANGELOG.md)
