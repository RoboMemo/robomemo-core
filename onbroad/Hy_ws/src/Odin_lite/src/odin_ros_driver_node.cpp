#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <map>
#include <set>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <functional>

#include <yaml-cpp/yaml.h>
#include <odin_lidar_api.h>
#include "logger/logger.h"
#if defined(ODIN_ROS2)
#include <geometry_msgs/msg/quaternion.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/string.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>
#include <rmw/qos_profiles.h>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#else
#include <geometry_msgs/Quaternion.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/String.h>
#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <sensor_msgs/Image.h>
#include <ddynamic_reconfigure/ddynamic_reconfigure.h>
#include <ros/package.h>
#endif

#include "service_types.h"
#if defined(ODIN_ROS2)
#include <rcpputils/filesystem_helper.hpp>
#endif

namespace odin_ros_driver {

namespace sdk = odin::sdk;

// Helper to get package source directory (for loading config files)
static std::string getPackageSourceDirectory() {
#if defined(ODIN_ROS2)
  rcpputils::fs::path current_file(__FILE__);
  auto path = current_file.parent_path();
  while (!path.empty() && !rcpputils::fs::exists(path / "package.xml")) {
    path = path.parent_path();
  }
  if (path.empty()) {
    return "";
  }
  return path.string();
#else
  // ROS1: use ros::package::getPath
  return ros::package::getPath("odin_ros_driver_rev1");
#endif
}

/**
 * @brief Thread-safe async queue for non-blocking callback processing
 */
template <typename T>
class AsyncQueue {
 public:
  using ProcessFunc = std::function<void(const T&)>;

  AsyncQueue(const std::string& name, size_t max_size = 100)
      : name_(name), max_size_(max_size), running_(false), drop_count_(0) {}

  ~AsyncQueue() { Stop(); }

  void Start(ProcessFunc process_func) {
    if (running_) return;
    process_func_ = process_func;
    running_ = true;
    worker_ = std::thread(&AsyncQueue::WorkerLoop, this);
  }

  void Stop() {
    if (!running_) return;
    running_ = false;
    cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  bool Push(T&& item) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.size() >= max_size_) {
      ++drop_count_;
      if (drop_count_ % 100 == 1) {
        LOG_WARN("AsyncQueue[%s]: queue full, dropped %lu items\n", name_.c_str(), drop_count_.load());
      }
      return false;
    }
    queue_.push(std::move(item));
    cv_.notify_one();
    return true;
  }

  size_t Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  uint64_t DropCount() const { return drop_count_; }

 private:
  void WorkerLoop() {
    while (running_) {
      T item;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || !running_; });
        if (!running_ && queue_.empty()) break;
        if (queue_.empty()) continue;
        item = std::move(queue_.front());
        queue_.pop();
      }
      if (process_func_) {
        process_func_(item);
      }
    }
  }

  std::string name_;
  size_t max_size_;
  std::atomic<bool> running_;
  std::atomic<uint64_t> drop_count_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<T> queue_;
  std::thread worker_;
  ProcessFunc process_func_;
};

// 通过 __FILE__ 定位 package 根目录
static std::string GetPackageRoot() {
  std::string path(__FILE__);  // .../src/odin_ros_driver_node.cpp
  for (int i = 0; i < 10; ++i) {
    size_t pos = path.rfind('/');
    if (pos == std::string::npos) break;
    path = path.substr(0, pos);
    struct stat st;
    std::string pkg_xml = path + "/package.xml";
    if (stat(pkg_xml.c_str(), &st) == 0) {
      return path;
    }
  }
  return "";
}

#if defined(ODIN_ROS2)
using PointCloud2Msg = sensor_msgs::msg::PointCloud2;
using PointFieldMsg = sensor_msgs::msg::PointField;
using CompressedImageMsg = sensor_msgs::msg::CompressedImage;
using ImageMsg = sensor_msgs::msg::Image;
using ImuMsg = sensor_msgs::msg::Imu;
using OdometryMsg = nav_msgs::msg::Odometry;
using QuaternionMsg = geometry_msgs::msg::Quaternion;
using RosTime = rclcpp::Time;

template <typename MsgT>
using PublisherType = typename rclcpp::Publisher<MsgT>::SharedPtr;
#else
using PointCloud2Msg = sensor_msgs::PointCloud2;
using PointFieldMsg = sensor_msgs::PointField;
using CompressedImageMsg = sensor_msgs::CompressedImage;
using ImageMsg = sensor_msgs::Image;
using ImuMsg = sensor_msgs::Imu;
using OdometryMsg = nav_msgs::Odometry;
using QuaternionMsg = geometry_msgs::Quaternion;
using RosTime = ros::Time;

template <typename MsgT>
using PublisherType = ros::Publisher;
#endif

#define PCLPROCESSTYPE uint16_t

class odinRosDriver {
 public:
#if defined(ODIN_ROS2)
  // Constructor with pre-discovered device and device index
  odinRosDriver(const rclcpp::Node::SharedPtr &node, const sdk::DiscoveredDevice &device,
                int device_index = -1)
      : node_(node),
        discovered_device_(device),
        device_sn_(device.sn),
        device_model_(device.model),
        device_index_(device_index) {}
#else
  // Constructor with pre-discovered device and device index
  // main_pnh: main node's private NodeHandle for reading global parameters (image_width, etc.)
  // device_pnh: device-specific NodeHandle for dynamic reconfigure
  odinRosDriver(const ros::NodeHandle &nh, const ros::NodeHandle &main_pnh,
                const ros::NodeHandle &device_pnh,
                const sdk::DiscoveredDevice &device, int device_index = -1)
      : nh_(nh),
        pnh_(device_pnh),
        main_pnh_(main_pnh),
        discovered_device_(device),
        device_sn_(device.sn),
        device_model_(device.model),
        device_index_(device_index) {}
#endif

  ~odinRosDriver() { Shutdown(); }

  bool Init() {
    printf("\n");
    printf("========================================\n");
    printf("  SDK VERSION: %s\n", sdk::GetSdkVersion().c_str());
    printf("========================================\n");
    printf("\n");
    if (!LoadParameters()) {
      return false;
    }
    SetupDynamicReconfigure();
    BuildCalibFilePath();
    BuildTopicPrefix();
    AdvertiseTopics();
    InitAsyncQueues();
    if (!ConnectDevice()) {
      return false;
    }
    RegisterCallbacks();
    // Publish device_info AFTER calibration is ready, so post_process_node can fetch it
    PublishTopicPrefix();
    return true;
  }

  void InitAsyncQueues() {
    // Create async queues with appropriate sizes
    raw_point_queue_ = std::make_unique<AsyncQueue<sdk::OdinPointCloudPacket>>("RawPoint", 50);
    slam_point_queue_ = std::make_unique<AsyncQueue<sdk::OdinSlamPacket>>("SlamPoint", 50);
    image0_queue_ = std::make_unique<AsyncQueue<sdk::OdinImagePacket>>("Image0", 30);
    image1_queue_ = std::make_unique<AsyncQueue<sdk::OdinImagePacket>>("Image1", 30);
    imu_queue_ = std::make_unique<AsyncQueue<sdk::OdinImuPacket>>("IMU", 200);
    odom_queue_ = std::make_unique<AsyncQueue<OdomQueueItem>>("Odom", 200);

    // Start worker threads with processing functions
    raw_point_queue_->Start([this](const sdk::OdinPointCloudPacket& pkt) {
      ProcessRawPointCloud(pkt);
    });
    slam_point_queue_->Start([this](const sdk::OdinSlamPacket& pkt) {
      ProcessSlamPointCloud(pkt);
    });
    image0_queue_->Start([this](const sdk::OdinImagePacket& pkt) {
      ProcessImage(pkt);
    });
    image1_queue_->Start([this](const sdk::OdinImagePacket& pkt) {
      ProcessImage2(pkt);
    });
    imu_queue_->Start([this](const sdk::OdinImuPacket& pkt) {
      ProcessImu(pkt);
    });
    odom_queue_->Start([this](const OdomQueueItem& item) {
      ProcessOdom(item.packet, item.odom_type);
    });
    LogInfo("Async queues initialized");
  }

  // Public setters for HotplugManager to forward parameter changes
  void SetConfidenceThreshold(int value) {
    m_confidence_threshold = value;
    LogInfo("confidence_threshold set to: %d", m_confidence_threshold);
  }

  void SetSensorMode(int mode) {
    sensor_mode_ = mode;
    const char *mode_name;
    switch (sensor_mode_) {
      case 0:
        mode_name = "HDR";
        break;
      case 1:
        mode_name = "High Peak";
        break;
      case 2:
        mode_name = "Low Peak";
        break;
      case 3:
        mode_name = "Adaptive";
        break;
    }
    LogInfo("sensor_mode set to: %s (%d)", mode_name, sensor_mode_);
    if (connected_ && device_handle_ != sdk::kInvalidDeviceHandle) {
      sdk::SetSensorMode(device_handle_, static_cast<uint8_t>(sensor_mode_));
    }
  }

 private:
  static odinRosDriver *FromClientData(void *data) { return static_cast<odinRosDriver *>(data); }

  // SDK callbacks - push to async queues immediately (non-blocking)
  static void PointCloudCallback(const sdk::OdinPointCloudPacket &packet, void *client) {
    if (auto *driver = FromClientData(client)) {
      driver->HandleRawPointCloud(packet);
    }
  }
  static void SlamCallback(const sdk::OdinSlamPacket &packet, void *client) {
    if (auto *driver = FromClientData(client)) {
      driver->HandleSlamPointCloud(packet);
    }
  }

  static void ImageCallback(const sdk::OdinImagePacket &packet, void *client) {
    if (auto *driver = FromClientData(client)) {
      driver->HandleImage(packet);
    }
  }

  static void ImageCallback2(const sdk::OdinImagePacket &packet, void *client) {
    if (auto *driver = FromClientData(client)) {
      driver->HandleImage2(packet);
    }
  }

  static void ImuCallback(const sdk::OdinImuPacket &packet, void *client) {
    if (auto *driver = FromClientData(client)) {
      driver->HandleImu(packet);
    }
  }

  static void OdomCallback(const sdk::OdinOdomPacket &packet, sdk::OdomSourceType odom_type,
                           void *client) {
    if (auto *driver = FromClientData(client)) {
      driver->HandleOdom(packet, odom_type);
    }
  }

  // Handle functions - push to async queue (fast, non-blocking)
  void HandleRawPointCloud(const sdk::OdinPointCloudPacket &packet) {
    if (raw_point_queue_) {
      raw_point_queue_->Push(sdk::OdinPointCloudPacket(packet));
    }
  }

  void HandleSlamPointCloud(const sdk::OdinSlamPacket &packet) {
    if (slam_point_queue_) {
      slam_point_queue_->Push(sdk::OdinSlamPacket(packet));
    }
  }

  void HandleImage(const sdk::OdinImagePacket &packet) {
    if (image0_queue_) {
      image0_queue_->Push(sdk::OdinImagePacket(packet));
    }
  }

  void HandleImage2(const sdk::OdinImagePacket &packet) {
    if (image1_queue_) {
      image1_queue_->Push(sdk::OdinImagePacket(packet));
    }
  }

  void HandleImu(const sdk::OdinImuPacket &packet) {
    if (imu_queue_) {
      imu_queue_->Push(sdk::OdinImuPacket(packet));
    }
  }

  void HandleOdom(const sdk::OdinOdomPacket &packet, sdk::OdomSourceType odom_type) {
    if (odom_queue_) {
      odom_queue_->Push(OdomQueueItem{packet, odom_type});
    }
  }

  template <typename T>
  T GetParam(const std::string &name, const T &default_value) {
#if defined(ODIN_ROS2)
    if (!node_) {
      return default_value;
    }
    if (!node_->has_parameter(name)) {
      node_->declare_parameter<T>(name, default_value);
    }
    T value = default_value;
    node_->get_parameter(name, value);
    return value;
#else
    // Use main_pnh_ to read global parameters set in launch file
    return main_pnh_.param(name, default_value);
#endif
  }

  void LoadChannelEnableConfig() {
    // Default: all channels enabled
    enable_raw_point_ = true;
    enable_slam_point_ = true;
    enable_image0_ = true;
    enable_image1_ = true;
    enable_imu_ = true;
    enable_odom_ = true;

    // Try to load from control_command.yaml
    std::string pkg_path = getPackageSourceDirectory();
    if (pkg_path.empty()) {
      LogWarn("Could not find package directory, using default channel config");
      return;
    }

    std::string config_file = pkg_path + "/config/control_command.yaml";
    try {
      YAML::Node config = YAML::LoadFile(config_file);
      if (!config["register_keys"]) {
        LogWarn("Missing 'register_keys' in %s, using defaults", config_file.c_str());
        return;
      }
      YAML::Node keys = config["register_keys"];

      if (keys["enable_raw_point"]) {
        enable_raw_point_ = keys["enable_raw_point"].as<int>() != 0;
      }
      if (keys["enable_slam_point"]) {
        enable_slam_point_ = keys["enable_slam_point"].as<int>() != 0;
      }
      if (keys["enable_image0"]) {
        enable_image0_ = keys["enable_image0"].as<int>() != 0;
      }
      if (keys["enable_image1"]) {
        enable_image1_ = keys["enable_image1"].as<int>() != 0;
      }
      if (keys["enable_imu"]) {
        enable_imu_ = keys["enable_imu"].as<int>() != 0;
      }
      if (keys["enable_odom"]) {
        enable_odom_ = keys["enable_odom"].as<int>() != 0;
      }

      LogInfo("Loaded channel config from %s", config_file.c_str());
    } catch (const std::exception& e) {
      LogWarn("Failed to load %s: %s, using defaults", config_file.c_str(), e.what());
    }
  }

  void AdvertiseTopics() {
#if defined(ODIN_ROS2)
    if (!node_) {
      return;
    }
    auto qos_profile = rclcpp::QoS(1)
                           .reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE)
                           .durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
    raw_cloud_pub_ =
        node_->create_publisher<PointCloud2Msg>(topic_prefix_ + "cloud/raw", qos_profile);
    slam_cloud_pub_ =
        node_->create_publisher<PointCloud2Msg>(topic_prefix_ + "cloud/slam", qos_profile);
    image_pub_ = node_->create_publisher<CompressedImageMsg>(topic_prefix_ + "camera0/compressed",
                                                             qos_profile);
    gray_image_pub_ = node_->create_publisher<ImageMsg>(topic_prefix_ + "gray", qos_profile);
    image2_pub_ = node_->create_publisher<CompressedImageMsg>(topic_prefix_ + "camera1/compressed",
                                                              qos_profile);
    imu_pub_ = node_->create_publisher<ImuMsg>(topic_prefix_ + "imu", qos_profile);
    odom_pub_ = node_->create_publisher<OdometryMsg>(topic_prefix_ + "odometry", qos_profile);
    odom_hf_pub_ = node_->create_publisher<OdometryMsg>(topic_prefix_ + "odometry_hf", qos_profile);

    // Setup calibration service
    calib_service_ = node_->create_service<odin_ros_driver_rev1::srv::GetCalibration>(
        topic_prefix_ + "get_calibration",
        [this](const std::shared_ptr<odin_ros_driver_rev1::srv::GetCalibration::Request> request,
               std::shared_ptr<odin_ros_driver_rev1::srv::GetCalibration::Response> response) {
          (void)request;
          if (calib_ready_ && !calib_yaml_content_.empty()) {
            response->success = true;
            response->yaml_content = calib_yaml_content_;
            response->message = "Calibration data retrieved successfully";
            LogInfo("Calibration service: sent %zu bytes", calib_yaml_content_.size());
          } else {
            response->success = false;
            response->yaml_content = "";
            response->message = "Calibration data not ready yet";
            LogWarn("Calibration service: data not ready");
          }
        });
    LogInfo("Calibration service ready: %sget_calibration", topic_prefix_.c_str());

    // Setup sensor mode service
    sensor_mode_service_ = node_->create_service<odin_ros_driver_rev1::srv::SetSensorMode>(
        topic_prefix_ + "set_sensor_mode",
        [this](const std::shared_ptr<odin_ros_driver_rev1::srv::SetSensorMode::Request> request,
               std::shared_ptr<odin_ros_driver_rev1::srv::SetSensorMode::Response> response) {
          if (request->mode > 3) {
            response->success = false;
            response->error_code = static_cast<int32_t>(sdk::OdinResult::kInvalidArgument);
            response->message = "Invalid mode (0=HDR 1=High Peak or 2=Low Peak 3=Adaptive)";
            LogWarn("SetSensorMode service: invalid mode %d", request->mode);
            return;
          }
          const char *mode_name;
          switch (request->mode) {
            case 0:
              mode_name = "HDR";
              break;
            case 1:
              mode_name = "High Peak";
              break;
            case 2:
              mode_name = "Low Peak";
              break;
            default:
              mode_name = "Adaptive";
              break;
          }
          if (!connected_ || device_handle_ == sdk::kInvalidDeviceHandle) {
            response->success = false;
            response->error_code = static_cast<int32_t>(sdk::OdinResult::kNotInitialized);
            response->message = "Device not connected";
            LogWarn("SetSensorMode service: device not connected");
            return;
          }
          sdk::OdinResult result = sdk::SetSensorMode(
              device_handle_, static_cast<uint8_t>(request->mode));
          response->error_code = static_cast<int32_t>(result);
          if (result == sdk::OdinResult::kOk) {
            sensor_mode_ = request->mode;
            response->success = true;
            response->message = mode_name;
            LogInfo("SetSensorMode service: set to %s (%d)", mode_name, sensor_mode_);
          } else {
            response->success = false;
            response->message = "SDK error: " + std::to_string(static_cast<int>(result));
            LogWarn("SetSensorMode service: SDK returned error %d", static_cast<int>(result));
          }
        });
    LogInfo("SensorMode service ready: %sset_sensor_mode", topic_prefix_.c_str());
#else
    raw_cloud_pub_ = nh_.advertise<PointCloud2Msg>(topic_prefix_ + "cloud/raw", 100);
    slam_cloud_pub_ = nh_.advertise<PointCloud2Msg>(topic_prefix_ + "cloud/slam", 100);
    image_pub_ = nh_.advertise<CompressedImageMsg>(topic_prefix_ + "camera0/compressed", 100);
    gray_image_pub_ = nh_.advertise<ImageMsg>(topic_prefix_ + "gray", 100);
    image2_pub_ = nh_.advertise<CompressedImageMsg>(topic_prefix_ + "camera1/compressed", 100);
    imu_pub_ = nh_.advertise<ImuMsg>(topic_prefix_ + "imu", 1000);
    odom_pub_ = nh_.advertise<OdometryMsg>(topic_prefix_ + "odometry", 1000);
    odom_hf_pub_ = nh_.advertise<OdometryMsg>(topic_prefix_ + "odometry_hf", 1000);

    // Setup calibration service
    calib_service_ = nh_.advertiseService(topic_prefix_ + "get_calibration",
                                          &odinRosDriver::HandleCalibrationService, this);
    LogInfo("Calibration service ready: %sget_calibration", topic_prefix_.c_str());

    // Setup sensor mode service
    sensor_mode_service_ = nh_.advertiseService(topic_prefix_ + "set_sensor_mode",
                                                &odinRosDriver::HandleSensorModeService, this);
    LogInfo("SensorMode service ready: %sset_sensor_mode", topic_prefix_.c_str());
#endif
  }

  template <typename MsgT>
  void PublishMessage(PublisherType<MsgT> &pub, const MsgT &msg) {
#if defined(ODIN_ROS2)
    if (pub) {
      pub->publish(msg);
    }
#else
    pub.publish(msg);
#endif
  }

  bool LoadParameters() {
    // Ports are now auto-generated by SDK, no need to configure them
    discovery_timeout_ms_ = GetParam<int>("discovery_timeout_ms", 2000);

    raw_frame_id_ = GetParam<std::string>("raw_frame_id", "odom");
    slam_frame_id_ = GetParam<std::string>("slam_frame_id", "odom");
    image_frame_id_ = GetParam<std::string>("image_frame_id", "odom");
    gray_frame_id_ = GetParam<std::string>("gray_image_frame_id", "odom");
    image2_frame_id_ = GetParam<std::string>("image2_frame_id", "odom");
    imu_frame_id_ = GetParam<std::string>("imu_frame_id", "odom");
    odom_frame_id_ = GetParam<std::string>("odom_frame_id", "odom");
    odom_child_frame_id_ = GetParam<std::string>("odom_child_frame_id", "base_link");

    odom_position_scale_ = GetParam<double>("odom_position_scale", 1e-6);
    odom_orientation_scale_ = GetParam<double>("odom_orientation_scale", 1e-6);
    odom_linear_velocity_scale_ = GetParam<double>("odom_linear_velocity_scale", 1e-6);
    odom_angular_velocity_scale_ = GetParam<double>("odom_angular_velocity_scale", 1e-6);

    std::string mode = GetParam<std::string>("operating_mode", "normal");
    desired_mode_ = sdk::OdinOperatingMode::kNormal;
    if (mode == "standby") {
      desired_mode_ = sdk::OdinOperatingMode::kStandby;
    } else if (mode != "normal") {
      LogWarn("Unknown operating_mode '%s', defaulting to 'normal'.", mode.c_str());
    }

    m_confidence_threshold = 7;

    // Image resolution parameters (0 = use first capability from device)
    image_width_ = GetParam<int>("image_width", 0);
    image_height_ = GetParam<int>("image_height", 0);
    image_fps_ = GetParam<int>("image_fps", 0);
    image_format_ = GetParam<std::string>("image_format", "mjpeg");

    // Channel enable parameters - load from control_command.yaml
    LoadChannelEnableConfig();

    return true;
  }

  void SetupDynamicReconfigure() {
#if defined(ODIN_ROS2)
    // ROS2: confidence_threshold slider
    rcl_interfaces::msg::ParameterDescriptor slider_desc;
    slider_desc.description = "Confidence threshold (0-255)";
    slider_desc.integer_range.resize(1);
    slider_desc.integer_range[0].from_value = 0;
    slider_desc.integer_range[0].to_value = 0xff;
    slider_desc.integer_range[0].step = 1;
    node_->declare_parameter<int>("confidence_threshold", m_confidence_threshold, slider_desc);

    rcl_interfaces::msg::ParameterDescriptor mode_desc;
    mode_desc.description = "Sensor mode: 0=HDR 1=High Peak, 2=Low Peak 3=Adaptive";
    mode_desc.integer_range.resize(1);
    mode_desc.integer_range[0].from_value = 0;
    mode_desc.integer_range[0].to_value = 3;
    mode_desc.integer_range[0].step = 1;
    node_->declare_parameter<int>("sensor_mode", sensor_mode_, mode_desc);

    // ROS2: log_level slider (0=ERROR, 1=INFO, 2=WARN, 3=DEBUG)
    rcl_interfaces::msg::ParameterDescriptor log_desc;
    log_desc.description = "Log level: 0=ERROR, 1=WARN,2=INFO, 3=DEBUG";
    log_desc.integer_range.resize(1);
    log_desc.integer_range[0].from_value = 0;
    log_desc.integer_range[0].to_value = 3;
    log_desc.integer_range[0].step = 1;
    node_->declare_parameter<int>("log_level", log_level_, log_desc);

    // Parameter callback
    param_callback_handle_ =
        node_->add_on_set_parameters_callback([this](const std::vector<rclcpp::Parameter> &params) {
          rcl_interfaces::msg::SetParametersResult result;
          result.successful = true;
          for (const auto &param : params) {
            if (param.get_name() == "confidence_threshold") {
              m_confidence_threshold = param.as_int();
              LogInfo("confidence_threshold changed to: %d", m_confidence_threshold);
            } else if (param.get_name() == "sensor_mode") {
              sensor_mode_ = param.as_int();
              const char *mode_name;
              switch (sensor_mode_) {
                case 0:
                  mode_name = "HDR";
                  break;
                case 1:
                  mode_name = "High Peak";
                  break;
                case 2:
                  mode_name = "Low Peak";
                  break;
                default:
                  mode_name = "Adaptive";
                  break;
              }
              LogInfo("sensor_mode changed to: %s (%d)", mode_name, sensor_mode_);
              if (connected_ && device_handle_ != sdk::kInvalidDeviceHandle) {
                sdk::SetSensorMode(device_handle_, static_cast<uint8_t>(sensor_mode_));
              }
            } else if (param.get_name() == "log_level") {
              log_level_ = param.as_int();
              if (log_level_ < 0) log_level_ = 0;
              if (log_level_ > 3) log_level_ = 3;
              log_config(static_cast<unsigned int>(log_level_), printf);
              const char *level_names[] = {"ERROR", "INFO", "WARN", "DEBUG"};
              LogInfo("log_level changed to: %s (%d)", level_names[log_level_], log_level_);
            }
          }
          return result;
        });
    LogInfo("Dynamic reconfigure initialized");
#else
    // ROS1: Setup ddynamic_reconfigure
    LogInfo("Setting up ddynamic_reconfigure with namespace: %s", pnh_.getNamespace().c_str());
    ddr_ = std::make_shared<ddynamic_reconfigure::DDynamicReconfigure>(pnh_);

    // confidence_threshold slider
    ddr_->registerVariable<int>("confidence_threshold", m_confidence_threshold,
                                boost::bind(&odinRosDriver::onConfidenceThresholdChanged, this, _1),
                                "Confidence threshold", 0, 255);

    // sensor_mode slider (0=HDR 1=High Peak, 2=Low Peak, 3=Adaptive)
    ddr_->registerVariable<int>("sensor_mode", sensor_mode_,
                                boost::bind(&odinRosDriver::onSensorModeChanged, this, _1),
                                "Sensor mode: 0=HDR 1=High Peak, 2=Low Peak 3=Adaptive", 0, 3);

    // log_level slider (0=ERROR, 1=INFO, 2=WARN, 3=DEBUG)
    ddr_->registerVariable<int>("log_level", log_level_,
                                boost::bind(&odinRosDriver::onLogLevelChanged, this, _1),
                                "Log level: 0=ERROR, 1=INFO, 2=WARN, 3=DEBUG", 0, 3);

    ddr_->publishServicesTopics();
    LogInfo("Dynamic reconfigure initialized");
#endif
  }

#if !defined(ODIN_ROS2)
  void onConfidenceThresholdChanged(int new_val) {
    m_confidence_threshold = new_val;
    LogInfo("confidence_threshold changed to: %d", m_confidence_threshold);
  }

  void onSensorModeChanged(int new_mode) {
    sensor_mode_ = new_mode;
    const char *mode_name;
    switch (sensor_mode_) {
      case 0:
        mode_name = "HDR";
        break;
      case 1:
        mode_name = "High Peak";
        break;
      case 2:
        mode_name = "Low Peak";
        break;
      default:
        mode_name = "Adaptive";
        break;
    }
    LogInfo("sensor_mode changed to: %s (%d)", mode_name, sensor_mode_);
    if (connected_ && device_handle_ != sdk::kInvalidDeviceHandle) {
      sdk::SetSensorMode(device_handle_, static_cast<uint8_t>(sensor_mode_));
    }
  }

  void onLogLevelChanged(int new_level) {
    log_level_ = new_level;
    if (log_level_ < 0) log_level_ = 0;
    if (log_level_ > 3) log_level_ = 3;
    log_config(static_cast<unsigned int>(log_level_), printf);
    const char *level_names[] = {"ERROR", "INFO", "WARN", "DEBUG"};
    LogInfo("log_level changed to: %s (%d)", level_names[log_level_], log_level_);
  }
#endif

  void RegisterCallbacks() {
    if (callbacks_registered_) {
      return;
    }
    sdk::RegisterPointCloudCallback(device_handle_, &odinRosDriver::PointCloudCallback, this);
    sdk::RegisterSlamCallback(device_handle_, &odinRosDriver::SlamCallback, this);
    sdk::RegisterImageCallback(device_handle_, &odinRosDriver::ImageCallback, this);
    sdk::RegisterImageCallback2(device_handle_, &odinRosDriver::ImageCallback2, this);
    sdk::RegisterImuCallback(device_handle_, &odinRosDriver::ImuCallback, this);
    sdk::RegisterOdomCallback(device_handle_, &odinRosDriver::OdomCallback, this);
    callbacks_registered_ = true;
  }

  // Device discovery moved to main() for multi-device support

  bool ConnectDevice() {
    device_handle_ = sdk::ConnectDevice(discovered_device_);
    if (device_handle_ == sdk::kInvalidDeviceHandle) {
      LogError("Failed to connect to device %s.", discovered_device_.ip.c_str());
      return false;
    }
    connected_ = true;
    LogInfo("Connected to %s handle=%u (ports auto-generated)", discovered_device_.ip.c_str(),
            static_cast<unsigned>(device_handle_));

    // Step 1: Set standby mode first to prepare for file transfer
    LogInfo("Setting device to standby mode for calibration fetch...");
    if (sdk::SetOperatingMode(device_handle_, sdk::OdinOperatingMode::kStandby) != sdk::OdinResult::kOk) {
      LogWarn("Failed to set standby mode, continuing anyway.");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Step 2: Fetch calibration file from device
    FetchCalibrationFile();

    // Step 3: Set desired operating mode
    if (sdk::SetOperatingMode(device_handle_, desired_mode_) != sdk::OdinResult::kOk) {
      LogWarn("Failed to send mode configuration command.");
    }

    std::string firmware_version;
    if (sdk::GetFirmwareVersion(device_handle_, firmware_version) == sdk::OdinResult::kOk) {
      printf("\n");
      printf("========================================\n");
      printf("  FIRMWARE VERSION: %s\n", firmware_version.c_str());
      printf("========================================\n");
      printf("\n");
    }

    // Step 4: Query and print all sensor capabilities
    QueryAndPrintSensorCapabilities();

    // Step 5: Configure all data streams (query capability per channel)
    ConfigureStreams(sdk::OdinTransportMode::kUdp);

    return true;
  }

  void ConfigureStreams(sdk::OdinTransportMode transport) {
    if (!connected_ || device_handle_ == sdk::kInvalidDeviceHandle) {
      return;
    }
    const char* transport_name = (transport == sdk::OdinTransportMode::kTcp) ? "TCP" : "UDP";
    LogInfo("Configuring data streams (%s)...", transport_name);

    // Channel name helper
    auto channelName = [](sdk::OdinDataChannel ch) -> const char* {
      switch (ch) {
        case sdk::OdinDataChannel::kRawPoint:  return "RawPoint";
        case sdk::OdinDataChannel::kSlamPoint: return "SlamPoint";
        case sdk::OdinDataChannel::kImage0:    return "Image0";
        case sdk::OdinDataChannel::kImage1:    return "Image1";
        case sdk::OdinDataChannel::kImu:       return "IMU";
        case sdk::OdinDataChannel::kOdom:      return "Odom";
        default: return "Unknown";
      }
    };

    // Build channels list based on enable parameters
    std::vector<sdk::OdinDataChannel> channels_to_start;
    if (enable_raw_point_)  channels_to_start.push_back(sdk::OdinDataChannel::kRawPoint);
    if (enable_slam_point_) channels_to_start.push_back(sdk::OdinDataChannel::kSlamPoint);
    if (enable_image0_)     channels_to_start.push_back(sdk::OdinDataChannel::kImage0);
    if (enable_image1_)     channels_to_start.push_back(sdk::OdinDataChannel::kImage1);
    if (enable_imu_)        channels_to_start.push_back(sdk::OdinDataChannel::kImu);
    if (enable_odom_)       channels_to_start.push_back(sdk::OdinDataChannel::kOdom);

    LogInfo("Channel config: raw=%d slam=%d img0=%d img1=%d imu=%d odom=%d",
            enable_raw_point_, enable_slam_point_, enable_image0_, enable_image1_,
            enable_imu_, enable_odom_);

    for (auto channel : channels_to_start) {
      // Query capability for this specific channel
      std::vector<sdk::OdinSensorCapability> caps;
      auto query_result = sdk::GetSensorCapability(device_handle_, caps, {channel}, 1000);
      
      // Skip if no capability returned for this channel
      if (query_result != sdk::OdinResult::kOk || caps.empty()) {
        LogWarn("%s: no capability found, skipping", channelName(channel));
        continue;
      }

      // Get config for this channel
      const sdk::OdinStreamCfg* cfg_ptr = nullptr;
      sdk::OdinStreamCfg cfg;
      
      if (!caps[0].modes.empty()) {
        // For Image channels, use launch parameters if specified, otherwise use first capability
        bool is_image = (channel == sdk::OdinDataChannel::kImage0 ||
                         channel == sdk::OdinDataChannel::kImage1);
        
        if (is_image && image_width_ > 0 && image_height_ > 0) {
          // Use launch parameters
          cfg.width = static_cast<uint16_t>(image_width_);
          cfg.height = static_cast<uint16_t>(image_height_);
          cfg.fps = static_cast<uint8_t>(image_fps_ > 0 ? image_fps_ : caps[0].modes[0].fps);
          // Parse image format from string
          if (image_format_ == "yuyv") {
            cfg.format = sdk::OdinDataFormat::kImageYuyv;
          } else if (image_format_ == "nv12") {
            cfg.format = sdk::OdinDataFormat::kImageNv12;
          } else if (image_format_ == "nv21") {
            cfg.format = sdk::OdinDataFormat::kImageNv21;
          } else if (image_format_ == "rgb24") {
            cfg.format = sdk::OdinDataFormat::kImageRgb24;
          } else {
            cfg.format = sdk::OdinDataFormat::kImageMjpeg;  // Default to MJPEG
          }
          cfg_ptr = &cfg;
          LogInfo("%s using launch params: %ux%u@%uFPS (%s)", 
                  channelName(channel), cfg.width, cfg.height, cfg.fps, image_format_.c_str());
        } else {
          // Use first capability
          const auto& cfg_default = caps[0].modes[0];
          cfg.width = cfg_default.width;
          cfg.height = cfg_default.height;
          cfg.fps = static_cast<uint8_t>(cfg_default.fps);
          cfg.format = cfg_default.format;
          LogInfo("%s capability: %ux%u@%uFPS (format=%d)", 
                  channelName(channel), cfg.width, cfg.height, cfg.fps, static_cast<int>(cfg.format));
          if (is_image) {
            cfg_ptr = &cfg;
          }
        }
      }

      auto result = sdk::StartStream(device_handle_, channel, transport, cfg_ptr);
      if (result == sdk::OdinResult::kOk) {
        if (cfg_ptr) {
          LogInfo("%s started: %ux%u@%uFPS", channelName(channel), cfg.width, cfg.height, cfg.fps);
        } else {
          LogInfo("%s started", channelName(channel));
        }
      } else {
        LogWarn("Failed to start %s stream", channelName(channel));
      }
    }

    LogInfo("Stream configuration complete");
  }

  std::vector<sdk::OdinSensorCapability> QueryAndPrintSensorCapabilities() {
    std::vector<sdk::OdinSensorCapability> capabilities;
    if (!connected_ || device_handle_ == sdk::kInvalidDeviceHandle) {
      return capabilities;
    }
    auto result = sdk::GetSensorCapability(device_handle_, capabilities, {}, 2000);
    if (result != sdk::OdinResult::kOk) {
      LogWarn("Failed to query sensor capabilities, result=%d", static_cast<int>(result));
      return capabilities;
    }

    // Helper lambda to convert channel to string
    auto channelToString = [](sdk::OdinDataChannel ch) -> const char* {
      switch (ch) {
        case sdk::OdinDataChannel::kRawPoint:  return "RawPoint";
        case sdk::OdinDataChannel::kSlamPoint: return "SlamPoint";
        case sdk::OdinDataChannel::kImage0:    return "Image0 (Left)";
        case sdk::OdinDataChannel::kImage1:    return "Image1 (Right)";
        case sdk::OdinDataChannel::kImu:       return "IMU";
        case sdk::OdinDataChannel::kOdom:      return "Odom";
        default: return "Unknown";
      }
    };

    // Helper lambda to convert format to string
    auto formatToString = [](sdk::OdinDataFormat fmt) -> const char* {
      switch (fmt) {
        case sdk::OdinDataFormat::kImageMjpeg:      return "MJPEG";
        case sdk::OdinDataFormat::kImageYuyv:       return "YUYV";
        case sdk::OdinDataFormat::kImageNv12:       return "NV12";
        case sdk::OdinDataFormat::kImageNv21:       return "NV21";
        case sdk::OdinDataFormat::kImageRgb24:      return "RGB24";
        case sdk::OdinDataFormat::kRawPointXyzic:   return "XYZIC";
        case sdk::OdinDataFormat::kRawPointXyz:     return "XYZ";
        case sdk::OdinDataFormat::kSlamPointXyzrgba: return "XYZRGBA";
        case sdk::OdinDataFormat::kSlamPointXyz:    return "XYZ";
        case sdk::OdinDataFormat::kImu6Axis:        return "6-Axis";
        case sdk::OdinDataFormat::kImu9Axis:        return "9-Axis";
        case sdk::OdinDataFormat::kOdomStandard:    return "Standard";
        default: return "Unknown";
      }
    };

    // Print formatted capabilities
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║                    SENSOR CAPABILITIES                           ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");

    for (const auto& cap : capabilities) {
      printf("║  Channel: %-20s                                 ║\n", channelToString(cap.channel));
      printf("║  ┌────────────────────────────────────────────────────────────┐ ║\n");
      printf("║  │  Mode   Width   Height    FPS    Format                   │ ║\n");
      printf("║  ├────────────────────────────────────────────────────────────┤ ║\n");

      int mode_idx = 0;
      for (const auto& cfg : cap.modes) {
        char width_str[16], height_str[16], fps_str[16];
        if (cfg.width > 0) {
          snprintf(width_str, sizeof(width_str), "%5u", cfg.width);
        } else {
          snprintf(width_str, sizeof(width_str), "  -  ");
        }
        if (cfg.height > 0) {
          snprintf(height_str, sizeof(height_str), "%5u", cfg.height);
        } else {
          snprintf(height_str, sizeof(height_str), "  -  ");
        }
        if (cfg.fps > 0) {
          snprintf(fps_str, sizeof(fps_str), "%3u", cfg.fps);
        } else {
          snprintf(fps_str, sizeof(fps_str), " - ");
        }

        printf("║  │  [%2d]  %s   %s    %s    %-20s │ ║\n",
               mode_idx++, width_str, height_str, fps_str, formatToString(cfg.format));
      }

      if (cap.modes.empty()) {
        printf("║  │  (No configurations available)                            │ ║\n");
      }

      printf("║  └────────────────────────────────────────────────────────────┘ ║\n");
      printf("║                                                                  ║\n");
    }

    if (capabilities.empty()) {
      printf("║  (No sensor capabilities reported by device)                    ║\n");
    }

    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    LogInfo("Sensor capabilities query complete: %zu channels", capabilities.size());
    return capabilities;
  }

  void RequestStandbyMode() {
    if (!connected_ || device_handle_ == sdk::kInvalidDeviceHandle) {
      return;
    }
    if (sdk::SetOperatingMode(device_handle_, sdk::OdinOperatingMode::kStandby) != sdk::OdinResult::kOk) {
      LogWarn("Failed to send standby mode command during shutdown.");
    } else {
      LogInfo("Requested standby mode before disconnect.");
    }
  }

  static void CalibProgressCallback(sdk::OdinDeviceHandle device, float progress, void *client) {
    (void)device;
    auto *driver = static_cast<odinRosDriver *>(client);
    if (driver && progress >= 0) {
      driver->LogInfo("Calibration download progress: %.1f%%", progress);
    }
  }

  void BuildCalibFilePath() {
    std::string pkg_root = GetPackageRoot();
    if (pkg_root.empty()) {
      LogWarn("Could not locate package root, calibration file path not set.");
      return;
    }
    if (device_sn_.empty()) {
      LogWarn("Device SN unknown, using default calibration filename.");
      calib_file_path_ = pkg_root + "/config/camera_calib.yaml";
    } else {
      calib_file_path_ = pkg_root + "/config/camera_calib_" + device_sn_ + ".yaml";
    }
    LogInfo("Calibration file path: %s", calib_file_path_.c_str());
  }

  void BuildTopicPrefix() {
    // 使用设备型号和索引生成 topic_prefix
    // 格式: /manifold/{model}/device{index}/
    // 设备 SN 通过 /odin_driver/device_info topic 发布
    std::string model = device_model_.empty() ? "odin" : device_model_;

    if (device_index_ < 0) {
      // 单设备或未指定索引时使用默认名称
      topic_prefix_ = "/manifold/" + model + "/device/";
    } else {
      topic_prefix_ = "/manifold/" + model + "/device" + std::to_string(device_index_) + "/";
    }
    LogInfo("Topic prefix: %s (SN: %s, Model: %s, IP: %s)", topic_prefix_.c_str(),
            device_sn_.c_str(), device_model_.c_str(), discovered_device_.ip.c_str());
  }

  void PublishTopicPrefix() {
#if defined(ODIN_ROS2)
    // Publish single topic_prefix for backward compatibility
    if (!topic_prefix_pub_) {
      topic_prefix_pub_ = node_->create_publisher<std_msgs::msg::String>(
          "odin_driver/topic_prefix", rclcpp::QoS(1).transient_local());
    }
    std_msgs::msg::String msg;
    msg.data = topic_prefix_;
    topic_prefix_pub_->publish(msg);

    // Publish device info as JSON for multi-device support
    if (!device_info_pub_) {
      device_info_pub_ = node_->create_publisher<std_msgs::msg::String>(
          "odin_driver/device_info", rclcpp::QoS(1).transient_local());
    }
    std_msgs::msg::String info_msg;
    info_msg.data = BuildDeviceInfoJson();
    device_info_pub_->publish(info_msg);
    LogInfo("Published device_info: %s", info_msg.data.c_str());
#else
    // Publish single topic_prefix for backward compatibility
    if (!topic_prefix_pub_) {
      topic_prefix_pub_ = nh_.advertise<std_msgs::String>("odin_driver/topic_prefix", 1, true);
    }
    std_msgs::String msg;
    msg.data = topic_prefix_;
    topic_prefix_pub_.publish(msg);

    // Publish device info as JSON for multi-device support
    // Use queue size 10 for multi-device support (latched messages from multiple publishers)
    if (!device_info_pub_) {
      device_info_pub_ = nh_.advertise<std_msgs::String>("odin_driver/device_info", 100, true);
    }
    std_msgs::String info_msg;
    info_msg.data = BuildDeviceInfoJson();
    device_info_pub_.publish(info_msg);
    LogInfo("Published device_info: %s", info_msg.data.c_str());
#endif
  }

  std::string BuildDeviceInfoJson() {
    // Format: {"sn":"xxx","prefix":"manifold/odin/xxx/","ip":"192.168.1.251"}
    std::ostringstream oss;
    oss << "{\"sn\":\"" << device_sn_ << "\","
        << "\"prefix\":\"" << topic_prefix_ << "\","
        << "\"ip\":\"" << discovered_device_.ip << "\"}";
    return oss.str();
  }

  void FetchCalibrationFile() {
    if (calib_file_path_.empty()) {
      LogWarn("Calibration file path not configured, skipping fetch.");
      return;
    }
    LogInfo("Fetching calibration file from device...");
    LogInfo("  Save path: %s", calib_file_path_.c_str());

    sdk::OdinResult result =
        sdk::GetDeviceFile(device_handle_, &odinRosDriver::CalibProgressCallback,
                           sdk::OdinFileType::kCalibrationFile, calib_file_path_);

    if (result != sdk::OdinResult::kOk) {
      LogError(
          "Failed to fetch calibration file from device (error %d). Calibration service will not "
          "be available.",
          static_cast<int>(result));
      calib_ready_ = false;
      return;
    }

    LogInfo("Calibration file fetched successfully: %s", calib_file_path_.c_str());
    // Read file content for service
    std::ifstream file(calib_file_path_);
    if (file.is_open()) {
      std::stringstream buffer;
      buffer << file.rdbuf();
      calib_yaml_content_ = buffer.str();
      calib_ready_ = true;
      LogInfo("Calibration content loaded (%zu bytes)", calib_yaml_content_.size());
    } else {
      LogError("Failed to read calibration file after download: %s", calib_file_path_.c_str());
      calib_ready_ = false;
    }
  }

#if !defined(ODIN_ROS2)
  bool HandleCalibrationService(odin_ros_driver_rev1::GetCalibration::Request &req,
                                odin_ros_driver_rev1::GetCalibration::Response &res) {
    (void)req;
    if (calib_ready_ && !calib_yaml_content_.empty()) {
      res.success = true;
      res.yaml_content = calib_yaml_content_;
      res.message = "Calibration data retrieved successfully";
      LogInfo("Calibration service: sent %zu bytes", calib_yaml_content_.size());
    } else {
      res.success = false;
      res.yaml_content = "";
      res.message = "Calibration data not ready yet";
      LogWarn("Calibration service: data not ready");
    }
    return true;
  }

  bool HandleSensorModeService(odin_ros_driver_rev1::SetSensorMode::Request &req,
                               odin_ros_driver_rev1::SetSensorMode::Response &res) {
    if (req.mode > 3) {
      res.success = false;
      res.error_code = static_cast<int32_t>(sdk::OdinResult::kInvalidArgument);
      res.message = "Invalid mode (must be 0=HDR 1=High Peak or 2=Low Peak 3=Adaptive)";
      LogWarn("SetSensorMode service: invalid mode %d", req.mode);
      return true;
    }
    const char *mode_name;
    switch (req.mode) {
      case 0:
        mode_name = "HDR";
        break;
      case 1:
        mode_name = "High Peak";
        break;
      case 2:
        mode_name = "Low Peak";
        break;
      default:
        mode_name = "Adaptive";
        break;
    }
    if (!connected_ || device_handle_ == sdk::kInvalidDeviceHandle) {
      res.success = false;
      res.error_code = static_cast<int32_t>(sdk::OdinResult::kNotInitialized);
      res.message = "Device not connected";
      LogWarn("SetSensorMode service: device not connected");
      return true;
    }
    sdk::OdinResult result =
        sdk::SetSensorMode(device_handle_, static_cast<uint8_t>(req.mode));
    res.error_code = static_cast<int32_t>(result);
    if (result == sdk::OdinResult::kOk) {
      sensor_mode_ = req.mode;
      res.success = true;
      res.message = mode_name;
      LogInfo("SetSensorMode service: set to %s (%d)", mode_name, sensor_mode_);
    } else {
      res.success = false;
      res.message = "SDK error: " + std::to_string(static_cast<int>(result));
      LogWarn("SetSensorMode service: SDK returned error %d", static_cast<int>(result));
    }
    return true;
  }
#endif

  void Shutdown() {
    // Stop async queues first to prevent callbacks during disconnect
    if (raw_point_queue_) raw_point_queue_->Stop();
    if (slam_point_queue_) slam_point_queue_->Stop();
    if (image0_queue_) image0_queue_->Stop();
    if (image1_queue_) image1_queue_->Stop();
    if (imu_queue_) imu_queue_->Stop();
    if (odom_queue_) odom_queue_->Stop();

    if (connected_ && device_handle_ != sdk::kInvalidDeviceHandle) {
      RequestStandbyMode();
      sdk::DisconnectDevice(device_handle_);
      connected_ = false;
      device_handle_ = sdk::kInvalidDeviceHandle;
    }
  }

  RosTime ToRosTime(uint64_t timestamp) const {
#if defined(ODIN_ROS2)
    if (!node_) {
      return RosTime(0);
    }
    if (timestamp == 0) {
      return node_->now();
    }
    return RosTime(static_cast<int64_t>(timestamp) * 1000LL);
#else
    if (timestamp == 0) {
      return ros::Time::now();
    }
    constexpr uint64_t kMicrosPerSec = 1000000ULL;
    ros::Time t;
    t.sec = static_cast<uint32_t>(timestamp / kMicrosPerSec);
    t.nsec = static_cast<uint32_t>((timestamp % kMicrosPerSec) * 1000ULL);
    return t;
#endif
  }

  // Process functions - called by async queue worker threads
  void ProcessRawPointCloud(const sdk::OdinPointCloudPacket &packet) {
    const size_t available_points = packet.payload.size() / sizeof(sdk::OdinRawPoint<float>);
    if (available_points == 0) {
      if (ShouldThrottle("raw_points_empty", 5.0)) {
        LogDebug("Received raw point cloud packet with empty payload.");
      }
      return;
    }
    const size_t point_count = std::min(static_cast<size_t>(packet.point_count), available_points);
    if (point_count < packet.point_count && ShouldThrottle("raw_points_truncated", 1.0)) {
      LogWarn("Raw point packet truncated: expected %u points got %zu.", packet.point_count,
              point_count);
    }

    // Rolling shutter scan parameters:
    // - 256: points per row (h_vld_seg * 16 = 16 * 16)
    // - 6: rows per group (v_pxl_outNum), captured simultaneously
    // - 32 groups total (v_roll_num), subframe_odr = 328.5 Hz
    constexpr int kPointsPerRow = 256;
    constexpr int kRowsPerGroup = 6;
    constexpr int kPointsPerGroup = kPointsPerRow * kRowsPerGroup;  // 1536 points
    constexpr float kSubframeInterval = 1.0f / 328.5f;              // ~3.044 ms between groups

    PointCloud2Msg msg;
    msg.header.stamp = ToRosTime(packet.timestamp);
    msg.header.frame_id = raw_frame_id_;
    msg.height = 1;
    msg.width = static_cast<uint32_t>(point_count);
    msg.is_bigendian = false;
    msg.is_dense = true;
    msg.fields.resize(6);

    constexpr uint32_t kOffsetX = 0;
    constexpr uint32_t kOffsetY = sizeof(float);
    constexpr uint32_t kOffsetZ = sizeof(float) * 2;
    constexpr uint32_t kOffsetIntensity = sizeof(float) * 3;
    constexpr uint32_t kOffsetConfidence = sizeof(float) * 3 + 1;
    constexpr uint32_t kOffsetTime = sizeof(float) * 3 + 2;
    constexpr uint32_t kPointStep = sizeof(float) * 4 + 2;

    msg.fields[0].name = "x";
    msg.fields[0].offset = kOffsetX;
    msg.fields[0].datatype = PointFieldMsg::FLOAT32;
    msg.fields[0].count = 1;
    msg.fields[1].name = "y";
    msg.fields[1].offset = kOffsetY;
    msg.fields[1].datatype = PointFieldMsg::FLOAT32;
    msg.fields[1].count = 1;
    msg.fields[2].name = "z";
    msg.fields[2].offset = kOffsetZ;
    msg.fields[2].datatype = PointFieldMsg::FLOAT32;
    msg.fields[2].count = 1;
    msg.fields[3].name = "intensity";
    msg.fields[3].offset = kOffsetIntensity;
    msg.fields[3].datatype = PointFieldMsg::UINT8;
    msg.fields[3].count = 1;
    msg.fields[4].name = "confidence";
    msg.fields[4].offset = kOffsetConfidence;
    msg.fields[4].datatype = PointFieldMsg::UINT8;
    msg.fields[4].count = 1;
    msg.fields[5].name = "offset_time";
    msg.fields[5].offset = kOffsetTime;
    msg.fields[5].datatype = PointFieldMsg::FLOAT32;
    msg.fields[5].count = 1;

    msg.point_step = kPointStep;
    msg.row_step = msg.width * msg.point_step;
    msg.data.resize(msg.row_step);

    const auto *points = reinterpret_cast<const sdk::OdinRawPoint<float> *>(packet.payload.data());
    uint8_t *dst = msg.data.data();
    for (size_t i = 0; i < point_count; ++i) {
      const int group_idx = static_cast<int>(i) / kPointsPerGroup;
      const float offset_time = group_idx * kSubframeInterval;

      float x = points[i].x * 0.001f;
      float y = points[i].y * 0.001f;
      float z = points[i].z * 0.001f;
      if (static_cast<float>(m_confidence_threshold) > points[i].confidence) {
        x = y = z = 0.0f;
      }

      std::memcpy(dst + kOffsetX, &x, sizeof(float));
      std::memcpy(dst + kOffsetY, &y, sizeof(float));
      std::memcpy(dst + kOffsetZ, &z, sizeof(float));
      std::memcpy(dst + kOffsetIntensity, &points[i].intensity, sizeof(uint8_t));
      std::memcpy(dst + kOffsetConfidence, &points[i].confidence, sizeof(uint8_t));
      std::memcpy(dst + kOffsetTime, &offset_time, sizeof(float));

      dst += kPointStep;
    }

    PublishMessage(raw_cloud_pub_, msg);
    HandleGrayImage(packet);
  }

  void ProcessSlamPointCloud(const sdk::OdinSlamPacket &packet) {
    // SLAM points are now stored as float xyz after transformation in SDK
    const size_t available_points = packet.payload.size() / sizeof(sdk::OdinSlamPoint<float>);
    if (available_points == 0) {
      if (ShouldThrottle("slam_points_empty", 5.0)) {
        LogDebug("Received slam point cloud packet with empty payload.");
      }
      return;
    }
    const size_t point_count = std::min(static_cast<size_t>(packet.point_count), available_points);
    if (point_count < packet.point_count && ShouldThrottle("slam_points_truncated", 1.0)) {
      LogWarn("SLAM point packet truncated: expected %u got %zu.", packet.point_count, point_count);
    }

    PointCloud2Msg msg;
    msg.header.stamp = ToRosTime(packet.timestamp);
    msg.header.frame_id = slam_frame_id_;
    msg.height = 1;
    msg.width = static_cast<uint32_t>(point_count);
    msg.is_bigendian = false;
    msg.is_dense = true;
    msg.fields.resize(4);

    auto pclType = PointFieldMsg::FLOAT32;
    msg.fields[0].name = "x";
    msg.fields[0].offset = 0;
    msg.fields[0].datatype = pclType;
    msg.fields[0].count = 1;

    msg.fields[1].name = "y";
    msg.fields[1].offset = sizeof(float);
    msg.fields[1].datatype = pclType;
    msg.fields[1].count = 1;

    msg.fields[2].name = "z";
    msg.fields[2].offset = sizeof(float) * 2;
    msg.fields[2].datatype = pclType;
    msg.fields[2].count = 1;

    msg.fields[3].name = "rgb";
    msg.fields[3].offset = sizeof(float) * 3;
    msg.fields[3].datatype = pclType;
    msg.fields[3].count = 1;

    msg.point_step = sizeof(sdk::OdinSlamPoint<float>);
    msg.row_step = msg.width * msg.point_step;
    msg.data.resize(msg.row_step);
    const auto *points = reinterpret_cast<const sdk::OdinSlamPoint<float> *>(packet.payload.data());
    uint8_t *dst = msg.data.data();
    for (size_t i = 0; i < point_count; ++i) {
      // xyz from SDK in mm, convert to meters for ROS
      const float x = points[i].x * 0.001f;
      const float y = points[i].y * 0.001f;
      const float z = points[i].z * 0.001f;
      std::memcpy(dst + 0, &x, sizeof(float));
      std::memcpy(dst + sizeof(float), &y, sizeof(float));
      std::memcpy(dst + sizeof(float) * 2, &z, sizeof(float));
      const uint32_t rgba =
          (static_cast<uint32_t>(points[i].a) << 24) | (static_cast<uint32_t>(points[i].r) << 16) |
          (static_cast<uint32_t>(points[i].g) << 8) | static_cast<uint32_t>(points[i].b);
      float rgb_packed = 0.0f;
      std::memcpy(&rgb_packed, &rgba, sizeof(uint32_t));
      std::memcpy(dst + sizeof(float) * 3, &rgb_packed, sizeof(float));
      dst += msg.point_step;
    }

    PublishMessage(slam_cloud_pub_, msg);
  }

  // SOI (0xFFD8) head，EOI (0xFFD9) tail
  bool IsJpegComplete(const std::vector<uint8_t> &data) {
    if (data.size() < 4) return false;
    // 检查 SOI (Start of Image)
    if (data[0] != 0xFF || data[1] != 0xD8) {
      LogWarn("no header\n");
      return false;
    }
    // EOI (End of Image)
    if (data[data.size() - 2] != 0xFF || data[data.size() - 1] != 0xD9) {
      // LogWarn("no tail,data[data.size() - 2:%02x  data[data.size() - 1]:%02x\n",data[data.size()
      // - 2], data[data.size() - 1]);
      return false;
    }
    return true;
  }

  void ProcessImage(const sdk::OdinImagePacket &packet) {
    if (!IsJpegComplete(packet.payload)) {
      if (ShouldThrottle("jpeg_incomplete_cam0", 5.0)) {
        LogWarn("Incomplete JPEG data (camera0), skipping publish");
      }
      return;
    }
    CompressedImageMsg msg;
    msg.header.stamp = ToRosTime(packet.timestamp);
    msg.header.frame_id = image_frame_id_;
    msg.format = "jpeg";
    msg.data.assign(packet.payload.begin(), packet.payload.end());
    PublishMessage(image_pub_, msg);

    // Publish resolution once after stream start (or restart)
    if (!resolution_sent_) {
      int w = 0, h = 0;
      if (ParseJpegResolution(packet.payload, w, h)) {
        LogInfo("Detected image resolution: %dx%d", w, h);
#if defined(ODIN_ROS2)
        if (!resolution_change_pub_) {
          resolution_change_pub_ = node_->create_publisher<std_msgs::msg::String>(
              "odin_driver/resolution_change", rclcpp::QoS(10));
        }
        std_msgs::msg::String res_msg;
#else
        if (!resolution_change_pub_) {
          resolution_change_pub_ =
              nh_.advertise<std_msgs::String>("odin_driver/resolution_change", 10);
        }
        std_msgs::String res_msg;
#endif
        res_msg.data = "{\"sn\":\"" + device_sn_ + "\",\"width\":" + std::to_string(w) +
                       ",\"height\":" + std::to_string(h) + "}";
        PublishMessage(resolution_change_pub_, res_msg);
        resolution_sent_ = true;
      }
    }
  }

  void HandleGrayImage(const sdk::OdinPointCloudPacket &packet) {
    ImageMsg msg;
    msg.header.stamp = ToRosTime(packet.timestamp);
    msg.header.frame_id = gray_frame_id_;
    msg.height = 192;
    msg.width = 256;
    msg.encoding = "mono8";
    msg.is_bigendian = false;
    msg.step = msg.width;

    size_t pixelNum = msg.height * msg.width;
    msg.data.resize(pixelNum);
    for (size_t i = 0; i < pixelNum; i++) {
      // Get pointer to the intensity field (after x,y,z float coordinates)
      const uint8_t *intensity_ptr =
          packet.payload.data() + i * sizeof(sdk::OdinRawPoint<float>) + 3 * sizeof(float);
      uint8_t gray_value = static_cast<uint8_t>(*intensity_ptr);
      msg.data[i] = gray_value;
    }

    PublishMessage(gray_image_pub_, msg);
  }
  void ProcessImage2(const sdk::OdinImagePacket &packet) {
    if (!IsJpegComplete(packet.payload)) {
      if (ShouldThrottle("jpeg_incomplete_cam1", 5.0)) {
        LogWarn("Incomplete JPEG data (camera1), skipping publish");
      }
      return;
    }
    CompressedImageMsg msg;
    msg.header.stamp = ToRosTime(packet.timestamp);
    msg.header.frame_id = image2_frame_id_;
    msg.format = "jpeg";
    msg.data.assign(packet.payload.begin(), packet.payload.end());
    PublishMessage(image2_pub_, msg);
  }

  void ProcessImu(const sdk::OdinImuPacket &packet) {
    const size_t samples = packet.payload.size() / sizeof(sdk::OdinImuSample);
    if (samples == 0) {
      if (ShouldThrottle("imu_empty", 5.0)) {
        LogDebug("Received IMU packet without samples.");
      }
      return;
    }
    const auto *values = reinterpret_cast<const sdk::OdinImuSample *>(packet.payload.data());
    const size_t count = std::min(samples, static_cast<size_t>(packet.sample_count));
    for (size_t i = 0; i < count; ++i) {
      ImuMsg msg;
      msg.header.stamp = ToRosTime(packet.timestamp);
      msg.header.frame_id = imu_frame_id_;
      msg.orientation_covariance[0] = -1.0;
      msg.orientation_covariance[4] = 0.0;
      msg.orientation_covariance[8] = 0.0;
      // msg.angular_velocity.x = -values[i].gyro_z;
      // msg.angular_velocity.y = -values[i].gyro_y;
      // msg.angular_velocity.z = -values[i].gyro_x ;
      msg.angular_velocity.x = values[i].gyro_x;
      msg.angular_velocity.y = values[i].gyro_y;
      msg.angular_velocity.z = values[i].gyro_z;
      msg.angular_velocity_covariance[0] = 0.0;
      msg.angular_velocity_covariance[4] = 0.0;
      msg.angular_velocity_covariance[8] = 0.0;
      // msg.linear_acceleration.x = -values[i].acc_z;
      // msg.linear_acceleration.y = -values[i].acc_y;
      // msg.linear_acceleration.z = - values[i].acc_x ;
      msg.linear_acceleration.x = values[i].acc_x;
      msg.linear_acceleration.y = values[i].acc_y;
      msg.linear_acceleration.z = values[i].acc_z;
      msg.linear_acceleration_covariance[0] = 0.0;
      msg.linear_acceleration_covariance[4] = 0.0;
      msg.linear_acceleration_covariance[8] = 0.0;
      PublishMessage(imu_pub_, msg);
    }
  }

  void ProcessOdom(const sdk::OdinOdomPacket &packet, sdk::OdomSourceType odom_type) {
    const size_t samples = packet.payload.size() / sizeof(sdk::OdinOdomData);
    if (samples == 0) {
      if (ShouldThrottle("odom_empty", 5.0)) {
        LogDebug("Received odom packet without samples.");
      }
      return;
    }
    const auto *values = reinterpret_cast<const sdk::OdinOdomData *>(packet.payload.data());
    const size_t count = std::min(samples, static_cast<size_t>(packet.data_count));

    // Determine which publisher to use based on odom_type
    bool is_high_frequency = (odom_type == sdk::OdomSourceType::kHighFrequency);

    for (size_t i = 0; i < count; ++i) {
      OdometryMsg msg;
      msg.header.stamp = ToRosTime(packet.timestamp);
      msg.header.frame_id = odom_frame_id_;
      msg.child_frame_id = odom_child_frame_id_;

      msg.pose.pose.position.x = values[i].pos[0] * 1e-6;
      msg.pose.pose.position.y = values[i].pos[1] * 1e-6;
      msg.pose.pose.position.z = values[i].pos[2] * 1e-6;

      msg.pose.pose.orientation.x = values[i].orient[0] * 1e-6;
      msg.pose.pose.orientation.y = values[i].orient[1] * 1e-6;
      msg.pose.pose.orientation.z = values[i].orient[2] * 1e-6;
      msg.pose.pose.orientation.w = values[i].orient[3] * 1e-6;
      NormalizeQuaternion(msg.pose.pose.orientation);

      msg.twist.twist.linear.x = values[i].linear_velocity[0] * 1e-6;
      msg.twist.twist.linear.y = values[i].linear_velocity[1] * 1e-6;
      msg.twist.twist.linear.z = values[i].linear_velocity[2] * 1e-6;

      msg.twist.twist.angular.x = values[i].angular_velocity[0] * 1e-6;
      msg.twist.twist.angular.y = values[i].angular_velocity[1] * 1e-6;
      msg.twist.twist.angular.z = values[i].angular_velocity[2] * 1e-6;

      // Publish to appropriate topic based on odom type
      if (is_high_frequency) {
        PublishMessage(odom_hf_pub_, msg);
      } else {
        PublishMessage(odom_pub_, msg);
      }
    }
  }

  double ApplyScale(int64_t value, double scale) const {
    return static_cast<double>(value) * scale;
  }

  void NormalizeQuaternion(QuaternionMsg &q) const {
    const double norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (norm > std::numeric_limits<double>::epsilon()) {
      q.x /= norm;
      q.y /= norm;
      q.z /= norm;
      q.w /= norm;
    } else {
      q.x = q.y = q.z = 0.0;
      q.w = 1.0;
    }
  }

  bool ShouldThrottle(const std::string &key, double period_sec) {
    const auto now = std::chrono::steady_clock::now();
    const auto it = throttle_slots_.find(key);
    if (it == throttle_slots_.end() ||
        std::chrono::duration<double>(now - it->second).count() >= period_sec) {
      throttle_slots_[key] = now;
      return true;
    }
    return false;
  }

  std::string LogPrefix() const {
    return "[" + (device_sn_.empty() ? "unknown" : device_sn_) + "] ";
  }

  template <typename... Args>
  void LogInfo(const char *fmt, Args... args) const {
    std::string prefixed_fmt = LogPrefix() + fmt;
#if defined(ODIN_ROS2)
    if (node_) {
      RCLCPP_INFO(node_->get_logger(), prefixed_fmt.c_str(), args...);
    }
#else
    ROS_INFO(prefixed_fmt.c_str(), args...);
#endif
  }

  template <typename... Args>
  void LogWarn(const char *fmt, Args... args) const {
    std::string prefixed_fmt = LogPrefix() + fmt;
#if defined(ODIN_ROS2)
    if (node_) {
      RCLCPP_WARN(node_->get_logger(), prefixed_fmt.c_str(), args...);
    }
#else
    ROS_WARN(prefixed_fmt.c_str(), args...);
#endif
  }

  template <typename... Args>
  void LogError(const char *fmt, Args... args) const {
    std::string prefixed_fmt = LogPrefix() + fmt;
#if defined(ODIN_ROS2)
    if (node_) {
      RCLCPP_ERROR(node_->get_logger(), prefixed_fmt.c_str(), args...);
    }
#else
    ROS_ERROR(prefixed_fmt.c_str(), args...);
#endif
  }

  template <typename... Args>
  void LogDebug(const char *fmt, Args... args) const {
    std::string prefixed_fmt = LogPrefix() + fmt;
#if defined(ODIN_ROS2)
    if (node_) {
      RCLCPP_DEBUG(node_->get_logger(), prefixed_fmt.c_str(), args...);
    }
#else
    ROS_DEBUG(prefixed_fmt.c_str(), args...);
#endif
  }

#if defined(ODIN_ROS2)
  rclcpp::Node::SharedPtr node_;
#else
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;       // Device-specific NodeHandle for dynamic reconfigure
  ros::NodeHandle main_pnh_;  // Main node's private NodeHandle for global parameters
#endif

  PublisherType<PointCloud2Msg> raw_cloud_pub_;
  PublisherType<PointCloud2Msg> slam_cloud_pub_;
  PublisherType<CompressedImageMsg> image_pub_;
  PublisherType<ImageMsg> gray_image_pub_;
  PublisherType<ImuMsg> imu_pub_;
  PublisherType<OdometryMsg> odom_pub_;
  PublisherType<OdometryMsg> odom_hf_pub_;
  PublisherType<CompressedImageMsg> image2_pub_;
#if defined(ODIN_ROS2)
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr topic_prefix_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr device_info_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr resolution_change_pub_;
#else
  ros::Publisher topic_prefix_pub_;
  ros::Publisher device_info_pub_;
  ros::Publisher resolution_change_pub_;
#endif

  sdk::DiscoveredDevice discovered_device_;  // 发现的设备信息
  sdk::OdinDeviceHandle device_handle_ = sdk::kInvalidDeviceHandle;
  sdk::OdinOperatingMode desired_mode_ = sdk::OdinOperatingMode::kNormal;
  bool connected_ = false;
  bool callbacks_registered_ = false;

  std::string device_sn_;     // 实际连接设备的SN
  std::string device_model_;  // 设备型号 (e.g., "ODIN2")
  std::string topic_prefix_;  // topic 前缀: /manifold/{model}/device0/
  int device_index_ = -1;     // 设备索引 (0, 1, 2, ...)
  int discovery_timeout_ms_ = 2000;

  std::string raw_frame_id_;
  std::string slam_frame_id_;
  std::string image_frame_id_;
  std::string gray_frame_id_;
  std::string imu_frame_id_;
  std::string odom_frame_id_;
  std::string image2_frame_id_;
  std::string odom_child_frame_id_;
  std::string calib_file_path_;
  std::string calib_yaml_content_;
  bool calib_ready_ = false;
#if defined(ODIN_ROS2)
  rclcpp::Service<odin_ros_driver_rev1::srv::GetCalibration>::SharedPtr calib_service_;
  rclcpp::Service<odin_ros_driver_rev1::srv::SetSensorMode>::SharedPtr sensor_mode_service_;
#else
  ros::ServiceServer calib_service_;
  ros::ServiceServer sensor_mode_service_;
#endif

  double odom_position_scale_ = 1e-6;
  double odom_orientation_scale_ = 1e-6;
  double odom_linear_velocity_scale_ = 1e-6;
  double odom_angular_velocity_scale_ = 1e-6;

  int m_confidence_threshold = 7;
  int sensor_mode_ = 1;  // 0=HDR, 1=High Peak, 2=Low Peak, 3=Adaptive
  int log_level_ = 1;    // 0=ERROR, 1=WARN, 2=INFO, 3=DEBUG

  // Image resolution parameters (0 = use first capability from device)
  int image_width_ = 0;
  int image_height_ = 0;
  int image_fps_ = 0;
  std::string image_format_ = "mjpeg";

  // Channel enable parameters
  bool enable_raw_point_ = true;
  bool enable_slam_point_ = true;
  bool enable_image0_ = true;
  bool enable_image1_ = true;
  bool enable_imu_ = true;
  bool enable_odom_ = true;
#if defined(ODIN_ROS2)
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
#else
  std::shared_ptr<ddynamic_reconfigure::DDynamicReconfigure> ddr_;
#endif

  uint64_t last_img_stamp_ = 0;
  uint64_t last_pcd_stamp_ = 0;

  // Resolution detection: publish once after stream start, reset on stream stop
  bool resolution_sent_ = false;

  // Async queues for non-blocking SDK callbacks
  struct OdomQueueItem {
    sdk::OdinOdomPacket packet;
    sdk::OdomSourceType odom_type;
  };
  std::unique_ptr<AsyncQueue<sdk::OdinPointCloudPacket>> raw_point_queue_;
  std::unique_ptr<AsyncQueue<sdk::OdinSlamPacket>> slam_point_queue_;
  std::unique_ptr<AsyncQueue<sdk::OdinImagePacket>> image0_queue_;
  std::unique_ptr<AsyncQueue<sdk::OdinImagePacket>> image1_queue_;
  std::unique_ptr<AsyncQueue<sdk::OdinImuPacket>> imu_queue_;
  std::unique_ptr<AsyncQueue<OdomQueueItem>> odom_queue_;

  // Parse JPEG SOF0 marker to extract image width and height
  static bool ParseJpegResolution(const std::vector<uint8_t> &data, int &width, int &height) {
    // Look for SOF0 marker (0xFF 0xC0)
    for (size_t i = 0; i + 8 < data.size(); ++i) {
      if (data[i] == 0xFF && (data[i + 1] == 0xC0 || data[i + 1] == 0xC2)) {
        height = (data[i + 5] << 8) | data[i + 6];
        width = (data[i + 7] << 8) | data[i + 8];
        return (width > 0 && height > 0);
      }
    }
    return false;
  }

  std::unordered_map<std::string, std::chrono::steady_clock::time_point> throttle_slots_;
};

}  // namespace odin_ros_driver

// =============================================================================
// Hotplug Device Manager - Unified for ROS1/ROS2
// =============================================================================

namespace {

// Logging macros for hotplug (work outside ros_compat namespace)
#if defined(ODIN_ROS2)
#define HP_LOG_INFO(...) printf(__VA_ARGS__)
#define HP_LOG_WARN(...) printf(__VA_ARGS__)
#define HP_LOG_ERROR(...) printf(__VA_ARGS__)
#define HP_LOG_FATAL(...) printf(__VA_ARGS__)
#else
#define HP_LOG_INFO(...) ROS_INFO(__VA_ARGS__)
#define HP_LOG_WARN(...) ROS_WARN(__VA_ARGS__)
#define HP_LOG_ERROR(...) ROS_ERROR(__VA_ARGS__)
#define HP_LOG_FATAL(...) ROS_FATAL(__VA_ARGS__)
#endif

// Device instance holder - unified structure
struct DeviceInstance {
#if defined(ODIN_ROS2)
  rclcpp::Node::SharedPtr node;
#endif
  std::unique_ptr<odin_ros_driver::odinRosDriver> driver;
  std::string sn;
  int device_index = -1;  // 设备索引，用于回收
};

// Helper to generate sanitized node name
inline std::string SanitizeNodeName(const std::string &sn, size_t index) {
  std::string name = "odin_driver_" + (sn.empty() ? std::to_string(index) : sn);
  for (char &c : name) {
    if (c == '-' || c == ' ') c = '_';
  }
  return name;
}

// Unified HotplugManager
class HotplugManager {
 public:
  std::mutex mutex;
  std::map<std::string, std::unique_ptr<DeviceInstance>> devices;
  std::set<std::string> pending_devices;  // Devices being initialized asynchronously
  std::vector<std::thread> init_threads;  // Track init threads for cleanup
  std::atomic<bool> running{true};
  size_t next_device_index = 0;     // 下一个可用索引
  std::set<int> available_indices;  // 已释放的索引池，优先复用

#if defined(ODIN_ROS2)
  rclcpp::executors::MultiThreadedExecutor *executor = nullptr;
  rclcpp::Node::SharedPtr manager_node;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr device_offline_pub;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr resolution_change_pub;

  void Init() {
    manager_node = std::make_shared<rclcpp::Node>("odin_hotplug_manager");
    device_offline_pub = manager_node->create_publisher<std_msgs::msg::String>(
        "odin_driver/device_offline", rclcpp::QoS(10).transient_local());
    resolution_change_pub = manager_node->create_publisher<std_msgs::msg::String>(
        "odin_driver/resolution_change", rclcpp::QoS(10));
    HP_LOG_INFO("[Hotplug] Manager node initialized (parameters on device nodes)\n");
  }

  void PublishDeviceOffline(const std::string &sn) {
    if (device_offline_pub) {
      std_msgs::msg::String msg;
      msg.data = sn;
      device_offline_pub->publish(msg);
      HP_LOG_INFO("[Hotplug] Published device_offline for: %s\n", sn.c_str());
    }
  }

  void PublishResolutionChange(const std::string &sn, int width, int height) {
    if (resolution_change_pub) {
      std_msgs::msg::String msg;
      msg.data = "{\"sn\":\"" + sn + "\",\"width\":" + std::to_string(width) +
                 ",\"height\":" + std::to_string(height) + "}";
      resolution_change_pub->publish(msg);
      HP_LOG_INFO("[Hotplug] Published resolution_change for %s: %dx%d\n", sn.c_str(), width,
                  height);
    }
  }
#else
  ros::NodeHandle *nh = nullptr;
  ros::NodeHandle *pnh = nullptr;
  ros::Publisher device_offline_pub;
  ros::Publisher resolution_change_pub;

  void Init() {
    if (nh) {
      device_offline_pub = nh->advertise<std_msgs::String>("odin_driver/device_offline", 10, true);
      resolution_change_pub = nh->advertise<std_msgs::String>("odin_driver/resolution_change", 10);
    }
  }

  void PublishDeviceOffline(const std::string &sn) {
    if (device_offline_pub) {
      std_msgs::String msg;
      msg.data = sn;
      device_offline_pub.publish(msg);
      HP_LOG_INFO("[Hotplug] Published device_offline for: %s", sn.c_str());
    }
  }

  void PublishResolutionChange(const std::string &sn, int width, int height) {
    if (resolution_change_pub) {
      std_msgs::String msg;
      msg.data = "{\"sn\":\"" + sn + "\",\"width\":" + std::to_string(width) +
                 ",\"height\":" + std::to_string(height) + "}";
      resolution_change_pub.publish(msg);
      HP_LOG_INFO("[Hotplug] Published resolution_change for %s: %dx%d", sn.c_str(), width, height);
    }
  }
#endif

  void OnAttach(const odin::sdk::DiscoveredDevice &dev) {
    std::string key = dev.sn.empty() ? dev.ip : dev.sn;

    // Quick check with lock - avoid duplicate initialization
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (devices.find(key) != devices.end()) {
        HP_LOG_INFO("[Hotplug] Device %s already connected, ignoring attach\n", key.c_str());
        return;
      }
      if (pending_devices.find(key) != pending_devices.end()) {
        HP_LOG_INFO("[Hotplug] Device %s already initializing, ignoring attach\n", key.c_str());
        return;
      }
      pending_devices.insert(key);
    }

#if defined(ODIN_ROS2)
    HP_LOG_INFO("[Hotplug] Device attached: SN=%s, IP=%s, Model=%s (async init)\n", dev.sn.c_str(),
                dev.ip.c_str(), dev.model.c_str());
#else
    HP_LOG_INFO("[Hotplug] Device attached: SN=%s, IP=%s, Model=%s (async init)", dev.sn.c_str(),
                dev.ip.c_str(), dev.model.c_str());
#endif

    // 从索引池获取索引，优先复用已释放的索引
    int idx;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (!available_indices.empty()) {
        idx = *available_indices.begin();
        available_indices.erase(available_indices.begin());
        HP_LOG_INFO("[Hotplug] Reusing device index %d\n", idx);
      } else {
        idx = static_cast<int>(next_device_index++);
        HP_LOG_INFO("[Hotplug] Allocating new device index %d\n", idx);
      }
    }
#if defined(ODIN_ROS2)
    auto *exec_ptr = executor;
#else
    ros::NodeHandle *nh_ptr = nh;
    ros::NodeHandle *pnh_ptr = pnh;
#endif

    // Async initialization in separate thread to avoid blocking existing devices
    init_threads.emplace_back([this, dev, key, idx
#if defined(ODIN_ROS2)
                               ,
                               exec_ptr
#else
                               ,
                               nh_ptr, pnh_ptr
#endif
    ]() {
      auto instance = std::make_unique<DeviceInstance>();
      instance->sn = key;
      instance->device_index = idx;  // 保存索引用于回收

#if defined(ODIN_ROS2)
      std::string node_name = SanitizeNodeName(dev.sn, idx);
      instance->node = std::make_shared<rclcpp::Node>(node_name);
      instance->driver = std::make_unique<odin_ros_driver::odinRosDriver>(instance->node, dev, idx);
#else
      // Create device-specific NodeHandle with SN as namespace for independent dynamic reconfigure
      // Pass both main_pnh (for global params like image_width) and device_pnh (for dynamic reconfigure)
      std::string device_ns = dev.sn.empty() ? "device_" + std::to_string(idx) : dev.sn;
      ros::NodeHandle device_pnh(*pnh_ptr, device_ns);
      instance->driver =
          std::make_unique<odin_ros_driver::odinRosDriver>(*nh_ptr, *pnh_ptr, device_pnh, dev, idx);
#endif

      bool init_success = instance->driver->Init();

      // Finalize with lock
      std::lock_guard<std::mutex> lock(mutex);
      pending_devices.erase(key);

      if (!init_success) {
        HP_LOG_ERROR("[Hotplug] Failed to initialize driver for device %s\n", key.c_str());
        return;
      }

#if defined(ODIN_ROS2)
      if (exec_ptr) {
        exec_ptr->add_node(instance->node);
      }
      // Note: Per-device parameters removed due to thread-safety issues
      // Use global default_confidence_threshold and default_sensor_mode instead
#endif

      devices[key] = std::move(instance);
      HP_LOG_INFO("[Hotplug] Device %s connected, total: %zu\n", key.c_str(), devices.size());
    });
  }

  void OnDetach(const odin::sdk::DiscoveredDevice &dev) {
    std::unique_ptr<DeviceInstance> instance;
    int released_index = -1;

    {
      std::lock_guard<std::mutex> lock(mutex);

      std::string key = dev.sn.empty() ? dev.ip : dev.sn;
      auto it = devices.find(key);
      if (it == devices.end()) {
        HP_LOG_WARN("[Hotplug] Device %s not found for detach\n", key.c_str());
        return;
      }

      HP_LOG_INFO("[Hotplug] Device detaching: SN=%s\n", dev.sn.c_str());
      instance = std::move(it->second);
      released_index = instance->device_index;
      devices.erase(it);

      // 回收索引到索引池
      if (released_index >= 0) {
        available_indices.insert(released_index);
        HP_LOG_INFO("[Hotplug] Recycled device index %d\n", released_index);
      }
    }

    if (instance) {
      PublishDeviceOffline(instance->sn);
#if defined(ODIN_ROS2)
      if (executor && instance->node) {
        executor->remove_node(instance->node);
      }
      instance->node.reset();
#endif
      instance->driver.reset();
      HP_LOG_INFO("[Hotplug] Device %s released, remaining: %zu\n", instance->sn.c_str(),
                  devices.size());
    }
  }

  void Cleanup() {
    // Wait for all pending init threads to complete
    for (auto &t : init_threads) {
      if (t.joinable()) {
        t.join();
      }
    }
    init_threads.clear();

    std::lock_guard<std::mutex> lock(mutex);
    for (auto &pair : devices) {
#if defined(ODIN_ROS2)
      if (executor && pair.second->node) {
        executor->remove_node(pair.second->node);
      }
      pair.second->node.reset();
#endif
      pair.second->driver.reset();
    }
    devices.clear();
    pending_devices.clear();
    HP_LOG_INFO("[Hotplug] All devices cleaned up\n");
  }
};

}  // anonymous namespace

// =============================================================================
// Main Entry Point
// =============================================================================

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IOLBF, 0);
  log_config(LOG_LEVEL_INFO, printf);

#if defined(ODIN_ROS2)
  rclcpp::init(argc, argv);
  printf("[odin_ros_driver] Starting with hotplug support (ROS2)...\n");

  HotplugManager manager;
  manager.Init();
  rclcpp::executors::MultiThreadedExecutor executor;
  manager.executor = &executor;
  executor.add_node(manager.manager_node);
#else
  ros::init(argc, argv, "odin_ros_driver");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  ROS_INFO("[odin_ros_driver] Starting with hotplug support (ROS1)...");

  HotplugManager manager;
  manager.nh = &nh;
  manager.pnh = &pnh;
  manager.Init();
#endif

  // Setup hotplug listener using SDK API
  odin::sdk::HotplugCallbacks callbacks;
  callbacks.on_attach = [&manager](const odin::sdk::DiscoveredDevice &dev) {
    manager.OnAttach(dev);
  };
  callbacks.on_detach = [&manager](const odin::sdk::DiscoveredDevice &dev) {
    manager.OnDetach(dev);
  };

  if (!odin::sdk::StartHotplugListener(callbacks, true)) {
    HP_LOG_FATAL("[odin_ros_driver] Failed to start hotplug listener!\n");
#if defined(ODIN_ROS2)
    rclcpp::shutdown();
#endif
    return 1;
  }

  HP_LOG_INFO("[odin_ros_driver] Hotplug listener started, running (Ctrl+C to stop)...\n");

#if defined(ODIN_ROS2)
  while (rclcpp::ok() && manager.running.load()) {
    executor.spin_some(std::chrono::milliseconds(100));
  }
#else
  ros::spin();
#endif

  HP_LOG_INFO("[odin_ros_driver] Shutting down...\n");
  odin::sdk::StopHotplugListener();
  manager.Cleanup();

#if defined(ODIN_ROS2)
  rclcpp::shutdown();
#endif

  HP_LOG_INFO("[odin_ros_driver] Shutdown complete\n");
  return 0;
}
