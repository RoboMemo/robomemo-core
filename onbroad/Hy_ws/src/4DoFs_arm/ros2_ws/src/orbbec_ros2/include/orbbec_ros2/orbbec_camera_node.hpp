#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <image_transport/image_transport.hpp>

// OrbbecSDK C++ API
#include <libobsensor/ObSensor.hpp>

#include <atomic>
#include <memory>
#include <string>

namespace orbbec_ros2
{

class OrbbecCameraNode : public rclcpp::Node
{
public:
  explicit OrbbecCameraNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~OrbbecCameraNode();

private:
  // ── Parameters ─────────────────────────────────────────────────────────────
  int color_width_{640};
  int color_height_{480};
  int color_fps_{30};
  int depth_width_{640};
  int depth_height_{480};
  int depth_fps_{30};
  std::string align_mode_{"HW"};   // "HW" | "SW" | "NONE"
  bool enable_point_cloud_{true};
  bool publish_tf_{true};
  std::string camera_frame_id_{"camera_link"};
  std::string color_optical_frame_{"camera_color_optical_frame"};
  std::string depth_optical_frame_{"camera_depth_optical_frame"};

  // ── SDK objects ─────────────────────────────────────────────────────────────
  ob::Pipeline pipeline_;
  ob::PointCloudFilter pc_filter_;
  OBCameraParam camera_param_{};

  // ── Publishers ──────────────────────────────────────────────────────────────
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr color_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr color_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;

  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_broadcaster_;

  // ── Internal helpers ────────────────────────────────────────────────────────
  void declareParameters();
  void startPipeline();
  void frameSetCallback(std::shared_ptr<ob::FrameSet> frameset);

  sensor_msgs::msg::Image::SharedPtr
  frameToImageMsg(const std::shared_ptr<ob::VideoFrame> & frame,
                  const std::string & encoding,
                  const std::string & frame_id);

  sensor_msgs::msg::CameraInfo
  buildCameraInfo(const OBCameraIntrinsic & intrinsic,
                  const OBCameraDistortion & distortion,
                  const std::string & frame_id);

  sensor_msgs::msg::PointCloud2::SharedPtr
  pointsFrameToMsg(const std::shared_ptr<ob::Frame> & frame,
                   const std::string & frame_id);

  void publishStaticTf();
};

}  // namespace orbbec_ros2
