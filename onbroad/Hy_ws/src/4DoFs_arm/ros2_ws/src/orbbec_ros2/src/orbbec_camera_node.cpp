#include "orbbec_ros2/orbbec_camera_node.hpp"

#include <sensor_msgs/msg/point_field.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <cstring>
#include <stdexcept>

namespace orbbec_ros2
{

// ── construction / destruction ─────────────────────────────────────────────

OrbbecCameraNode::OrbbecCameraNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("orbbec_camera_node", options)
{
  declareParameters();

  // Publishers
  color_pub_      = create_publisher<sensor_msgs::msg::Image>("/camera/color/image_raw", 10);
  color_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("/camera/color/camera_info", 10);
  depth_pub_      = create_publisher<sensor_msgs::msg::Image>("/camera/depth/image_raw", 10);
  depth_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>("/camera/depth/camera_info", 10);
  cloud_pub_      = create_publisher<sensor_msgs::msg::PointCloud2>("/camera/depth_registered/points", 10);

  tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

  startPipeline();

  if (publish_tf_) {
    publishStaticTf();
  }

  RCLCPP_INFO(get_logger(), "OrbbecCameraNode started.");
}

OrbbecCameraNode::~OrbbecCameraNode()
{
  try {
    pipeline_.stop();
  } catch (...) {}
}

// ── parameter declaration ──────────────────────────────────────────────────

void OrbbecCameraNode::declareParameters()
{
  color_width_   = declare_parameter("color_width",   640);
  color_height_  = declare_parameter("color_height",  480);
  color_fps_     = declare_parameter("color_fps",     30);
  depth_width_   = declare_parameter("depth_width",   640);
  depth_height_  = declare_parameter("depth_height",  480);
  depth_fps_     = declare_parameter("depth_fps",     30);
  align_mode_    = declare_parameter("align_mode",    std::string("HW"));
  enable_point_cloud_ = declare_parameter("enable_point_cloud", true);
  publish_tf_    = declare_parameter("publish_tf",    true);
  camera_frame_id_     = declare_parameter("camera_frame_id",     std::string("camera_link"));
  color_optical_frame_ = declare_parameter("color_optical_frame", std::string("camera_color_optical_frame"));
  depth_optical_frame_ = declare_parameter("depth_optical_frame", std::string("camera_depth_optical_frame"));
}

// ── SDK pipeline startup ───────────────────────────────────────────────────

void OrbbecCameraNode::startPipeline()
{
  auto config = std::make_shared<ob::Config>();

  // ── Color stream ─────────────────────────────────────────────────────────
  auto color_profiles = pipeline_.getStreamProfileList(OB_SENSOR_COLOR);
  std::shared_ptr<ob::VideoStreamProfile> color_profile;
  try {
    color_profile = color_profiles->getVideoStreamProfile(
      color_width_, color_height_, OB_FORMAT_RGB888, color_fps_);
  } catch (...) {
    color_profile = color_profiles->getVideoStreamProfile(
      OB_WIDTH_ANY, OB_HEIGHT_ANY, OB_FORMAT_RGB888, OB_FPS_ANY);
    RCLCPP_WARN(get_logger(),
      "Requested color resolution not available; using default.");
  }
  config->enableStream(color_profile);

  // ── Depth stream (D2C-aligned or free) ───────────────────────────────────
  OBAlignMode align = ALIGN_DISABLE;
  if (align_mode_ == "HW") {
    align = ALIGN_D2C_HW_MODE;
  } else if (align_mode_ == "SW") {
    align = ALIGN_D2C_SW_MODE;
  }

  if (align != ALIGN_DISABLE) {
    auto depth_list = pipeline_.getD2CDepthProfileList(color_profile, align);
    auto depth_profile = depth_list->getVideoStreamProfile(
      OB_WIDTH_ANY, OB_HEIGHT_ANY, OB_FORMAT_ANY, OB_FPS_ANY);
    config->enableStream(depth_profile);
    config->setAlignMode(align);
  } else {
    auto depth_profiles = pipeline_.getStreamProfileList(OB_SENSOR_DEPTH);
    auto depth_profile = depth_profiles->getVideoStreamProfile(
      depth_width_, depth_height_, OB_FORMAT_Y16, depth_fps_);
    config->enableStream(depth_profile);
  }

  pipeline_.enableFrameSync();

  // Cache camera parameters for CameraInfo and point-cloud filter
  camera_param_ = pipeline_.getCameraParam();

  // Configure point cloud filter once
  pc_filter_.setCameraParam(camera_param_);
  pc_filter_.setCreatePointFormat(OB_FORMAT_RGB_POINT);

  // Start with callback
  pipeline_.start(config,
    [this](std::shared_ptr<ob::FrameSet> fs) { frameSetCallback(fs); });
}

// ── Frame callback ─────────────────────────────────────────────────────────

void OrbbecCameraNode::frameSetCallback(std::shared_ptr<ob::FrameSet> frameset)
{
  if (!frameset) return;

  auto now = rclcpp::Clock{}.now();

  // ── Color ──────────────────────────────────────────────────────────────
  auto color_frame = frameset->colorFrame();
  if (color_frame && (color_pub_->get_subscription_count() > 0 ||
                      color_info_pub_->get_subscription_count() > 0)) {
    auto img_msg = frameToImageMsg(color_frame, "rgb8", color_optical_frame_);
    if (img_msg) {
      img_msg->header.stamp = now;
      color_pub_->publish(*img_msg);

      auto info = buildCameraInfo(
        camera_param_.rgbIntrinsic, camera_param_.rgbDistortion, color_optical_frame_);
      info.header.stamp = now;
      color_info_pub_->publish(info);
    }
  }

  // ── Depth ──────────────────────────────────────────────────────────────
  auto depth_frame = frameset->depthFrame();
  if (depth_frame && (depth_pub_->get_subscription_count() > 0 ||
                      depth_info_pub_->get_subscription_count() > 0)) {
    auto img_msg = frameToImageMsg(depth_frame, "16UC1", depth_optical_frame_);
    if (img_msg) {
      img_msg->header.stamp = now;
      depth_pub_->publish(*img_msg);

      auto info = buildCameraInfo(
        camera_param_.depthIntrinsic, camera_param_.depthDistortion, depth_optical_frame_);
      info.header.stamp = now;
      depth_info_pub_->publish(info);
    }
  }

  // ── Point Cloud ────────────────────────────────────────────────────────
  if (enable_point_cloud_ && depth_frame && color_frame &&
      cloud_pub_->get_subscription_count() > 0) {
    // Must update scale each frame
    pc_filter_.setPositionDataScaled(depth_frame->getValueScale());
    try {
      auto pc_frame = pc_filter_.process(frameset);
      if (pc_frame) {
        auto cloud_msg = pointsFrameToMsg(pc_frame, depth_optical_frame_);
        if (cloud_msg) {
          cloud_msg->header.stamp = now;
          cloud_pub_->publish(*cloud_msg);
        }
      }
    } catch (const std::exception & e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "PointCloud filter error: %s", e.what());
    }
  }
}

// ── Conversion helpers ─────────────────────────────────────────────────────

sensor_msgs::msg::Image::SharedPtr
OrbbecCameraNode::frameToImageMsg(
  const std::shared_ptr<ob::VideoFrame> & frame,
  const std::string & encoding,
  const std::string & frame_id)
{
  auto msg = std::make_shared<sensor_msgs::msg::Image>();
  msg->header.frame_id = frame_id;
  msg->width  = frame->width();
  msg->height = frame->height();
  msg->encoding = encoding;
  msg->is_bigendian = false;

  uint32_t bytes_per_pixel = (encoding == "16UC1") ? 2 : 3;
  msg->step = msg->width * bytes_per_pixel;
  msg->data.resize(msg->step * msg->height);
  std::memcpy(msg->data.data(), frame->data(), msg->data.size());
  return msg;
}

sensor_msgs::msg::CameraInfo
OrbbecCameraNode::buildCameraInfo(
  const OBCameraIntrinsic & intr,
  const OBCameraDistortion & dist,
  const std::string & frame_id)
{
  sensor_msgs::msg::CameraInfo info;
  info.header.frame_id = frame_id;
  info.width  = static_cast<uint32_t>(intr.width);
  info.height = static_cast<uint32_t>(intr.height);
  info.distortion_model = "plumb_bob";

  // D = [k1, k2, p1, p2, k3]
  info.d = {dist.k1, dist.k2, dist.p1, dist.p2, dist.k3};

  // K (row-major 3x3)
  info.k = {
    intr.fx,   0.0,    intr.cx,
    0.0,    intr.fy,  intr.cy,
    0.0,    0.0,    1.0
  };

  // R = identity
  info.r = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  // P = [K | 0]
  info.p = {
    intr.fx,   0.0,    intr.cx,  0.0,
    0.0,    intr.fy,  intr.cy,  0.0,
    0.0,    0.0,    1.0,      0.0
  };
  return info;
}

sensor_msgs::msg::PointCloud2::SharedPtr
OrbbecCameraNode::pointsFrameToMsg(
  const std::shared_ptr<ob::Frame> & frame,
  const std::string & frame_id)
{
  const auto * pts = reinterpret_cast<const OBColorPoint *>(frame->data());
  const uint32_t num_points = static_cast<uint32_t>(
    frame->dataSize() / sizeof(OBColorPoint));

  if (num_points == 0) return nullptr;

  auto msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
  msg->header.frame_id = frame_id;
  msg->height = 1;
  msg->width  = num_points;
  msg->is_dense = false;
  msg->is_bigendian = false;

  // Define fields: x, y, z (float32) + rgb (float32 packed)
  auto add_field = [](sensor_msgs::msg::PointField f,
                      const std::string & name, uint32_t offset,
                      uint8_t datatype) {
    f.name     = name;
    f.offset   = offset;
    f.datatype = datatype;
    f.count    = 1;
    return f;
  };

  sensor_msgs::msg::PointField pf;
  msg->fields.push_back(add_field(pf, "x",   0,  sensor_msgs::msg::PointField::FLOAT32));
  msg->fields.push_back(add_field(pf, "y",   4,  sensor_msgs::msg::PointField::FLOAT32));
  msg->fields.push_back(add_field(pf, "z",   8,  sensor_msgs::msg::PointField::FLOAT32));
  msg->fields.push_back(add_field(pf, "rgb", 12, sensor_msgs::msg::PointField::FLOAT32));

  msg->point_step = 16;  // 4 fields × 4 bytes
  msg->row_step   = msg->point_step * num_points;
  msg->data.resize(msg->row_step);

  uint8_t * out = msg->data.data();
  for (uint32_t i = 0; i < num_points; ++i) {
    // SDK units are mm → convert to meters
    float x = pts[i].x / 1000.0f;
    float y = pts[i].y / 1000.0f;
    float z = pts[i].z / 1000.0f;

    // Pack RGB into a float (PCL convention)
    uint32_t rgb_packed =
      (static_cast<uint32_t>(pts[i].r) << 16) |
      (static_cast<uint32_t>(pts[i].g) <<  8) |
      (static_cast<uint32_t>(pts[i].b));
    float rgb_float;
    std::memcpy(&rgb_float, &rgb_packed, sizeof(float));

    std::memcpy(out,      &x,         4);
    std::memcpy(out + 4,  &y,         4);
    std::memcpy(out + 8,  &z,         4);
    std::memcpy(out + 12, &rgb_float, 4);
    out += 16;
  }
  return msg;
}

// ── Static TF ─────────────────────────────────────────────────────────────

void OrbbecCameraNode::publishStaticTf()
{
  // camera_link → camera_color_optical_frame
  // Standard ROS optical convention: rotate -90° about Z then -90° about X
  tf2::Quaternion q_optical;
  q_optical.setRPY(-M_PI / 2.0, 0.0, -M_PI / 2.0);

  geometry_msgs::msg::TransformStamped color_tf;
  color_tf.header.stamp    = rclcpp::Clock{}.now();
  color_tf.header.frame_id = camera_frame_id_;
  color_tf.child_frame_id  = color_optical_frame_;
  color_tf.transform.translation.x = 0;
  color_tf.transform.translation.y = 0;
  color_tf.transform.translation.z = 0;
  color_tf.transform.rotation.x = q_optical.x();
  color_tf.transform.rotation.y = q_optical.y();
  color_tf.transform.rotation.z = q_optical.z();
  color_tf.transform.rotation.w = q_optical.w();

  // camera_link → camera_depth_optical_frame
  // Depth-to-color offset from SDK (camera_param_.transform.trans[0..2] in mm)
  geometry_msgs::msg::TransformStamped depth_tf;
  depth_tf.header.stamp    = rclcpp::Clock{}.now();
  depth_tf.header.frame_id = camera_frame_id_;
  depth_tf.child_frame_id  = depth_optical_frame_;
  depth_tf.transform.translation.x = camera_param_.transform.trans[0] / 1000.0;
  depth_tf.transform.translation.y = camera_param_.transform.trans[1] / 1000.0;
  depth_tf.transform.translation.z = camera_param_.transform.trans[2] / 1000.0;
  depth_tf.transform.rotation.x = q_optical.x();
  depth_tf.transform.rotation.y = q_optical.y();
  depth_tf.transform.rotation.z = q_optical.z();
  depth_tf.transform.rotation.w = q_optical.w();

  tf_broadcaster_->sendTransform({color_tf, depth_tf});
}

}  // namespace orbbec_ros2

// ── main ────────────────────────────────────────────────────────────────────

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<orbbec_ros2::OrbbecCameraNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
