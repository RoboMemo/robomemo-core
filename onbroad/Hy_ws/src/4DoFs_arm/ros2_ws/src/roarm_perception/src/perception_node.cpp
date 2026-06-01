#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <pcl/common/common.h>

#include <Eigen/Dense>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

#include <vector>
#include <string>
#include <memory>

using PointT = pcl::PointXYZRGB;
using Cloud  = pcl::PointCloud<PointT>;

class PerceptionNode : public rclcpp::Node
{
public:
  PerceptionNode()
  : rclcpp::Node("perception_node")
  {
    // Parameters
    declare_parameter("input_cloud_topic",    std::string("/camera/depth_registered/points"));
    declare_parameter("target_frame",         std::string("base_link"));
    declare_parameter("voxel_leaf_size",      0.005);
    declare_parameter("passthrough_z_min",    0.05);
    declare_parameter("passthrough_z_max",    1.20);
    declare_parameter("sac_distance_threshold", 0.010);
    declare_parameter("sac_max_iterations",   1000);
    declare_parameter("ec_cluster_tolerance", 0.020);
    declare_parameter("ec_min_cluster_size",  100);
    declare_parameter("ec_max_cluster_size",  25000);

    input_topic_  = get_parameter("input_cloud_topic").as_string();
    target_frame_ = get_parameter("target_frame").as_string();
    voxel_leaf_   = get_parameter("voxel_leaf_size").as_double();
    pt_z_min_     = get_parameter("passthrough_z_min").as_double();
    pt_z_max_     = get_parameter("passthrough_z_max").as_double();
    sac_dist_     = get_parameter("sac_distance_threshold").as_double();
    sac_iter_     = get_parameter("sac_max_iterations").as_int();
    ec_tol_       = get_parameter("ec_cluster_tolerance").as_double();
    ec_min_       = get_parameter("ec_min_cluster_size").as_int();
    ec_max_       = get_parameter("ec_max_cluster_size").as_int();

    // TF
    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Publishers
    poses_pub_   = create_publisher<geometry_msgs::msg::PoseArray>(
      "/perception/detected_objects", 10);
    markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/perception/object_markers", 10);
    cloud_pub_   = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/perception/filtered_cloud", 10);

    // Subscriber
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::PointCloud2::UniquePtr msg) { cloudCallback(std::move(msg)); });

    // Trigger service
    trigger_srv_ = create_service<std_srvs::srv::Trigger>(
      "/perception/trigger_detection",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res)
      {
        triggered_ = true;
        res->success = true;
        res->message = "Detection triggered.";
      });

    RCLCPP_INFO(get_logger(), "PerceptionNode ready. Listening on %s", input_topic_.c_str());
  }

private:
  // ── config ────────────────────────────────────────────────────────────────
  std::string input_topic_, target_frame_;
  double voxel_leaf_, pt_z_min_, pt_z_max_, sac_dist_, ec_tol_;
  int    sac_iter_, ec_min_, ec_max_;

  // ── ROS handles ──────────────────────────────────────────────────────────
  std::shared_ptr<tf2_ros::Buffer>             tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener>  tf_listener_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr    poses_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    cloud_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr             trigger_srv_;

  std::atomic<bool> triggered_{false};

  // ── callback ──────────────────────────────────────────────────────────────
  void cloudCallback(sensor_msgs::msg::PointCloud2::UniquePtr msg)
  {
    // Only process when triggered, or always if no one has triggered yet
    if (!triggered_ && poses_pub_->get_subscription_count() == 0) return;
    triggered_ = false;

    // ── 1. Transform to target_frame ─────────────────────────────────────
    sensor_msgs::msg::PointCloud2 cloud_tf;
    try {
      tf_buffer_->transform(*msg, cloud_tf, target_frame_,
        tf2::durationFromSec(0.1));
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
        "TF error: %s", ex.what());
      return;
    }

    Cloud::Ptr cloud(new Cloud);
    pcl::fromROSMsg(cloud_tf, *cloud);

    if (cloud->empty()) return;

    // ── 2. VoxelGrid downsample ──────────────────────────────────────────
    Cloud::Ptr voxeled(new Cloud);
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(voxel_leaf_, voxel_leaf_, voxel_leaf_);
    vg.filter(*voxeled);

    // ── 3. PassThrough on Z ──────────────────────────────────────────────
    Cloud::Ptr passed(new Cloud);
    pcl::PassThrough<PointT> pt;
    pt.setInputCloud(voxeled);
    pt.setFilterFieldName("z");
    pt.setFilterLimits(pt_z_min_, pt_z_max_);
    pt.filter(*passed);

    if (passed->empty()) return;

    // ── 4. Statistical outlier removal ───────────────────────────────────
    Cloud::Ptr cleaned(new Cloud);
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(passed);
    sor.setMeanK(30);
    sor.setStddevMulThresh(1.5);
    sor.filter(*cleaned);

    // ── 5. RANSAC plane segmentation (remove dominant plane = table) ─────
    pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(sac_iter_);
    seg.setDistanceThreshold(sac_dist_);
    seg.setInputCloud(cleaned);
    seg.segment(*inliers, *coeffs);

    Cloud::Ptr objects(new Cloud);
    if (!inliers->indices.empty()) {
      pcl::ExtractIndices<PointT> ei;
      ei.setInputCloud(cleaned);
      ei.setIndices(inliers);
      ei.setNegative(true);  // keep everything NOT on the plane
      ei.filter(*objects);
    } else {
      objects = cleaned;
    }

    // Publish filtered cloud for debug
    if (cloud_pub_->get_subscription_count() > 0) {
      sensor_msgs::msg::PointCloud2 filtered_msg;
      pcl::toROSMsg(*objects, filtered_msg);
      filtered_msg.header.frame_id = target_frame_;
      filtered_msg.header.stamp    = msg->header.stamp;
      cloud_pub_->publish(filtered_msg);
    }

    if (objects->empty()) return;

    // ── 6. Euclidean cluster extraction ──────────────────────────────────
    auto tree = std::make_shared<pcl::search::KdTree<PointT>>();
    tree->setInputCloud(objects);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(ec_tol_);
    ec.setMinClusterSize(ec_min_);
    ec.setMaxClusterSize(ec_max_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(objects);
    ec.extract(cluster_indices);

    // ── 7. Compute pose per cluster ──────────────────────────────────────
    geometry_msgs::msg::PoseArray pose_array;
    pose_array.header.frame_id = target_frame_;
    pose_array.header.stamp    = msg->header.stamp;

    visualization_msgs::msg::MarkerArray marker_array;
    // Delete old markers first
    visualization_msgs::msg::Marker del_marker;
    del_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(del_marker);

    int marker_id = 0;
    for (const auto & ci : cluster_indices) {
      Cloud::Ptr cluster(new Cloud);
      for (auto idx : ci.indices) {
        cluster->push_back((*objects)[idx]);
      }

      // Centroid
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*cluster, centroid);

      // PCA for orientation
      pcl::PCA<PointT> pca;
      pca.setInputCloud(cluster);
      Eigen::Matrix3f eigen_vecs = pca.getEigenVectors();

      // Ensure right-handed coordinate system
      eigen_vecs.col(2) = eigen_vecs.col(0).cross(eigen_vecs.col(1));

      // Convert to tf2 quaternion
      tf2::Matrix3x3 rot_mat(
        eigen_vecs(0, 0), eigen_vecs(0, 1), eigen_vecs(0, 2),
        eigen_vecs(1, 0), eigen_vecs(1, 1), eigen_vecs(1, 2),
        eigen_vecs(2, 0), eigen_vecs(2, 1), eigen_vecs(2, 2));
      tf2::Quaternion q;
      rot_mat.getRotation(q);
      q.normalize();

      geometry_msgs::msg::Pose pose;
      pose.position.x = centroid[0];
      pose.position.y = centroid[1];
      pose.position.z = centroid[2];
      pose.orientation.x = q.x();
      pose.orientation.y = q.y();
      pose.orientation.z = q.z();
      pose.orientation.w = q.w();
      pose_array.poses.push_back(pose);

      // Bounding box marker
      PointT min_pt, max_pt;
      pcl::getMinMax3D(*cluster, min_pt, max_pt);

      visualization_msgs::msg::Marker box;
      box.header = pose_array.header;
      box.ns     = "objects";
      box.id     = marker_id++;
      box.type   = visualization_msgs::msg::Marker::CUBE;
      box.action = visualization_msgs::msg::Marker::ADD;
      box.pose   = pose;
      box.scale.x = max_pt.x - min_pt.x;
      box.scale.y = max_pt.y - min_pt.y;
      box.scale.z = max_pt.z - min_pt.z;
      box.color.r = 0.2f; box.color.g = 0.8f; box.color.b = 0.2f;
      box.color.a = 0.4f;
      box.lifetime = rclcpp::Duration::from_seconds(1.0);
      marker_array.markers.push_back(box);
    }

    poses_pub_->publish(pose_array);
    markers_pub_->publish(marker_array);

    RCLCPP_DEBUG(get_logger(), "Detected %zu objects.", cluster_indices.size());
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PerceptionNode>());
  rclcpp::shutdown();
  return 0;
}
