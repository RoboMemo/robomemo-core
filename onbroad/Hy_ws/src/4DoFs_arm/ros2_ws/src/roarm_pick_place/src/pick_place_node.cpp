#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>

#include "roarm_moveit/srv/move_point_cmd.hpp"
#include "roarm_pick_place/action/pick_place.hpp"

#include <thread>
#include <chrono>
#include <string>
#include <memory>
#include <vector>
#include <optional>

using PickPlaceAction = roarm_pick_place::action::PickPlace;
using GoalHandle      = rclcpp_action::ServerGoalHandle<PickPlaceAction>;

// ── helpers ───────────────────────────────────────────────────────────────

static std::string stateName(const std::string & s, float pct)
{
  return s + " (" + std::to_string(static_cast<int>(pct)) + "%)";
}

// ── node ─────────────────────────────────────────────────────────────────

class PickPlaceNode : public rclcpp::Node
{
public:
  PickPlaceNode()
  : rclcpp::Node("pick_place_node",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true))
  {
    // Parameters
    approach_z_   = get_parameter_or("approach_offset_z", 0.08);
    retreat_z_    = get_parameter_or("retreat_offset_z", 0.10);
    home_x_       = get_parameter_or("home_x", 0.18);
    home_y_       = get_parameter_or("home_y", 0.00);
    home_z_       = get_parameter_or("home_z", 0.15);
    gripper_open_ = get_parameter_or("gripper_open_rad", 1.5f);
    gripper_close_= get_parameter_or("gripper_close_rad", 0.2f);
    settle_s_     = get_parameter_or("gripper_settle_s", 1.0);
    move_svc_     = get_parameter_or("move_point_service", std::string("move_point_cmd"));
    gripper_topic_= get_parameter_or("gripper_topic", std::string("gripper_cmd"));
    det_topic_    = get_parameter_or("detection_topic", std::string("/perception/detected_objects"));
    planning_group_ = get_parameter_or("planning_group", std::string("hand"));
    cart_step_    = get_parameter_or("cartesian_step", 0.010);
    cart_jump_    = get_parameter_or("cartesian_jump_threshold", 0.0);
    cart_min_frac_= get_parameter_or("cartesian_min_fraction", 0.90);

    // Service clients
    move_client_ = create_client<roarm_moveit::srv::MovePointCmd>(move_svc_);

    // Gripper publisher
    gripper_pub_ = create_publisher<std_msgs::msg::Float32>(gripper_topic_, 10);

    // Detected objects subscriber
    det_sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
      det_topic_, 10,
      [this](geometry_msgs::msg::PoseArray::SharedPtr msg) {
        latest_objects_ = msg;
      });

    // Action server
    action_server_ = rclcpp_action::create_server<PickPlaceAction>(
      this, "/pick_place/execute",
      [this](const rclcpp_action::GoalUUID &, std::shared_ptr<const PickPlaceAction::Goal>) {
        return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
      },
      [this](const std::shared_ptr<GoalHandle>) {
        return rclcpp_action::CancelResponse::ACCEPT;
      },
      [this](std::shared_ptr<GoalHandle> gh) {
        std::thread{[this, gh]() { execute(gh); }}.detach();
      });

    RCLCPP_INFO(get_logger(), "PickPlaceNode ready.");
  }

private:
  // ── config ────────────────────────────────────────────────────────────
  double approach_z_, retreat_z_, home_x_, home_y_, home_z_;
  float  gripper_open_, gripper_close_;
  double settle_s_, cart_step_, cart_jump_, cart_min_frac_;
  std::string move_svc_, gripper_topic_, det_topic_, planning_group_;

  // ── ROS handles ────────────────────────────────────────────────────────
  rclcpp::Client<roarm_moveit::srv::MovePointCmd>::SharedPtr move_client_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr       gripper_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr det_sub_;
  rclcpp_action::Server<PickPlaceAction>::SharedPtr          action_server_;

  geometry_msgs::msg::PoseArray::SharedPtr latest_objects_;

  // ── helpers ────────────────────────────────────────────────────────────

  void sendFeedback(std::shared_ptr<GoalHandle> & gh,
                    const std::string & state, float pct)
  {
    auto fb = std::make_shared<PickPlaceAction::Feedback>();
    fb->current_state   = state;
    fb->progress_percent = pct;
    gh->publish_feedback(fb);
    RCLCPP_INFO(get_logger(), "[%s] %.0f%%", state.c_str(), pct);
  }

  bool moveToPoint(double x, double y, double z)
  {
    if (!move_client_->wait_for_service(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_logger(), "move_point_cmd service not available.");
      return false;
    }
    auto req = std::make_shared<roarm_moveit::srv::MovePointCmd::Request>();
    req->x = x;
    req->y = y;   // NOTE: movepointcmd negates Y internally, pass +Y as-is
    req->z = z;
    auto future = move_client_->async_send_request(req);
    if (rclcpp::spin_until_future_complete(
          shared_from_this(), future, std::chrono::seconds(15))
        != rclcpp::FutureReturnCode::SUCCESS) {
      RCLCPP_ERROR(get_logger(), "move_point_cmd call failed or timed out.");
      return false;
    }
    return future.get()->success;
  }

  void setGripper(float rad)
  {
    std_msgs::msg::Float32 msg;
    msg.data = rad;
    gripper_pub_->publish(msg);
    std::this_thread::sleep_for(
      std::chrono::milliseconds(static_cast<int>(settle_s_ * 1000)));
  }

  bool cartesianMove(const std::vector<geometry_msgs::msg::Pose> & waypoints)
  {
    // Construct a transient MoveGroupInterface for Cartesian path
    auto mg_node = rclcpp::Node::make_shared("_pick_place_mg_helper");
    moveit::planning_interface::MoveGroupInterface mg(mg_node, planning_group_);

    moveit_msgs::msg::RobotTrajectory traj;
    double frac = mg.computeCartesianPath(
      waypoints, cart_step_, cart_jump_, traj);

    if (frac < cart_min_frac_) {
      RCLCPP_WARN(get_logger(),
        "Cartesian path coverage %.1f%% < required %.1f%%",
        frac * 100.0, cart_min_frac_ * 100.0);
      return false;
    }
    return mg.execute(traj) ==
      moveit::planning_interface::MoveItErrorCode::SUCCESS;
  }

  // ── main execute ────────────────────────────────────────────────────────

  void execute(std::shared_ptr<GoalHandle> gh)
  {
    const auto & goal = *gh->get_goal();
    auto result = std::make_shared<PickPlaceAction::Result>();

    // ── SCAN / SELECT TARGET ──────────────────────────────────────────────
    sendFeedback(gh, "SCAN", 5.0f);

    // Use provided target_pose if position is non-zero, else use first detection
    geometry_msgs::msg::Pose target = goal.target_pose;
    bool has_target = (target.position.x != 0.0 || target.position.y != 0.0);

    if (!has_target) {
      // Wait up to 5s for a detection
      rclcpp::Rate rate(10);
      int timeout_iters = 50;
      while (timeout_iters-- > 0 && !latest_objects_) {
        rclcpp::spin_some(shared_from_this());
        rate.sleep();
      }
      if (!latest_objects_ || latest_objects_->poses.empty()) {
        RCLCPP_ERROR(get_logger(), "No objects detected.");
        result->success = false; result->message = "No objects detected.";
        gh->succeed(result); return;
      }
      target = latest_objects_->poses[0];  // pick closest/first
    }

    sendFeedback(gh, "TARGET_SELECTED", 10.0f);

    // ── PRE-GRASP ─────────────────────────────────────────────────────────
    sendFeedback(gh, "PLAN_PREGRASP", 20.0f);
    if (!moveToPoint(target.position.x, target.position.y,
                     target.position.z + approach_z_)) {
      result->success = false; result->message = "Pre-grasp planning failed.";
      gh->succeed(result); return;
    }

    // ── OPEN GRIPPER ──────────────────────────────────────────────────────
    sendFeedback(gh, "OPEN_GRIPPER", 35.0f);
    setGripper(goal.gripper_open_rad > 0.0f ? goal.gripper_open_rad : gripper_open_);

    // ── GRASP (Cartesian straight-down) ──────────────────────────────────
    sendFeedback(gh, "PLAN_GRASP", 45.0f);
    {
      geometry_msgs::msg::Pose grasp_pose = target;
      // Use current pose as start, move straight down to object
      std::vector<geometry_msgs::msg::Pose> wps = {grasp_pose};
      if (!cartesianMove(wps)) {
        // Fallback: use /move_point_cmd
        if (!moveToPoint(target.position.x, target.position.y, target.position.z)) {
          result->success = false; result->message = "Grasp planning failed.";
          gh->succeed(result); return;
        }
      }
    }

    // ── CLOSE GRIPPER ─────────────────────────────────────────────────────
    sendFeedback(gh, "CLOSE_GRIPPER", 55.0f);
    setGripper(goal.gripper_close_rad > 0.0f ? goal.gripper_close_rad : gripper_close_);

    // ── RETREAT ───────────────────────────────────────────────────────────
    sendFeedback(gh, "PLAN_RETREAT", 65.0f);
    if (!moveToPoint(target.position.x, target.position.y,
                     target.position.z + retreat_z_)) {
      // Non-fatal: try to home anyway
      RCLCPP_WARN(get_logger(), "Retreat failed, attempting home.");
    }

    // ── TRANSPORT TO PLACE ────────────────────────────────────────────────
    sendFeedback(gh, "PLAN_PLACE", 75.0f);
    double px = goal.place_x != 0.0f ? goal.place_x : home_x_;
    double py = goal.place_y != 0.0f ? goal.place_y : home_y_;
    double pz = goal.place_z != 0.0f ? goal.place_z + approach_z_ : home_z_;
    if (!moveToPoint(px, py, pz)) {
      RCLCPP_WARN(get_logger(), "Place transport failed, returning home.");
      moveToPoint(home_x_, home_y_, home_z_);
      result->success = false; result->message = "Place transport failed.";
      gh->succeed(result); return;
    }

    // ── OPEN GRIPPER (release) ────────────────────────────────────────────
    sendFeedback(gh, "OPEN_GRIPPER_PLACE", 88.0f);
    setGripper(gripper_open_);

    // ── HOME ──────────────────────────────────────────────────────────────
    sendFeedback(gh, "HOME", 95.0f);
    moveToPoint(home_x_, home_y_, home_z_);

    sendFeedback(gh, "DONE", 100.0f);
    result->success = true;
    result->message = "Pick-and-place complete.";
    gh->succeed(result);
  }

  template<typename T>
  T get_parameter_or(const std::string & name, T default_val)
  {
    if (has_parameter(name)) {
      return get_parameter(name).get_value<T>();
    }
    declare_parameter(name, default_val);
    return default_val;
  }
};

// ── main ───────────────────────────────────────────────────────────────────

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PickPlaceNode>());
  rclcpp::shutdown();
  return 0;
}
