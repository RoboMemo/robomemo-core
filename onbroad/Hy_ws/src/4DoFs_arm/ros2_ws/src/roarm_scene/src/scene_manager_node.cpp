#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <geometry_msgs/msg/pose.hpp>

class SceneManagerNode : public rclcpp::Node
{
public:
  SceneManagerNode()
  : rclcpp::Node("scene_manager_node",
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true))
  {
    // Parameters
    table_x_  = get_param("table_x",      0.25);
    table_y_  = get_param("table_y",      0.00);
    table_z_  = get_param("table_z",     -0.005);
    table_sx_ = get_param("table_size_x", 0.60);
    table_sy_ = get_param("table_size_y", 0.80);
    table_sz_ = get_param("table_size_z", 0.01);
    add_wall_ = get_param("add_back_wall", true);
    wall_x_   = get_param("wall_x",  -0.40);
    wall_y_   = get_param("wall_y",   0.00);
    wall_z_   = get_param("wall_z",   0.20);
    wall_sx_  = get_param("wall_size_x", 0.02);
    wall_sy_  = get_param("wall_size_y", 1.00);
    wall_sz_  = get_param("wall_size_z", 0.60);

    clear_srv_ = create_service<std_srvs::srv::Trigger>(
      "/scene/clear_scene",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res)
      {
        psi_.removeCollisionObjects({"table", "back_wall"});
        res->success = true; res->message = "Scene cleared.";
      });

    // Give move_group time to start before adding objects
    using namespace std::chrono_literals;
    init_timer_ = create_wall_timer(3s, [this]() {
      init_timer_->cancel();
      addStaticObjects();
    });

    RCLCPP_INFO(get_logger(), "SceneManagerNode ready. Will add objects in 3s.");
  }

private:
  double table_x_, table_y_, table_z_, table_sx_, table_sy_, table_sz_;
  bool   add_wall_;
  double wall_x_,  wall_y_,  wall_z_,  wall_sx_,  wall_sy_,  wall_sz_;

  moveit::planning_interface::PlanningSceneInterface psi_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_srv_;
  rclcpp::TimerBase::SharedPtr init_timer_;

  void addStaticObjects()
  {
    std::vector<moveit_msgs::msg::CollisionObject> objects;

    // ── Table ────────────────────────────────────────────────────────────
    objects.push_back(makeBox("table", "base_link",
      table_x_, table_y_, table_z_ - table_sz_ / 2.0,
      table_sx_, table_sy_, table_sz_));

    // ── Back wall ─────────────────────────────────────────────────────────
    if (add_wall_) {
      objects.push_back(makeBox("back_wall", "base_link",
        wall_x_, wall_y_, wall_z_,
        wall_sx_, wall_sy_, wall_sz_));
    }

    psi_.applyCollisionObjects(objects);
    RCLCPP_INFO(get_logger(), "Static collision objects added (%zu).", objects.size());
  }

  moveit_msgs::msg::CollisionObject makeBox(
    const std::string & id, const std::string & frame_id,
    double cx, double cy, double cz,
    double sx, double sy, double sz)
  {
    moveit_msgs::msg::CollisionObject obj;
    obj.header.frame_id = frame_id;
    obj.header.stamp    = now();
    obj.id              = id;

    shape_msgs::msg::SolidPrimitive prim;
    prim.type = shape_msgs::msg::SolidPrimitive::BOX;
    prim.dimensions = {sx, sy, sz};

    geometry_msgs::msg::Pose pose;
    pose.position.x  = cx;
    pose.position.y  = cy;
    pose.position.z  = cz;
    pose.orientation.w = 1.0;

    obj.primitives.push_back(prim);
    obj.primitive_poses.push_back(pose);
    obj.operation = moveit_msgs::msg::CollisionObject::ADD;
    return obj;
  }

  template<typename T>
  T get_param(const std::string & name, T def)
  {
    if (!has_parameter(name)) declare_parameter(name, def);
    return get_parameter(name).get_value<T>();
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SceneManagerNode>());
  rclcpp::shutdown();
  return 0;
}
