cmake_minimum_required(VERSION 3.22)
project(roarm_pick_place)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_core REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(roarm_moveit REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate action interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/PickPlace.action"
  DEPENDENCIES geometry_msgs
)

add_executable(pick_place_node
  src/pick_place_node.cpp
)

ament_target_dependencies(pick_place_node
  rclcpp
  rclcpp_action
  geometry_msgs
  std_msgs
  std_srvs
  moveit_ros_planning_interface
  moveit_core
  tf2_ros
  tf2_geometry_msgs
  roarm_moveit
)

rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(pick_place_node "${cpp_typesupport_target}")

install(TARGETS pick_place_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
