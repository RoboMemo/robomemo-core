cmake_minimum_required(VERSION 3.22)
project(roarm_scene)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_srvs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

add_executable(scene_manager_node
  src/scene_manager_node.cpp
)

ament_target_dependencies(scene_manager_node
  rclcpp
  std_srvs
  moveit_ros_planning_interface
  moveit_core
  moveit_msgs
  shape_msgs
  geometry_msgs
)

install(TARGETS scene_manager_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
