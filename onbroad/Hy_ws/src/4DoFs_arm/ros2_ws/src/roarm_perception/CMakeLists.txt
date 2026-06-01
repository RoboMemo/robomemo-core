cmake_minimum_required(VERSION 3.22)
project(roarm_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS
  common filters segmentation search kdtree features io)

add_executable(perception_node
  src/perception_node.cpp
)

target_include_directories(perception_node PRIVATE
  include
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(perception_node
  ${PCL_LIBRARIES}
)

ament_target_dependencies(perception_node
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  std_srvs
  tf2_ros
  tf2_sensor_msgs
  tf2_geometry_msgs
  pcl_conversions
)

install(TARGETS perception_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
