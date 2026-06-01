cmake_minimum_required(VERSION 3.22)
project(orbbec_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Locate OrbbecSDK from the submodule (relative to this package) ──────────
# Package is at: ros2_ws/src/orbbec_ros2/
# Submodule is at: ../../../submodules/OrbbecSDK  (from repo root)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(OrbbecSDK_CONFIG_FILE "${REPO_ROOT}/submodules/OrbbecSDK/OrbbecSDKConfig.cmake")

if(NOT EXISTS "${OrbbecSDK_CONFIG_FILE}")
  # Fallback: try the local ~/OrbbecSDK clone
  set(OrbbecSDK_CONFIG_FILE "$ENV{HOME}/OrbbecSDK/OrbbecSDKConfig.cmake")
endif()

if(NOT EXISTS "${OrbbecSDK_CONFIG_FILE}")
  message(FATAL_ERROR "OrbbecSDKConfig.cmake not found. Expected at ${OrbbecSDK_CONFIG_FILE}")
endif()
include("${OrbbecSDK_CONFIG_FILE}")

# ── ROS 2 dependencies ────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# ── Camera node executable ────────────────────────────────────────────────────
add_executable(orbbec_camera_node
  src/orbbec_camera_node.cpp
)

target_include_directories(orbbec_camera_node PRIVATE
  include
  ${OrbbecSDK_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(orbbec_camera_node
  OrbbecSDK::OrbbecSDK
  ${OpenCV_LIBS}
)

ament_target_dependencies(orbbec_camera_node
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2_ros
  tf2
  image_transport
  cv_bridge
)

# Make sure the SDK shared libraries are found at runtime via RPATH
set_target_properties(orbbec_camera_node PROPERTIES
  INSTALL_RPATH "$ORIGIN/../lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS orbbec_camera_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install OrbbecSDK runtime .so files alongside our node
if(DEFINED OrbbecSDK_DLL_FILES)
  install(FILES ${OrbbecSDK_DLL_FILES}
    DESTINATION lib
  )
endif()
if(DEFINED OrbbecSDK_RUNTIME_RESOURCE_FILES)
  install(FILES ${OrbbecSDK_RUNTIME_RESOURCE_FILES}
    DESTINATION lib
  )
endif()

install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
