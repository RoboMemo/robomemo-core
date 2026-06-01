cmake_minimum_required(VERSION 3.10)
project(odin_ros_driver_rev1)

# Default to Release build with -O3 optimization (preserves precision)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Print build configuration
message(STATUS "========================================")
message(STATUS "Build Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CXX Flags: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "  JPEG Decoder: OpenCV cv::imdecode (libjpeg-turbo accelerated)")
message(STATUS "========================================")

# Enable comprehensive warnings
add_compile_options(
  -Wall
  -Wextra
  -Wno-unused-parameter
  -Wno-sign-compare
  -Wno-unused-function
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# ROS Version Detection
# =============================================================================
set(ODIN_ROS_VERSION 1)
if(DEFINED ENV{ROS_VERSION})
  set(ODIN_ROS_VERSION $ENV{ROS_VERSION})
endif()
message(STATUS "Building for ROS${ODIN_ROS_VERSION}")

# =============================================================================
# ODIN SDK Location (直接指定目录，可添加多个)
# =============================================================================
set(ODIN_SDK_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/module/sdk_api/include
  ${CMAKE_CURRENT_SOURCE_DIR}/module/sdk_api/sdk
  # 添加更多头文件目录...
)

set(ODIN_SDK_LIBRARY_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/module/sdk_api/build/sdk
  # 添加更多库目录...
)

set(ODIN_SDK_LIBRARIES
  ${CMAKE_CURRENT_SOURCE_DIR}/module/sdk_api/build/sdk/libodin_sdk.a
  # 添加更多库文件...
)

# =============================================================================
# Common Dependencies (shared by ROS1 and ROS2)
# =============================================================================
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

# =============================================================================
# ROS-Specific Setup
# =============================================================================
if(ODIN_ROS_VERSION EQUAL 2)
  # --- ROS2 packages ---
  add_definitions(-DODIN_ROS2=1)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(stereo_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(cv_bridge REQUIRED)
  find_package(pcl_conversions REQUIRED)
  find_package(rcpputils REQUIRED)
  find_package(rosidl_default_generators REQUIRED)

  rosidl_generate_interfaces(${PROJECT_NAME}
    "srv/GetCalibration.srv"
    "srv/SetSensorMode.srv"
  )
  rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
else()
  # --- ROS1 packages ---
  find_package(catkin REQUIRED COMPONENTS
    roscpp roslib sensor_msgs stereo_msgs nav_msgs std_msgs geometry_msgs
    cv_bridge ddynamic_reconfigure message_generation
  )
  add_service_files(FILES GetCalibration.srv SetSensorMode.srv)
  generate_messages(DEPENDENCIES std_msgs)
  catkin_package(
    CATKIN_DEPENDS roscpp roslib sensor_msgs stereo_msgs nav_msgs std_msgs
                   geometry_msgs cv_bridge ddynamic_reconfigure message_runtime
  )
  
  # Suppress ROS1 rosconsole macro warnings and notes on Ubuntu 20.04
  add_compile_options(
    -Wno-gnu-zero-variadic-macro-arguments
    -Wno-format-security
  )
endif()

# =============================================================================
# Jetson Platform Detection & Hardware JPEG Decoder (NvJpegDecoder)
# =============================================================================
# >>> Toggle: Set to ON to force software decoder for CPU testing <<<
set(FORCE_SOFTWARE_DECODER OFF)

set(USE_JETSON_DECODER OFF)
set(JETSON_DECODER_SOURCES "")

if(EXISTS "/etc/nv_tegra_release")
  message(STATUS "Jetson platform detected")
  
  # Read JetPack version from /etc/nv_tegra_release
  # Format: "# R36 (release), REVISION: 4.4, ..." or "# R35 (release), REVISION: 6.0, ..."
  file(READ "/etc/nv_tegra_release" NV_TEGRA_RELEASE_CONTENT)
  string(REGEX MATCH "R([0-9]+)" JETPACK_MAJOR_MATCH "${NV_TEGRA_RELEASE_CONTENT}")
  string(REGEX REPLACE "R([0-9]+)" "\\1" JETPACK_MAJOR_VERSION "${JETPACK_MAJOR_MATCH}")
  message(STATUS "  JetPack major version: R${JETPACK_MAJOR_VERSION}")
  
  # Check JetPack version compatibility for hardware decoder
  # R36+ (Ubuntu 22.04): Hardware decoder supported
  # R35 (Ubuntu 20.04): Hardware decoder NOT supported (OpenCV libjpeg-turbo ABI conflict)
  set(JETPACK_HW_DECODE_SUPPORTED OFF)
  if(JETPACK_MAJOR_VERSION GREATER_EQUAL 36)
    set(JETPACK_HW_DECODE_SUPPORTED ON)
    message(STATUS "  JetPack R${JETPACK_MAJOR_VERSION}: Hardware JPEG decoder SUPPORTED")
  else()
    message(STATUS "")
    message(STATUS "╔═══════════════════════════════════════════════════════════════════════╗")
    message(STATUS "║  JetPack R${JETPACK_MAJOR_VERSION} detected - Hardware JPEG decoder NOT SUPPORTED        ║")
    message(STATUS "╠═══════════════════════════════════════════════════════════════════════╣")
    message(STATUS "║  Reason: OpenCV in R35 uses libjpeg-turbo (776 bytes struct)          ║")
    message(STATUS "║          NvJpegDecoder requires libjpeg-8b (656 bytes struct)         ║")
    message(STATUS "║          ABI mismatch causes runtime crash                            ║")
    message(STATUS "╠═══════════════════════════════════════════════════════════════════════╣")
    message(STATUS "║  Solution: Upgrade to JetPack 6.x (R36+) with Ubuntu 22.04            ║")
    message(STATUS "║            Or use software decoder (auto-selected below)              ║")
    message(STATUS "╚═══════════════════════════════════════════════════════════════════════╝")
    message(STATUS "")
  endif()
  
  # Check for Jetson Multimedia API
  set(JETSON_MULTIMEDIA_API_DIR /usr/src/jetson_multimedia_api)
  
  if(NOT FORCE_SOFTWARE_DECODER AND JETPACK_HW_DECODE_SUPPORTED AND EXISTS "${JETSON_MULTIMEDIA_API_DIR}/include/NvJpegDecoder.h")
    message(STATUS "  Jetson Multimedia API found")
    
    # Find required libraries
    find_library(NVBUF_UTILS NAMES nvbufsurface PATHS /usr/lib/aarch64-linux-gnu/nvidia /usr/lib/aarch64-linux-gnu/tegra)
    find_library(NVBUF_TRANSFORM NAMES nvbufsurftransform PATHS /usr/lib/aarch64-linux-gnu/nvidia /usr/lib/aarch64-linux-gnu/tegra)
    find_library(NVJPEG_LIB NAMES nvjpeg PATHS /usr/lib/aarch64-linux-gnu/nvidia /usr/lib/aarch64-linux-gnu/tegra)
    
    if(NVBUF_UTILS AND NVBUF_TRANSFORM AND NVJPEG_LIB)
      message(STATUS "  nvbufsurface: ${NVBUF_UTILS}")
      message(STATUS "  nvbufsurftransform: ${NVBUF_TRANSFORM}")
      message(STATUS "  nvjpeg: ${NVJPEG_LIB}")
      
      set(USE_JETSON_DECODER ON)
      
      # Include paths for Jetson Multimedia API
      include_directories(
        ${JETSON_MULTIMEDIA_API_DIR}/include
        ${JETSON_MULTIMEDIA_API_DIR}/include/libjpeg-8b
        ${JETSON_MULTIMEDIA_API_DIR}/samples/common/classes
      )
      
      # Source files to compile
      set(JETSON_DECODER_SOURCES
        ${JETSON_MULTIMEDIA_API_DIR}/samples/common/classes/NvJpegDecoder.cpp
        ${JETSON_MULTIMEDIA_API_DIR}/samples/common/classes/NvElement.cpp
        ${JETSON_MULTIMEDIA_API_DIR}/samples/common/classes/NvElementProfiler.cpp
        ${JETSON_MULTIMEDIA_API_DIR}/samples/common/classes/NvBuffer.cpp
        ${JETSON_MULTIMEDIA_API_DIR}/samples/common/classes/NvLogging.cpp
      )
      
      message(STATUS "  JPEG Decoder: NvJpegDecoder Hardware Decoder")
    else()
      message(STATUS "  Required libraries not found - using software decoder")
    endif()
  else()
    message(STATUS "  Jetson Multimedia API not found - using software decoder")
  endif()
else()
  message(STATUS "Non-Jetson platform - using software decoder")
endif()

if(NOT USE_JETSON_DECODER)
  message(STATUS "  JPEG Decoder: OpenCV cv::imdecode (libjpeg-turbo)")
endif()

# Summary (use WARNING to ensure visibility in colcon build)
if(USE_JETSON_DECODER)
  message(STATUS "│  JPEG Decoder: Hardware (NvJpegDecoder)                     │")
  message(STATUS "│  Platform: Jetson R${JETPACK_MAJOR_VERSION}                                          │")
else()
  message(STATUS "│  JPEG Decoder: Software (OpenCV cv::imdecode)               │")
  if(EXISTS "/etc/nv_tegra_release")
    message(STATUS "│  Platform: Jetson R${JETPACK_MAJOR_VERSION} (HW decode not supported)               │")
  else()
    message(STATUS "│  Platform: Non-Jetson (x86/other)                           │")
  endif()
endif()

# =============================================================================
# Common Include Directories
# =============================================================================
include_directories(
  include
  include/utility
  include/camera
  ${ODIN_SDK_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)
link_directories(${ODIN_SDK_LIBRARY_DIRS})
if(ODIN_ROS_VERSION EQUAL 1)
  include_directories(${catkin_INCLUDE_DIRS})
endif()

# =============================================================================
# Source Files (使用 aux_source_directory 自动收集)
# =============================================================================
aux_source_directory(src POST_PROCESS_SOURCES)
list(FILTER POST_PROCESS_SOURCES EXCLUDE REGEX "odin_ros_driver_node\\.cpp$")

# =============================================================================
# Executables
# =============================================================================

# --- odin_ros_driver_node ---
add_executable(odin_ros_driver_node src/odin_ros_driver_node.cpp)
target_link_libraries(odin_ros_driver_node ${ODIN_SDK_LIBRARIES} OpenSSL::SSL OpenSSL::Crypto Threads::Threads yaml-cpp)

# --- post_process_node ---
if(USE_JETSON_DECODER)
  add_executable(post_process_node ${POST_PROCESS_SOURCES} ${JETSON_DECODER_SOURCES})
  target_compile_definitions(post_process_node PRIVATE USE_JETSON_DECODER)
  target_link_libraries(post_process_node ${OpenCV_LIBS} ${PCL_LIBRARIES} yaml-cpp ${NVBUF_UTILS} ${NVBUF_TRANSFORM} ${NVJPEG_LIB})
else()
  add_executable(post_process_node ${POST_PROCESS_SOURCES})
  target_link_libraries(post_process_node ${OpenCV_LIBS} ${PCL_LIBRARIES} yaml-cpp)
endif()
target_compile_definitions(post_process_node PRIVATE ROS_VERSION_MAJOR=${ODIN_ROS_VERSION})

# =============================================================================
# ROS-Specific Linking & Install
# =============================================================================
if(ODIN_ROS_VERSION EQUAL 2)
  # ROS2 dependencies
  ament_target_dependencies(odin_ros_driver_node rclcpp geometry_msgs nav_msgs sensor_msgs std_msgs)
  ament_target_dependencies(post_process_node rclcpp sensor_msgs stereo_msgs std_msgs cv_bridge pcl_conversions rcpputils)
  target_link_libraries(odin_ros_driver_node ${cpp_typesupport_target})
  target_link_libraries(post_process_node ${cpp_typesupport_target})

  # ROS2 install
  install(TARGETS odin_ros_driver_node post_process_node DESTINATION lib/${PROJECT_NAME})
  install(DIRECTORY launch rviz DESTINATION share/${PROJECT_NAME})
  ament_package()
else()
  # ROS1 dependencies
  target_link_libraries(odin_ros_driver_node ${catkin_LIBRARIES})
  target_link_libraries(post_process_node ${catkin_LIBRARIES})
  add_dependencies(odin_ros_driver_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  add_dependencies(post_process_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

  # ROS1 install
  install(TARGETS odin_ros_driver_node post_process_node RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
  install(DIRECTORY launch rviz DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
endif()
