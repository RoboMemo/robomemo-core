from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory("orbbec_ros2")
    params_file = os.path.join(pkg, "config", "camera_params.yaml")

    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=params_file),

        Node(
            package="orbbec_ros2",
            executable="orbbec_camera_node",
            name="orbbec_camera_node",
            output="screen",
            parameters=[LaunchConfiguration("params_file")],
        ),
    ])
