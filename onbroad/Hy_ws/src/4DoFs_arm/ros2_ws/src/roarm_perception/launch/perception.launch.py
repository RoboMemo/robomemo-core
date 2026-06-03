from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory("roarm_perception")
    params = os.path.join(pkg, "config", "perception_params.yaml")

    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=params),

        Node(
            package="roarm_perception",
            executable="perception_node",
            name="perception_node",
            output="screen",
            parameters=[LaunchConfiguration("params_file")],
        ),
    ])
