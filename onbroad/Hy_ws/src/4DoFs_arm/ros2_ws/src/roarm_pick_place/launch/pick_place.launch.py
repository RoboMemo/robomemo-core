from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory("roarm_pick_place")
    params = os.path.join(pkg, "config", "pick_place_params.yaml")

    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=params),

        Node(
            package="roarm_pick_place",
            executable="pick_place_node",
            name="pick_place_node",
            output="screen",
            parameters=[LaunchConfiguration("params_file")],
        ),
    ])
