import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    # ── package paths ────────────────────────────────────────────────────────
    bringup_pkg     = get_package_share_directory("roarm_bringup")
    roarm_moveit_pkg = get_package_share_directory("roarm_moveit")
    roarm_scene_pkg  = get_package_share_directory("roarm_scene")

    # ── launch arguments ─────────────────────────────────────────────────────
    camera_mode_arg = DeclareLaunchArgument(
        "camera_mode",
        default_value="eye_to_hand",
        description="eye_to_hand (default) or eye_in_hand",
    )
    use_hardware_arg = DeclareLaunchArgument(
        "use_real_hardware",
        default_value="true",
        description="Launch hardware driver node.",
    )
    serial_port_arg = DeclareLaunchArgument(
        "serial_port",
        default_value="/dev/ttyUSB0",
        description="Serial port for roarm_driver.",
    )
    enable_octomap_arg = DeclareLaunchArgument(
        "enable_octomap",
        default_value="true",
        description="Enable live octomap collision avoidance.",
    )
    enable_rviz_arg = DeclareLaunchArgument(
        "enable_rviz",
        default_value="true",
        description="Launch RViz2.",
    )

    camera_mode      = LaunchConfiguration("camera_mode")
    use_real_hardware = LaunchConfiguration("use_real_hardware")
    serial_port      = LaunchConfiguration("serial_port")
    enable_octomap   = LaunchConfiguration("enable_octomap")
    enable_rviz      = LaunchConfiguration("enable_rviz")

    # ── MoveIt config with camera URDF overlay ───────────────────────────────
    sensors_yaml = os.path.join(roarm_scene_pkg, "config", "sensors_3d_orbbec.yaml")

    moveit_config = (
        MoveItConfigsBuilder("roarm_description", package_name="roarm_moveit")
        .robot_description(
            file_path=os.path.join(bringup_pkg, "urdf", "roarm_with_camera.urdf.xacro"),
            mappings={
                "camera_mode": "eye_to_hand",   # passed as string; runtime value
                "initial_positions_file": os.path.join(
                    roarm_moveit_pkg, "config", "initial_positions.yaml"
                ),
            },
        )
        .robot_description_semantic(
            file_path=os.path.join(roarm_moveit_pkg, "config", "roarm_description.srdf")
        )
        .robot_description_kinematics(
            file_path=os.path.join(roarm_moveit_pkg, "config", "kinematics.yaml")
        )
        .joint_limits(
            file_path=os.path.join(roarm_moveit_pkg, "config", "joint_limits.yaml")
        )
        .trajectory_execution(
            file_path=os.path.join(roarm_moveit_pkg, "config", "moveit_controllers.yaml")
        )
        .planning_pipelines(pipelines=["ompl"])
        .sensors_3d(sensors_yaml)
        .to_moveit_configs()
    )

    # ── LD_LIBRARY_PATH: add orbbec_ros2 install/lib for SDK .so files ───────
    orbbec_lib_env = SetEnvironmentVariable(
        name="LD_LIBRARY_PATH",
        value=[
            os.path.join(
                get_package_share_directory("orbbec_ros2"), "..", "..", "lib"
            ),
            ":",
            EnvironmentVariable("LD_LIBRARY_PATH", default_value=""),
        ],
    )

    # ── robot_state_publisher ─────────────────────────────────────────────────
    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description],
    )

    # ── ros2_control_node ─────────────────────────────────────────────────────
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description,
            os.path.join(roarm_moveit_pkg, "config", "ros2_controllers.yaml"),
        ],
        output="screen",
    )

    # ── controller spawners ───────────────────────────────────────────────────
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    hand_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["hand_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    # ── roarm_driver (hardware only) ──────────────────────────────────────────
    roarm_driver_node = Node(
        package="roarm_driver",
        executable="roarm_driver",
        name="roarm_driver",
        output="screen",
        parameters=[{"serial_port": serial_port}],
        condition=IfCondition(use_real_hardware),
    )

    # ── move_group ────────────────────────────────────────────────────────────
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )

    # ── Orbbec camera ─────────────────────────────────────────────────────────
    camera_node = Node(
        package="orbbec_ros2",
        executable="orbbec_camera_node",
        name="orbbec_camera_node",
        output="screen",
        parameters=[
            os.path.join(get_package_share_directory("orbbec_ros2"),
                         "config", "camera_params.yaml")
        ],
        condition=IfCondition(use_real_hardware),
    )

    # ── Perception ────────────────────────────────────────────────────────────
    perception_node = Node(
        package="roarm_perception",
        executable="perception_node",
        name="perception_node",
        output="screen",
        parameters=[
            os.path.join(get_package_share_directory("roarm_perception"),
                         "config", "perception_params.yaml")
        ],
    )

    # ── Scene manager ─────────────────────────────────────────────────────────
    scene_node = Node(
        package="roarm_scene",
        executable="scene_manager_node",
        name="scene_manager_node",
        output="screen",
        parameters=[
            os.path.join(get_package_share_directory("roarm_scene"),
                         "config", "scene_objects.yaml")
        ],
    )

    # ── Pick-and-place ────────────────────────────────────────────────────────
    pick_place_node = Node(
        package="roarm_pick_place",
        executable="pick_place_node",
        name="pick_place_node",
        output="screen",
        parameters=[
            os.path.join(get_package_share_directory("roarm_pick_place"),
                         "config", "pick_place_params.yaml")
        ],
    )

    # ── movepointcmd + setgrippercmd (from roarm_moveit_cmd) ─────────────────
    movepointcmd_node = Node(
        package="roarm_moveit_cmd",
        executable="movepointcmd",
        name="movepointcmd",
        output="screen",
    )
    setgrippercmd_node = Node(
        package="roarm_moveit_cmd",
        executable="setgrippercmd",
        name="setgrippercmd",
        output="screen",
    )

    # ── RViz ──────────────────────────────────────────────────────────────────
    rviz_config = os.path.join(roarm_moveit_pkg, "config", "moveit.rviz")
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
        ],
        condition=IfCondition(enable_rviz),
    )

    return LaunchDescription([
        # Args
        camera_mode_arg,
        use_hardware_arg,
        serial_port_arg,
        enable_octomap_arg,
        enable_rviz_arg,
        # Env
        orbbec_lib_env,
        # Core
        rsp_node,
        ros2_control_node,
        joint_state_broadcaster_spawner,
        hand_controller_spawner,
        gripper_controller_spawner,
        roarm_driver_node,
        move_group_node,
        # Vision
        camera_node,
        perception_node,
        # Scene
        scene_node,
        # Control
        movepointcmd_node,
        setgrippercmd_node,
        pick_place_node,
        # UI
        rviz_node,
    ])
