from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    robot_ns_arg = DeclareLaunchArgument("robot_ns", default_value="dsr01")
    robot_system_arg = DeclareLaunchArgument("robot_system", default_value="0")
    start_bringup_arg = DeclareLaunchArgument("start_bringup", default_value="true")
    bringup_mode_arg = DeclareLaunchArgument("bringup_mode", default_value="real")
    bringup_model_arg = DeclareLaunchArgument("bringup_model", default_value="e0509")
    bringup_host_arg = DeclareLaunchArgument("bringup_host", default_value="110.120.1.40")
    motions_dir_arg = DeclareLaunchArgument(
        "motions_dir",
        default_value=PathJoinSubstitution(
            [
                EnvironmentVariable("HOME"),
                "cube_solver_ws",
                "src",
                "cube_solver",
                "cube_solver",
                "motions",
            ]
        ),
    )
    auto_start_arg = DeclareLaunchArgument("auto_start", default_value="true")
    touch_empty_preamble = ExecuteProcess(
        cmd=["bash", "-lc", "touch /tmp/empty_preamble.drl"],
        shell=False,
    )

    run_dsr_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("dsr_bringup2"), "launch", "dsr_bringup2_rviz.launch.py"]
            )
        ),
        launch_arguments={
            "name": LaunchConfiguration("robot_ns"),
            "mode": LaunchConfiguration("bringup_mode"),
            "model": LaunchConfiguration("bringup_model"),
            "host": LaunchConfiguration("bringup_host"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("start_bringup")),
    )

    run_perception_drl = Node(
        package="cube_solver",
        executable="drl_block_runner",
        name="drl_block_runner_perception",
        output="screen",
        parameters=[
            {
                "robot_ns": LaunchConfiguration("robot_ns"),
                "robot_system": LaunchConfiguration("robot_system"),
                "script_path": PathJoinSubstitution(
                    [LaunchConfiguration("motions_dir"), "Perception.drl"]
                ),
                "prepend_path": "/tmp/empty_preamble.drl",
                "auto_run": True,
                "timeout_sec": 120.0,
            }
        ],
    )

    run_cube_perception = Node(
        package="cube_solver",
        executable="cube_perception_node",
        name="cube_perception",
        output="screen",
        parameters=[
            {
                "auto_start": LaunchConfiguration("auto_start"),
                "manual_mode": False,
                "face_order": ["R", "B", "L", "F", "D", "U"],
                "publish_solution_after_scan": True,
            }
        ],
    )

    run_cube_master = Node(
        package="cube_solver",
        executable="cube_master_node",
        name="cube_master_node",
        output="screen",
        parameters=[
            {
                "robot_ns": LaunchConfiguration("robot_ns"),
                "robot_system": LaunchConfiguration("robot_system"),
                "auto_execute_solution": True,
                "use_drl_chunks": True,
                "motions_dir": LaunchConfiguration("motions_dir"),
            }
        ],
    )

    return LaunchDescription(
        [
            robot_ns_arg,
            robot_system_arg,
            start_bringup_arg,
            bringup_mode_arg,
            bringup_model_arg,
            bringup_host_arg,
            motions_dir_arg,
            auto_start_arg,
            touch_empty_preamble,
            run_dsr_bringup,
            run_perception_drl,
            run_cube_perception,
            run_cube_master,
        ]
    )
