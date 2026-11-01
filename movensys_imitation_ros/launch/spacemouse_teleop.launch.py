"""Free 6-DoF SpaceMouse teleop through MoveIt Servo (TWIST mode): driver plus teleop node.

For parking the rod at push height and reach checks. Needs sim_bridge and moveit.launch.py
up and NO other Servo writer (planar_guard, quest_servo_teleop); the node holds with
``hold:pose_writer`` while one exists. Stop it before launching the guard.

    ros2 launch movensys_imitation_ros spacemouse_teleop.launch.py mode:=sim
    ros2 launch movensys_imitation_ros spacemouse_teleop.launch.py mode:=real motion_mode:=planar
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("movensys_imitation_ros")
    config = os.path.join(share, "config", "dobot_cr3a")
    mode = LaunchConfiguration("mode")
    use_sim_time = PythonExpression(["'", mode, "' != 'real'"])
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "mode", default_value="sim", description="sim | hil | real (real: wall clock)"
            ),
            DeclareLaunchArgument(
                "motion_mode", default_value="full",
                description="full | translation | rotation | planar | vertical | yaw",
            ),
            DeclareLaunchArgument(
                "driver", default_value="true", description="also start spacenav_node"
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(share, "launch", "spacenav.launch.py")),
                condition=IfCondition(LaunchConfiguration("driver")),
            ),
            Node(
                package="movensys_imitation_ros",
                executable="spacemouse_teleop",
                name="spacemouse_teleop",
                output="screen",
                parameters=[
                    os.path.join(config, "task.yaml"),
                    os.path.join(config, "spacemouse_teleop.yaml"),
                    {
                        "use_sim_time": use_sim_time,
                        "motion_mode": LaunchConfiguration("motion_mode"),
                    },
                ],
            ),
        ]
    )
