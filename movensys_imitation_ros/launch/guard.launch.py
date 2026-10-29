"""Launch the planar guard with the dobot_cr3a task and guard parameters (plan P2.2).

``device:=quest`` (default) or ``device:=spacemouse`` appends
``config/dobot_cr3a/joy_<device>.yaml`` after ``guard.yaml`` (stick topic, axes, deadzone,
dead-man). With the SpaceMouse the
``spacenav_node`` driver is included too (``driver:=false`` if it already runs elsewhere).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("movensys_imitation_ros")
    config = os.path.join(share, "config", "dobot_cr3a")
    mode = LaunchConfiguration("mode")
    use_sim_time = PythonExpression(["'", mode, "' != 'real'"])
    device = LaunchConfiguration("device")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "mode", default_value="sim", description="sim | hil | real (real: wall clock)"
            ),
            DeclareLaunchArgument(
                "source", default_value="stick", description="guard input: stick | pose"
            ),
            DeclareLaunchArgument(
                "device", default_value="quest", description="stick device: quest | spacemouse"
            ),
            DeclareLaunchArgument(
                "driver", default_value="true",
                description="also start spacenav_node when device is spacemouse",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(share, "launch", "spacenav.launch.py")),
                condition=IfCondition(PythonExpression([
                    "'", device, "' == 'spacemouse' and '", LaunchConfiguration("driver"),
                    "' == 'true'",
                ])),
            ),
            Node(
                package="movensys_imitation_ros",
                executable="planar_guard",
                name="planar_guard",
                output="screen",
                parameters=[
                    os.path.join(config, "task.yaml"),
                    os.path.join(config, "guard.yaml"),
                    PathJoinSubstitution([config, ["joy_", device, ".yaml"]]),
                    {"use_sim_time": use_sim_time, "source": LaunchConfiguration("source")},
                ],
            ),
        ]
    )
