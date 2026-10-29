"""Launch the planar guard with the dobot_cr3a task and guard parameters (plan P2.2)."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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
                "source", default_value="stick", description="guard input: stick | pose"
            ),
            Node(
                package="movensys_imitation_ros",
                executable="planar_guard",
                name="planar_guard",
                output="screen",
                parameters=[
                    os.path.join(config, "task.yaml"),
                    os.path.join(config, "guard.yaml"),
                    {"use_sim_time": use_sim_time, "source": LaunchConfiguration("source")},
                ],
            ),
        ]
    )
