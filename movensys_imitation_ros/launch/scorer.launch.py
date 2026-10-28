"""Launch pusht_scorer from the project venv with the task and scorer parameters (plan P2.3).

The scorer needs numpy 2, OpenCV and shapely, so it runs as ``python -m`` with the ``python``
found on PATH (the venv, when env.bash was sourced) instead of the colcon-installed script.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PythonExpression


def generate_launch_description():
    share = get_package_share_directory("movensys_imitation_ros")
    config = os.path.join(share, "config", "dobot_cr3a")
    mode = LaunchConfiguration("mode")
    use_sim_time = PythonExpression(["'", mode, "' != 'real'"])
    scorer_mode = PythonExpression(["'sim' if '", mode, "' != 'real' else 'real'"])
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "mode", default_value="sim", description="sim | hil | real (real: wall clock)"
            ),
            ExecuteProcess(
                cmd=[
                    "python", "-m", "movensys_imitation_ros.pusht_scorer", "--ros-args",
                    "--params-file", os.path.join(config, "task.yaml"),
                    "--params-file", os.path.join(config, "scorer.yaml"),
                    "-p", ["use_sim_time:=", use_sim_time],
                    "-p", ["mode:=", scorer_mode],
                ],
                output="screen",
            ),
        ]
    )
