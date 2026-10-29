"""Launch pusht_scorer from the project venv with the task and scorer parameters (plan P2.3).

The scorer needs numpy 2, OpenCV and shapely, so it runs as ``python -m`` with the ``python``
found on PATH (the venv, when env.bash was sourced) instead of the colcon-installed script. The
``mode`` and ``use_sim_time`` overrides go in a params file passed last (``launch_support``).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction

from movensys_imitation_ros.launch_support import sim_time_and_mode, venv_process


def launch_setup(context):
    share = get_package_share_directory("movensys_imitation_ros")
    config = os.path.join(share, "config", "dobot_cr3a")
    files = [os.path.join(config, "task.yaml"), os.path.join(config, "scorer.yaml")]
    return [venv_process(context, "pusht_scorer", files, sim_time_and_mode(context))]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "mode", default_value="sim", description="sim | hil | real (real: wall clock)"
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
