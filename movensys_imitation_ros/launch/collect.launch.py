"""Launch the collection stack (plan P2.4): guard, scorer, recorder and episode manager.

Preconditions as for guard.launch.py: scene playing, sim_bridge and MoveIt up, the rod parked
at push height, the stick device connected (``device:=quest`` or ``device:=spacemouse``, which
also starts ``spacenav_node``) and ``quest_servo_teleop`` / ``spacemouse_teleop`` NOT running.
The scorer, the recorder and the manager need the venv, so they run as ``python -m`` with the
``python`` on PATH (env.bash); their launch overrides go in a params file passed last
(``launch_support``). Keys come from ``python -m movensys_imitation_ros.collect_keys`` in
another shell.

    ros2 launch movensys_imitation_ros collect.launch.py mode:=sim operator:=name device:=spacemouse
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node

from movensys_imitation_ros.launch_support import sim_time_and_mode, venv_process


def launch_setup(context):
    share = get_package_share_directory("movensys_imitation_ros")
    config = os.path.join(share, "config", "dobot_cr3a")
    task_yaml = os.path.join(config, "task.yaml")
    common = sim_time_and_mode(context)
    return [
        venv_process(
            context, "pusht_scorer", [task_yaml, os.path.join(config, "scorer.yaml")], common
        ),
        venv_process(
            context, "episode_recorder", [task_yaml, os.path.join(config, "recorder.yaml")],
            common,
        ),
        venv_process(
            context, "episode_manager", [task_yaml, os.path.join(config, "recorder.yaml")],
            {
                **common,
                "operator": LaunchConfiguration("operator"),
                "teleop_source": LaunchConfiguration("source"),
                "input_device": LaunchConfiguration("device"),
                "seed_start": LaunchConfiguration("seed_start"),
                "n_episodes": LaunchConfiguration("n_episodes"),
                "auto_start": LaunchConfiguration("auto_start"),
            },
        ),
    ]


def generate_launch_description():
    share = get_package_share_directory("movensys_imitation_ros")
    config = os.path.join(share, "config", "dobot_cr3a")
    mode = LaunchConfiguration("mode")
    use_sim_time = PythonExpression(["'", mode, "' != 'real'"])
    device = LaunchConfiguration("device")
    return LaunchDescription(
        [
            DeclareLaunchArgument("mode", default_value="sim", description="sim | hil | real"),
            DeclareLaunchArgument(
                "source", default_value="stick", description="guard input while recording"
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
            DeclareLaunchArgument("operator", default_value="", description="sidecar metadata"),
            DeclareLaunchArgument("seed_start", default_value="1000"),
            DeclareLaunchArgument("n_episodes", default_value="10"),
            DeclareLaunchArgument("auto_start", default_value="false"),
            Node(
                package="movensys_imitation_ros",
                executable="planar_guard",
                name="planar_guard",
                output="screen",
                parameters=[
                    os.path.join(config, "task.yaml"),
                    os.path.join(config, "guard.yaml"),
                    PathJoinSubstitution([config, ["joy_", device, ".yaml"]]),
                    {
                        "use_sim_time": use_sim_time,
                        "source": LaunchConfiguration("source"),
                        "latch_mode": PythonExpression(
                            ["'nominal' if '", mode, "' == 'sim' else 'measured'"]
                        ),
                    },
                ],
            ),
            OpaqueFunction(function=launch_setup),
        ]
    )
