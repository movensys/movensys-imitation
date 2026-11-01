"""Start the 3Dconnexion SpaceMouse driver (``spacenav_node`` from ros-jazzy-spacenav).

Needs ``spacenavd`` running on the host (``systemctl is-active spacenavd``). The driver
publishes its Joy on every 1 ms tick (measured 1000 Hz at rest, 2026-10-29), which costs a
Python subscriber about half a core, so the driver's Joy is remapped to ``/spacenav/joy_raw``
and a ``topic_tools`` throttle republishes it as ``/spacenav/joy`` at ``joy_hz`` (60 Hz, the
Quest controller's rate; the consumers tick at 50 Hz with ``joy_stale`` 0.2 s). Measured
steady-state guard CPU on the lab box: 9 % idle, 17 % at 60 Hz, 25 % at 100 Hz, 50 % on
the raw stream. Axes are x, y, z,
roll, pitch, yaw in about -1..1 (raw counts over ``full_scale`` 512), zero at rest
(``zero_when_static``); buttons as numbered by spacenavd, the array growing to the highest
key pressed so far. ``/spacenav/twist`` and the two offsets stay unthrottled (about 32 Hz at
rest, event-driven in motion). Included by guard.launch.py, collect.launch.py and
spacemouse_teleop.launch.py with ``device:=spacemouse``; start it alone to inspect the device:

    ros2 launch movensys_imitation_ros spacenav.launch.py
    ros2 run movensys_imitation_ros spacemouse_probe
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "joy_hz", default_value="60", description="/spacenav/joy rate after throttling"
            ),
            Node(
                package="spacenav",
                executable="spacenav_node",
                name="spacenav",
                output="screen",
                parameters=[{"zero_when_static": True}],
                remappings=[("spacenav/joy", "spacenav/joy_raw")],
            ),
            Node(
                package="topic_tools",
                executable="throttle",
                name="spacenav_joy_throttle",
                output="screen",
                # Wall clock on purpose: no use_sim_time here, Isaac runs below real time.
                arguments=[
                    "messages", "/spacenav/joy_raw", LaunchConfiguration("joy_hz"),
                    "/spacenav/joy",
                ],
            ),
        ]
    )
