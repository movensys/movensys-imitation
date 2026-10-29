"""Start the 3Dconnexion SpaceMouse driver (``spacenav_node`` from ros-jazzy-spacenav).

Needs ``spacenavd`` running on the host (``systemctl is-active spacenavd``). Publishes
``/spacenav/joy`` (axes x, y, z, roll, pitch, yaw in about -1..1, buttons as numbered by
spacenavd), ``/spacenav/twist`` and the two offsets. ``zero_when_static`` makes the node
publish zeros about every 30 ms while the puck is at rest, so the consumers' Joy watchdogs
stay quiet. Included by guard.launch.py, collect.launch.py and spacemouse_teleop.launch.py
with ``device:=spacemouse``; start it alone to inspect the device:

    ros2 launch movensys_imitation_ros spacenav.launch.py
    ros2 topic echo /spacenav/joy
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="spacenav",
                executable="spacenav_node",
                name="spacenav",
                output="screen",
                parameters=[{"zero_when_static": True}],
            ),
        ]
    )
