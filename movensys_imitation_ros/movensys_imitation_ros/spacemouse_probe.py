"""
spacemouse_probe: read the SpaceMouse axes and keys off ``/spacenav/joy`` for calibration.

Bring-up helper (doc/4_sim.md, "SpaceMouse input"). Prints, five times a second, every axis
deflected beyond ``--threshold`` with its sign, and every key press with its index, so the
operator can push the puck one way at a time and press the key meant as dead-man. Ctrl-C
prints a summary: per-axis extremes and the keys seen. Needs ``spacenav_node`` running
(``ros2 launch movensys_imitation_ros spacenav.launch.py``).

    ros2 run movensys_imitation_ros spacemouse_probe            # or python3 -m ...
"""

from __future__ import annotations

import argparse
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

AXIS_NAMES = ("0 x (away/toward)", "1 y (left/right)", "2 z (up/down)",
              "3 roll (tilt sideways)", "4 pitch (tilt away/toward)", "5 yaw (twist)")


class Probe(Node):
    def __init__(self, topic: str, threshold: float) -> None:
        super().__init__("spacemouse_probe")
        self.threshold = threshold
        self.axes: list[float] = []
        self.buttons: list[int] = []
        self.lo: list[float] = []
        self.hi: list[float] = []
        self.pressed: set[int] = set()
        self.count = 0
        self.create_subscription(Joy, topic, self._on_joy, 10)
        self.create_timer(0.2, self._report)
        self.get_logger().info(
            f"listening on {topic}; push the puck one axis at a time, press the key you want "
            f"as dead-man; Ctrl-C for the summary"
        )

    def _on_joy(self, msg: Joy) -> None:
        self.count += 1
        axes = list(msg.axes)
        if len(self.lo) < len(axes):
            self.lo += [math.inf] * (len(axes) - len(self.lo))
            self.hi += [-math.inf] * (len(axes) - len(self.hi))
        for i, v in enumerate(axes):
            self.lo[i] = min(self.lo[i], v)
            self.hi[i] = max(self.hi[i], v)
        buttons = list(msg.buttons)
        for i, b in enumerate(buttons):
            was = self.buttons[i] if i < len(self.buttons) else 0
            if b and not was:
                self.pressed.add(i)
                print(f"key {i} pressed   (buttons now has {len(buttons)} entries)", flush=True)
            elif was and not b:
                print(f"key {i} released", flush=True)
        self.axes, self.buttons = axes, buttons

    def _report(self) -> None:
        active = [
            f"axis {AXIS_NAMES[i] if i < len(AXIS_NAMES) else i} = {v:+.2f}"
            for i, v in enumerate(self.axes) if abs(v) > self.threshold
        ]
        if active:
            print("   ".join(active), flush=True)

    def summary(self) -> None:
        print(f"\n{self.count} Joy messages; {len(self.axes)} axes, {len(self.buttons)} buttons")
        for i in range(len(self.lo)):
            name = AXIS_NAMES[i] if i < len(AXIS_NAMES) else str(i)
            print(f"  axis {name:<22} min {self.lo[i]:+.2f}  max {self.hi[i]:+.2f}")
        print(f"  keys pressed: {sorted(self.pressed) or 'none'}")


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--topic", default="/spacenav/joy")
    parser.add_argument("--threshold", type=float, default=0.2)
    ns, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args)
    node = Probe(ns.topic, ns.threshold)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
