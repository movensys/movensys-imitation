"""
collect_keys: keyboard front end for the episode manager (P2.4).

The manager runs under ``ros2 launch`` without a terminal, so keys live in this small client:
``s`` start the next episode, ``x`` stop and save the current one, ``d`` stop and discard,
``a`` abort the run, ``q`` quit. Run from any sourced shell::

    python -m movensys_imitation_ros.collect_keys
"""

from __future__ import annotations

import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

KEYS = {"s": "start", "x": "stop", "d": "discard", "a": "abort"}


class Keys(Node):
    def __init__(self) -> None:
        super().__init__("collect_keys")
        self.clients_ = {cmd: self.create_client(Trigger, f"/episode_manager/{cmd}")
                         for cmd in KEYS.values()}
        self.state = ""
        self.create_subscription(String, "/episode_manager/state", self._on_state, 10)

    def _on_state(self, msg: String) -> None:
        if msg.data != self.state:
            self.state = msg.data
            print(f"\r[manager] {self.state}", end="\r\n", flush=True)

    def send(self, cmd: str) -> None:
        client = self.clients_[cmd]
        if not client.wait_for_service(timeout_sec=2.0):
            print(f"\r{cmd}: manager service not available", end="\r\n", flush=True)
            return
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        res = future.result()
        text = f"{res.success} {res.message}" if res is not None else "no reply"
        print(f"\r{cmd}: {text}", end="\r\n", flush=True)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Keys()
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    print("s start | x stop+save | d discard | a abort | q quit", end="\r\n", flush=True)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == "q":
                    break
                if key in KEYS:
                    node.send(KEYS[key])
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
