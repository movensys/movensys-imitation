"""
collect_keys: keyboard front end for the episode manager (P2.4).

The manager runs under ``ros2 launch`` without a terminal, so keys live in this small client:
``s`` start the next episode, ``x`` stop and save the current one, ``d`` stop and discard,
``a`` abort the run, ``r`` then ``y`` delete the last saved episode and record its seed again
(between episodes only), ``q`` quit. On the real robot the manager waits in ``reset:place`` for the
T to be put on its start pose (``score_viewer`` shows it); ``s`` there means "placed, check it"
and ``d`` gives the episode up. Run from any sourced shell::

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
REDO = "redo"  # key r, confirmed with y: it deletes a saved episode


class Keys(Node):
    def __init__(self) -> None:
        super().__init__("collect_keys")
        self.clients_ = {cmd: self.create_client(Trigger, f"/episode_manager/{cmd}")
                         for cmd in (*KEYS.values(), REDO)}
        self.state = ""
        self.create_subscription(String, "/episode_manager/state", self._on_state, 10)

    def _on_state(self, msg: String) -> None:
        if msg.data != self.state:
            self.state = msg.data
            print(f"\r[manager] {self.state}", end="\r\n", flush=True)
            if self.state == "reset:place":
                print("\rplace the T on the orange outline (score_viewer), hand away, then s "
                      "(d gives up)", end="\r\n", flush=True)

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
    print("s start | x stop+save | d discard | r redo last | a abort | q quit", end="\r\n",
          flush=True)
    confirm_redo = False
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if confirm_redo:
                    confirm_redo = False
                    if key == "y":
                        node.send(REDO)
                    else:
                        print("\rredo cancelled", end="\r\n", flush=True)
                    continue
                if key == "q":
                    break
                if key == "r":
                    confirm_redo = True
                    print("\rdelete the last saved episode and record its seed again? y to "
                          "confirm, any other key cancels", end="\r\n", flush=True)
                elif key in KEYS:
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
