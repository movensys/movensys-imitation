"""ROS 2 smoke check for the PushT Isaac scene (plan P2.1 interfaces, Appendix A).

With Isaac Sim playing ``9a_pusht_simulation.usd`` (GUI, or ``build_scene.py --play``), run from
the project venv with ROS 2 Jazzy sourced:

    python sim/isaacsim/smoke_check.py [--image out.png] [--no-teleport]

Checks the topics the ROS nodes depend on (rates, image size and encoding, frame id), then
teleports the T through ``/pusht/t_block/pose_cmd`` and verifies the published pose follows and
stays put. Exit code 0 when everything passes.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, JointState


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Checker(Node):
    def __init__(self) -> None:
        super().__init__("pusht_smoke_check")
        self.stamps: dict[str, list[float]] = {}
        self.last: dict[str, object] = {}
        for name, typ in (
            ("/joint_states", JointState),
            ("/clock", Clock),
            ("/image_top/rgb", Image),
            ("/image_top/camera_info", CameraInfo),
            ("/image_hand/rgb", Image),
            ("/image_hand/camera_info", CameraInfo),
            ("/pusht/t_block/pose", Pose),
        ):
            self.stamps[name] = []
            self.create_subscription(typ, name, lambda m, n=name: self._on(n, m), 10)
        self.cmd = self.create_publisher(Pose, "/pusht/t_block/pose_cmd", 10)

    def _on(self, name: str, msg) -> None:
        self.stamps[name].append(time.monotonic())
        self.last[name] = msg

    def spin_for(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def rate(self, name: str, window: float) -> float:
        now = time.monotonic()
        return sum(1 for t in self.stamps[name] if t > now - window) / window


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", help="save the last /image_top/rgb frame here (.png via Pillow)")
    ap.add_argument("--hand-image", help="save the last /image_hand/rgb frame here (.png)")
    ap.add_argument("--no-teleport", action="store_true")
    ap.add_argument("--window", type=float, default=5.0, help="seconds to measure rates over")
    args = ap.parse_args()

    rclpy.init()
    node = Checker()
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append((name, bool(ok), detail))

    node.spin_for(args.window + 1.0)
    for topic, lo, hi in (
        ("/joint_states", 30, 400),
        ("/clock", 30, 400),
        ("/image_top/rgb", 20, 40),
        ("/image_top/camera_info", 20, 40),
        ("/image_hand/rgb", 20, 40),
        ("/image_hand/camera_info", 20, 40),
        ("/pusht/t_block/pose", 20, 40),
    ):
        r = node.rate(topic, args.window)
        check(f"{topic} rate", lo <= r <= hi, f"{r:.1f} Hz (expect {lo} to {hi})")

    for ns, frame, save_to in (
        ("/image_top", "camera_top_color_optical_frame", args.image),
        ("/image_hand", "camera_hand_color_optical_frame", args.hand_image),
    ):
        img = node.last.get(f"{ns}/rgb")
        if img is not None:
            check(
                f"{ns} image size",
                (img.width, img.height) == (1280, 720),
                f"{img.width} x {img.height} {img.encoding}",
            )
            check(f"{ns} image frame_id", img.header.frame_id == frame, img.header.frame_id)
            if save_to:
                from PIL import Image as PILImage

                rows = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.step)
                arr = rows[:, : img.width * 3].reshape(img.height, img.width, 3)
                PILImage.fromarray(arr.copy(), "RGB").save(save_to)
                print(f"saved {save_to}")
        info = node.last.get(f"{ns}/camera_info")
        if info is not None:
            check(
                f"{ns} camera_info",
                (info.width, info.height) == (1280, 720),
                f"fx={info.k[0]:.1f} fy={info.k[4]:.1f} cx={info.k[2]:.1f} cy={info.k[5]:.1f}",
            )
    js = node.last.get("/joint_states")
    if js is not None:
        check("joint names", all(f"joint{i}" in js.name for i in range(1, 7)), ", ".join(js.name))

    pose = node.last.get("/pusht/t_block/pose")
    if pose is not None:
        check(
            "block at rest on the table",
            abs(pose.position.z) < 0.003,
            f"z={pose.position.z * 1000:.1f} mm",
        )

    if not args.no_teleport and pose is not None:
        target = (0.10, -0.05, math.radians(90))
        msg = Pose()
        msg.position.x, msg.position.y, msg.position.z = target[0], target[1], 0.001
        msg.orientation.w, msg.orientation.z = math.cos(target[2] / 2), math.sin(target[2] / 2)
        for _ in range(5):
            node.cmd.publish(msg)
            node.spin_for(0.1)
        node.spin_for(1.0)
        p1 = node.last["/pusht/t_block/pose"]
        node.spin_for(2.0)
        p2 = node.last["/pusht/t_block/pose"]
        err = math.hypot(p1.position.x - target[0], p1.position.y - target[1])
        yaw_err = abs((yaw_of(p1.orientation) - target[2] + math.pi) % (2 * math.pi) - math.pi)
        drift = math.hypot(p2.position.x - p1.position.x, p2.position.y - p1.position.y)
        check("teleport position", err < 0.002, f"error {err * 1000:.2f} mm")
        check("teleport yaw", yaw_err < math.radians(1), f"error {math.degrees(yaw_err):.2f} deg")
        check(
            "block stays put after teleport",
            drift < 0.001 and abs(p2.position.z) < 0.003,
            f"drift {drift * 1000:.2f} mm over 2 s, z={p2.position.z * 1000:.1f} mm",
        )

    node.destroy_node()
    rclpy.shutdown()
    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
    failed = [n for n, ok, _ in results if not ok]
    print(f"{len(results) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
