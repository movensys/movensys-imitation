"""
score_viewer: the overhead camera with the live coverage and IoU drawn on it.

Shows ``/t_pose_estimator/debug`` (the frame with the goal outline in magenta and the fitted
block in yellow) or, with ``--raw``, the plain ``/image_top/rgb``, and writes the first two
fields of ``/pusht/score`` on top. Run from the venv next to ``scorer.launch.py``::

    python -m movensys_imitation_ros.score_viewer [--raw] [--scale 0.75]

Quit with q or Esc in the window.
"""

from __future__ import annotations

import argparse
import math
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from movensys_imitation.tasks.pusht import scoring as sc

WINDOW = "pusht score"
SCORE_STALE = 1.0  # s; older scores are shown as "no score"


class ScoreViewer(Node):
    def __init__(self, image_topic: str) -> None:
        super().__init__("score_viewer")
        self.frame = None
        self.score: tuple[float, float, float] | None = None
        self.score_t = -math.inf
        self.create_subscription(Image, image_topic, self._on_image, qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, "/pusht/score", self._on_score, 10)

    def _on_image(self, msg: Image) -> None:
        try:
            rgb = sc.decode_image(msg.data, msg.width, msg.height, msg.step, msg.encoding)
        except ValueError as e:
            self.get_logger().warning(f"image: {e}", throttle_duration_sec=5.0)
            return
        self.frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _on_score(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 3:
            self.score = (msg.data[0], msg.data[1], msg.data[2])
            self.score_t = time.monotonic()

    def render(self, scale: float):
        if self.frame is None:
            return None
        img = self.frame.copy()
        fresh = self.score is not None and time.monotonic() - self.score_t < SCORE_STALE
        if not fresh:
            lines, colour = ["no score"], (0, 0, 255)
        elif math.isnan(self.score[0]):
            lines, colour = ["coverage  --", "iou       --"], (0, 165, 255)
        else:
            cov, iou, success = self.score
            lines = [f"coverage  {cov:.3f}", f"iou       {iou:.3f}"]
            colour = (0, 220, 0) if success >= 0.5 else (255, 255, 255)
        cv2.rectangle(img, (10, 10), (400, 30 + 45 * len(lines)), (0, 0, 0), -1)
        for i, text in enumerate(lines):
            cv2.putText(img, text, (22, 55 + 45 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 2,
                        cv2.LINE_AA)
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return img


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", action="store_true", help="plain camera frame, no outlines")
    ap.add_argument("--image-topic", help="override the image topic")
    ap.add_argument("--scale", type=float, default=1.0, help="window scale factor")
    args = ap.parse_args(argv)
    topic = args.image_topic or ("/image_top/rgb" if args.raw else "/t_pose_estimator/debug")

    rclpy.init()
    node = ScoreViewer(topic)
    node.get_logger().info(f"showing {topic}; q or Esc quits")
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)
            img = node.render(args.scale)
            if img is not None:
                cv2.imshow(WINDOW, img)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
