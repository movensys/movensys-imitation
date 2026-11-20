"""
score_viewer: the overhead camera with the live coverage and IoU drawn on it.

Shows ``/image_top/rgb`` with the goal outline in magenta and the block pose from
``/pusht/t_block/pose`` in yellow (both projected through ``camera.yaml``), and writes the first
two fields of ``/pusht/score`` on top. ``--raw`` leaves the outlines out; ``--grid`` shows the
estimator's own ``/t_pose_estimator/debug`` image instead, which adds the push-area grid. Run
from the venv next to ``scorer.launch.py``::

    python -m movensys_imitation_ros.score_viewer [--raw | --grid] [--scale 0.75]

Quit with q or Esc in the window.
"""

from __future__ import annotations

import argparse
import math
import time

import cv2
import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from movensys_imitation.tasks.pusht import geometry as g
from movensys_imitation.tasks.pusht import pose_vision as pv
from movensys_imitation.tasks.pusht import scoring as sc
from movensys_imitation_ros.pusht_scorer import default_calibration_dir, pose_to_tuple

WINDOW = "pusht score"
SCORE_STALE = 1.0  # s; older scores are shown as "no score"
POSE_STALE = 0.5  # s; older block poses are not drawn


class ScoreViewer(Node):
    def __init__(self, image_topic: str, cam: pv.CameraModel | None = None) -> None:
        """``cam``: draw the goal and block outlines on the frames; None shows them as they are."""
        super().__init__("score_viewer")
        self.cam = cam
        self.frame = None  # RGB
        self.pose: tuple[float, float, float] | None = None
        self.pose_t = -math.inf
        self.score: tuple[float, float, float] | None = None
        self.score_t = -math.inf
        self.create_subscription(Image, image_topic, self._on_image, qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, "/pusht/score", self._on_score, 10)
        if cam is not None:
            self.create_subscription(Pose, "/pusht/t_block/pose", self._on_pose, 10)

    def _on_image(self, msg: Image) -> None:
        try:
            self.frame = sc.decode_image(msg.data, msg.width, msg.height, msg.step, msg.encoding)
        except ValueError as e:
            self.get_logger().warning(f"image: {e}", throttle_duration_sec=5.0)

    def _on_pose(self, msg: Pose) -> None:
        self.pose = pose_to_tuple(msg)
        self.pose_t = time.monotonic()

    def _on_score(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 3:
            self.score = (msg.data[0], msg.data[1], msg.data[2])
            self.score_t = time.monotonic()

    def render(self, scale: float):
        if self.frame is None:
            return None
        rgb = self.frame.copy()
        if self.cam is not None:
            pv.draw_block_outline(rgb, self.cam, g.goal_pose(), (255, 0, 255), 1, height=0.0)
            if self.pose is not None and time.monotonic() - self.pose_t < POSE_STALE:
                pv.draw_block_outline(rgb, self.cam, self.pose, (255, 255, 0), 2)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
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
    ap.add_argument("--grid", action="store_true",
                    help="the estimator's debug image: outlines and the push-area grid")
    ap.add_argument("--mode", default="real", help="calibration to draw the outlines with")
    ap.add_argument("--scale", type=float, default=1.0, help="window scale factor")
    args = ap.parse_args(argv)
    topic = "/t_pose_estimator/debug" if args.grid else "/image_top/rgb"
    cam = None
    if not (args.raw or args.grid):
        try:
            cam = pv.CameraModel.load(default_calibration_dir(args.mode))
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"no camera.yaml for mode {args.mode} ({e}); showing the plain frame")

    rclpy.init()
    node = ScoreViewer(topic, cam)
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
