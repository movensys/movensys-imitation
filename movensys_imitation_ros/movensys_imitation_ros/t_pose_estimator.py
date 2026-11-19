"""
t_pose_estimator: block pose from the overhead camera (real robot).

Fits the T's pose to each frame with :class:`pose_vision.PoseEstimator` (colour mask of the
block, the red pusher masked as unknown, the full prism rendered through the calibrated camera)
and publishes it where Isaac publishes the ground truth in sim, so the scorer's geometric
backend, the recorder's ``block_pose`` and the manager read the real block the same way.

Subscribes
    /image_top/rgb              sensor_msgs/Image
Publishes
    pose_topic                  geometry_msgs/Pose           only for fits that pass the checks
                                (real: /pusht/t_block/pose; sim: /pusht/t_block/pose_vision so
                                Isaac's ground truth stays untouched for validation)
    ~/quality                   std_msgs/Float32MultiArray   QUALITY_FIELDS for every frame
    ~/state                     std_msgs/String              ok | hold:<why> | fault:<why>
    ~/debug                     sensor_msgs/Image (rgb8)     overlay, only while subscribed

A frame without a good fit publishes no pose: the scorer's pose goes stale after
``pose_stale`` and its coverage turns NaN, which also resets the manager's success run.
Needs ``camera.yaml`` from ``setup_camera_pose`` in the calibration directory. Runs in the venv
(``python -m movensys_imitation_ros.t_pose_estimator``).
"""

from __future__ import annotations

import math
from pathlib import Path
import time

from geometry_msgs.msg import Pose
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String

from movensys_imitation.tasks.pusht import geometry as g
from movensys_imitation.tasks.pusht import pose_vision as pv
from movensys_imitation.tasks.pusht import scoring as sc
from movensys_imitation_ros.pusht_scorer import default_calibration_dir

QUALITY_FIELDS = ("ok", "iou", "visible", "cost", "n_evals", "mask_px", "latency_ms")

ESTIMATOR_DEFAULTS = {
    "estimator.block.hue": [55, 95],
    "estimator.block.sat_min": 80,
    "estimator.block.val": [20, 245],
    "estimator.occluder.hue": [170, 10],
    "estimator.occluder.sat_min": 100,
    "estimator.occluder.val": [40, 255],
    "estimator.open_px": 3,
    "estimator.occluder_dilate_px": 7,
    "estimator.area_margin_m": 0.05,
    "estimator.roi_margin_px": 25,
    "estimator.min_component_frac": 0.05,
    "estimator.min_iou": 0.80,
    "estimator.min_visible": 0.50,
    "estimator.max_iter": 250,
    "estimator.track_jump_m": 0.03,
}


def nested_estimator_dict(params: dict) -> dict:
    """Nest ``estimator.block.hue`` style parameters for ``estimator_config_from_dict``."""
    out: dict = {}
    for name, value in params.items():
        if not name.startswith("estimator."):
            continue
        parts = name.split(".")[1:]
        d = out
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = list(value) if isinstance(value, (list, tuple)) else value
    return out


def pose_msg(pose) -> Pose:
    msg = Pose()
    msg.position.x, msg.position.y = float(pose[0]), float(pose[1])
    msg.position.z = g.BLOCK_HEIGHT / 2
    w, x, y, z = g.yaw_to_quat_wxyz(pose[2])
    msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = w, x, y, z
    return msg


class TPoseEstimator(Node):
    def __init__(self) -> None:
        super().__init__("t_pose_estimator")
        p = self._declare_parameters()
        self.mode = p["mode"]
        self.image_stale = float(p["image_stale"])
        pose_topic = p["pose_topic"] or (
            "/pusht/t_block/pose" if self.mode == "real" else "/pusht/t_block/pose_vision"
        )
        cal_dir = Path(p["calibration_dir"]) if p["calibration_dir"] else None
        cal_dir = cal_dir or default_calibration_dir(self.mode)
        self.est: pv.PoseEstimator | None = None
        try:
            cam = pv.CameraModel.load(cal_dir)
            cfg = pv.estimator_config_from_dict(nested_estimator_dict(p))
            self.est = pv.PoseEstimator(cam, cfg)
            c = cam.camera_centre()
            self.get_logger().info(
                f"camera {cal_dir / pv.CAMERA_YAML}: {cam.image_size[0]} x {cam.image_size[1]}, "
                f"fx {cam.K[0, 0]:.0f}, at sheet ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:.3f}) m, "
                f"reprojection {cam.reprojection_px:.2f} px; block {cfg.block.to_dict()}, "
                f"occluder {cfg.occluder.to_dict()}"
            )
        except (FileNotFoundError, KeyError, ValueError) as e:
            self.get_logger().error(
                f"no usable camera model in {cal_dir} ({e}); run setup_camera_pose first"
            )

        self.image: Image | None = None
        self.image_t = -math.inf
        self.last_stamp = None
        self.state = ""
        self.n_ok = 0
        self.n_frames = 0
        self.create_subscription(Image, p["image_topic"], self._on_image, qos_profile_sensor_data)
        self.pub_pose = self.create_publisher(Pose, pose_topic, 10)
        self.pub_quality = self.create_publisher(Float32MultiArray, "~/quality", 10)
        self.pub_state = self.create_publisher(String, "~/state", 10)
        self.pub_debug = self.create_publisher(Image, "~/debug", 1)
        self.create_timer(1.0 / float(p["rate_hz"]), self._tick)
        self.create_timer(10.0, self._report)
        self.get_logger().info(f"mode {self.mode}, pose on {pose_topic} at {p['rate_hz']:.0f} Hz")

    def _declare_parameters(self) -> dict:
        defaults = {
            "mode": "real",
            "calibration_dir": "",
            "image_topic": "/image_top/rgb",
            "pose_topic": "",
            "rate_hz": 10.0,
            "image_stale": 0.3,
            **ESTIMATOR_DEFAULTS,
        }
        return {k: self.declare_parameter(k, v).value for k, v in defaults.items()}

    def _on_image(self, msg: Image) -> None:
        self.image, self.image_t = msg, time.monotonic()

    def _set_state(self, text: str) -> None:
        if text != self.state:
            self.state = text
            self.pub_state.publish(String(data=text))
            if text == "ok":
                self.get_logger().info(f"state {text}")
            else:
                self.get_logger().warning(f"state {text}")

    def _tick(self) -> None:
        if self.est is None:
            self._set_state("fault:no_camera_model")
            return
        msg = self.image
        if msg is None or time.monotonic() - self.image_t > self.image_stale:
            self._set_state("hold:no_image")
            return
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp == self.last_stamp:
            return  # same frame as last tick
        self.last_stamp = stamp
        t0 = time.perf_counter()
        try:
            frame = sc.decode_image(msg.data, msg.width, msg.height, msg.step, msg.encoding)
            r = self.est.estimate(frame)
        except ValueError as e:
            self._set_state(f"fault:{e}")
            return
        latency = (time.perf_counter() - t0) * 1000
        self.n_frames += 1
        if r.ok:
            self.n_ok += 1
            self.pub_pose.publish(pose_msg(r.pose))
            self._set_state("ok")
        else:
            self._set_state(f"hold:{r.note}")
        q = Float32MultiArray()
        n = len(QUALITY_FIELDS)
        q.layout.dim.append(MultiArrayDimension(label=",".join(QUALITY_FIELDS), size=n, stride=n))
        q.data = [float(r.ok), r.iou, r.visible, r.cost, float(r.n_evals), float(r.mask_px),
                  latency]
        self.pub_quality.publish(q)
        if self.pub_debug.get_subscription_count() > 0:
            self._publish_debug(msg, frame, r)

    def _publish_debug(self, src: Image, frame: np.ndarray, r: pv.PoseResult) -> None:
        img = frame.copy()
        if self.est.unknown_mask is not None:
            img[self.est.unknown_mask] = (img[self.est.unknown_mask] // 2) + (0, 0, 120)
        pv.draw_sheet(img, self.est.cam)
        pv.draw_block_outline(img, self.est.cam, g.goal_pose(), (255, 0, 255), 1, height=0.0)
        if r.pose is not None:
            pv.draw_block_outline(img, self.est.cam, r.pose,
                                  (255, 255, 0) if r.ok else (255, 80, 0), 2)
        out = Image()
        out.header = src.header
        out.height, out.width = img.shape[:2]
        out.encoding = "rgb8"
        out.step = out.width * 3
        out.data = np.ascontiguousarray(img).tobytes()
        self.pub_debug.publish(out)

    def _report(self) -> None:
        if self.n_frames:
            self.get_logger().info(
                f"{self.n_ok}/{self.n_frames} frames with a good fit in the last 10 s"
            )
        self.n_ok = self.n_frames = 0


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
