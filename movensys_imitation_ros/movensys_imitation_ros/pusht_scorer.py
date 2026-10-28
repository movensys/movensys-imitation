"""
pusht_scorer: coverage and IoU of the T against the goal (plan P2.3).

Runs the pixel backend on the overhead image and, when the ground-truth pose topic is alive
(sim), the geometric backend next to it, and logs their difference. Runs from the project venv
(``python -m movensys_imitation_ros.pusht_scorer``): it needs numpy 2, OpenCV and shapely.

Inputs
    /image_top/rgb           sensor_msgs/Image           overhead frame, decoded with numpy
    /pusht/t_block/pose      geometry_msgs/Pose          T pose in the sheet frame (sim only)
Outputs
    /pusht/score             std_msgs/Float32MultiArray  SCORE_FIELDS at rate_hz
    /pusht/t_mask            sensor_msgs/Image (mono8)   the T mask in the crop (optional)
    ~/state                  std_msgs/String             "<state>:<reason>" on change
Services
    ~/score                  std_srvs/Trigger            fresh score of the latest frame as JSON
                                                         (episode end, after the arm retracts)

The primary coverage and IoU (first two fields) come from the geometric backend in sim and from
the pixel backend on the real robot (``primary: auto``); both backends are always in the message.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
from std_srvs.srv import Trigger

from movensys_imitation.tasks.pusht import geometry as g
from movensys_imitation.tasks.pusht import scoring as sc

SCORE_FIELDS = (
    "coverage",
    "iou",
    "success",
    "coverage_pix",
    "iou_pix",
    "pix_valid",
    "area_ratio",
    "coverage_geom",
    "iou_geom",
)


def default_calibration_dir(mode: str) -> Path:
    """``config/dobot_cr3a/calibration/<mode>`` from the installed share, else the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("movensys_imitation_ros"))
        cand = share / "config" / "dobot_cr3a" / "calibration" / mode
        if cand.exists():
            return cand
    except Exception:  # noqa: B902  not installed; fall through to the source tree
        pass
    return Path(__file__).resolve().parents[1] / "config" / "dobot_cr3a" / "calibration" / mode


def pose_to_tuple(p: Pose) -> tuple[float, float, float]:
    q = p.orientation
    return (p.position.x, p.position.y, g.quat_wxyz_to_yaw(q.w, q.x, q.y, q.z))


class PushtScorer(Node):
    """See the module docstring."""

    def __init__(self) -> None:
        super().__init__("pusht_scorer")
        p = self._declare_parameters()
        self.mode = p["mode"]
        self.primary = p["primary"]
        if self.primary == "auto":
            self.primary = "geometric" if self.mode == "sim" else "pixel"
        self.image_stale = p["image_stale"]
        self.pose_stale = p["pose_stale"]
        self.log_period = p["log_period"]
        self.publish_mask = p["publish_mask"]
        self.mask_cfg = sc.MaskConfig(
            open_px=int(p["segmentation.open_px"]),
            area_tolerance=float(p["segmentation.area_tolerance"]),
        )

        cal_dir = Path(p["calibration_dir"]) if p["calibration_dir"] else None
        cal_dir = cal_dir or default_calibration_dir(self.mode)
        self.scorer: sc.PixelScorer | None = None
        try:
            self.scorer = sc.PixelScorer(sc.Calibration.load(cal_dir), self.mask_cfg)
            cal = self.scorer.cal
            self.get_logger().info(
                f"calibration {cal_dir}: crop {cal.crop}, {cal.px_per_m:.0f} px/m, "
                f"T {cal.t_area_px} px, threshold {cal.threshold.to_dict()}"
            )
        except (FileNotFoundError, KeyError, ValueError) as e:
            self.get_logger().error(
                f"no usable calibration in {cal_dir} ({e}); run setup_camera first. "
                "Pixel scores are NaN until then."
            )

        self.image = None
        self.image_t = -math.inf
        self.pose: tuple[float, float, float] | None = None
        self.pose_t = -math.inf
        self.state = ""
        self.last_json = "{}"
        self.diffs: list[float] = []
        self.last_log = 0.0

        self.create_subscription(Image, p["image_topic"], self._on_image, qos_profile_sensor_data)
        self.create_subscription(Pose, p["pose_topic"], self._on_pose, 10)
        self.pub_score = self.create_publisher(Float32MultiArray, "/pusht/score", 10)
        self.pub_mask = self.create_publisher(Image, "/pusht/t_mask", 1)
        self.pub_state = self.create_publisher(String, "~/state", 10)
        self.create_service(Trigger, "~/score", self._on_score_request)
        self.create_timer(1.0 / p["rate_hz"], self._tick)
        self.get_logger().info(
            f"mode {self.mode}, primary backend {self.primary}, {p['rate_hz']:.0f} Hz"
        )

    def _declare_parameters(self) -> dict:
        defaults = {
            "mode": "sim",
            "primary": "auto",
            "image_topic": "/image_top/rgb",
            "pose_topic": "/pusht/t_block/pose",
            "calibration_dir": "",
            "rate_hz": 10.0,
            "publish_mask": True,
            "image_stale": 0.5,
            "pose_stale": 0.5,
            "log_period": 2.0,
            "segmentation.open_px": 5,
            "segmentation.area_tolerance": 0.15,
        }
        out = {}
        for name, default in defaults.items():
            out[name] = self.declare_parameter(name, default).value
        if out["primary"] not in ("auto", "geometric", "pixel"):
            raise ValueError(f"primary must be auto, geometric or pixel, not {out['primary']!r}")
        return out

    # ------------------------------------------------------------------ callbacks

    def _on_image(self, msg: Image) -> None:
        self.image = msg
        self.image_t = time.monotonic()

    def _on_pose(self, msg: Pose) -> None:
        self.pose = pose_to_tuple(msg)
        self.pose_t = time.monotonic()

    def _on_score_request(self, _req, resp):
        self._tick()
        resp.success = self.state.startswith("ok")
        resp.message = self.last_json
        return resp

    # ------------------------------------------------------------------ scoring

    def _set_state(self, state: str, reason: str = "") -> None:
        text = f"{state}:{reason}" if reason else state
        if text != self.state:
            self.state = text
            self.pub_state.publish(String(data=text))
            # Two call sites on purpose: rclpy pins the severity per call site.
            if state == "ok":
                self.get_logger().info(f"state {text}")
            else:
                self.get_logger().warning(f"state {text}")

    def _tick(self) -> None:
        now = time.monotonic()
        pix = geo = None
        fields = {k: math.nan for k in SCORE_FIELDS}
        reason = ""

        image_fresh = self.image is not None and now - self.image_t < self.image_stale
        if self.scorer is not None and image_fresh:
            msg = self.image
            try:
                frame = sc.decode_image(msg.data, msg.width, msg.height, msg.step, msg.encoding)
                pix = self.scorer.score(frame)
            except ValueError as e:
                reason = f"image:{e}"
            else:
                fields.update(
                    coverage_pix=pix.coverage,
                    iou_pix=pix.iou,
                    pix_valid=float(pix.valid),
                    area_ratio=pix.area_ratio,
                )
                if self.publish_mask and self.scorer.last is not None:
                    self._publish_mask(msg, self.scorer.last.mask)
        elif self.scorer is None:
            reason = "no_calibration"
        else:
            reason = "no_image"

        if self.pose is not None and now - self.pose_t < self.pose_stale:
            geo = sc.geometric_score(self.pose)
            fields.update(coverage_geom=geo.coverage, iou_geom=geo.iou)

        chosen = geo if self.primary == "geometric" else pix
        if chosen is not None:
            fields.update(coverage=chosen.coverage, iou=chosen.iou, success=float(chosen.success))
            if chosen is pix and not pix.valid:
                reason = reason or "occluded"
        elif self.primary == "geometric" and geo is None:
            reason = reason or "no_pose"

        if pix is not None and geo is not None and pix.valid:
            d = pix.coverage - geo.coverage
            self.diffs.append(d)
            if now - self.last_log > self.log_period:
                self.last_log = now
                ad = [abs(x) for x in self.diffs]
                self.get_logger().info(
                    f"coverage pixel {pix.coverage:.3f} geometric {geo.coverage:.3f} "
                    f"diff {d:+.3f} | valid frames {len(ad)}, mean |diff| "
                    f"{sum(ad) / len(ad):.4f}, max |diff| {max(ad):.4f}"
                )

        if reason in ("no_calibration", "no_image") and self.primary == "pixel":
            self._set_state("fault", reason)
        elif reason:
            self._set_state("ok", reason) if chosen is not None else self._set_state("hold", reason)
        else:
            self._set_state("ok")

        out = Float32MultiArray()
        n = len(SCORE_FIELDS)
        dim = MultiArrayDimension(label=",".join(SCORE_FIELDS), size=n, stride=n)
        out.layout.dim.append(dim)
        out.data = [float(fields[k]) for k in SCORE_FIELDS]
        self.pub_score.publish(out)
        self.last_json = json.dumps(
            {
                "state": self.state,
                "primary": self.primary,
                "pixel": sc.score_as_dict(pix) if pix else None,
                "geometric": sc.score_as_dict(geo) if geo else None,
                "pose": list(self.pose) if geo else None,
            }
        )

    def _publish_mask(self, src: Image, mask) -> None:
        out = Image()
        out.header = src.header
        out.height, out.width = mask.shape
        out.encoding = "mono8"
        out.is_bigendian = 0
        out.step = out.width
        out.data = (mask.astype("uint8") * 255).tobytes()
        self.pub_mask.publish(out)

    def summary(self) -> str:
        if not self.diffs:
            return "no frame had both backends valid"
        ad = [abs(x) for x in self.diffs]
        return (
            f"{len(ad)} frames with both backends: mean |diff| {sum(ad) / len(ad):.4f}, "
            f"max |diff| {max(ad):.4f}, within 0.02: {sum(x <= 0.02 for x in ad)}/{len(ad)}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PushtScorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(node.summary())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
