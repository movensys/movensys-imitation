"""
setup_camera: one-time camera calibration for the pixel scorer (plan P2.3, reused by P4.3).

With the T on the goal (sim: teleported to ``goal_pose()``; real: placed on the printed outline
by hand) it grabs one overhead frame, finds the T, fits the square crop around the goal, refines
the colour threshold from the T's own pixels and writes the calibration directory
(``calibration.yaml``, ``goal_mask.png``, ``reference.png``, ``goal_overlay.png``). Run from
the project venv while the scene plays and the arm is parked away from the sheet::

    python -m movensys_imitation_ros.setup_camera --mode sim
    python -m movensys_imitation_ros.setup_camera --mode real --camera-height 0.70

The initial colour range and the refinement margins come from ``scorer.yaml``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
import yaml

from movensys_imitation.tasks.pusht import geometry as g
from movensys_imitation.tasks.pusht import scoring as sc

PKG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SCORER_YAML = PKG_DIR / "config" / "dobot_cr3a" / "scorer.yaml"


def load_scorer_yaml(path: Path) -> dict:
    doc = yaml.safe_load(path.read_text())
    return doc["pusht_scorer"]["ros__parameters"]


def mask_config_from_yaml(params: dict) -> tuple[sc.MaskConfig, int, int]:
    seg = params.get("segmentation", {})
    init = params.get("initial_threshold", {})
    ref = params.get("refine", {})
    cfg = sc.MaskConfig(
        sc.HsvRange.from_dict(init) if init else sc.HsvRange(),
        int(seg.get("open_px", 5)),
        float(seg.get("area_tolerance", 0.15)),
    )
    return cfg, int(ref.get("hue_margin", 8)), int(ref.get("sv_margin", 40))


def pose_msg(pose) -> Pose:
    msg = Pose()
    msg.position.x, msg.position.y, msg.position.z = pose[0], pose[1], 0.001
    w, x, y, z = g.yaw_to_quat_wxyz(pose[2])
    msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = w, x, y, z
    return msg


def pose_error(p: Pose, target) -> tuple[float, float]:
    q = p.orientation
    yaw = g.quat_wxyz_to_yaw(q.w, q.x, q.y, q.z)
    d = math.hypot(p.position.x - target[0], p.position.y - target[1])
    dyaw = abs((yaw - target[2] + math.pi) % (2 * math.pi) - math.pi)
    return d, dyaw


class Grabber(Node):
    """Latest image, camera_info and (sim) block pose; teleport publisher."""

    def __init__(self, image_topic: str, info_topic: str, pose_topic: str) -> None:
        super().__init__("setup_camera")
        self.image: Image | None = None
        self.info: CameraInfo | None = None
        self.pose: Pose | None = None
        self.image_t = 0.0
        self.create_subscription(Image, image_topic, self._on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, info_topic, self._on_info, qos_profile_sensor_data)
        self.create_subscription(Pose, pose_topic, self._on_pose, 10)
        self.cmd = self.create_publisher(Pose, "/pusht/t_block/pose_cmd", 10)

    def _on_image(self, msg: Image) -> None:
        self.image, self.image_t = msg, time.monotonic()

    def _on_info(self, msg: CameraInfo) -> None:
        self.info = msg

    def _on_pose(self, msg: Pose) -> None:
        self.pose = msg

    def spin_for(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def teleport(self, pose, settle: float) -> None:
        # A publish before Isaac's subscriber has discovered this publisher is dropped (the first
        # teleport of a fresh node, 2026-10-28), so wait for the match first.
        t0 = time.monotonic()
        while self.cmd.get_subscription_count() == 0 and time.monotonic() - t0 < 30.0:
            self.spin_for(0.1)
        if self.cmd.get_subscription_count() == 0:
            self.get_logger().warning(
                "no subscriber on /pusht/t_block/pose_cmd; is the scene playing?"
            )
        msg = pose_msg(pose)
        for _ in range(5):
            self.cmd.publish(msg)
            self.spin_for(0.1)
        self.spin_for(settle)

    def fresh_frame(self, timeout: float = 5.0):
        """Wait for a frame newer than now, decode it."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            self.spin_for(0.1)
            if self.image is not None and self.image_t > t0:
                m = self.image
                return sc.decode_image(m.data, m.width, m.height, m.step, m.encoding), m
        raise TimeoutError("no overhead image received")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=("sim", "real"), default="sim")
    ap.add_argument(
        "--out", type=Path, help="calibration directory (default config/.../calibration/<mode>)"
    )
    ap.add_argument("--image-topic", default="/image_top/rgb")
    ap.add_argument("--info-topic", default="/image_top/camera_info")
    ap.add_argument("--pose-topic", default="/pusht/t_block/pose")
    ap.add_argument("--scorer-yaml", type=Path, default=DEFAULT_SCORER_YAML)
    ap.add_argument("--no-teleport", action="store_true", help="sim: the T is already on the goal")
    ap.add_argument("--settle", type=float, default=2.0, help="seconds after the teleport")
    ap.add_argument(
        "--camera-height", type=float, help="camera above the table [m], for a scale cross-check"
    )
    ap.add_argument("--yes", action="store_true", help="real: do not wait for Enter")
    ap.add_argument("--force", action="store_true", help="write even if the checks fail")
    args = ap.parse_args(argv)

    out = args.out or PKG_DIR / "config" / "dobot_cr3a" / "calibration" / args.mode
    cfg, hue_margin, sv_margin = mask_config_from_yaml(load_scorer_yaml(args.scorer_yaml))
    goal = g.goal_pose()
    warnings: list[str] = []

    rclpy.init()
    node = Grabber(args.image_topic, args.info_topic, args.pose_topic)
    try:
        if args.mode == "sim" and not args.no_teleport:
            print(f"teleporting the T to the goal {tuple(round(v, 4) for v in goal)} ...")
            node.teleport(goal, args.settle)
            if node.pose is None:
                warnings.append(f"no pose on {args.pose_topic}; cannot confirm the teleport")
            else:
                d, dyaw = pose_error(node.pose, goal)
                print(
                    f"block pose error after teleport: {d * 1000:.2f} mm, "
                    f"{math.degrees(dyaw):.2f} deg"
                )
                if d > 0.002 or dyaw > math.radians(1):
                    warnings.append("the T did not settle on the goal pose")
        elif args.mode == "real" and not args.yes:
            input("Place the T on the printed outline, retract the arm, then press Enter ... ")

        frame, msg = node.fresh_frame()
        print(f"frame {msg.width} x {msg.height} {msg.encoding} from {msg.header.frame_id}")
        meta = {
            "mode": args.mode,
            "image_topic": args.image_topic,
            "frame_id": msg.header.frame_id,
            "stamp": f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}",
            "initial_threshold": cfg.threshold.to_dict(),
            "refine": {"hue_margin": hue_margin, "sv_margin": sv_margin},
        }
        if node.info is not None:
            k = node.info.k
            meta["camera_info"] = {
                "fx": float(k[0]), "fy": float(k[4]), "cx": float(k[2]), "cy": float(k[5])
            }
        if args.camera_height:
            meta["camera_height_m"] = args.camera_height

        cal, result = sc.calibrate_from_frame(
            frame, cfg, hue_margin=hue_margin, sv_margin=sv_margin, meta=meta
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print(f"T found: {result.area} px in {result.n_components} component(s) inside the crop")
    print(f"crop (x0, y0, w, h) = {cal.crop}; {cal.px_per_m:.1f} px/m on the T's top face")
    print(f"threshold {cal.threshold.to_dict()}; T HSV median {cal.t_hsv_median}")
    if result.n_components > 1:
        warnings.append(
            f"{result.n_components - 1} extra blue component(s) in the crop; check the overlay"
        )
    if "camera_info" in meta and args.camera_height:
        expected = meta["camera_info"]["fx"] / (args.camera_height - g.BLOCK_HEIGHT)
        print(f"camera_info predicts {expected:.1f} px/m at the top face")
        if abs(expected / cal.px_per_m - 1) > 0.10:
            warnings.append("pixel scale from the T disagrees with camera_info by more than 10 %")
    sheet_px = max(g.SHEET_SIZE) * cal.px_per_m * (1 - g.BLOCK_HEIGHT / (args.camera_height or 0.7))
    if cal.crop[2] < sheet_px:
        warnings.append(f"crop {cal.crop[2]} px is smaller than the sheet ({sheet_px:.0f} px)")

    for w in warnings:
        print(f"WARNING: {w}")
    if warnings and not args.force:
        print("not written (use --force to write anyway)")
        return 1
    cal.save(out, reference_rgb=frame)
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
