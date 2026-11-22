"""
setup_camera_pose: camera pose from the printed goal T, for the vision pose estimator.

The printed goal T (dark on the light sheet) has 8 corners at known sheet coordinates
(``geometry.goal_pose()``); ``solvePnP`` on them gives the camera pose in the sheet frame. The
result goes to ``camera.yaml`` (intrinsics, distortion, pose, reprojection error) and
``camera_overlay.png`` (push-area outline, 5 cm grid, goal T, detected corners, the policy image
crop in white, and the fitted block if one is in view) in the calibration directory, next to
the pixel scorer's files.

Run with the printed goal T fully visible: the block off the goal and the rod away from it.
Run it again whenever the camera or the sheet moves. Intrinsics come from, in order: ``--fx``
(with ``--fy --cx --cy``), the ``camera_info`` topic, or ``--estimate-focal`` (from the T
alone: rough, a fallback only)::

    python -m movensys_imitation_ros.setup_camera_pose --mode real
    python -m movensys_imitation_ros.setup_camera_pose --mode real --image frame.png --fx 634

``--image`` works without ROS (an RGB/BGR file as written by OpenCV).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

from movensys_imitation.tasks.pusht import geometry as g
from movensys_imitation.tasks.pusht import pose_vision as pv

PKG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ESTIMATOR_YAML = PKG_DIR / "config" / "dobot_cr3a" / "pose_estimator.yaml"


def estimator_settings(path: Path) -> dict:
    """Return the ``estimator`` block of ``pose_estimator.yaml`` as a plain dict."""
    if not path.exists():
        return {}
    doc = yaml.safe_load(path.read_text()) or {}
    return dict(doc.get("t_pose_estimator", {}).get("ros__parameters", {}).get("estimator", {}))


def grab_ros(image_topic: str, info_topic: str, wait_info: float = 3.0):
    """One fresh frame (RGB) and the camera_info K, D (None when not published)."""
    import rclpy

    from movensys_imitation_ros.setup_camera import Grabber

    rclpy.init()
    node = Grabber(image_topic, info_topic, "/pusht/t_block/pose")
    try:
        frame, msg = node.fresh_frame()
        node.spin_for(0.0 if node.info is not None else wait_info)
        info = node.info
        K = np.array(info.k, float).reshape(3, 3) if info is not None else None
        D = np.array(info.d, float) if info is not None and len(info.d) else None
        if info is not None and (info.width, info.height) != (msg.width, msg.height):
            print(f"WARNING: camera_info is {info.width} x {info.height}, image "
                  f"{msg.width} x {msg.height}; ignoring camera_info")
            K = D = None
        meta = {"image_topic": image_topic, "frame_id": msg.header.frame_id,
                "stamp": f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}"}
        return frame, K, D, meta
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=("sim", "real"), default="real")
    ap.add_argument("--out", type=Path, help="default config/dobot_cr3a/calibration/<mode>")
    ap.add_argument("--image", type=Path, help="calibrate from this file instead of ROS")
    ap.add_argument("--image-topic", default="/image_top/rgb")
    ap.add_argument("--info-topic", default="/image_top/camera_info")
    ap.add_argument("--fx", type=float)
    ap.add_argument("--fy", type=float)
    ap.add_argument("--cx", type=float)
    ap.add_argument("--cy", type=float)
    ap.add_argument("--estimate-focal", action="store_true",
                    help="no camera_info: take the focal length from the T (rough)")
    ap.add_argument("--goal", type=float, nargs=3, metavar=("X", "Y", "YAW_DEG"),
                    help="printed goal pose if it is not geometry.goal_pose()")
    ap.add_argument("--estimator-yaml", type=Path, default=DEFAULT_ESTIMATOR_YAML)
    ap.add_argument("--max-reproj", type=float, default=1.5, help="px; refuse to write above")
    ap.add_argument("--yes", action="store_true", help="do not wait for Enter")
    ap.add_argument("--force", action="store_true", help="write even if the checks fail")
    args = ap.parse_args(argv)

    out = args.out or PKG_DIR / "config" / "dobot_cr3a" / "calibration" / args.mode
    goal = g.goal_pose() if args.goal is None else (
        args.goal[0], args.goal[1], math.radians(args.goal[2]))
    warnings: list[str] = []

    if args.image:
        bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"cannot read {args.image}")
            return 1
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        K_info = D_info = None
        meta = {"image_file": str(args.image)}
    else:
        if not args.yes:
            input("Goal T fully visible (block off the goal, rod away from it)? Press Enter ... ")
        frame, K_info, D_info, meta = grab_ros(args.image_topic, args.info_topic)
    h, w = frame.shape[:2]
    meta.update(mode=args.mode, goal_pose=[float(v) for v in goal])
    print(f"frame {w} x {h}")

    t = pv.find_goal_t(frame, goal)
    model = g.block_polygon(goal)
    print(f"goal T found: {int(t.mask.sum())} px, corner shape error {t.homography_px:.2f} px")
    centre = ((w - 1) / 2, (h - 1) / 2)
    try:
        f_t = pv.estimate_focal(t.corners, model, centre)
        print(f"focal length implied by the T alone: {f_t:.0f} px (rough)")
    except ValueError as e:
        f_t = None
        print(f"focal length not observable from the T: {e}")

    dist = np.zeros(5)
    if args.fx:
        fy = args.fy or args.fx
        K = pv.intrinsics_matrix(args.fx, fy, args.cx if args.cx is not None else centre[0],
                                 args.cy if args.cy is not None else centre[1])
        meta["intrinsics"] = "command line"
    elif K_info is not None:
        K, dist = K_info, (D_info if D_info is not None else np.zeros(5))
        meta["intrinsics"] = "camera_info"
    elif args.estimate_focal and f_t is not None:
        K = pv.intrinsics_matrix(f_t, f_t, *centre)
        meta["intrinsics"] = "estimated from the goal T"
        warnings.append("intrinsics estimated from the T; use camera_info for accuracy")
    else:
        print("no intrinsics: pass --fx (and --fy --cx --cy), publish camera_info, "
              "or use --estimate-focal")
        return 1
    print(f"K: fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f} "
          f"({meta['intrinsics']}); distortion {np.round(dist, 4).tolist()}")
    if f_t is not None and abs(f_t / K[0, 0] - 1) > 0.10:
        # The single-T estimate is rough (a few % to ~15 % off at this tilt); the reprojection
        # check below is what judges a measured K. Informational only.
        print(f"note: the T alone implies fx {f_t:.0f}, {abs(f_t / K[0, 0] - 1):.0%} off K "
              "(expected; the reprojection error is the real check)")

    cam, t = pv.calibrate_from_goal(frame, K, dist, goal, meta=meta)
    c = cam.camera_centre()
    R, _ = cv2.Rodrigues(cam.rvec)
    tilt = math.degrees(math.acos(float(np.clip(-R[2] @ (0, 0, 1), -1, 1))))
    print(f"camera at sheet ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:.3f}) m, optical axis {tilt:.1f} deg "
          f"from vertical, reprojection {cam.reprojection_px:.2f} px")
    if cam.reprojection_px > args.max_reproj:
        warnings.append(f"reprojection {cam.reprojection_px:.2f} px > {args.max_reproj}")
    _, box = pv.area_roi(cam, 0.0)
    if box[2] * box[3] == 0 or box[0] == 0 or box[1] == 0 or box[0] + box[2] >= w \
            or box[1] + box[3] >= h:
        warnings.append("the push area reaches the image edge; part of it is not visible")
    crop = None
    try:
        crop = pv.policy_crop(cam)
        print(f"policy image crop (x0, y0, w, h) = {crop} (recorder and policy runner)")
    except ValueError as e:
        warnings.append(f"no policy crop: {e}")

    overlay = frame.copy()
    if crop is not None:
        cv2.rectangle(overlay, crop[:2], (crop[0] + crop[2] - 1, crop[1] + crop[3] - 1),
                      (255, 255, 255), 1)
    pv.draw_sheet(overlay, cam)
    pv.draw_block_outline(overlay, cam, goal, (255, 0, 255), 2, height=0.0)
    for p in t.corners:
        cv2.circle(overlay, tuple(int(round(v)) for v in p), 3, (255, 0, 0), -1)
    est = pv.PoseEstimator(cam, pv.estimator_config_from_dict(estimator_settings(
        args.estimator_yaml)))
    r = est.estimate(frame)
    if r.pose is not None:
        pv.draw_block_outline(overlay, cam, r.pose, (255, 255, 0), 2)
        print(f"block in view at ({r.pose[0]:+.4f}, {r.pose[1]:+.4f}) m, "
              f"{math.degrees(r.pose[2]):+.1f} deg, fit IoU {r.iou:.3f}"
              f"{'' if r.ok else ' (' + r.note + ')'}")

    for wmsg in warnings:
        print(f"WARNING: {wmsg}")
    if warnings and not args.force:
        print("not written (use --force to write anyway)")
        return 1
    path = cam.save(out)
    cv2.imwrite(str(out / pv.CAMERA_OVERLAY_PNG), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"written {path} and {out / pv.CAMERA_OVERLAY_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
