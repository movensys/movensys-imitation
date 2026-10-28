"""
scorer_validation: pixel versus geometric coverage on teleported end states (plan P2.3 evidence).

Teleports the T to a set of poses (near the goal within the 0.95 threshold band, moderately off,
and random starts), lets it settle, scores the frame with the pixel backend and the published
pose with the geometric backend, and writes one CSV row per pose. The done criterion is agreement
within 0.02 on every valid row. Run from the venv while the scene plays and the arm is parked
away from the sheet::

    python -m movensys_imitation_ros.scorer_validation --out results/sim/scorer_validation.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np
import rclpy

from movensys_imitation.tasks.pusht import geometry as g
from movensys_imitation.tasks.pusht import scoring as sc
from movensys_imitation_ros.pusht_scorer import default_calibration_dir, pose_to_tuple
from movensys_imitation_ros.setup_camera import Grabber, pose_error

FIELDS = (
    "idx", "kind", "x", "y", "yaw", "reached_mm", "reached_deg", "coverage_geom", "iou_geom",
    "coverage_pix", "iou_pix", "pix_valid", "area_ratio", "diff", "note",
)


def sample_poses(n_near: int, n_mid: int, n_far: int, seed: int):
    rng = np.random.default_rng(seed)
    goal = g.goal_pose()
    poses = [("goal", goal)]
    for _ in range(n_near):
        dx, dy = rng.uniform(-0.003, 0.003, 2)
        dyaw = math.radians(rng.uniform(-3, 3))
        poses.append(("near", (goal[0] + dx, goal[1] + dy, goal[2] + dyaw)))
    for _ in range(n_mid):
        r, a = rng.uniform(0.005, 0.03), rng.uniform(-math.pi, math.pi)
        dyaw = math.radians(rng.uniform(-20, 20))
        pose = (goal[0] + r * math.cos(a), goal[1] + r * math.sin(a), goal[2] + dyaw)
        poses.append(("mid", pose))
    for i in range(n_far):
        poses.append(("far", g.sample_start_pose(2000 + i)))
    return poses


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--calibration", type=Path, default=default_calibration_dir("sim"))
    ap.add_argument("--out", type=Path, default=Path("results/sim/scorer_validation.csv"))
    ap.add_argument("--n-near", type=int, default=10)
    ap.add_argument("--n-mid", type=int, default=5)
    ap.add_argument("--n-far", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--settle", type=float, default=1.5)
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--mask-dir", type=Path, help="save the crop and mask of every pose here")
    ap.add_argument("--image-topic", default="/image_top/rgb")
    ap.add_argument("--pose-topic", default="/pusht/t_block/pose")
    args = ap.parse_args(argv)

    scorer = sc.PixelScorer(sc.Calibration.load(args.calibration))
    poses = sample_poses(args.n_near, args.n_mid, args.n_far, args.seed)
    rows = []

    rclpy.init()
    node = Grabber(args.image_topic, "/image_top/camera_info", args.pose_topic)
    try:
        for idx, (kind, target) in enumerate(poses):
            node.teleport(target, args.settle)
            frame, _ = node.fresh_frame()
            if node.pose is None:
                raise RuntimeError(f"no pose on {args.pose_topic}")
            measured = pose_to_tuple(node.pose)
            d, dyaw = pose_error(node.pose, target)
            pix = scorer.score(frame)
            geo = sc.geometric_score(measured)
            diff = pix.coverage - geo.coverage if pix.valid else math.nan
            rows.append({
                "idx": idx, "kind": kind, "x": f"{measured[0]:.4f}", "y": f"{measured[1]:.4f}",
                "yaw": f"{measured[2]:.4f}", "reached_mm": f"{d * 1000:.2f}",
                "reached_deg": f"{math.degrees(dyaw):.2f}",
                "coverage_geom": f"{geo.coverage:.4f}", "iou_geom": f"{geo.iou:.4f}",
                "coverage_pix": f"{pix.coverage:.4f}", "iou_pix": f"{pix.iou:.4f}",
                "pix_valid": int(pix.valid), "area_ratio": f"{pix.area_ratio:.3f}",
                "diff": f"{diff:+.4f}" if pix.valid else "", "note": pix.note,
            })
            print(
                f"{idx:2d} {kind:4s} geom {geo.coverage:.3f} pix {pix.coverage:.3f} "
                f"diff {diff:+.3f} area {pix.area_ratio:.2f} {pix.note}"
            )
            if args.mask_dir:
                import cv2

                args.mask_dir.mkdir(parents=True, exist_ok=True)
                crop = scorer.cal.apply_crop(frame)
                bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(args.mask_dir / f"{idx:02d}_crop.png"), bgr)
                mask8 = scorer.last.mask.astype(np.uint8) * 255
                cv2.imwrite(str(args.mask_dir / f"{idx:02d}_mask.png"), mask8)
        node.teleport(g.sample_start_pose(0), 0.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    valid = [r for r in rows if r["pix_valid"]]
    diffs = [abs(float(r["diff"])) for r in valid]
    ok = len(valid) == len(rows) and all(d <= args.tolerance for d in diffs)
    mean_d = float(np.mean(diffs)) if diffs else math.nan
    max_d = max(diffs) if diffs else math.nan
    print(
        f"{len(rows)} poses, {len(valid)} valid, mean |diff| {mean_d:.4f}, max |diff| {max_d:.4f}, "
        f"tolerance {args.tolerance}: {'PASS' if ok else 'FAIL'} -> {args.out}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
