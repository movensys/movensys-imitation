"""
reset_check: teleport repeatability (P2.4): N resets, the T at rest after each.

Teleports the T to ``sample_start_pose(seed)`` for N seeds, checks the published pose lands
within 2 mm and 1 deg, then that the T does not move between 0.5 s and 1.5 s after the
teleport (0.5 mm, 0.2 deg: velocity zero). Needs only the scene playing. Run from the venv::

    python -m movensys_imitation_ros.reset_check --n 10 --out results/sim/reset_check.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
import time

import rclpy

from movensys_imitation.tasks.pusht import geometry as g
from movensys_imitation_ros.episode_manager import (
    Driver, DriverError, spin_in_background, stop_spinning,
)

FIELDS = ("idx", "seed", "x", "y", "yaw", "reached_mm", "reached_deg", "drift_mm", "drift_deg",
          "pass", "note")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("results/sim/reset_check.csv"))
    ap.add_argument("--settle", type=float, default=0.5)
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--drift-mm", type=float, default=0.5)
    ap.add_argument("--drift-deg", type=float, default=0.2)
    args = ap.parse_args(argv)

    rclpy.init()
    drv = Driver("reset_check")
    spinner = spin_in_background(drv)
    rows = []
    try:
        for i in range(args.n):
            seed = args.seed_start + i
            pose = g.sample_start_pose(seed)
            row = {"idx": i, "seed": seed, "x": f"{pose[0]:.4f}", "y": f"{pose[1]:.4f}",
                   "yaw": f"{pose[2]:.4f}", "pass": 0, "note": ""}
            try:
                d, dyaw = drv.teleport(pose, settle=args.settle, retries=1)
                drift, drift_yaw = drv.block_motion(args.window)
            except DriverError as e:
                row["note"] = str(e)
                rows.append(row)
                print(f"{i:2d} seed {seed}: FAIL {e}")
                continue
            ok = drift * 1000 <= args.drift_mm and math.degrees(drift_yaw) <= args.drift_deg
            row.update(
                reached_mm=f"{d * 1000:.2f}", reached_deg=f"{math.degrees(dyaw):.2f}",
                drift_mm=f"{drift * 1000:.3f}", drift_deg=f"{math.degrees(drift_yaw):.3f}",
                **{"pass": int(ok)},
            )
            rows.append(row)
            print(
                f"{i:2d} seed {seed}: reached {d * 1000:.2f} mm, drift {drift * 1000:.3f} mm "
                f"{math.degrees(drift_yaw):.3f} deg over {args.window} s: "
                f"{'PASS' if ok else 'FAIL'}"
            )
        drv.teleport(g.sample_start_pose(0), settle=0.5, retries=1)
        time.sleep(0.2)
    finally:
        stop_spinning(drv, spinner)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    n_pass = sum(int(r["pass"]) for r in rows)
    print(f"{n_pass}/{len(rows)} resets at rest -> {args.out}")
    return 0 if n_pass == len(rows) == args.n else 1


if __name__ == "__main__":
    sys.exit(main())
