r"""
replay_check: recorded actions through the guard must reproduce the final T pose (P2.4).

For each episode of a recorded dataset: lift, home the tip to the episode's first
``observation.state``, teleport the T to its first ``block_pose``, lower, then publish the
recorded 10 Hz ``action`` sequence as a timed path on ``/pusht/waypoints`` (guard in ``pose``
source) and compare the T pose and coverage at the end with the recorded ones. The guard's
dead-man applies: ``ros2 param set /planar_guard require_deadman false`` for this check (sim
only) or hold the grip.
Run from the venv with the scene playing, the guard and the scorer up::

    python -m movensys_imitation_ros.replay_check --root datasets/movensys/pusht_cr3a_sim_v1 \
        --out results/sim/replay_check.csv

Tolerances (plan P2.4 criterion made numeric): coverage within 0.05, junction within 10 mm,
yaw within 5 deg. The replayed path is the 10 Hz target linearly interpolated, not the original
50 Hz stream, and PhysX contact is not bit-reproducible, so exact agreement is not expected.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
import time

import numpy as np
import rclpy

from movensys_imitation_ros import manager_core as mc
from movensys_imitation_ros.episode_manager import (
    Driver, DriverError, spin_in_background, stop_spinning,
)
from movensys_imitation_ros.recorder_core import read_sidecar

FIELDS = (
    "episode", "seed", "n_frames", "start_err_mm", "rec_x", "rec_y", "rec_yaw", "rep_x", "rep_y",
    "rep_yaw", "pos_err_mm", "yaw_err_deg", "rec_coverage", "rep_coverage", "cov_diff", "pass",
    "note",
)


def load_episodes(repo_id: str, root: Path, wanted: list[int] | None) -> list[dict]:
    """Per episode: actions, first state, first and last block pose, last coverage."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root)
    table = ds.hf_dataset.with_format("numpy")
    ep_index = np.asarray(table["episode_index"]).reshape(-1)
    actions = np.asarray(table["action"])
    states = np.asarray(table["observation.state"])
    blocks = np.asarray(table["block_pose"])
    coverage = np.asarray(table["coverage"]).reshape(-1)
    fps = ds.fps
    out = []
    for ep in sorted({int(e) for e in ep_index}):
        if wanted and ep not in wanted:
            continue
        sel = ep_index == ep
        out.append({
            "episode": ep,
            "fps": fps,
            "actions": actions[sel],
            "first_state": states[sel][0],
            "first_block": tuple(float(v) for v in blocks[sel][0]),
            "last_block": tuple(float(v) for v in blocks[sel][-1]),
            "last_coverage": float(coverage[sel][-1]),
        })
    return out


def replay_one(drv: Driver, ep: dict, settle: float) -> dict:
    ox, oy = ep["origin"]  # the dataset is in the sheet frame, the guard in the planning frame
    drv.set_guard_source("pose")
    drv.lift(True)
    drv.go_to((float(ep["first_state"][0]) + ox, float(ep["first_state"][1]) + oy))
    d, _ = drv.teleport(ep["first_block"])
    drv.lift(False)
    time.sleep(settle)
    t0 = drv.now() + 0.5
    actions = [(float(a[0]) + ox, float(a[1]) + oy) for a in ep["actions"]]
    path = mc.path_from_actions(actions, t0, ep["fps"])
    drv.publish_path(path)
    duration = len(path) / ep["fps"] + 0.5
    time.sleep(duration + 1.0)
    drv.publish_target(path[-1][1:])  # clear the path, hold the last target
    drv.lift(True)
    time.sleep(settle)
    score = drv.score()
    primary = mc.primary_score(score) or {}
    rep_block = drv.block_pose
    pos_err, yaw_err = mc.pose_difference(rep_block, ep["last_block"])
    rep_cov = float(primary.get("coverage", math.nan))
    return {
        "start_err_mm": d * 1000, "rep_block": rep_block, "pos_err": pos_err,
        "yaw_err": yaw_err, "rep_coverage": rep_cov,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo-id", default="movensys/pusht_cr3a_sim_v1")
    ap.add_argument("--root", type=Path, default=Path("datasets/movensys/pusht_cr3a_sim_v1"))
    ap.add_argument("--episodes", type=int, nargs="*", help="episode indices (default: all)")
    ap.add_argument("--out", type=Path, default=Path("results/sim/replay_check.csv"))
    ap.add_argument("--tol-coverage", type=float, default=0.05)
    ap.add_argument("--tol-pos", type=float, default=0.010)
    ap.add_argument("--tol-yaw-deg", type=float, default=5.0)
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument(
        "--area-center", type=float, nargs=2, default=(-0.03, -0.05),
        help="sheet origin in the planning frame when the sidecar does not record it",
    )
    args = ap.parse_args(argv)

    episodes = load_episodes(args.repo_id, args.root, args.episodes)
    if not episodes:
        print("no episodes found", file=sys.stderr)
        return 2
    sidecar = {m["episode_index"]: m for m in read_sidecar(args.root)}
    seeds = {k: m.get("seed") for k, m in sidecar.items()}
    for ep in episodes:
        origin = sidecar.get(ep["episode"], {}).get("sheet_origin_wm", args.area_center)
        ep["origin"] = (float(origin[0]), float(origin[1]))

    rclpy.init()
    drv = Driver("replay_check")
    spinner = spin_in_background(drv)
    rows = []
    try:
        for ep in episodes:
            note = ""
            try:
                r = replay_one(drv, ep, args.settle)
            except DriverError as e:
                note = str(e)
                r = None
            if r is None:
                rows.append({"episode": ep["episode"], "seed": seeds.get(ep["episode"]),
                             "n_frames": len(ep["actions"]), "pass": 0, "note": note})
                print(f"episode {ep['episode']}: FAILED to replay: {note}")
                continue
            cov_diff = r["rep_coverage"] - ep["last_coverage"]
            ok = (
                r["pos_err"] <= args.tol_pos
                and math.degrees(r["yaw_err"]) <= args.tol_yaw_deg
                and abs(cov_diff) <= args.tol_coverage
            )
            rb, lb = r["rep_block"], ep["last_block"]
            rows.append({
                "episode": ep["episode"], "seed": seeds.get(ep["episode"]),
                "n_frames": len(ep["actions"]), "start_err_mm": f"{r['start_err_mm']:.2f}",
                "rec_x": f"{lb[0]:.4f}", "rec_y": f"{lb[1]:.4f}", "rec_yaw": f"{lb[2]:.4f}",
                "rep_x": f"{rb[0]:.4f}", "rep_y": f"{rb[1]:.4f}", "rep_yaw": f"{rb[2]:.4f}",
                "pos_err_mm": f"{r['pos_err'] * 1000:.1f}",
                "yaw_err_deg": f"{math.degrees(r['yaw_err']):.2f}",
                "rec_coverage": f"{ep['last_coverage']:.4f}",
                "rep_coverage": f"{r['rep_coverage']:.4f}", "cov_diff": f"{cov_diff:+.4f}",
                "pass": int(ok), "note": note,
            })
            print(
                f"episode {ep['episode']}: pos {r['pos_err'] * 1000:.1f} mm, "
                f"yaw {math.degrees(r['yaw_err']):.2f} deg, coverage {ep['last_coverage']:.3f} "
                f"-> {r['rep_coverage']:.3f}: {'PASS' if ok else 'FAIL'}"
            )
    finally:
        stop_spinning(drv, spinner)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    n_pass = sum(int(r.get("pass", 0)) for r in rows)
    print(f"{n_pass}/{len(rows)} episodes within tolerance -> {args.out}")
    return 0 if n_pass == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
