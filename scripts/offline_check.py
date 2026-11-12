#!/usr/bin/env python
"""Open-loop check of a checkpoint on recorded episodes before any robot motion (plan P2.8).

Feeds the held-out episodes frame by frame through ``PolicyRunner`` (the same pre/post
processing the sim runner uses) and compares the predicted next target with the recorded
action. A constant offset or an error near the action range means a normalisation, crop or
orientation bug. Two trivial baselines are printed on the same frames: "hold" (predict the
current tip position) and "repeat" (predict the previous action); the policy has learnt once it
beats "hold". Because pushes are multimodal, the direction agreement (fraction of frames whose
predicted displacement from the tip is within 45 deg of the recorded one, over frames where the
recorded displacement exceeds 5 mm) is the more telling number.

    python scripts/offline_check.py outputs/sim/diffusion_v3_s1/checkpoints/last/pretrained_model \
        --episodes 60 61 62 63 64 --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def direction_agreement(state, target, pred, min_mm: float = 5.0, max_deg: float = 45.0) -> float:
    """Fraction of frames whose predicted displacement points the recorded way.

    Within ``max_deg`` of the recorded displacement, over frames where that displacement
    exceeds ``min_mm``; NaN if there are none.
    """
    d_rec = target - state
    d_pred = pred - state
    keep = np.linalg.norm(d_rec, axis=1) > min_mm / 1000
    if not keep.any():
        return float("nan")
    d_rec, d_pred = d_rec[keep], d_pred[keep]
    cos = (d_rec * d_pred).sum(1) / (
        np.linalg.norm(d_rec, axis=1) * np.maximum(np.linalg.norm(d_pred, axis=1), 1e-9)
    )
    return float((cos >= np.cos(np.radians(max_deg))).mean())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("pretrained", type=Path)
    ap.add_argument("--repo-id", default="movensys/pusht_cr3a_sim_v3")
    ap.add_argument("--root", type=Path, default=ROOT / "datasets/movensys/pusht_cr3a_sim_v3")
    ap.add_argument("--episodes", type=int, nargs="+", default=[60, 61, 62, 63, 64])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=None, help="per-frame CSV")
    args, overrides = ap.parse_known_args()
    bad = [o for o in overrides if not o.startswith("--policy.")]
    if bad:
        ap.error(f"unknown arguments {bad}; policy overrides start with --policy.")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from movensys_imitation.data.convert_gym_pusht import to_uint8_hwc
    from movensys_imitation.inference import ACTION, OBS_IMAGE, OBS_STATE, PolicyRunner

    runner = PolicyRunner(args.pretrained, tuple(overrides), device=args.device)
    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=args.episodes)
    print(f"{args.pretrained}: {runner.cfg.type}, n_obs_steps {runner.n_obs_steps}, "
          f"{len(ds)} frames in episodes {args.episodes}, device {runner.cfg.device}")

    rows = []
    per_episode = {}
    current = None
    for i in range(len(ds)):
        item = ds[i]
        ep = int(item["episode_index"])
        if ep != current:
            runner.reset()
            current = ep
        image = to_uint8_hwc(item[OBS_IMAGE])
        state = item[OBS_STATE].numpy()
        target = item[ACTION].numpy()
        pred = runner.act(image, state)
        err = pred - target
        rows.append((ep, int(item["frame_index"]), *state, *target, *pred, *err,
                     runner.last_latency_ms))
        per_episode.setdefault(ep, []).append((err, runner.last_latency_ms, state, target, pred))

    print(f"{'episode':>8} {'frames':>6} {'mean|err| mm':>13} {'p95|err| mm':>12} "
          f"{'bias x mm':>10} {'bias y mm':>10} {'dir agree':>10} {'ms/call':>8}")
    all_err, all_ms, all_state, all_target, all_pred, all_prev = [], [], [], [], [], []
    for ep, vals in sorted(per_episode.items()):
        err = np.array([v[0] for v in vals])
        ms = np.array([v[1] for v in vals])
        state = np.array([v[2] for v in vals])
        target = np.array([v[3] for v in vals])
        pred = np.array([v[4] for v in vals])
        norm = np.linalg.norm(err, axis=1) * 1000
        agree = direction_agreement(state, target, pred)
        print(f"{ep:>8} {len(vals):>6} {norm.mean():>13.1f} {np.percentile(norm, 95):>12.1f} "
              f"{err[:, 0].mean() * 1000:>10.1f} {err[:, 1].mean() * 1000:>10.1f} "
              f"{agree:>10.2f} {np.median(ms):>8.1f}")
        all_err.append(err)
        all_ms.append(ms)
        all_state.append(state)
        all_target.append(target)
        all_pred.append(pred)
        all_prev.append(np.vstack([state[:1], target[:-1]]))  # previous action, episode-local
    err = np.concatenate(all_err)
    ms = np.concatenate(all_ms)
    state, target, pred, prev = (
        np.concatenate(v) for v in (all_state, all_target, all_pred, all_prev)
    )
    norm = np.linalg.norm(err, axis=1) * 1000
    hold = np.linalg.norm(target - state, axis=1) * 1000
    repeat = np.linalg.norm(target - prev, axis=1) * 1000
    summary = {
        "pretrained": str(args.pretrained), "overrides": overrides, "episodes": args.episodes,
        "frames": int(len(norm)), "mean_err_mm": float(norm.mean()),
        "p95_err_mm": float(np.percentile(norm, 95)),
        "bias_mm": [float(err[:, 0].mean() * 1000), float(err[:, 1].mean() * 1000)],
        "direction_agreement": float(direction_agreement(state, target, pred)),
        "baseline_hold_mm": float(hold.mean()), "baseline_repeat_mm": float(repeat.mean()),
        "baseline_repeat_direction_agreement": float(direction_agreement(state, target, prev)),
        "latency_ms_median": float(np.median(ms)), "latency_ms_p95": float(np.percentile(ms, 95)),
    }
    print(f"{'all':>8} {len(norm):>6} {summary['mean_err_mm']:>13.1f} "
          f"{summary['p95_err_mm']:>12.1f} {summary['bias_mm'][0]:>10.1f} "
          f"{summary['bias_mm'][1]:>10.1f} {summary['direction_agreement']:>10.2f} "
          f"{summary['latency_ms_median']:>8.1f}")
    print(f"baselines on the same frames: hold {summary['baseline_hold_mm']:.1f} mm, "
          f"repeat {summary['baseline_repeat_mm']:.1f} mm "
          f"(direction agreement {summary['baseline_repeat_direction_agreement']:.2f})")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "frame", "state_x", "state_y", "action_x", "action_y",
                        "pred_x", "pred_y", "err_x", "err_y", "latency_ms"])
            w.writerows(rows)
        args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out} and {args.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
