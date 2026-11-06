"""T visibility per frame in a recorded sim dataset: is the T hidden by the arm in the policy image?

For every frame of ``observation.images.top`` (the 256 px policy crop) the dark blue T is
segmented with the sim calibration's colour threshold and its area is compared with the
reference area scaled to the policy image. Prints, per episode, the fraction of frames with the
T under 85 % and under 50 % visible, and the same over all frames. Run from the venv::

    python scripts/t_visibility.py --root datasets/movensys/pusht_cr3a_sim_v3 [--episodes 0 1 2]

The straight-down camera (dataset v1, 2026-11-03) had the T under 50 % visible in 56 to 67 % of
the pushing frames; the oblique camera (v2) is expected to be well under 20 %.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from movensys_imitation.tasks.pusht import scoring as sc  # noqa: E402

DEFAULT_CAL = ROOT / "movensys_imitation_ros/config/dobot_cr3a/calibration/sim"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT / "datasets/movensys/pusht_cr3a_sim_v3")
    ap.add_argument("--repo-id", default="movensys/pusht_cr3a_sim_v3")
    ap.add_argument("--calibration", type=Path, default=DEFAULT_CAL)
    ap.add_argument("--episodes", type=int, nargs="*", help="default: all")
    ap.add_argument("--open-px", type=int, default=3, help="opening kernel at 256 px (scorer: 5 at full res)")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo_id, root=str(args.root))
    cal = sc.Calibration.load(args.calibration)
    cfg = sc.MaskConfig(threshold=cal.threshold, open_px=args.open_px)
    sample = ds[0]["observation.images.top"]
    size = sample.shape[-1]
    # Area scale from the full-resolution crop to the policy image (landscape crops are squashed).
    scale = (size / cal.crop[2]) * (size / cal.crop[3])
    ref_area = cal.t_area_px * scale
    episodes = args.episodes if args.episodes else sorted(set(int(e) for e in ds.hf_dataset["episode_index"]))
    ep_index = np.asarray(ds.hf_dataset["episode_index"])
    totals = []
    print(f"reference T area in the {size} px image: {ref_area:.0f} px (crop {cal.crop})")
    for ep in episodes:
        idx = np.nonzero(ep_index == ep)[0]
        vis = []
        for i in idx:
            img = ds[int(i)]["observation.images.top"]  # torch CHW float 0..1
            rgb = (np.asarray(img).transpose(1, 2, 0) * 255).round().astype(np.uint8)
            vis.append(sc.segment_t(rgb, cfg).area / ref_area)
        vis = np.asarray(vis)
        totals.append(vis)
        print(
            f"episode {ep:3d}: {len(vis):4d} frames, T under 85 %: {np.mean(vis < 0.85) * 100:5.1f} %, "
            f"under 50 %: {np.mean(vis < 0.50) * 100:5.1f} %, median visible {np.median(vis) * 100:5.1f} %"
        )
    allv = np.concatenate(totals)
    print(
        f"all: {len(allv)} frames, T under 85 %: {np.mean(allv < 0.85) * 100:.1f} %, "
        f"under 50 %: {np.mean(allv < 0.50) * 100:.1f} %"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
