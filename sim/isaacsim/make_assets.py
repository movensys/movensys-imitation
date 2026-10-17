"""Generate the PushT sheet artwork used by the Isaac scene (and later printed).

Run from the project venv (needs Pillow and numpy):

    python sim/isaacsim/make_assets.py

Writes into ``sim/isaacsim/assets/``:

* ``target_sheet.png``   the 0.45 x 0.45 m sheet: goal T outline and 30 start marks, no markers
* ``pusht_sheet.json``   every number the artwork was made from, for ``task.yaml`` and the scorer

The sheet carries no fiducials (plan v2.1): scoring segments the dark blue T in the overhead image,
so everything printed on the sheet is thin and coloured to stay clear of the T's colour threshold.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from movensys_imitation.tasks.pusht import geometry as g  # noqa: E402

ASSETS = Path(__file__).resolve().parent / "assets"
PX_PER_M = 4000  # 4 px per mm
BLOCK_RGB = (20, 40, 110)  # dark blue T (the block material in the scene)
GOAL_RGB = (40, 160, 70)  # gym-pusht's green goal
MARK_RGB = (60, 110, 220)  # start marks: light blue, thin lines; the T threshold must not pick them up


def px(m: float) -> int:
    return int(round(m * PX_PER_M))


def to_pixel(img_h: int, x_m: float, y_m: float, origin_m: float):
    """Sheet-frame metres (origin at the centre, y up) to image pixels (y down)."""
    return (px(x_m + origin_m), img_h - px(y_m + origin_m))


def draw_polyline(draw, img_h, pts_m, origin_m, fill, width):
    pts = [to_pixel(img_h, x, y, origin_m) for x, y in pts_m]
    draw.line(pts + pts[:1], fill=fill, width=width, joint="curve")


def make_sheet() -> Image.Image:
    side = px(g.SHEET_SIZE)
    img = Image.new("RGB", (side, side), "white")
    draw = ImageDraw.Draw(img)
    half = g.SHEET_SIZE / 2

    # Goal outline.
    draw_polyline(draw, side, g.block_polygon(g.goal_pose()), half, GOAL_RGB, px(0.003))

    # Start marks: dot at the junction, a tick toward the block's +y (bar side), the seed number.
    font = ImageFont.load_default(size=px(0.006))
    for seed, (x, y, yaw) in enumerate(g.start_marks()):
        tick = g.transform_points(np.array([[0.0, 0.0], [0.0, 0.012]]), (x, y, yaw))
        p0, p1 = (to_pixel(side, *p, half) for p in tick)
        draw.line([p0, p1], fill=MARK_RGB, width=px(0.001))
        r = px(0.0015)
        draw.ellipse([p0[0] - r, p0[1] - r, p0[0] + r, p0[1] + r], fill=MARK_RGB)
        label = g.transform_points(np.array([[0.006, -0.004]]), (x, y, yaw))[0]
        draw.text(to_pixel(side, *label, half), str(seed), fill=MARK_RGB, font=font, anchor="mm")

    # Push-area border (the sheet edge).
    draw.rectangle([0, 0, side - 1, side - 1], outline=MARK_RGB, width=px(0.001))
    return img


def write_spec() -> dict:
    spec = {
        "frame": "sheet frame: origin at the sheet centre, aligned with world_manipulator, "
        "robot on +y",
        "markers": "none; scoring is in overhead pixel space (plan v2.1)",
        "unit_m": g.UNIT,
        "block": {
            "bar_xy_m": list(g.BAR),
            "stem_xy_m": list(g.STEM),
            "height_m": g.BLOCK_HEIGHT,
            "mass_kg": g.BLOCK_MASS,
            "rgb": list(BLOCK_RGB),
            "pose_origin": "bar-stem junction; yaw 0 = bar along +x on the +y side",
            "polygon_local_m": g.block_polygon_local().tolist(),
            "centroid_local_m": g.block_centroid_local().tolist(),
        },
        "sheet": {
            "size_m": g.SHEET_SIZE,
            "inner_size_m": g.INNER_SIZE,
            "goal_rgb": list(GOAL_RGB),
            "mark_rgb": list(MARK_RGB),
            "px_per_m": PX_PER_M,
        },
        "goal_pose": list(g.goal_pose()),
        "start_marks": {str(i): list(p) for i, p in enumerate(g.start_marks())},
        "eval_seeds": "0..49 via sample_start_pose(seed); start marks are seeds 0..29",
    }
    (ASSETS / "pusht_sheet.json").write_text(json.dumps(spec, indent=2) + "\n")
    return spec


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    sheet = make_sheet()
    sheet.save(ASSETS / "target_sheet.png")
    spec = write_spec()
    goal = [round(v, 4) for v in spec["goal_pose"]]
    print(f"sheet {sheet.size} px, goal {goal}")
    print(f"wrote {ASSETS}")


if __name__ == "__main__":
    main()
