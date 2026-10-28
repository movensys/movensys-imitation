"""Plan P2.3: hand-computed geometric cases, pixel backend on synthetic overhead frames."""

from __future__ import annotations

import math

import numpy as np
import pytest

from movensys_imitation.tasks.pusht import geometry as g
from movensys_imitation.tasks.pusht import scoring as sc

U = g.UNIT
GOAL = g.goal_pose()

# Synthetic camera: scene_9a's optics (fx 634 px) at 0.70 m give 946 px/m on the T's top face.
PX_PER_M = 946.0
FRAME = (1280, 720)
CENTRE = (640.0, 360.0)
T_RGB = (78, 110, 175)  # what the (20, 40, 110) T renders to in sim (linear diffuseColor)
MARK_RGB = (60, 110, 220)
GOAL_RGB = (40, 160, 70)
TIP_RGB = (40, 40, 45)
SHADOW_RGB = (200, 200, 205)


def offset(pose, dx=0.0, dy=0.0, dyaw=0.0):
    return (pose[0] + dx, pose[1] + dy, pose[2] + dyaw)


# ------------------------------------------------------------------------------- geometric


def test_geometric_identity_and_far():
    s = sc.geometric_score(GOAL)
    assert s.coverage == pytest.approx(1.0) and s.iou == pytest.approx(1.0) and s.success
    s = sc.geometric_score(offset(GOAL, 0.3, 0.0))
    assert s.coverage == 0.0 and s.iou == 0.0 and not s.success


def test_geometric_one_unit_shift_along_the_bar():
    # Shift one unit along the bar: 3 of the 4 bar units still overlap, the stem none.
    c, s = math.cos(GOAL[2]), math.sin(GOAL[2])
    moved = (GOAL[0] + U * c, GOAL[1] + U * s, GOAL[2])
    r = sc.geometric_score(moved)
    assert r.coverage == pytest.approx(3 / 7, abs=1e-9)
    assert r.iou == pytest.approx(3 / 11, abs=1e-9)


def test_geometric_half_turn_about_the_junction():
    # Bar onto the stem's first unit and stem onto the bar's middle unit: 2 of 7.
    r = sc.geometric_score(offset(GOAL, dyaw=math.pi))
    assert r.coverage == pytest.approx(2 / 7, abs=1e-9)
    assert r.iou == pytest.approx(2 / 12, abs=1e-9)


def test_geometric_is_symmetric_and_threshold_distance():
    a, b = offset(GOAL, 0.004, -0.002, 0.03), GOAL
    assert sc.geometric_score(a, b).coverage == pytest.approx(sc.geometric_score(b, a).coverage)
    # The 0.95 threshold is about 3 mm of pure translation across the bar's length.
    along = sc.geometric_score(offset(GOAL, 0.003 * math.cos(GOAL[2]), 0.003 * math.sin(GOAL[2])))
    assert 0.93 < along.coverage < 0.96


# ------------------------------------------------------------------------------- synthetic frames


def synthetic_frame(block_pose, distractors=True, tip_uv=None, tip_r=10, shadow=False, t_rgb=T_RGB):
    img = np.full((FRAME[1], FRAME[0], 3), 235, np.uint8)  # the table
    hw, hh = g.SHEET_SIZE[0] / 2 * PX_PER_M, g.SHEET_SIZE[1] / 2 * PX_PER_M
    x0, x1 = int(CENTRE[0] - hw), int(CENTRE[0] + hw)
    y0, y1 = int(CENTRE[1] - hh), int(CENTRE[1] + hh)
    img[y0:y1, x0:x1] = 250  # the white sheet
    if distractors:
        sc.draw_block(img, GOAL, PX_PER_M, CENTRE, GOAL_RGB, filled=False)
        for seed in range(g.N_START_MARKS):
            x, y, yaw = g.sample_start_pose(seed)
            u, v = CENTRE[0] + x * PX_PER_M, CENTRE[1] - y * PX_PER_M
            img[int(v) - 1 : int(v) + 2, int(u) - 1 : int(u) + 2] = MARK_RGB
            du, dv = 8 * math.cos(yaw), -8 * math.sin(yaw)
            for t in np.linspace(0, 1, 10):
                img[int(v + t * dv), int(u + t * du)] = MARK_RGB
    if shadow:
        sc.draw_block(img, offset(block_pose, 0.004, -0.004), PX_PER_M, CENTRE, SHADOW_RGB)
    sc.draw_block(img, block_pose, PX_PER_M, CENTRE, t_rgb)
    if tip_uv is not None:
        import cv2

        cv2.circle(img, (int(tip_uv[0]), int(tip_uv[1])), tip_r, TIP_RGB, -1)
    return img


@pytest.fixture(scope="module")
def calibration():
    cal, result = sc.calibrate_from_frame(synthetic_frame(GOAL), sc.MaskConfig())
    assert result.n_components == 1
    return cal


def test_calibration_crop_and_scale(calibration):
    cal = calibration
    assert cal.px_per_m == pytest.approx(PX_PER_M, rel=0.03)
    assert cal.crop[2] == cal.crop[3]
    assert 0.6 * PX_PER_M < cal.crop[2] <= FRAME[1]
    assert cal.t_area_px == pytest.approx(sc.T_AREA_M2 * PX_PER_M**2, rel=0.05)
    # The crop is centred on the goal, which is the sheet centre.
    assert cal.crop[0] + cal.centre_uv[0] == pytest.approx(CENTRE[0], abs=2)
    assert cal.crop[1] + cal.centre_uv[1] == pytest.approx(CENTRE[1], abs=2)
    assert cal.threshold.hue[0] <= cal.t_hsv_median[0] <= cal.threshold.hue[1]


@pytest.mark.parametrize(
    "pose",
    [
        GOAL,
        offset(GOAL, 0.002, 0.001, 0.02),
        offset(GOAL, -0.003, 0.002, -0.05),
        offset(GOAL, 0.0, 0.0, 0.1),
        offset(GOAL, 0.015, 0.0, 0.0),
        offset(GOAL, -0.05, 0.03, 0.5),
        offset(GOAL, 0.2, 0.1, 2.0),
        g.sample_start_pose(3),
    ],
)
def test_pixel_matches_geometric_within_tolerance(calibration, pose):
    scorer = sc.PixelScorer(calibration)
    pix = scorer.score(synthetic_frame(pose, shadow=True))
    geo = sc.geometric_score(pose)
    assert pix.valid, pix.note
    assert pix.coverage == pytest.approx(geo.coverage, abs=0.02)
    assert pix.iou == pytest.approx(geo.iou, abs=0.02)
    assert pix.success == geo.success


def test_distractors_do_not_change_the_mask(calibration):
    scorer = sc.PixelScorer(calibration)
    pose = offset(GOAL, 0.004, 0.0, 0.1)
    plain = scorer.score(synthetic_frame(pose, distractors=False))
    busy = scorer.score(synthetic_frame(pose, distractors=True, tip_uv=(400, 300), shadow=True))
    assert busy.valid and plain.valid
    assert busy.coverage == pytest.approx(plain.coverage, abs=0.005)
    assert busy.area_ratio == pytest.approx(1.0, abs=0.03)


def test_occlusion_is_flagged_and_a_lower_bound(calibration):
    scorer = sc.PixelScorer(calibration)
    pose = offset(GOAL, 0.002, 0.0, 0.0)
    geo = sc.geometric_score(pose)
    tip = (CENTRE[0] + 20, CENTRE[1] + 10)  # on the T
    # The 10 px tip hides about 5 % of the T: inside the tolerance, still valid, a lower bound.
    small = scorer.score(synthetic_frame(pose, tip_uv=tip, tip_r=10))
    assert small.valid and 0.9 < small.area_ratio < 1.0
    assert small.coverage <= geo.coverage + 0.01
    # A wrist-sized occluder hides a third of it: flagged, never a false success.
    big = scorer.score(synthetic_frame(pose, tip_uv=tip, tip_r=25))
    assert not big.valid and "hidden" in big.note
    assert big.coverage <= geo.coverage + 0.01 and not big.success


def test_no_t_in_frame(calibration):
    scorer = sc.PixelScorer(calibration)
    s = scorer.score(synthetic_frame(GOAL, t_rgb=(250, 250, 250)))
    assert s.coverage == 0.0 and not s.valid and s.note == "no T found"
    with pytest.raises(ValueError):
        sc.calibrate_from_frame(synthetic_frame(GOAL, t_rgb=(250, 250, 250)), sc.MaskConfig())


def test_calibration_round_trip(tmp_path, calibration):
    frame = synthetic_frame(GOAL)
    d = calibration.save(tmp_path / "cal", reference_rgb=frame)
    for name in (sc.CALIBRATION_YAML, sc.GOAL_MASK_PNG, sc.REFERENCE_PNG, sc.OVERLAY_PNG):
        assert (d / name).exists()
    loaded = sc.Calibration.load(d)
    assert loaded.crop == calibration.crop and loaded.image_size == calibration.image_size
    assert loaded.threshold == calibration.threshold
    assert np.array_equal(loaded.goal_mask, calibration.goal_mask)
    assert sc.PixelScorer(loaded).score(frame).coverage == pytest.approx(1.0, abs=0.005)
    with pytest.raises(ValueError):
        sc.PixelScorer(loaded).score(frame[:-1])


def test_refine_threshold_contains_the_samples():
    rng = np.random.default_rng(0)
    hsv = np.stack(
        [rng.integers(110, 120, 500), rng.integers(120, 160, 500), rng.integers(140, 180, 500)], 1
    )
    t = sc.refine_threshold(hsv, hue_margin=2, sv_margin=5)
    assert t.hue[0] <= 110 and t.hue[1] >= 119 and t.sat_min <= 120
    assert t.val[0] <= 140 and t.val[1] >= 179
    assert t.mask(hsv[None, :, :]).all()


def test_decode_image_handles_step_padding_and_bgr():
    w, h, step = 4, 3, 16
    rgb = np.arange(w * h * 3, dtype=np.uint8).reshape(h, w, 3)
    rows = np.zeros((h, step), np.uint8)
    rows[:, : w * 3] = rgb.reshape(h, w * 3)
    assert np.array_equal(sc.decode_image(rows.tobytes(), w, h, step, "rgb8"), rgb)
    assert np.array_equal(sc.decode_image(rows.tobytes(), w, h, step, "bgr8"), rgb[..., ::-1])
    with pytest.raises(ValueError):
        sc.decode_image(rows.tobytes(), w, h, step, "mono8")


def test_score_dict_and_success_rule():
    s = sc.Score(0.95, 0.9)
    assert s.success and sc.score_as_dict(s)["success"]
    assert not sc.Score(0.95, 0.9, valid=False).success
    assert not sc.Score(0.949, 0.9).success
