"""Vision pose estimator (real-robot scorer): calibration from the printed goal T and the block
pose fit, on synthetic frames from a known camera."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from movensys_imitation.tasks.pusht import geometry as g
from movensys_imitation.tasks.pusht import pose_vision as pv
from movensys_imitation.tasks.pusht import scoring as sc

SIZE = (1280, 800)
FX = 634.0
# About where the real D455 sits (2026-11-19 calibration): beside the sheet, 45 deg oblique.
CAM_POS = (0.51, -0.13, 0.48)
GOAL = g.goal_pose()

SHEET_RGB = (228, 228, 224)
GOAL_RGB = (35, 35, 38)
TOP_RGB = (25, 165, 75)
SIDE_RGB = (15, 120, 55)
ROD_RGB = (210, 30, 40)


@pytest.fixture(scope="module")
def cam() -> pv.CameraModel:
    return pv.look_at_camera(SIZE, FX, CAM_POS)


def fill(img, poly_px, rgb):
    p = np.round(poly_px * 16).astype(np.int32)
    cv2.fillPoly(img, [p], rgb, cv2.LINE_AA, 4)


def scene(cam, block=None, goal=True, rod=None, noise=2.0, seed=0) -> np.ndarray:
    """White sheet, the printed goal T, the block (darker sides, lit top), a red vertical rod."""
    w, h = cam.image_size
    img = np.full((h, w, 3), (90, 95, 100), np.uint8)
    hx, hy = 0.32, 0.18
    fill(img, cam.project(np.array([(-hx, -hy, 0), (hx, -hy, 0), (hx, hy, 0), (-hx, hy, 0)],
                                   float)), SHEET_RGB)
    if goal:
        fill(img, cam.project(np.column_stack([g.block_polygon(GOAL), np.zeros(8)])), GOAL_RGB)
    if block is not None:
        faces = pv.prism_faces_px(cam, block)
        for f in faces[2:]:
            fill(img, f, SIDE_RGB)
        fill(img, faces[0], TOP_RGB)
    if rod is not None:
        x, y, r = rod
        a, b = cam.project(np.array([(x, y, 0.005), (x, y, 0.30)]))
        rad = float(np.linalg.norm(cam.project(np.array([(x + r, y, 0.005)]))[0] - a))
        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), ROD_RGB,
                 max(2, int(2 * rad)), cv2.LINE_AA)
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)


def pose_err(a, b) -> tuple[float, float]:
    return math.hypot(a[0] - b[0], a[1] - b[1]), abs(pv._wrap(a[2] - b[2]))


# ------------------------------------------------------------------------- camera model


def test_project_backproject_roundtrip(cam):
    pts = np.array([(0.1, -0.05), (-0.2, 0.1), (0.0, 0.0)])
    for z in (0.0, 0.03):
        uv = cam.project(np.column_stack([pts, np.full(3, z)]))
        assert np.allclose(cam.backproject_to_plane(uv, z), pts, atol=1e-9)


def test_camera_yaml_roundtrip(cam, tmp_path):
    cam.meta = {"source": "test"}
    cam.save(tmp_path)
    back = pv.CameraModel.load(tmp_path)
    assert back.image_size == cam.image_size and back.meta == {"source": "test"}
    pts = np.array([(0.1, 0.05, 0.0), (-0.2, 0.1, 0.03)])
    assert np.allclose(back.project(pts), cam.project(pts))


def test_look_at_camera_is_above_and_oblique(cam):
    assert np.allclose(cam.camera_centre(), CAM_POS)
    assert np.allclose(cam.project(np.zeros((1, 3)))[0], ((SIZE[0] - 1) / 2, (SIZE[1] - 1) / 2))


# --------------------------------------------------------------------- goal calibration


def test_find_goal_t_orders_corners(cam):
    img = scene(cam, block=None)
    t = pv.find_goal_t(img)
    truth = cam.project(np.column_stack([g.block_polygon(GOAL), np.zeros(8)]))
    assert np.max(np.linalg.norm(t.corners - truth, axis=1)) < 0.6
    assert t.homography_px < 0.5


def test_find_goal_t_ignores_the_block_and_rod(cam):
    img = scene(cam, block=g.sample_start_pose(3), rod=(0.12, 0.05, 0.012))
    t = pv.find_goal_t(img)
    truth = cam.project(np.column_stack([g.block_polygon(GOAL), np.zeros(8)]))
    assert np.max(np.linalg.norm(t.corners - truth, axis=1)) < 0.6


def test_calibrate_from_goal_recovers_the_camera(cam):
    img = scene(cam)
    est, _ = pv.calibrate_from_goal(img, cam.K)
    assert est.reprojection_px < 0.5
    assert np.linalg.norm(est.camera_centre() - np.array(CAM_POS)) < 0.01
    # What matters for scoring: points on the push area land where they should.
    probe = np.array(
        [(x, y, z) for x in (-0.3, 0, 0.3) for y in (-0.15, 0, 0.15) for z in (0, 0.03)]
    )
    assert np.max(np.linalg.norm(est.project(probe) - cam.project(probe), axis=1)) < 2.0


def test_no_goal_t_raises(cam):
    with pytest.raises(ValueError, match="no printed goal T"):
        pv.find_goal_t(scene(cam, goal=False))


def test_estimate_focal_is_close(cam):
    img = scene(cam)
    t = pv.find_goal_t(img)
    f = pv.estimate_focal(t.corners, g.block_polygon(GOAL), cam.K[:2, 2])
    assert f == pytest.approx(FX, rel=0.05)


# ------------------------------------------------------------------------------ pose fit

POSES = [
    GOAL,
    (GOAL[0] + 0.003, GOAL[1] - 0.002, GOAL[2] + math.radians(2)),
    *[g.sample_start_pose(s) for s in range(4)],
]


@pytest.mark.parametrize("pose", POSES)
def test_fit_recovers_the_pose(cam, pose):
    img = scene(cam, block=pose, goal=False, seed=1)
    r = pv.PoseEstimator(cam).estimate(img)
    assert r.ok, r.note
    d, dyaw = pose_err(r.pose, pose)
    assert d < 0.001 and dyaw < math.radians(0.5), (d, math.degrees(dyaw))
    assert r.visible == pytest.approx(1.0)
    cov_true = sc.geometric_score(pose).coverage
    assert sc.geometric_score(r.pose).coverage == pytest.approx(cov_true, abs=0.01)


def test_fit_with_the_goal_printed_underneath(cam):
    pose = (GOAL[0] + 0.004, GOAL[1] + 0.003, GOAL[2] - math.radians(3))
    r = pv.PoseEstimator(cam).estimate(scene(cam, block=pose, goal=True))
    d, dyaw = pose_err(r.pose, pose)
    assert r.ok and d < 0.001 and dyaw < math.radians(0.5)


def test_fit_through_the_rod(cam):
    pose = g.sample_start_pose(2)
    c = g.transform_points(g.block_centroid_local()[None, :], pose)[0]
    rod = (c[0] + 0.01, c[1], 0.012)  # stands right in front of the block
    img = scene(cam, block=pose, rod=rod)
    est = pv.PoseEstimator(cam)
    r = est.estimate(img)
    assert est.unknown_mask.any()
    assert r.ok, r.note
    assert r.visible < 1.0
    d, dyaw = pose_err(r.pose, pose)
    assert d < 0.002 and dyaw < math.radians(1.0), (d, math.degrees(dyaw))


def test_no_block_gives_no_pose(cam):
    r = pv.PoseEstimator(cam).estimate(scene(cam))
    assert r.pose is None and not r.ok and r.note == "no block in view"


def test_tracking_starts_from_the_last_pose(cam):
    pose = g.sample_start_pose(1)
    est = pv.PoseEstimator(cam)
    first = est.estimate(scene(cam, block=pose))
    moved = (pose[0] + 0.004, pose[1] - 0.003, pose[2] + math.radians(2))
    second = est.estimate(scene(cam, block=moved, seed=3))
    assert first.ok and second.ok
    d, dyaw = pose_err(second.pose, moved)
    assert d < 0.001 and dyaw < math.radians(0.5)
    assert second.n_evals < first.n_evals


def test_wrong_frame_size_raises(cam):
    with pytest.raises(ValueError, match="camera model expects"):
        pv.PoseEstimator(cam).estimate(np.zeros((720, 1280, 3), np.uint8))


def test_config_from_dict():
    cfg = pv.estimator_config_from_dict(
        {"block": {"hue": [50, 90], "sat_min": 70, "val": [10, 250]}, "min_iou": 0.7}
    )
    assert cfg.block.hue == (50, 90) and cfg.min_iou == 0.7
    assert cfg.occluder == pv.default_occluder_threshold()
    with pytest.raises(KeyError):
        pv.estimator_config_from_dict({"nope": 1})
