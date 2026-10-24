import math

import numpy as np
import pytest

from movensys_imitation.tasks.pusht import geometry as g


def _shoelace(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_block_polygon_area_and_extent():
    poly = g.block_polygon_local()
    u = g.UNIT
    assert _shoelace(poly) == pytest.approx(4 * u * u + 3 * u * u)
    assert poly[:, 0].min() == pytest.approx(-2 * u) and poly[:, 0].max() == pytest.approx(2 * u)
    assert poly[:, 1].min() == pytest.approx(-3 * u) and poly[:, 1].max() == pytest.approx(u)


def test_centroid_is_on_the_stem_side():
    c = g.block_centroid_local()
    assert c[0] == 0.0 and -3 * g.UNIT < c[1] < 0.0


def test_goal_centroid_at_sheet_centre():
    pose = g.goal_pose()
    c = g.transform_points(g.block_centroid_local()[None, :], pose)[0]
    assert np.allclose(c, 0.0, atol=1e-12)
    assert pose[2] == pytest.approx(math.pi / 4)


def test_start_pose_deterministic_and_inside_inner_square():
    assert g.sample_start_pose(7) == g.sample_start_pose(7)
    assert g.sample_start_pose(7) != g.sample_start_pose(8)
    for seed in range(200):
        pose = g.sample_start_pose(seed)
        c = g.transform_points(g.block_centroid_local()[None, :], pose)[0]
        assert np.all(np.abs(c) <= np.array(g.INNER_SIZE) / 2)
        assert -math.pi <= pose[2] < math.pi
    assert len(g.start_marks()) == g.N_START_MARKS


def test_quaternion_round_trip():
    for yaw in (-3.0, -1.0, 0.0, 0.5, 2.9):
        assert g.quat_wxyz_to_yaw(*g.yaw_to_quat_wxyz(yaw)) == pytest.approx(yaw)
