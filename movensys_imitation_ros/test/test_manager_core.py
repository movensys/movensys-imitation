"""Episode manager decisions (P2.4): home choice, stop rule, keep-or-drop, replay path."""

import math

import pytest

from movensys_imitation.tasks.pusht import geometry as g
from movensys_imitation_ros import manager_core as mc

CFG = mc.ManagerConfig()


def test_pairs():
    assert mc.pairs([1, 2, 3, 4]) == ((1.0, 2.0), (3.0, 4.0))
    with pytest.raises(ValueError):
        mc.pairs([1, 2, 3])


def test_home_clearance_zero_inside_the_t():
    pose = (-0.03, 0.08, 0.0)  # junction on the centre candidate, bar along +x on the +y side
    assert mc.home_clearance((-0.03, 0.085), pose) == 0.0
    far = mc.home_clearance((0.19, 0.08), pose)  # bar ends at x = -0.03 + 0.06
    assert far == pytest.approx(0.19 - 0.03 - mc.TIP_RADIUS, abs=1e-6)


def test_choose_home_avoids_the_t():
    pose = (-0.03, 0.08, 0.0)
    home, clearance = mc.choose_home(mc.HOME_CANDIDATES, pose, 0.04)
    assert home == (-0.25, 0.08)  # left and right both 0.16 - r from a bar end: tie -> first
    assert clearance > 0.1


def test_choose_home_every_collection_seed_has_a_clear_candidate():
    for seed in range(1000, 1200):
        _, clearance = mc.choose_home(mc.HOME_CANDIDATES, g.sample_start_pose(seed), 0.04)
        assert clearance >= 0.04


def test_choose_home_converts_candidates_to_the_sheet_frame():
    pose = (0.0, 0.0, 0.0)  # T on the sheet origin
    origin = (-0.03, -0.05)  # sheet origin in the planning frame
    on_the_t = (-0.03, -0.05)  # planning-frame point that is the sheet origin
    far = (0.25, 0.10)
    home, clearance = mc.choose_home((on_the_t, far), pose, 0.04, origin)
    assert home == far
    assert clearance == pytest.approx(mc.home_clearance((0.28, 0.15), pose))
    with pytest.raises(ValueError):
        mc.choose_home((on_the_t,), pose, 0.04, origin)


def test_choose_home_raises_when_blocked():
    pose = (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="no home candidate"):
        mc.choose_home(((0.0, 0.01),), pose, 0.04)
    with pytest.raises(ValueError, match="no home candidates"):
        mc.choose_home((), pose, 0.04)


@pytest.mark.parametrize(
    "coverage, n, stop, discard, expected",
    [
        (0.3, 10, False, False, None),
        (math.nan, 10, False, False, None),
        (0.95, 10, False, False, mc.REASON_SUCCESS),
        (0.3, 300, False, False, mc.REASON_MAX_STEPS),
        (0.3, 299, False, False, None),
        (0.99, 300, True, False, mc.REASON_STOP),
        (0.99, 300, True, True, mc.REASON_DISCARD),
    ],
)
def test_stop_reason(coverage, n, stop, discard, expected):
    assert mc.stop_reason(coverage, n, CFG, stop, discard) == expected


SCORE = {"primary": "geometric", "geometric": {"coverage": 0.97, "iou": 0.9, "valid": True},
         "pixel": {"coverage": 0.5, "iou": 0.4, "valid": False}}


def test_primary_score_follows_the_scorer():
    assert mc.primary_score(SCORE)["coverage"] == 0.97
    assert mc.primary_score({**SCORE, "primary": "pixel"})["coverage"] == 0.5
    assert mc.primary_score(None) is None
    assert mc.primary_score({"primary": "geometric", "geometric": None}) is None


def test_decide_outcome():
    o = mc.decide_outcome(mc.REASON_SUCCESS, SCORE, True, CFG)
    assert o.save and o.success and o.coverage == 0.97 and o.iou == 0.9
    o = mc.decide_outcome(mc.REASON_MAX_STEPS, {**SCORE, "primary": "pixel"}, True, CFG)
    assert o.save and not o.success  # pixel backend invalid: never a success
    assert not mc.decide_outcome(mc.REASON_DISCARD, SCORE, True, CFG).save
    assert not mc.decide_outcome(mc.REASON_SUCCESS, SCORE, False, CFG).save
    keep = mc.ManagerConfig(discard_on_timing_fault=False)
    assert mc.decide_outcome(mc.REASON_SUCCESS, SCORE, False, keep).save
    drop_fail = mc.ManagerConfig(keep_failures=False)
    assert not mc.decide_outcome(mc.REASON_MAX_STEPS, {**SCORE, "primary": "pixel"}, True,
                                 drop_fail).save
    o = mc.decide_outcome(mc.REASON_STOP, None, True, CFG)
    assert o.save and not o.success and math.isnan(o.coverage)


def test_path_from_actions_is_timed_at_fps():
    path = mc.path_from_actions([(0.0, 0.0), (0.1, 0.0), (0.2, 0.1)], 100.0, 10)
    assert [p[0] for p in path] == pytest.approx([100.0, 100.1, 100.2])
    assert path[2][1:] == (0.2, 0.1)


def test_pose_difference_wraps_yaw():
    d, dyaw = mc.pose_difference((0.0, 0.0, math.pi - 0.01), (0.003, 0.004, -math.pi + 0.01))
    assert d == pytest.approx(0.005) and dyaw == pytest.approx(0.02)


def test_episode_meta_fields():
    m = mc.episode_meta(1000, (0.1, 0.2, 0.3), (-0.03, 0.08), "me", "sim", {"x": 1})
    assert m["seed"] == 1000 and m["home_xy"] == [-0.03, 0.08] and m["x"] == 1
