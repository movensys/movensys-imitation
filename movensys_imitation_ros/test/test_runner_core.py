"""Runner core (P2.8): recorder-state gate, slow-mode step limit, area clamp, frame shift."""

import math

import pytest

from movensys_imitation_ros import runner_core as rn


def test_parse_recorder_state():
    assert rn.parse_recorder_state("recording:12") == ("recording", 12)
    assert rn.parse_recorder_state("idle") == ("idle", -1)
    assert rn.parse_recorder_state("fault:save") == ("fault", -1)


def test_gate_follows_recorder():
    g = rn.EpisodeGate("recorder")
    assert g.update("idle") is None
    assert g.update("recording:0") == "start"
    assert g.update("recording:1") == "run"
    assert g.update("stopped:57") == "stop"
    assert g.update("idle") is None
    assert g.update("recording:0") == "start"


def test_gate_always():
    g = rn.EpisodeGate("always")
    assert g.update("") == "start"
    assert g.update("idle") == "run"
    with pytest.raises(ValueError):
        rn.EpisodeGate("sometimes")


def test_limit_step():
    assert rn.limit_step((0, 0), (0.004, 0.003), 0.010) == (0.004, 0.003)
    x, y = rn.limit_step((0, 0), (0.08, 0.06), 0.010)
    assert math.isclose(math.hypot(x, y), 0.010)
    assert math.isclose(y / x, 0.75)
    assert rn.limit_step((1, 1), (5, 5), 0) == (5.0, 5.0)


def test_clamp_area_and_frames():
    assert rn.clamp_area((0.5, -0.4), (0.60, 0.30)) == pytest.approx((0.30, -0.15))
    assert rn.clamp_area((0.5, -0.4), (0.60, 0.30), margin=0.02) == pytest.approx((0.28, -0.13))
    origin = (-0.03, -0.05)
    sheet = rn.to_sheet((0.07, 0.05), origin)
    assert sheet == pytest.approx((0.10, 0.10))
    assert rn.to_planning(sheet, origin) == pytest.approx((0.07, 0.05))
    assert rn.finite_xy((0.1, 0.2)) and not rn.finite_xy((float("nan"), 0.0))
