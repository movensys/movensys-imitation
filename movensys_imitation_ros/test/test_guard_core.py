"""Hand-computed cases for the planar guard core (plan P2.2)."""

import math

import pytest

from movensys_imitation_ros.guard_core import (
    GuardConfig,
    GuardCore,
    Inputs,
    SOURCE_POSE,
    SOURCE_STICK,
    STATE_FAULT,
    STATE_HOLD,
    STATE_INIT,
    STATE_MOVING,
)

DT = 0.02
CENTRE = (-0.03, -0.05)


def make(**overrides) -> GuardCore:
    return GuardCore(GuardConfig(**overrides))


def inp(eef=CENTRE, **kw) -> Inputs:
    return Inputs(dt=DT, eef_xy=eef, **kw)


# --- area ---------------------------------------------------------------------------------


def test_area_bounds_from_scene_numbers():
    c = GuardConfig()
    assert c.area_min == pytest.approx((-0.33, -0.20))
    assert c.area_max == pytest.approx((0.27, 0.10))


@pytest.mark.parametrize(
    "point, expected",
    [
        ((0.0, 0.0), (0.0, 0.0)),
        ((1.0, 0.0), (0.27, 0.0)),
        ((-1.0, 0.0), (-0.33, 0.0)),
        ((0.0, 1.0), (0.0, 0.10)),
        ((0.0, -1.0), (0.0, -0.20)),
        ((1.0, -1.0), (0.27, -0.20)),
    ],
)
def test_clamp_to_each_edge_and_corner(point, expected):
    assert make().clamp(*point) == pytest.approx(expected)


# --- stick mapping ------------------------------------------------------------------------


def test_stick_deadzone_gives_zero():
    g = make(deadzone=0.15)
    assert g.stick_to_velocity(0.1, 0.1) == (0.0, 0.0)
    assert g.stick_to_velocity(0.0, 0.0) == (0.0, 0.0)


def test_stick_full_deflection_gives_max_speed_along_axis():
    g = make(max_speed=0.12, expo=2.0)
    assert g.stick_to_velocity(1.0, 0.0) == pytest.approx((0.12, 0.0))
    assert g.stick_to_velocity(0.0, -1.0) == pytest.approx((0.0, -0.12))


def test_stick_expo_is_quadratic_above_deadzone():
    g = make(max_speed=1.0, deadzone=0.0, expo=2.0)
    assert g.stick_to_velocity(0.5, 0.0) == pytest.approx((0.25, 0.0))
    g = make(max_speed=1.0, deadzone=0.2, expo=2.0)
    # magnitude 0.6 -> (0.6 - 0.2) / 0.8 = 0.5 -> 0.25
    assert g.stick_to_velocity(0.6, 0.0) == pytest.approx((0.25, 0.0))


def test_stick_is_symmetric_and_magnitude_saturates():
    g = make(max_speed=0.1, expo=1.0, deadzone=0.0)
    vx, vy = g.stick_to_velocity(1.0, 1.0)
    assert math.hypot(vx, vy) == pytest.approx(0.1)
    assert g.stick_to_velocity(-1.0, 0.0) == pytest.approx((-0.1, 0.0))


def test_stick_yaw_and_inversion():
    g = make(max_speed=1.0, expo=1.0, deadzone=0.0, stick_yaw=math.pi / 2)
    assert g.stick_to_velocity(1.0, 0.0) == pytest.approx((0.0, 1.0))
    g = make(max_speed=1.0, expo=1.0, deadzone=0.0, stick_invert_y=True)
    assert g.stick_to_velocity(0.0, 1.0) == pytest.approx((0.0, -1.0))


# --- step limit ---------------------------------------------------------------------------


def test_step_limit_caps_the_move_and_keeps_direction():
    g = make(step_limit=0.005)
    new = g.limit_step((0.0, 0.0), (0.03, 0.04))
    assert math.hypot(*new) == pytest.approx(0.005)
    assert new[0] / new[1] == pytest.approx(0.75)
    assert g.limit_step((0.0, 0.0), (0.001, 0.0)) == (0.001, 0.0)


# --- cycle behaviour ----------------------------------------------------------------------


def test_no_eef_pose_is_a_fault_and_nothing_is_published():
    g = make()
    out = g.update(inp(eef=None))
    assert out.publish is False and out.state == STATE_FAULT and out.reason == "eef_stale"
    out = g.update(inp(eef_age=1.0))
    assert out.publish is False and out.reason == "eef_stale"
    assert g.target is None


def test_first_cycle_anchors_on_the_measured_eef():
    g = make()
    out = g.update(inp(eef=(0.1, -0.1), deadman=False))
    assert out.publish is True and out.state == STATE_HOLD and out.reason == "deadman"
    assert out.target == pytest.approx((0.1, -0.1))


def test_eef_outside_area_at_start_is_a_fault():
    g = make()
    out = g.update(inp(eef=(0.5, 0.0)))
    assert out.publish is False and out.reason == "eef_outside_area"
    assert g.state == STATE_FAULT and g.target is None
    # Just outside, within the EEF-distance limit: anchor on the clamped point.
    out = g.update(inp(eef=(0.29, 0.0), deadman=False))
    assert out.publish is True and out.target == pytest.approx((0.27, 0.0))


def test_stick_integrates_velocity_per_cycle():
    g = make(max_speed=0.1, expo=1.0, deadzone=0.0)
    out = g.update(inp(stick=(1.0, 0.0)))
    assert out.state == STATE_MOVING
    assert out.target == pytest.approx((CENTRE[0] + 0.1 * DT, CENTRE[1]))


def test_centred_stick_holds():
    g = make()
    g.update(inp(stick=(1.0, 0.0)))
    before = g.target
    out = g.update(inp(stick=(0.0, 0.0)))
    assert out.state == STATE_HOLD and out.reason == "stick_centred"
    assert out.target == before


def test_released_deadman_holds_in_both_sources():
    for source in (SOURCE_STICK, SOURCE_POSE):
        g = GuardCore(GuardConfig(), source=source)
        g.update(inp(stick=(1.0, 0.0), pose_cmd=(0.0, 0.0)))
        before = g.target
        out = g.update(inp(stick=(1.0, 0.0), pose_cmd=(0.0, 0.0), deadman=False))
        assert out.publish is True and out.state == STATE_HOLD and out.reason == "deadman"
        assert out.target == before


def test_stale_joy_holds():
    g = make(joy_stale=0.2)
    out = g.update(inp(stick=(1.0, 0.0), joy_age=0.5))
    assert out.publish is True and out.state == STATE_HOLD and out.reason == "joy_stale"


def test_clamp_stops_the_target_at_the_edge_without_windup():
    g = make(max_speed=1.0, expo=1.0, deadzone=0.0, step_limit=1.0, eef_distance_limit=10.0)
    eef = (0.26, 0.0)
    g.update(inp(eef=eef, stick=(1.0, 0.0)))
    for _ in range(20):
        out = g.update(inp(eef=eef, stick=(1.0, 0.0)))
    assert out.target == pytest.approx((0.27, 0.0))
    # Reverse: the very next cycle moves back; no accumulated overshoot to unwind.
    out = g.update(inp(eef=eef, stick=(-1.0, 0.0)))
    assert out.target[0] < 0.27 - 0.01


def test_eef_distance_limit_holds_the_lead():
    g = make(max_speed=1.0, expo=1.0, deadzone=0.0, step_limit=0.01, eef_distance_limit=0.03)
    eef = CENTRE
    for _ in range(3):
        out = g.update(inp(eef=eef, stick=(1.0, 0.0)))
    assert out.state == STATE_MOVING
    assert out.target[0] - eef[0] == pytest.approx(0.03)
    out = g.update(inp(eef=eef, stick=(1.0, 0.0)))
    assert out.state == STATE_HOLD and out.reason == "eef_distance"
    assert out.target[0] - eef[0] == pytest.approx(0.03)
    # The arm catches up: motion resumes.
    out = g.update(inp(eef=(eef[0] + 0.02, eef[1]), stick=(1.0, 0.0)))
    assert out.state == STATE_MOVING


def test_servo_halt_reanchors_on_the_measured_eef():
    g = make(max_speed=1.0, expo=1.0, deadzone=0.0, step_limit=0.01)
    g.update(inp(stick=(1.0, 0.0)))
    g.update(inp(stick=(1.0, 0.0)))
    assert g.target[0] > CENTRE[0]
    out = g.update(inp(stick=(1.0, 0.0), servo_halted=True))
    assert out.publish is True and out.state == STATE_HOLD and out.reason == "servo_halted"
    assert out.target == pytest.approx(CENTRE)


def test_pose_source_steps_toward_the_command_and_clamps_it():
    g = GuardCore(GuardConfig(step_limit=0.005, eef_distance_limit=1.0), source=SOURCE_POSE)
    out = g.update(inp(pose_cmd=(1.0, CENTRE[1])))
    assert out.state == STATE_MOVING
    assert out.target == pytest.approx((CENTRE[0] + 0.005, CENTRE[1]))
    out = g.update(inp(pose_cmd=(CENTRE[0] + 0.005, CENTRE[1])))
    assert out.state == STATE_HOLD and out.reason == "at_target"
    out = g.update(inp(pose_cmd=(1.0, CENTRE[1]), pose_age=2.0))
    assert out.state == STATE_HOLD and out.reason == "pose_stale"


def test_reset_and_set_source():
    g = make()
    g.update(inp(stick=(1.0, 0.0)))
    g.reset()
    assert g.target is None and g.state == STATE_INIT
    g.set_source(SOURCE_POSE)
    assert g.source == SOURCE_POSE
    with pytest.raises(ValueError):
        g.set_source("hand")


# --- lift ramp (P2.4) ------------------------------------------------------------------------


def test_ramp_moves_at_most_step_limit_and_lands_exactly():
    core = make(step_limit=0.005)
    z = 0.0
    steps = 0
    while z != 0.08:
        z = core.ramp(z, 0.08)
        steps += 1
    assert steps == 16  # 0.08 / 0.005
    assert core.ramp(0.08, 0.0) == pytest.approx(0.075)
    assert core.ramp(0.003, 0.0) == 0.0
