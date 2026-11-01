"""Hand-computed cases for the SpaceMouse twist teleop core."""

import math

import pytest

from movensys_imitation_ros.spacemouse_core import (
    joy_to_twist,
    MODES,
    shape,
    STATE_FAULT,
    STATE_HOLD,
    STATE_MOVING,
    TeleopConfig,
    TeleopCore,
    TeleopInputs,
    ZERO,
)

REST = [0.0] * 6


def cfg(**kw) -> TeleopConfig:
    base = {"linear_gain": 1.0, "angular_gain": 1.0, "deadzone": 0.0}
    base.update(kw)
    return TeleopConfig(**base)


def inp(axes=REST, deadman=True, joy_age=0.0, **kw) -> TeleopInputs:
    return TeleopInputs(axes=list(axes), joy_age=joy_age, deadman=deadman, **kw)


# --- shaping ------------------------------------------------------------------------------------


@pytest.mark.parametrize("v", [0.0, 0.03, -0.05, 0.05])
def test_shape_deadzone_is_zero(v):
    assert shape(v, 0.05, 1.0) == 0.0


def test_shape_rescales_from_the_deadzone_edge_and_keeps_sign():
    assert shape(0.525, 0.05, 1.0) == pytest.approx(0.5)
    assert shape(-0.525, 0.05, 1.0) == pytest.approx(-0.5)
    assert shape(1.0, 0.05, 1.0) == pytest.approx(1.0)


def test_shape_clips_overshoot_and_applies_expo():
    assert shape(3.0, 0.0, 1.0) == 1.0
    assert shape(-3.0, 0.0, 1.0) == -1.0
    assert shape(0.5, 0.0, 2.0) == pytest.approx(0.25)


# --- mapping ------------------------------------------------------------------------------------


def test_identity_map_at_unit_gains():
    axes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert joy_to_twist(axes, cfg(), "full") == pytest.approx(tuple(axes))


def test_gains_are_fractions_of_servo_scale():
    axes = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    t = joy_to_twist(axes, cfg(linear_gain=0.05, angular_gain=0.1), "full")
    assert t == pytest.approx((0.05, 0, 0, 0, 0, 0.1))


def test_axis_map_and_sign():
    c = cfg(axis_map=(1, 0, 2, 3, 4, 5), axis_sign=(1, -1, 1, 1, 1, -1))
    t = joy_to_twist([0.2, 0.4, 0.0, 0.0, 0.0, 0.6], c, "full")
    assert t == pytest.approx((0.4, -0.2, 0, 0, 0, -0.6))


def test_yaw_rotates_translation_and_roll_pitch_together():
    c = cfg(yaw=math.pi / 2)
    t = joy_to_twist([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], c, "full")
    assert t == pytest.approx((0.0, 1.0, 0.0, 0.0, 1.0, 0.0), abs=1e-12)


@pytest.mark.parametrize("mode,mask", MODES.items())
def test_modes_mask_components(mode, mask):
    t = joy_to_twist([1.0] * 6, cfg(), mode)
    assert t == pytest.approx(mask)


def test_short_axes_or_unknown_mode_is_none():
    assert joy_to_twist([0.0] * 5, cfg(), "full") is None
    assert joy_to_twist(REST, cfg(), "sideways") is None


# --- core ---------------------------------------------------------------------------------------


def test_stale_joy_publishes_nothing():
    core = TeleopCore(cfg())
    out = core.update(inp([1.0] * 6, joy_age=0.3))
    assert (out.publish, out.state, out.reason) == (False, STATE_HOLD, "joy_stale")


def test_release_sends_a_zero_burst_then_goes_silent():
    core = TeleopCore(cfg(stop_cycles=3))
    assert core.update(inp([0.5] + [0.0] * 5)).state == STATE_MOVING
    outs = [core.update(inp([0.5] + [0.0] * 5, deadman=False)) for _ in range(5)]
    assert [o.publish for o in outs] == [True, True, True, False, False]
    assert all(o.twist == ZERO and o.reason == "deadman" for o in outs)


def test_release_without_engagement_is_silent():
    core = TeleopCore(cfg())
    out = core.update(inp(deadman=False))
    assert (out.publish, out.reason) == (False, "deadman")


def test_held_at_rest_streams_zeros():
    core = TeleopCore(cfg())
    out = core.update(inp())
    assert (out.publish, out.twist, out.state, out.reason) == (True, ZERO, STATE_HOLD, "centred")


def test_moving_passes_the_twist():
    core = TeleopCore(cfg(linear_gain=0.05))
    out = core.update(inp([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    assert out.state == STATE_MOVING and out.twist == pytest.approx((0, 0.05, 0, 0, 0, 0))


def test_floor_blocks_downward_only():
    core = TeleopCore(cfg())
    down = [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]
    out = core.update(inp(down, eef_z=0.26, z_floor=0.2631))
    assert (out.publish, out.twist, out.state, out.reason) == (True, ZERO, STATE_HOLD, "z_floor")
    up = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert core.update(inp(up, eef_z=0.26, z_floor=0.2631)).twist == pytest.approx(tuple(up))
    assert core.update(inp(down, eef_z=0.30, z_floor=0.2631)).twist == pytest.approx(tuple(down))


def test_floor_lets_the_other_axes_move():
    core = TeleopCore(cfg())
    out = core.update(inp([1.0, 0.0, -1.0, 0.0, 0.0, 0.0], eef_z=0.1, z_floor=0.2631))
    assert out.state == STATE_MOVING and out.reason == "z_floor"
    assert out.twist == pytest.approx((1.0, 0, 0, 0, 0, 0))


def test_bad_axes_is_a_fault():
    core = TeleopCore(cfg())
    out = core.update(inp([0.0] * 3))
    assert (out.publish, out.state, out.reason) == (False, STATE_FAULT, "bad_axes")
