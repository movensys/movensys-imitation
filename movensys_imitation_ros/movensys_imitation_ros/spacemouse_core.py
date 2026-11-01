"""
SpaceMouse twist teleop logic without ROS.

:class:`TeleopCore` turns the six SpaceMouse axes into a unitless MoveIt Servo twist
(each component in -1..1, Servo scales by ``scale.linear`` / ``scale.rotational``) once per
control cycle: per-axis deadzone and expo, a motion-mode mask (``MODES``), a yaw rotation of
the operator's axes onto the robot base frame, then the linear and angular gains. The
dead-man, the Joy watchdog and the tool floor are decided here too, so the node only moves
messages.

No command is published on stale input (Servo halts on its own timeout). A released dead-man
publishes a short burst of zero twists so Servo stops cleanly, then goes silent; a held
dead-man with the puck at rest streams zeros so the operator keeps the arm "engaged".
"""

from __future__ import annotations

from dataclasses import dataclass
import math

STATE_HOLD = "hold"
STATE_MOVING = "moving"
STATE_FAULT = "fault"

MODES: dict[str, tuple[float, ...]] = {
    "full": (1, 1, 1, 1, 1, 1),
    "translation": (1, 1, 1, 0, 0, 0),
    "rotation": (0, 0, 0, 1, 1, 1),
    "planar": (1, 1, 0, 0, 0, 1),
    "vertical": (0, 0, 1, 0, 0, 0),
    "yaw": (0, 0, 0, 0, 0, 1),
}
"""Per-component gain ``(tx, ty, tz, rx, ry, rz)`` for each motion mode (same presets as
the Quest teleop)."""

ZERO = (0.0,) * 6


@dataclass
class TeleopConfig:
    """Mapping parameters; defaults are the conservative first-run values (2026-10-29)."""

    axis_map: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    """Joy axis index feeding each twist component ``(tx, ty, tz, rx, ry, rz)``."""
    axis_sign: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    yaw: float = 0.0
    """Rotation (rad, about +z) of the operator's x, y onto the robot base frame; applied to
    the translation and to the roll/pitch pair alike so they never disagree."""
    deadzone: float = 0.05
    expo: float = 1.0
    linear_gain: float = 0.05
    """Fraction of Servo's ``scale.linear`` at full deflection (0.05 x 0.4 m/s = 2 cm/s)."""
    angular_gain: float = 0.05
    """Fraction of Servo's ``scale.rotational`` (0.05 x 0.8 rad/s = 2.3 deg/s)."""
    joy_stale: float = 0.2
    stop_cycles: int = 5
    """Zero twists published after the dead-man is released."""


@dataclass
class TeleopInputs:
    axes: list
    joy_age: float
    deadman: bool
    mode: str = "full"
    eef_z: float | None = None
    """Measured end-effector z, or None when the floor is not enforced."""
    z_floor: float = -math.inf
    """Below this z, downward translation is zeroed (tool floor above the table)."""


@dataclass
class TeleopOutput:
    publish: bool
    twist: tuple[float, ...]
    state: str
    reason: str = ""


def shape(v: float, deadzone: float, expo: float) -> float:
    """Per-axis deadzone, power-law expo on the rest, clipped to -1..1, sign kept."""
    m = min(abs(v), 1.0)
    if m <= deadzone:
        return 0.0
    g = ((m - deadzone) / (1.0 - deadzone)) ** expo
    return math.copysign(g, v)


def joy_to_twist(axes, cfg: TeleopConfig, mode: str) -> tuple[float, ...] | None:
    """Unitless twist from the Joy axes, or None when an axis is missing or the mode unknown."""
    mask = MODES.get(mode)
    if mask is None or max(cfg.axis_map) >= len(axes):
        return None
    v = [
        shape(cfg.axis_sign[i] * axes[cfg.axis_map[i]], cfg.deadzone, cfg.expo) * mask[i]
        for i in range(6)
    ]
    cs, sn = math.cos(cfg.yaw), math.sin(cfg.yaw)
    tx, ty = cs * v[0] - sn * v[1], sn * v[0] + cs * v[1]
    rx, ry = cs * v[3] - sn * v[4], sn * v[3] + cs * v[4]
    lg, ag = cfg.linear_gain, cfg.angular_gain
    return (tx * lg, ty * lg, v[2] * lg, rx * ag, ry * ag, v[5] * ag)


class TeleopCore:
    """See the module docstring."""

    def __init__(self, config: TeleopConfig | None = None):
        self.config = config or TeleopConfig()
        self.stop_left = 0
        self.engaged = False
        self.state = STATE_HOLD

    def update(self, inp: TeleopInputs) -> TeleopOutput:
        """Advance one cycle."""
        c = self.config
        if inp.joy_age > c.joy_stale:
            self.engaged = False
            return self._out(False, ZERO, STATE_HOLD, "joy_stale")
        if not inp.deadman:
            if self.engaged:
                self.stop_left = c.stop_cycles
            self.engaged = False
            if self.stop_left > 0:
                self.stop_left -= 1
                return self._out(True, ZERO, STATE_HOLD, "deadman")
            return self._out(False, ZERO, STATE_HOLD, "deadman")
        self.engaged = True
        twist = joy_to_twist(inp.axes, c, inp.mode)
        if twist is None:
            return self._out(False, ZERO, STATE_FAULT, "bad_axes")
        reason = ""
        if inp.eef_z is not None and inp.eef_z <= inp.z_floor and twist[2] < 0.0:
            twist = twist[:2] + (0.0,) + twist[3:]
            reason = "z_floor"
        if all(v == 0.0 for v in twist):
            return self._out(True, ZERO, STATE_HOLD, reason or "centred")
        return self._out(True, twist, STATE_MOVING, reason)

    def _out(self, publish: bool, twist, state: str, reason: str) -> TeleopOutput:
        self.state = state
        return TeleopOutput(publish, tuple(twist), state, reason)
