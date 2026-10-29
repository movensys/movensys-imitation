"""
Planar guard logic without ROS (plan P2.2).

:class:`GuardCore` owns the guarded x, y target in the planning frame and advances it once per
control cycle from one of two sources: the Quest thumbstick, integrated as a planar velocity, or
an absolute pose command (policy runner, replay, hand teleop). Every cycle it clamps the target to
the push area, bounds the step, and refuses to lead the measured end effector by more than the
EEF-distance limit. z and orientation are not handled here: the node latches them from the
measured pose at start and overwrites them on every outgoing command.

Faults (no end-effector pose, end effector outside the area at start) make :attr:`Output.publish`
false, so the node stops streaming and Servo halts on its own command timeout. Everything else is
a hold: the last target is republished and the arm stays where it is.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

EPS = 1e-9
"""Float tolerance for "moved" and "at the limit" decisions (metres)."""

STATE_INIT = "init"
STATE_FAULT = "fault"
STATE_HOLD = "hold"
STATE_MOVING = "moving"

SOURCE_STICK = "stick"
SOURCE_POSE = "pose"
SOURCES = (SOURCE_STICK, SOURCE_POSE)


def deadman_held(
    axes, buttons, axis: int = -1, button: int = -1, threshold: float = 0.6
) -> bool:
    """
    Return True while the operator holds the dead-man.

    ``button`` >= 0 reads ``Joy.buttons[button]`` (SpaceMouse: a momentary side key, held with
    the free hand); otherwise ``axis`` >= 0 reads ``Joy.axes[axis] >= threshold`` (Quest: the
    analog grip). A missing index, or neither selector set, is "not held".
    """
    if button >= 0:
        return button < len(buttons) and buttons[button] != 0
    if axis >= 0:
        return axis < len(axes) and axes[axis] >= threshold
    return False


@dataclass
class GuardConfig:
    """Guard parameters; defaults are plan Section 5 and scene_9a.json."""

    area_center: tuple[float, float] = (-0.03, -0.05)
    area_size: tuple[float, float] = (0.60, 0.30)
    step_limit: float = 0.005
    eef_distance_limit: float = 0.030
    max_speed: float = 0.12
    deadzone: float = 0.15
    expo: float = 2.0
    stick_yaw: float = 0.0
    stick_invert_x: bool = False
    stick_invert_y: bool = False
    joy_stale: float = 0.2
    eef_stale: float = 0.5
    pose_stale: float = 0.5

    @property
    def area_min(self) -> tuple[float, float]:
        return (
            self.area_center[0] - self.area_size[0] / 2,
            self.area_center[1] - self.area_size[1] / 2,
        )

    @property
    def area_max(self) -> tuple[float, float]:
        return (
            self.area_center[0] + self.area_size[0] / 2,
            self.area_center[1] + self.area_size[1] / 2,
        )


@dataclass
class Inputs:
    """Everything the core needs for one cycle; ages are seconds since arrival."""

    dt: float
    eef_xy: tuple[float, float] | None
    eef_age: float = 0.0
    servo_halted: bool = False
    deadman: bool = True
    joy_age: float = 0.0
    stick: tuple[float, float] | None = None
    pose_cmd: tuple[float, float] | None = None
    pose_age: float = 0.0


@dataclass
class Output:
    """Result of one cycle."""

    publish: bool
    target: tuple[float, float] | None
    state: str
    reason: str = ""


class GuardCore:
    """Guarded planar target; see the module docstring."""

    def __init__(self, config: GuardConfig | None = None, source: str = SOURCE_STICK):
        self.config = config or GuardConfig()
        self.source = source
        self.target: tuple[float, float] | None = None
        self.state = STATE_INIT

    # --- pure helpers ---------------------------------------------------------------------

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        """Clamp a point to the push area."""
        lo, hi = self.config.area_min, self.config.area_max
        return (min(max(x, lo[0]), hi[0]), min(max(y, lo[1]), hi[1]))

    def inside(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Return True if the point is within ``margin`` of the push area."""
        cx, cy = self.clamp(x, y)
        return math.hypot(cx - x, cy - y) <= margin

    def stick_to_velocity(self, sx: float, sy: float) -> tuple[float, float]:
        """
        Map a thumbstick deflection to a planar velocity in the planning frame.

        Radial deadzone, then a power-law expo on the remaining magnitude, direction kept;
        optional per-axis inversion; rotation by ``stick_yaw``; scaled to ``max_speed``.
        """
        c = self.config
        m = math.hypot(sx, sy)
        if m <= c.deadzone or m == 0.0:
            return (0.0, 0.0)
        gain = min((m - c.deadzone) / (1.0 - c.deadzone), 1.0) ** c.expo
        ux, uy = sx / m, sy / m
        if c.stick_invert_x:
            ux = -ux
        if c.stick_invert_y:
            uy = -uy
        cs, sn = math.cos(c.stick_yaw), math.sin(c.stick_yaw)
        rx, ry = cs * ux - sn * uy, sn * ux + cs * uy
        return (rx * gain * c.max_speed, ry * gain * c.max_speed)

    def limit_step(
        self, current: tuple[float, float], desired: tuple[float, float]
    ) -> tuple[float, float]:
        """Move from ``current`` toward ``desired`` by at most ``step_limit``."""
        dx, dy = desired[0] - current[0], desired[1] - current[1]
        n = math.hypot(dx, dy)
        if n <= self.config.step_limit or n == 0.0:
            return desired
        k = self.config.step_limit / n
        return (current[0] + dx * k, current[1] + dy * k)

    # --- state ------------------------------------------------------------------------------

    def ramp(self, current: float, goal: float) -> float:
        """Move a scalar (the lift offset) toward ``goal`` by at most ``step_limit``."""
        d = goal - current
        if abs(d) <= self.config.step_limit:
            return goal
        return current + math.copysign(self.config.step_limit, d)

    def reset(self) -> None:
        """Forget the target; the next cycle re-anchors on the measured end effector."""
        self.target = None
        self.state = STATE_INIT

    def set_source(self, source: str) -> None:
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
        self.source = source

    def update(self, inp: Inputs) -> Output:
        """Advance one cycle."""
        c = self.config
        if inp.eef_xy is None or inp.eef_age > c.eef_stale:
            return self._out(False, STATE_FAULT, "eef_stale")
        eef = inp.eef_xy

        if self.target is None:
            if not self.inside(*eef, margin=c.eef_distance_limit):
                return self._out(False, STATE_FAULT, "eef_outside_area")
            self.target = self.clamp(*eef)

        if inp.servo_halted:
            # Servo stopped (singularity, collision, joint bound). Re-anchor on the measured pose
            # so the held target stops pushing into the stop and the operator can back out.
            self.target = self.clamp(*eef)
            return self._out(True, STATE_HOLD, "servo_halted")

        desired, reason = self._desired(inp)
        if desired is None:
            return self._out(True, STATE_HOLD, reason)

        new = self.limit_step(self.target, self.clamp(*desired))
        if math.hypot(new[0] - eef[0], new[1] - eef[1]) > c.eef_distance_limit + EPS:
            return self._out(True, STATE_HOLD, "eef_distance")
        moved = math.hypot(new[0] - self.target[0], new[1] - self.target[1]) > EPS
        self.target = new
        return self._out(True, STATE_MOVING if moved else STATE_HOLD, "" if moved else reason)

    def _desired(self, inp: Inputs) -> tuple[tuple[float, float] | None, str]:
        c = self.config
        if not inp.deadman:
            return None, "deadman"
        if self.source == SOURCE_STICK:
            if inp.stick is None or inp.joy_age > c.joy_stale:
                return None, "joy_stale"
            vx, vy = self.stick_to_velocity(*inp.stick)
            if vx == 0.0 and vy == 0.0:
                return None, "stick_centred"
            return (self.target[0] + vx * inp.dt, self.target[1] + vy * inp.dt), ""
        if inp.pose_cmd is None or inp.pose_age > c.pose_stale:
            return None, "pose_stale"
        return inp.pose_cmd, "at_target"

    def _out(self, publish: bool, state: str, reason: str) -> Output:
        self.state = state
        return Output(publish, self.target, state, reason)
