"""Policy runner logic without ROS (plan P2.8).

The runner drives the planar guard in ``pose`` source with one target per 10 Hz tick while the
recorder is recording; everything else (reset, teleport, lift, scoring, saving) stays with the
episode manager. This module holds the pieces that can be unit tested: the recorder-state gate,
the step limit of the ``slow`` mode, the area clamp and the frame shift.
"""

from __future__ import annotations

import math

RUN_MODES = ("shadow", "slow", "normal")
TRIGGERS = ("recorder", "always")


def parse_recorder_state(state: str) -> tuple[str, int]:
    """``"recording:12"`` -> ``("recording", 12)``; states without a count give -1."""
    phase, _, rest = state.partition(":")
    try:
        return phase, int(rest)
    except ValueError:
        return phase, -1


class EpisodeGate:
    """Decide per tick whether the policy runs, from the recorder's ``~/state`` topic.

    ``update`` returns ``"start"`` on the first tick of a recording (the policy history must be
    reset), ``"run"`` while it continues, ``"stop"`` once when it ends and ``None`` otherwise.
    With trigger ``always`` the policy runs on every tick after an initial ``"start"``.
    """

    def __init__(self, trigger: str = "recorder"):
        if trigger not in TRIGGERS:
            raise ValueError(f"trigger must be one of {TRIGGERS}, got {trigger!r}")
        self.trigger = trigger
        self.active = False

    def update(self, recorder_state: str) -> str | None:
        if self.trigger == "always":
            want = True
        else:
            want = parse_recorder_state(recorder_state)[0] == "recording"
        if want and not self.active:
            self.active = True
            return "start"
        if want:
            return "run"
        if self.active:
            self.active = False
            return "stop"
        return None


def limit_step(state_xy, target_xy, max_step: float) -> tuple[float, float]:
    """Pull ``target`` back to within ``max_step`` of ``state`` (``slow`` mode, metres)."""
    dx, dy = target_xy[0] - state_xy[0], target_xy[1] - state_xy[1]
    d = math.hypot(dx, dy)
    if max_step <= 0 or d <= max_step:
        return (float(target_xy[0]), float(target_xy[1]))
    k = max_step / d
    return (float(state_xy[0] + dx * k), float(state_xy[1] + dy * k))


def clamp_area(xy, area_size, margin: float = 0.0) -> tuple[float, float]:
    """Clamp a sheet-frame point to the push area (centre 0, 0) minus ``margin`` per side."""
    hx, hy = area_size[0] / 2 - margin, area_size[1] / 2 - margin
    return (float(min(max(xy[0], -hx), hx)), float(min(max(xy[1], -hy), hy)))


def to_sheet(xy, origin) -> tuple[float, float]:
    return (float(xy[0] - origin[0]), float(xy[1] - origin[1]))


def to_planning(xy, origin) -> tuple[float, float]:
    return (float(xy[0] + origin[0]), float(xy[1] + origin[1]))


def finite_xy(xy) -> bool:
    return len(xy) >= 2 and all(math.isfinite(float(v)) for v in xy[:2])
