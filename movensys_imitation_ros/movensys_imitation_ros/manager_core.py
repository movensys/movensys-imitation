"""
Episode manager decisions without ROS (plan P2.4).

The manager node (:mod:`episode_manager`) drives reset, record, score and save; the pieces that
can be wrong in a way a unit test catches live here: where to park the tip so the T can be
teleported in without touching it, when an episode stops, whether it is kept, and how recorded
actions become a timed path for the replay check.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from shapely.geometry import Point

from movensys_imitation.tasks.pusht import geometry as g
from movensys_imitation.tasks.pusht import scoring as sc

TIP_RADIUS = 0.010
HOME_CANDIDATES = ((-0.03, 0.08), (-0.25, 0.08), (0.19, 0.08))
"""Default parking points (task.yaml ``home_candidates``), near the robot inside the push area."""

REASON_DISCARD = "discard"
REASON_STOP = "stop"
REASON_SUCCESS = "success"
REASON_MAX_STEPS = "max_steps"


@dataclass
class ManagerConfig:
    success_threshold: float = sc.SUCCESS_THRESHOLD
    max_steps: int = 300
    min_home_clearance: float = 0.04
    """Metres between the tip edge and the T outline at the chosen home."""
    home_candidates: tuple[tuple[float, float], ...] = HOME_CANDIDATES
    discard_on_timing_fault: bool = True
    keep_failures: bool = True
    """Save episodes that end without success (P2.6 filters on the sidecar's ``success``)."""


def pairs(flat) -> tuple[tuple[float, float], ...]:
    """``[x0, y0, x1, y1, ...]`` (a ROS parameter) to pairs."""
    flat = [float(v) for v in flat]
    if len(flat) % 2:
        raise ValueError("home_candidates needs an even number of values")
    return tuple((flat[i], flat[i + 1]) for i in range(0, len(flat), 2))


def home_clearance(xy, block_pose) -> float:
    """Distance from the tip's edge at ``xy`` to the T outline at ``block_pose`` (0 if inside)."""
    return max(0.0, sc.block_shape(block_pose).distance(Point(xy)) - TIP_RADIUS)


def choose_home(candidates, block_pose, min_clearance: float) -> tuple[tuple[float, float], float]:
    """Pick the candidate farthest from the T; ``ValueError`` if none clears ``min_clearance``."""
    if not candidates:
        raise ValueError("no home candidates")
    best = max(candidates, key=lambda xy: home_clearance(xy, block_pose))
    clearance = home_clearance(best, block_pose)
    if clearance < min_clearance:
        raise ValueError(
            f"no home candidate clears the T at {tuple(round(v, 3) for v in block_pose)} by "
            f"{min_clearance} m (best {clearance:.3f} m)"
        )
    return best, clearance


def stop_reason(
    coverage: float,
    n_frames: int,
    cfg: ManagerConfig,
    stop_requested: bool = False,
    discard_requested: bool = False,
) -> str | None:
    """Return why the episode ends now, or None to keep recording."""
    if discard_requested:
        return REASON_DISCARD
    if stop_requested:
        return REASON_STOP
    if not math.isnan(coverage) and coverage >= cfg.success_threshold:
        return REASON_SUCCESS
    if n_frames >= cfg.max_steps:
        return REASON_MAX_STEPS
    return None


def primary_score(score_json: dict | None) -> dict | None:
    """Return the scorer service's primary backend result (``coverage``, ``iou``, ``success``)."""
    if not score_json:
        return None
    primary = score_json.get("primary", "geometric")
    return score_json.get(primary) or None


@dataclass
class Outcome:
    save: bool
    success: bool
    reason: str
    coverage: float = math.nan
    iou: float = math.nan


def decide_outcome(
    reason: str, score_json: dict | None, timing_ok: bool, cfg: ManagerConfig
) -> Outcome:
    """Keep or drop the episode; success comes from the end score, not from the live stop."""
    score = primary_score(score_json)
    coverage = float(score["coverage"]) if score else math.nan
    iou = float(score["iou"]) if score else math.nan
    success = bool(score and score.get("valid", True) and coverage >= cfg.success_threshold)
    if reason == REASON_DISCARD:
        return Outcome(False, success, "operator discard", coverage, iou)
    if not timing_ok and cfg.discard_on_timing_fault:
        return Outcome(False, success, "timing fault", coverage, iou)
    if not success and not cfg.keep_failures:
        return Outcome(False, success, "failure not kept", coverage, iou)
    return Outcome(True, success, reason, coverage, iou)


def episode_meta(seed: int, start_pose, home, operator: str, mode: str, extra: dict | None = None):
    """Build the sidecar fields known before an episode starts."""
    meta = {
        "seed": int(seed),
        "start_pose": [round(float(v), 5) for v in start_pose],
        "home_xy": [round(float(v), 4) for v in home],
        "operator": operator,
        "mode": mode,
    }
    if extra:
        meta.update(extra)
    return meta


def path_from_actions(actions, t0: float, fps: float) -> list[tuple[float, float, float]]:
    """Turn recorded 10 Hz actions into ``(t, x, y)`` waypoints from ``t0`` (replay check).

    The guard interpolates linearly between them, so the replayed arm path is the recorded
    target sampled at ``fps``, not the original 50 Hz stream.
    """
    return [(t0 + i / fps, float(a[0]), float(a[1])) for i, a in enumerate(actions)]


def pose_difference(a, b) -> tuple[float, float]:
    """Position (m) and yaw (rad, wrapped) difference between two block poses."""
    d = math.hypot(a[0] - b[0], a[1] - b[1])
    dyaw = abs((a[2] - b[2] + math.pi) % (2 * math.pi) - math.pi)
    return d, dyaw


def start_pose_for_seed(seed: int) -> tuple[float, float, float]:
    return g.sample_start_pose(seed)
