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

TIP_RADIUS = 0.015  # 30 mm rod (plan v2.6)
HOME_CANDIDATES = ((-0.03, 0.08), (-0.25, 0.08), (0.19, 0.08))
"""Default parking points (task.yaml ``home_candidates``), near the robot inside the push area."""

FIRST_COLLECTION_SEED = 1000
"""Collection seeds are 1000 and up (plan Section 5); 0 to 49 evaluate."""

START_UNIFORM = "uniform"
START_NEAR_GOAL = "near_goal"
START_MODES = (START_UNIFORM, START_NEAR_GOAL)
"""Where the T starts (``start_mode``): anywhere in the inner rectangle at any yaw, or over the
goal and badly rotated (extra data for the state the policy gets stuck in)."""

REASON_DISCARD = "discard"
REASON_STOP = "stop"
REASON_SUCCESS = "success"
REASON_MAX_STEPS = "max_steps"


@dataclass
class ManagerConfig:
    success_threshold: float = sc.SUCCESS_THRESHOLD
    success_hold: int = 5
    """Consecutive ``/pusht/score`` samples at or above the threshold before the episode ends on
    success (0.5 s at the scorer's 10 Hz). A single crossing while the T is still sliding ended
    episodes that then scored below the threshold once the T came to rest (2026-11-03), and the
    recorder, ticking at the same 10 Hz, never stored the crossing sample."""
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


def choose_home(
    candidates, block_pose, min_clearance: float, origin=(0.0, 0.0)
) -> tuple[tuple[float, float], float]:
    """Pick the candidate farthest from the T; ``ValueError`` if none clears ``min_clearance``.

    Candidates are in the planning frame, the T pose in the sheet frame whose origin is
    ``origin`` (the push-area centre) in the planning frame. The chosen home is returned in
    the planning frame.
    """
    if not candidates:
        raise ValueError("no home candidates")
    ox, oy = origin

    def clearance_of(xy):
        return home_clearance((xy[0] - ox, xy[1] - oy), block_pose)

    best = max(candidates, key=clearance_of)
    clearance = clearance_of(best)
    if clearance < min_clearance:
        raise ValueError(
            f"no home candidate clears the T at {tuple(round(v, 3) for v in block_pose)} by "
            f"{min_clearance} m (best {clearance:.3f} m)"
        )
    return best, clearance


@dataclass
class Placement:
    """Verdict on a hand-placed T (real-robot reset); ``note`` says what to fix when not ok."""

    ok: bool
    d: float = math.nan  # m from the requested start pose
    dyaw: float = math.nan  # rad
    note: str = ""


def check_placement(
    measured, age: float, target, home_sheet_xy, tol_m: float, tol_rad: float,
    max_age: float = 1.0, min_clearance: float = 0.02,
) -> Placement:
    """Judge the T the operator placed against the requested start pose (all in the sheet frame).

    ``measured`` is the latest pose from ``t_pose_estimator`` and ``age`` its age in seconds: the
    estimator publishes good fits only, so a hand over the T or a T outside the push area shows
    up as a stale pose. ``home_sheet_xy`` is the parked tip, which lowers next and must not land
    on the T.
    """
    if measured is None or age > max_age:
        return Placement(False, note="no block pose from the camera (T hidden or out of view)")
    d, dyaw = pose_difference(measured, target)
    if d > tol_m or dyaw > tol_rad:
        return Placement(
            False, d, dyaw,
            f"T is {d * 1000:.0f} mm and {math.degrees(dyaw):.0f} deg from the start pose "
            f"(allowed {tol_m * 1000:.0f} mm, {math.degrees(tol_rad):.0f} deg)",
        )
    clearance = home_clearance(home_sheet_xy, measured)
    if clearance < min_clearance:
        return Placement(
            False, d, dyaw, f"T is {clearance * 1000:.0f} mm from the parked tip; move it away"
        )
    return Placement(True, d, dyaw)


def success_run(run: int, coverage: float, threshold: float) -> int:
    """Consecutive samples at or above ``threshold`` after this one (NaN or below resets to 0)."""
    if math.isnan(coverage) or coverage < threshold:
        return 0
    return run + 1


def stop_reason(
    success_samples: int,
    n_frames: int,
    cfg: ManagerConfig,
    stop_requested: bool = False,
    discard_requested: bool = False,
) -> str | None:
    """Return why the episode ends now, or None to keep recording.

    ``success_samples`` is the current :func:`success_run`: success needs ``cfg.success_hold``
    consecutive samples so a T that is still moving through the goal does not end the episode.
    """
    if discard_requested:
        return REASON_DISCARD
    if stop_requested:
        return REASON_STOP
    if success_samples >= max(1, cfg.success_hold):
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


def start_pose_for_seed(
    seed: int,
    start_mode: str = START_UNIFORM,
    near_goal_radius: float = g.NEAR_GOAL_RADIUS,
    near_goal_yaw_deg=tuple(math.degrees(v) for v in g.NEAR_GOAL_YAW_ERROR),
) -> tuple[float, float, float]:
    """Start pose of the T for ``seed`` under ``start_mode`` (one of :data:`START_MODES`).

    ``near_goal_radius`` (m) and ``near_goal_yaw_deg`` (min, max size of the yaw error to the
    goal) only apply to ``near_goal``; see :func:`geometry.sample_near_goal_pose`.
    """
    if start_mode == START_UNIFORM:
        return g.sample_start_pose(seed)
    if start_mode == START_NEAR_GOAL:
        if len(near_goal_yaw_deg) != 2:
            raise ValueError(f"near_goal_yaw_deg needs [min, max], got {list(near_goal_yaw_deg)}")
        yaw_error = tuple(math.radians(float(v)) for v in near_goal_yaw_deg)
        return g.sample_near_goal_pose(seed, float(near_goal_radius), yaw_error)
    raise ValueError(f"start_mode {start_mode!r} is not one of {START_MODES}")


def next_seed(sidecar: list[dict], first: int = FIRST_COLLECTION_SEED) -> int:
    """Return the seed a new session continues from: one past the largest in the sidecar.

    The manager's seed counter lives in memory, so a relaunch that left ``seed_start`` at its
    default recorded seed 1000 again (episodes 0 and 1 of ``pusht_cr3a_sim_v1``, 2026-11-03).
    Lines without a seed (old or foreign metadata) are ignored; an empty sidecar starts at
    ``first``.
    """
    seeds = [int(line["seed"]) for line in sidecar if line.get("seed") is not None]
    return max(seeds) + 1 if seeds else int(first)
