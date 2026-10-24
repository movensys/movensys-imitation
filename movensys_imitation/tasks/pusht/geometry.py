"""PushT task geometry shared by the Isaac scene, the scorer and the sheet artwork.

Everything is in metres and radians, in the *sheet frame*: origin at the centre of the target
sheet, x and y along the sheet edges, z up. In Isaac Sim the sheet frame is the prim
``/World/pusht`` and is aligned with the robot planning frame ``world_manipulator`` (the robot
base sits on the +y side of the sheet). On the real table the sheet frame is the printed sheet
itself; it is used for the start marks and the three-mark workspace alignment, never for scoring,
which happens in overhead-camera pixel space (plan v2.1, no markers).

The T block has the gym-pusht proportions at :data:`UNIT` metres per unit: a 4 x 1 bar and a
1 x 3 stem. A block pose ``(x, y, yaw)`` is the pose of the *bar-stem junction* (centre of the
bar's inner edge); at ``yaw = 0`` the bar is along +x on the +y side of the junction and the
stem extends toward -y.
"""

from __future__ import annotations

import math

import numpy as np

UNIT = 0.030
"""Metres per gym-pusht unit (plan Section 5, v2.4).

40 mm from v2.0 to v2.3; 30 mm since 2026-10-24 so the T keeps a workable ratio to the push area,
which the CR3A's reach caps at about 0.36 m (see :data:`SHEET_SIZE`). A 30 mm stem is still about
27 px wide in the 1280 x 720 overhead image from 0.7 m.
"""

BAR = (4 * UNIT, 1 * UNIT)  # (length along x, width along y)
STEM = (1 * UNIT, 3 * UNIT)
BLOCK_HEIGHT = 0.030
BLOCK_MASS = 0.150  # printed at 30 mm per unit: 150 to 250 g depending on infill

SHEET_SIZE = 0.36
"""Side of the square target sheet, equal to the push area (plan Section 5, v2.4).

Reduced from 0.45 m on 2026-10-24: with the tool vertical at push height the CR3A reaches the
table without a Servo singularity slow-down only 0.28 to 0.48 m from its base axis, and at all
only 0.18 to 0.57 m, so a 0.45 m square does not fit at any centre. At 0.36 m the near and far
edges lie in Servo's slow-down zone (condition number 17 to 30) but stay reachable.
"""
INNER_SIZE = 0.22
"""Side of the square in which start positions are sampled (about 0.6 x the sheet)."""

GOAL_YAW = math.pi / 4
"""Goal orientation, as in gym-pusht (goal pose (256, 256, pi/4))."""

N_START_MARKS = 30
"""Start marks printed on the real sheet: seeds 0 to 29 of :func:`sample_start_pose`."""


def block_polygon_local() -> np.ndarray:
    """T outline (counter-clockwise, 8 vertices) in the block frame, metres."""
    bx, by = BAR[0] / 2, BAR[1]
    sx, sy = STEM[0] / 2, STEM[1]
    return np.array(
        [
            (-bx, 0.0),
            (-sx, 0.0),
            (-sx, -sy),
            (sx, -sy),
            (sx, 0.0),
            (bx, 0.0),
            (bx, by),
            (-bx, by),
        ]
    )


def block_centroid_local() -> np.ndarray:
    """Area centroid of the T in the block frame (it is not at the junction)."""
    a_bar, a_stem = BAR[0] * BAR[1], STEM[0] * STEM[1]
    c_bar, c_stem = BAR[1] / 2, -STEM[1] / 2
    return np.array([0.0, (a_bar * c_bar + a_stem * c_stem) / (a_bar + a_stem)])


def transform_points(points: np.ndarray, pose) -> np.ndarray:
    """Map local ``points`` (N, 2) through a planar pose ``(x, y, yaw)``."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return points @ rot.T + np.array([x, y])


def block_polygon(pose) -> np.ndarray:
    """T outline in the sheet frame for block pose ``(x, y, yaw)``."""
    return transform_points(block_polygon_local(), pose)


def goal_pose() -> tuple[float, float, float]:
    """Goal block pose: T centroid at the sheet centre, yaw :data:`GOAL_YAW`."""
    c = transform_points(block_centroid_local()[None, :], (0.0, 0.0, GOAL_YAW))[0]
    return (float(-c[0]), float(-c[1]), GOAL_YAW)


def sample_start_pose(seed: int) -> tuple[float, float, float]:
    """Deterministic start pose for ``seed``: centroid uniform in the inner square, yaw uniform.

    Seeds 0 to 49 are the evaluation set, 1000 and up the collection set (plan Section 5).
    """
    rng = np.random.default_rng(seed)
    half = INNER_SIZE / 2
    cx, cy = rng.uniform(-half, half, size=2)
    yaw = rng.uniform(-math.pi, math.pi)
    c = transform_points(block_centroid_local()[None, :], (0.0, 0.0, yaw))[0]
    return (float(cx - c[0]), float(cy - c[1]), float(yaw))


def start_marks() -> list[tuple[float, float, float]]:
    """The :data:`N_START_MARKS` poses printed on the real sheet (seeds 0 to 29)."""
    return [sample_start_pose(i) for i in range(N_START_MARKS)]


def yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) for a rotation of ``yaw`` about z."""
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def quat_wxyz_to_yaw(w: float, x: float, y: float, z: float) -> float:
    """Yaw about z of a quaternion (w, x, y, z); exact for planar rotations."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
