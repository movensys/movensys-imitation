"""T pose from the overhead camera (real robot): camera model, goal-T calibration, pose fit.

The pixel scorer (:mod:`scoring`) compares colour masks in image space. With the oblique camera
that overstates coverage near the goal: the scale changes across the sheet and the side faces of
the block, the same colour as its top, are counted as T (``doc/4_sim.md``, 2026-11-05). This
module instead estimates the block pose ``(x, y, yaw)`` in the sheet frame, so coverage comes
from the geometric backend exactly as in sim, and the pose feeds ``/pusht/t_block/pose``.

Three pieces, no ROS dependency:

* :class:`CameraModel`: pinhole intrinsics plus the sheet-to-camera pose (OpenCV ``rvec``,
  ``tvec``), stored as ``camera.yaml`` next to the pixel calibration.
* :func:`calibrate_from_goal`: the printed goal T (dark on a light sheet) has 8 corners at known
  sheet coordinates (:func:`geometry.goal_pose`); they give the camera pose with ``solvePnP``.
  :func:`estimate_focal` recovers a focal length from the same corners when no ``camera_info``
  is available (a diagnostic; prefer the RealSense factory intrinsics).
* :class:`PoseEstimator`: segments the block (colour), masks the pusher (another colour) as
  unknown, then fits the pose by rendering the full prism (top face, bottom face and the eight
  sides) and minimising the mismatch with the mask over the known pixels. Occlusion only removes
  evidence, never adds wrong evidence, and :attr:`PoseResult.visible` says how much was seen.

Conventions follow :mod:`geometry` (sheet frame, z up, block pose at the bar-stem junction) and
:mod:`scoring` (RGB uint8 images, OpenCV HSV).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import minimize

from movensys_imitation.tasks.pusht import geometry as g
from movensys_imitation.tasks.pusht.scoring import HsvRange, to_hsv

CAMERA_YAML = "camera.yaml"
CAMERA_OVERLAY_PNG = "camera_overlay.png"

_SUB = 4  # fillPoly fractional bits: 1/16 px vertex precision
_SCALE = float(1 << _SUB)


# ------------------------------------------------------------------------------ camera model


@dataclass
class CameraModel:
    """Pinhole camera with distortion and its pose: ``x_cam = R(rvec) @ x_sheet + tvec``."""

    image_size: tuple[int, int]  # (width, height)
    K: np.ndarray  # (3, 3)
    dist: np.ndarray  # (N,) OpenCV distortion coefficients, may be empty
    rvec: np.ndarray  # (3,)
    tvec: np.ndarray  # (3,)
    reprojection_px: float = math.nan
    meta: dict = field(default_factory=dict)

    def project(self, pts: np.ndarray) -> np.ndarray:
        """Sheet-frame points (N, 3) to pixels (N, 2)."""
        uv, _ = cv2.projectPoints(
            np.asarray(pts, np.float64).reshape(-1, 1, 3), self.rvec, self.tvec, self.K, self.dist
        )
        return uv.reshape(-1, 2)

    def camera_centre(self) -> np.ndarray:
        """Camera position in the sheet frame."""
        R, _ = cv2.Rodrigues(self.rvec)
        return (-R.T @ self.tvec.reshape(3)).reshape(3)

    def backproject_to_plane(self, uv: np.ndarray, z: float = 0.0) -> np.ndarray:
        """Pixels (N, 2) to sheet-frame (N, 2) on the plane at height ``z``."""
        pts = cv2.undistortPoints(
            np.asarray(uv, np.float64).reshape(-1, 1, 2), self.K, self.dist
        ).reshape(-1, 2)
        R, _ = cv2.Rodrigues(self.rvec)
        c = self.camera_centre()
        rays = np.column_stack([pts, np.ones(len(pts))]) @ R  # = (R.T @ d).T: sheet-frame rays
        s = (z - c[2]) / rays[:, 2]
        return c[:2] + rays[:, :2] * s[:, None]

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        doc = {
            "image_size": [int(v) for v in self.image_size],
            "K": np.asarray(self.K, float).tolist(),
            "dist": np.asarray(self.dist, float).ravel().tolist(),
            "rvec": np.asarray(self.rvec, float).ravel().tolist(),
            "tvec": np.asarray(self.tvec, float).ravel().tolist(),
            "reprojection_px": float(self.reprojection_px),
            "camera_centre_sheet_m": [round(float(v), 5) for v in self.camera_centre()],
            "meta": dict(self.meta),
        }
        path = d / CAMERA_YAML
        path.write_text(yaml.safe_dump(doc, sort_keys=False))
        return path

    @classmethod
    def load(cls, directory: str | Path) -> CameraModel:
        p = Path(directory)
        doc = yaml.safe_load((p / CAMERA_YAML if p.is_dir() else p).read_text())
        return cls(
            tuple(int(v) for v in doc["image_size"]),
            np.array(doc["K"], np.float64),
            np.array(doc.get("dist") or [], np.float64),
            np.array(doc["rvec"], np.float64),
            np.array(doc["tvec"], np.float64),
            float(doc.get("reprojection_px", math.nan)),
            dict(doc.get("meta") or {}),
        )


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def look_at_camera(
    image_size, fx: float, position, target=(0.0, 0.0, 0.0), cx=None, cy=None
) -> CameraModel:
    """A distortion-free camera at ``position`` (sheet frame) looking at ``target``.

    Image up is the sheet +z direction projected (as scene_9a's camera). Used by the tests and
    to describe the sim camera.
    """
    w, h = image_size
    pos = np.asarray(position, float)
    fwd = np.asarray(target, float) - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])  # rows: camera x, y, z in sheet coordinates
    rvec, _ = cv2.Rodrigues(R)
    cx = (w - 1) / 2 if cx is None else cx
    cy = (h - 1) / 2 if cy is None else cy
    K = intrinsics_matrix(fx, fx, cx, cy)
    return CameraModel((w, h), K, np.zeros(5), rvec.ravel(), (-R @ pos).ravel())


# ------------------------------------------------------------------------- block rendering


def prism_points(pose, height: float = g.BLOCK_HEIGHT) -> np.ndarray:
    """The 16 block vertices in the sheet frame: bottom face (0..7), then top face (8..15)."""
    poly = g.block_polygon(pose)
    z0, z1 = np.zeros((8, 1)), np.full((8, 1), height)
    return np.vstack([np.hstack([poly, z0]), np.hstack([poly, z1])])


def prism_faces_px(cam: CameraModel, pose, height: float = g.BLOCK_HEIGHT) -> list[np.ndarray]:
    """Projected faces of the block (each an (N, 2) float pixel polygon); their union is the
    silhouette, so no visibility ordering is needed."""
    uv = cam.project(prism_points(pose, height))
    bottom, top = uv[:8], uv[8:]
    sides = [np.array([bottom[i], bottom[(i + 1) % 8], top[(i + 1) % 8], top[i]]) for i in range(8)]
    return [top, bottom, *sides]


def render_silhouette(
    cam: CameraModel, pose, roi=None, height: float = g.BLOCK_HEIGHT, antialias: bool = True
) -> np.ndarray:
    """Block silhouette as float32 coverage in [0, 1] over ``roi`` (x0, y0, w, h), default full
    frame. Anti-aliased edges make the fit cost vary smoothly with sub-pixel pose changes."""
    x0, y0, w, h = roi if roi is not None else (0, 0, *cam.image_size)
    canvas = np.zeros((h, w), np.uint8)
    faces = prism_faces_px(cam, pose, height)
    pts = [np.round((f - (x0, y0)) * _SCALE).astype(np.int32) for f in faces]
    line = cv2.LINE_AA if antialias else cv2.LINE_8
    for p in pts:  # one call per face: fillPoly on several polygons uses even-odd filling
        cv2.fillPoly(canvas, [p], 255, lineType=line, shift=_SUB)
    return canvas.astype(np.float32) / 255.0


def draw_block_outline(image: np.ndarray, cam: CameraModel, pose, rgb, thickness: int = 1,
                       height: float = g.BLOCK_HEIGHT) -> None:
    """Draw the top and bottom outlines and vertical edges of the block (in place)."""
    uv = cam.project(prism_points(pose, height))
    p = np.round(uv * _SCALE).astype(np.int32)
    col = tuple(int(c) for c in rgb)
    cv2.polylines(image, [p[:8]], True, col, thickness, cv2.LINE_AA, _SUB)
    cv2.polylines(image, [p[8:]], True, col, thickness, cv2.LINE_AA, _SUB)
    for i in range(8):
        cv2.line(image, tuple(p[i]), tuple(p[i + 8]), col, thickness, cv2.LINE_AA, _SUB)


def draw_sheet(image: np.ndarray, cam: CameraModel, rgb=(0, 160, 255), grid: float = 0.05,
               sheet_size=g.SHEET_SIZE) -> None:
    """Draw the push-area outline and a ``grid`` metre grid on the table plane (in place)."""
    hx, hy = sheet_size[0] / 2, sheet_size[1] / 2
    col = tuple(int(c) for c in rgb)

    def seg(a, b, thick):
        uv = cam.project(np.array([[*a, 0.0], [*b, 0.0]]))
        p = np.round(uv * _SCALE).astype(np.int32)
        cv2.line(image, tuple(p[0]), tuple(p[1]), col, thick, cv2.LINE_AA, _SUB)

    for x in np.arange(-hx, hx + 1e-9, grid):
        seg((x, -hy), (x, hy), 1)
    for y in np.arange(-hy, hy + 1e-9, grid):
        seg((-hx, y), (hx, y), 1)
    for a, b in [((-hx, -hy), (hx, -hy)), ((hx, -hy), (hx, hy)), ((hx, hy), (-hx, hy)),
                 ((-hx, hy), (-hx, -hy))]:
        seg(a, b, 2)
    o = cam.project(np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0]], float))
    p = np.round(o * _SCALE).astype(np.int32)
    cv2.arrowedLine(image, tuple(p[0]), tuple(p[1]), (255, 0, 0), 2, cv2.LINE_AA, _SUB, 0.2)
    cv2.arrowedLine(image, tuple(p[0]), tuple(p[2]), (0, 200, 0), 2, cv2.LINE_AA, _SUB, 0.2)


def area_roi(cam: CameraModel, margin: float = 0.05, sheet_size=g.SHEET_SIZE,
             height: float = g.BLOCK_HEIGHT) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Image mask and bounding box (x0, y0, w, h) of the push area plus ``margin`` metres, from
    the table up to the top of a block standing at its edge."""
    hx, hy = sheet_size[0] / 2 + margin, sheet_size[1] / 2 + margin
    corners = np.array([(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)])
    w, h = cam.image_size
    mask = np.zeros((h, w), np.uint8)
    pts = []
    for z in (0.0, height):
        uv = cam.project(np.column_stack([corners, np.full(4, z)]))
        pts.append(uv)
        cv2.fillPoly(mask, [np.round(uv * _SCALE).astype(np.int32)], 255, cv2.LINE_8, _SUB)
    allp = np.vstack(pts)
    x0, y0 = np.clip(np.floor(allp.min(0)).astype(int), 0, [w - 1, h - 1])
    x1, y1 = np.clip(np.ceil(allp.max(0)).astype(int) + 1, 1, [w, h])
    return mask > 0, (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def policy_crop(cam: CameraModel, margin: float = 0.03, sheet_size=g.SHEET_SIZE,
                height: float = g.BLOCK_HEIGHT) -> tuple[int, int, int, int]:
    """The policy image crop ``(x0, y0, w, h)`` for this camera: a square around the push area.

    The real-robot counterpart of :func:`scoring.fit_crop`: the bounding box of the push area
    plus ``margin`` metres (table to block top), grown to a square about its centre so the
    recorder's resize does not distort it. Each side is capped by the frame, as in ``fit_crop``.
    It is a function of ``camera.yaml`` alone, so the crop follows the sheet when the camera is
    calibrated again. Raises ``ValueError`` when the area is not inside the frame.
    """
    hx, hy = sheet_size[0] / 2 + margin, sheet_size[1] / 2 + margin
    corners = np.array([(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)])
    uv = np.vstack([cam.project(np.column_stack([corners, np.full(4, z)])) for z in (0.0, height)])
    w_img, h_img = cam.image_size
    lo, hi = uv.min(0), uv.max(0)
    if lo[0] < 0 or lo[1] < 0 or hi[0] > w_img - 1 or hi[1] > h_img - 1:
        raise ValueError(
            f"the push area projects to u {lo[0]:.0f}..{hi[0]:.0f}, v {lo[1]:.0f}..{hi[1]:.0f}, "
            f"outside the {w_img} x {h_img} frame; move the camera"
        )
    side = int(math.ceil(max(hi - lo))) + 1
    w, h = min(side, w_img), min(side, h_img)
    cx, cy = (lo + hi) / 2
    x0 = min(max(int(round(cx - w / 2)), 0), w_img - w)
    y0 = min(max(int(round(cy - h / 2)), 0), h_img - h)
    return (x0, y0, w, h)


# ------------------------------------------------------------------- goal-T calibration


@dataclass
class DarkT:
    corners: np.ndarray  # (8, 2) sub-pixel corners, ordered as geometry.block_polygon_local()
    contour: np.ndarray  # (N, 2) the component's outer contour
    mask: np.ndarray  # bool (H, W)
    homography_px: float  # rms corner error of the best homography fit (shape check)


def dark_mask(rgb: np.ndarray, contrast: int = 60, sat_max: int = 80, val_max: int = 110,
              background_px: int = 101) -> np.ndarray:
    """Pixels much darker than their surroundings and nearly grey (the printed goal T).

    Local contrast instead of a fixed level: the sheet is brighter near the camera than far away
    (V 207 against 180 on the 2026-11-19 frame)."""
    hsv = to_hsv(rgb)
    v = hsv[..., 2]
    bg = cv2.medianBlur(v, background_px | 1)
    m = ((bg.astype(np.int16) - v) > contrast) & (hsv[..., 1] < sat_max) & (v < val_max)
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0


def _line_intersection(l1, l2) -> np.ndarray:
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    A = np.array([[vx1, -vx2], [vy1, -vy2]])
    t = np.linalg.solve(A, np.array([x2 - x1, y2 - y1]))
    return np.array([x1 + t[0] * vx1, y1 + t[0] * vy1])


def _refine_corners(contour: np.ndarray, approx_idx: np.ndarray) -> np.ndarray:
    """Fit a line to each contour run between approximate corners; intersect neighbours."""
    k = len(approx_idx)
    lines = []
    for i in range(k):
        a, b = approx_idx[i], approx_idx[(i + 1) % k]
        run = contour[a:b + 1] if b > a else np.vstack([contour[a:], contour[: b + 1]])
        trim = max(1, len(run) // 6)  # corners are rounded by the opening: drop their ends
        seg = run[trim:-trim] if len(run) > 2 * trim + 2 else run
        lines.append(cv2.fitLine(seg.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())
    pts = []
    for i in range(k):
        try:
            pts.append(_line_intersection(lines[i - 1], lines[i]))
        except np.linalg.LinAlgError:
            pts.append(contour[approx_idx[i]].astype(float))
    return np.array(pts)


def _signed_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def match_t_corners(img_pts: np.ndarray, model_pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Reorder 8 image corners to match ``model_pts`` (sheet frame, z = 0).

    The camera looks down on the table, so the outline keeps its orientation once the image v
    axis is flipped; that rules out the mirror matches (the T is mirror symmetric). Of the 8
    cyclic shifts the one with the smallest homography error wins (the T has no rotational
    symmetry). Returns the ordered corners and the rms homography error in pixels.
    """
    pts = np.asarray(img_pts, float)
    flipped = pts * (1, -1)
    if np.sign(_signed_area(flipped)) != np.sign(_signed_area(model_pts)):
        pts = pts[::-1]
    best, best_err = None, math.inf
    for s in range(8):
        cand = np.roll(pts, -s, axis=0)
        H, _ = cv2.findHomography(model_pts, cand, 0)
        if H is None:
            continue
        proj = cv2.perspectiveTransform(model_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = float(np.sqrt(np.mean(np.sum((proj - cand) ** 2, axis=1))))
        if err < best_err:
            best, best_err = cand, err
    if best is None:
        raise ValueError("no homography fits the T corners")
    return best, best_err


def find_goal_t(rgb: np.ndarray, goal=None, min_area: int = 400, **dark_kw) -> DarkT:
    """Find the printed goal T: the largest dark, T-shaped (8 corners) component."""
    goal = g.goal_pose() if goal is None else goal
    model = g.block_polygon(goal)
    m = dark_mask(rgb, **dark_kw)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
    reasons = []
    for i in order:
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            break
        comp = lab == i
        cs, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea).reshape(-1, 2)
        hull = cv2.convexHull(c)
        solidity = cv2.contourArea(c) / max(cv2.contourArea(hull), 1.0)
        approx = None
        for eps in (0.01, 0.015, 0.02, 0.03, 0.04):
            a = cv2.approxPolyDP(c.reshape(-1, 1, 2), eps * cv2.arcLength(c, True), True)
            if len(a) == 8:
                approx = a.reshape(-1, 2)
                break
        if approx is None or not 0.45 < solidity < 0.8:
            reasons.append(f"component {stats[i, cv2.CC_STAT_AREA]} px: not T-shaped "
                           f"(solidity {solidity:.2f})")
            continue
        idx = np.array([int(np.argmin(np.sum((c - p) ** 2, axis=1))) for p in approx])
        o = np.argsort(idx)
        corners = _refine_corners(c, idx[o])
        corners, err = match_t_corners(corners, model)
        return DarkT(corners, c, comp, err)
    raise ValueError("no printed goal T found" + (": " + "; ".join(reasons[:3]) if reasons else ""))


def estimate_focal(img_pts: np.ndarray, model_pts: np.ndarray, centre) -> float:
    """Focal length (square pixels, zero skew, principal point ``centre``) from one plane view
    (Zhang's two homography constraints, least squares). A small target gives a rough value;
    use it to sanity-check ``camera_info``, not to replace it."""
    T = np.array([[1, 0, -centre[0]], [0, 1, -centre[1]], [0, 0, 1]], float)
    H, _ = cv2.findHomography(model_pts, img_pts, 0)
    H = T @ H
    h1, h2 = H[:, 0], H[:, 1]
    a = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    b = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    # a / f^2 + b = 0 for both constraints: least squares in 1 / f^2.
    inv_f2 = -float(a @ b) / float(a @ a)
    if inv_f2 <= 0:
        raise ValueError("focal length not observable from this view (too frontal?)")
    return float(math.sqrt(1.0 / inv_f2))


def calibrate_from_goal(
    rgb: np.ndarray, K: np.ndarray, dist=None, goal=None, meta: dict | None = None, **find_kw
) -> tuple[CameraModel, DarkT]:
    """Camera pose from the printed goal T at ``goal`` (default :func:`geometry.goal_pose`)."""
    goal = g.goal_pose() if goal is None else goal
    dist = np.zeros(5) if dist is None else np.asarray(dist, float)
    t = find_goal_t(rgb, goal, **find_kw)
    obj = np.column_stack([g.block_polygon(goal), np.zeros(8)])
    ok, rvec, tvec = cv2.solvePnP(obj, t.corners, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise ValueError("solvePnP failed on the goal T corners")
    rvec, tvec = cv2.solvePnPRefineLM(obj, t.corners, K, dist, rvec, tvec)
    h, w = rgb.shape[:2]
    cam = CameraModel((w, h), np.asarray(K, float), dist, rvec.ravel(), tvec.ravel(),
                      meta=dict(meta or {}))
    err = cam.project(obj) - t.corners
    cam.reprojection_px = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
    cam.meta.setdefault("goal_pose", [float(v) for v in goal])
    if cam.camera_centre()[2] <= 0:
        raise ValueError("camera pose comes out below the table; corner matching failed")
    return cam, t


# ---------------------------------------------------------------------------- pose fitting


def default_block_threshold() -> HsvRange:
    """Green printed T (2026-11-19 frame: median HSV (77, 216, 137))."""
    return HsvRange((55, 95), 80, (20, 245))


def default_occluder_threshold() -> HsvRange:
    """Red pusher rod and collar (hue wraps around 0)."""
    return HsvRange((170, 10), 100, (40, 255))


@dataclass
class EstimatorConfig:
    block: HsvRange = field(default_factory=default_block_threshold)
    occluder: HsvRange = field(default_factory=default_occluder_threshold)
    open_px: int = 3
    occluder_dilate_px: int = 7
    """Pixels this close to the pusher count as unknown (its anti-aliased rim is neither)."""
    area_margin_m: float = 0.05
    """Search the block only inside the push area plus this margin."""
    roi_margin_px: int = 25
    min_component_frac: float = 0.05
    """Mask components smaller than this fraction of the largest are dropped, unless they lie
    within the block's reach of it (the pusher can cut the T in two)."""
    min_iou: float = 0.80
    min_visible: float = 0.50
    max_iter: int = 250
    track_jump_m: float = 0.03
    """A previous pose farther than this from the mask centre is not used as a start."""


@dataclass
class PoseResult:
    pose: tuple[float, float, float] | None
    iou: float = math.nan
    """Fit IoU over the known (not occluded) pixels."""
    visible: float = math.nan
    """Fraction of the rendered block outside the occluder mask."""
    cost: float = math.nan
    n_evals: int = 0
    mask_px: int = 0
    ok: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pose"] = list(self.pose) if self.pose is not None else None
        return d


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class PoseEstimator:
    """Fits the block pose to one frame. Keeps the last good pose to start the next fit."""

    def __init__(self, cam: CameraModel, cfg: EstimatorConfig | None = None,
                 height: float = g.BLOCK_HEIGHT) -> None:
        self.cam = cam
        self.cfg = cfg or EstimatorConfig()
        self.height = height
        self.area_mask, self.area_box = area_roi(cam, self.cfg.area_margin_m, height=height)
        self.last: tuple[float, float, float] | None = None
        self.block_mask: np.ndarray | None = None  # last full-frame block mask (debug)
        self.unknown_mask: np.ndarray | None = None
        self._k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, self.cfg.open_px),) * 2)
        d = max(1, self.cfg.occluder_dilate_px)
        self._k_occ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))

    def reset(self) -> None:
        self.last = None

    # -- masks -------------------------------------------------------------------------

    def masks(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Block and unknown (occluder) masks, full frame, restricted to the push area."""
        x0, y0, w, h = self.area_box
        sub = rgb[y0:y0 + h, x0:x0 + w]
        hsv = to_hsv(sub)
        area = self.area_mask[y0:y0 + h, x0:x0 + w]
        blk = self.cfg.block.mask(hsv) & area
        if self.cfg.open_px > 1:
            blk = cv2.morphologyEx(blk.astype(np.uint8), cv2.MORPH_OPEN, self._k_open) > 0
        occ = cv2.dilate(self.cfg.occluder.mask(hsv).astype(np.uint8), self._k_occ) > 0
        blk &= ~occ
        blk = self._keep_block_components(blk)
        H, W = rgb.shape[:2]
        full_b = np.zeros((H, W), bool)
        full_u = np.zeros((H, W), bool)
        full_b[y0:y0 + h, x0:x0 + w] = blk
        full_u[y0:y0 + h, x0:x0 + w] = occ
        return full_b, full_u

    def _keep_block_components(self, m: np.ndarray) -> np.ndarray:
        n, lab, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        if n <= 2:
            return m
        areas = st[1:, cv2.CC_STAT_AREA]
        i0 = int(np.argmax(areas)) + 1
        reach = 1.5 * math.hypot(st[i0, cv2.CC_STAT_WIDTH], st[i0, cv2.CC_STAT_HEIGHT])
        keep = np.zeros(n, bool)
        keep[i0] = True
        for i in range(1, n):
            if i != i0 and areas[i - 1] >= self.cfg.min_component_frac * areas[i0 - 1]:
                keep[i] = math.dist(cen[i], cen[i0]) < reach
        return keep[lab]

    # -- fit ---------------------------------------------------------------------------

    def _initial_poses(self, block: np.ndarray) -> list[tuple[float, float, float]]:
        """Candidates from the mask moments on the table plane, plus the last pose."""
        ys, xs = np.nonzero(block)
        step = max(1, len(xs) // 4000)
        pts = self.cam.backproject_to_plane(np.column_stack([xs[::step], ys[::step]]),
                                            self.height / 2)
        c = pts.mean(0)
        cov = np.cov((pts - c).T)
        evals, evecs = np.linalg.eigh(cov)
        axis = math.atan2(evecs[1, 1], evecs[0, 1])  # major axis
        off = g.block_centroid_local()
        cands = []
        for k in range(4):
            yaw = _wrap(axis + k * math.pi / 2)
            r = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            j = c - r @ off
            cands.append((float(j[0]), float(j[1]), yaw))
        if self.last is not None:
            lc = g.transform_points(off[None, :], self.last)[0]
            if np.linalg.norm(lc - c) < self.cfg.track_jump_m:
                cands.insert(0, self.last)
        return cands

    def _roi(self, block: np.ndarray, unknown: np.ndarray, poses) -> tuple[int, int, int, int]:
        ys, xs = np.nonzero(block)
        boxes = [np.array([[xs.min(), ys.min()], [xs.max(), ys.max()]], float)]
        for p in poses:
            uv = self.cam.project(prism_points(p, self.height))
            boxes.append(np.array([uv.min(0), uv.max(0)]))
        b = np.vstack(boxes)
        m = self.cfg.roi_margin_px
        W, H = self.cam.image_size
        x0, y0 = np.clip(np.floor(b.min(0)).astype(int) - m, 0, [W - 1, H - 1])
        x1, y1 = np.clip(np.ceil(b.max(0)).astype(int) + m, 1, [W, H])
        return int(x0), int(y0), int(x1 - x0), int(y1 - y0)

    def estimate(self, rgb: np.ndarray) -> PoseResult:
        if rgb.shape[1::-1] != tuple(self.cam.image_size):
            raise ValueError(f"frame is {rgb.shape[1]} x {rgb.shape[0]}, camera model expects "
                             f"{self.cam.image_size[0]} x {self.cam.image_size[1]}")
        block, unknown = self.masks(rgb)
        self.block_mask, self.unknown_mask = block, unknown
        n_px = int(block.sum())
        if n_px < 50:
            return PoseResult(None, mask_px=n_px, note="no block in view")
        cands = self._initial_poses(block)
        x0, y0, w, h = roi = self._roi(block, unknown, cands)
        obs = block[y0:y0 + h, x0:x0 + w].astype(np.float32)
        known = (~unknown[y0:y0 + h, x0:x0 + w]).astype(np.float32)
        obs_area = float(obs.sum())
        n_evals = 0

        def cost_of(p) -> float:
            nonlocal n_evals
            n_evals += 1
            r = render_silhouette(self.cam, p, roi, self.height)
            return float(np.sum(known * np.abs(r - obs))) / obs_area

        # Coarse: best start, then a local Nelder-Mead in (mm, mm, deg) units.
        scored = sorted(((cost_of(p), i, p) for i, p in enumerate(cands)), key=lambda s: s[0])
        start = np.array(scored[0][2], float)
        tracked = self.last is not None and scored[0][2] == self.last
        scale = np.array([1e-3, 1e-3, math.radians(1.0)])

        def f(z):
            return cost_of(tuple(start + z * scale))

        step = 1.5 if tracked else 4.0  # mm and degrees
        simplex = np.array([[0, 0, 0], [step, 0, 0], [0, step, 0], [0, 0, step]], float)
        res = minimize(f, np.zeros(3), method="Nelder-Mead",
                       options={"initial_simplex": simplex, "xatol": 0.05, "fatol": 1e-5,
                                "maxfev": self.cfg.max_iter})
        z = res.x
        pose = (float(start[0] + z[0] * scale[0]), float(start[1] + z[1] * scale[1]),
                _wrap(float(start[2] + z[2] * scale[2])))
        hard = render_silhouette(self.cam, pose, roi, self.height, antialias=False) > 0.5
        o = obs > 0.5
        k = known > 0.5
        inter = float(np.sum(hard & o & k))
        union = float(np.sum((hard | o) & k))
        iou = inter / union if union > 0 else 0.0
        model_px = float(hard.sum())
        visible = float(np.sum(hard & k)) / model_px if model_px > 0 else 0.0
        ok = iou >= self.cfg.min_iou and visible >= self.cfg.min_visible
        note = "" if ok else (f"fit IoU {iou:.2f} < {self.cfg.min_iou}" if iou < self.cfg.min_iou
                              else f"visible {visible:.2f} < {self.cfg.min_visible}")
        if ok:
            self.last = pose
        return PoseResult(pose, iou, visible, float(res.fun), n_evals, n_px, ok, note)


# --------------------------------------------------------------------------- configuration


def estimator_config_from_dict(d: dict | None) -> EstimatorConfig:
    """``EstimatorConfig`` from a (yaml) dict; missing keys keep their defaults."""
    d = dict(d or {})
    cfg = EstimatorConfig()
    for key in ("block", "occluder"):
        if key in d:
            setattr(cfg, key, HsvRange.from_dict(d.pop(key)))
    for k, v in d.items():
        if not hasattr(cfg, k):
            raise KeyError(f"unknown estimator setting {k!r}")
        setattr(cfg, k, type(getattr(cfg, k))(v))
    return cfg
