"""PushT scoring (plan P2.3): coverage and IoU of the T against the goal, two backends.

* **Geometric**: shapely on block poses ``(x, y, yaw)`` in the sheet frame (:mod:`geometry`).
  Sim ground truth from ``/pusht/t_block/pose``; unavailable on the real robot.
* **Pixel**: the T mask from the overhead crop (HSV colour threshold, morphological opening, the
  largest connected component, an area check) against a goal mask captured once by
  ``setup_camera`` with the T placed on the goal (plan Section 5, Chi et al. 2023). Works the same
  in sim and on the real camera; needs no calibration beyond that one frame.

Coverage is ``area(T ∩ goal) / area(goal)`` as in gym-pusht, success at
:data:`SUCCESS_THRESHOLD`; IoU is ``area(T ∩ goal) / area(T ∪ goal)`` (the paper's metric). The
pixel coverage is a lower bound while the arm occludes the T: occlusion only removes T pixels.
:attr:`Score.valid` is false when the mask area is outside :attr:`MaskConfig.area_tolerance` of
the reference area, so a reading taken during a push is flagged, never silently wrong.

Pixel conventions: images are RGB uint8 (H, W, 3); masks are bool (H, W) of the crop; HSV is
OpenCV's (H 0..179, S 0..255, V 0..255); image +x is sheet +x and image *up* is sheet +y
(``scene_9a.json``). This module has no ROS dependency; the node and the setup script wrap it.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from shapely.geometry import Polygon

from movensys_imitation.tasks.pusht import geometry as g

SUCCESS_THRESHOLD = 0.95
"""Coverage at or above this is a success (plan Section 7; gym-pusht's ``success_threshold``)."""

T_AREA_M2 = g.BAR[0] * g.BAR[1] + g.STEM[0] * g.STEM[1]
"""Footprint of the T in square metres (7 square units)."""

CALIBRATION_YAML = "calibration.yaml"
GOAL_MASK_PNG = "goal_mask.png"
REFERENCE_PNG = "reference.png"
OVERLAY_PNG = "goal_overlay.png"


@dataclass
class Score:
    """One scoring result. ``area_ratio`` and ``note`` are filled by the pixel backend only."""

    coverage: float
    iou: float
    valid: bool = True
    area_ratio: float = math.nan
    note: str = ""

    @property
    def success(self) -> bool:
        return self.valid and self.coverage >= SUCCESS_THRESHOLD


# ------------------------------------------------------------------------- geometric backend


def block_shape(pose) -> Polygon:
    """The T footprint as a shapely polygon for block pose ``(x, y, yaw)``."""
    return Polygon(g.block_polygon(pose))


def geometric_score(pose, goal=None) -> Score:
    """Coverage and IoU of the T at ``pose`` against the T at ``goal`` (default: the goal pose)."""
    goal_shape = block_shape(g.goal_pose() if goal is None else goal)
    shape = block_shape(pose)
    inter = shape.intersection(goal_shape).area
    union = shape.union(goal_shape).area
    return Score(inter / goal_shape.area, inter / union if union > 0 else 0.0)


# ----------------------------------------------------------------------------- pixel backend


@dataclass
class HsvRange:
    """Inclusive HSV box (OpenCV scale). ``hue`` may wrap (lo > hi) for red; blue does not."""

    hue: tuple[int, int] = (95, 135)
    sat_min: int = 60
    val: tuple[int, int] = (20, 230)

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        lo, hi = self.hue
        in_hue = (h >= lo) & (h <= hi) if lo <= hi else (h >= lo) | (h <= hi)
        return in_hue & (s >= self.sat_min) & (v >= self.val[0]) & (v <= self.val[1])

    def to_dict(self) -> dict:
        return {"hue": list(self.hue), "sat_min": int(self.sat_min), "val": list(self.val)}

    @classmethod
    def from_dict(cls, d: dict) -> HsvRange:
        return cls(
            tuple(int(x) for x in d["hue"]), int(d["sat_min"]), tuple(int(x) for x in d["val"])
        )


@dataclass
class MaskConfig:
    """Segmentation parameters (``scorer.yaml``); the threshold itself lives in the calibration."""

    threshold: HsvRange = field(default_factory=HsvRange)
    open_px: int = 5
    """Diameter of the elliptical opening kernel: removes the 1 mm start marks and speckle."""
    area_tolerance: float = 0.15
    """Mask area within this fraction of the reference area counts as an unoccluded T."""


@dataclass
class MaskResult:
    mask: np.ndarray
    """bool (H, W): the chosen connected component, or all false."""
    raw_area: int
    """Pixels passing the colour threshold before opening."""
    n_components: int
    area: int
    centroid: tuple[float, float]
    """(u, v) of the chosen component in crop pixels; NaN when empty."""


def to_hsv(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)


def segment_t(crop_rgb: np.ndarray, cfg: MaskConfig) -> MaskResult:
    """Threshold, open, keep the largest 8-connected component."""
    raw = cfg.threshold.mask(to_hsv(crop_rgb))
    opened = raw.astype(np.uint8)
    if cfg.open_px > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_px, cfg.open_px))
        opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN, k)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if n <= 1:
        return MaskResult(np.zeros(raw.shape, bool), int(raw.sum()), 0, 0, (math.nan, math.nan))
    areas = stats[1:, cv2.CC_STAT_AREA]
    i = int(np.argmax(areas)) + 1
    return MaskResult(
        labels == i, int(raw.sum()), n - 1, int(areas[i - 1]), tuple(map(float, cents[i]))
    )


def mask_score(mask: np.ndarray, goal_mask: np.ndarray, ref_area: int, tolerance: float) -> Score:
    """Coverage and IoU of ``mask`` against ``goal_mask``; valid if the area fits the reference."""
    inter = int(np.count_nonzero(mask & goal_mask))
    union = int(np.count_nonzero(mask | goal_mask))
    goal_area = int(np.count_nonzero(goal_mask))
    area = int(np.count_nonzero(mask))
    ratio = area / ref_area if ref_area > 0 else math.nan
    valid = abs(ratio - 1.0) <= tolerance
    if valid:
        note = ""
    elif area == 0:
        note = "no T found"
    elif ratio < 1.0:
        note = "T partly hidden (occluded or at the crop edge)"
    else:
        note = "mask too large (arm, shadow or marks merged into the T)"
    return Score(
        inter / goal_area if goal_area > 0 else 0.0,
        inter / union if union > 0 else 0.0,
        valid,
        ratio,
        note,
    )


def refine_threshold(
    hsv_pixels: np.ndarray, hue_margin: int = 8, sv_margin: int = 40, percentile: float = 1.0
) -> HsvRange:
    """A threshold around the observed T colour: percentile envelope plus margins.

    ``hsv_pixels`` is (N, 3) from the T pixels of the calibration frame. The margins leave room
    for shadows and the darker side faces of the block seen off-axis; they are tuned in P2.5.
    """
    lo = np.percentile(hsv_pixels, percentile, axis=0)
    hi = np.percentile(hsv_pixels, 100 - percentile, axis=0)
    return HsvRange(
        (max(0, int(lo[0]) - hue_margin), min(179, int(hi[0]) + hue_margin)),
        max(0, int(lo[1]) - sv_margin),
        (max(0, int(lo[2]) - sv_margin), min(255, int(hi[2]) + sv_margin)),
    )


# -------------------------------------------------------------------------------- calibration


def fit_crop(
    centroid_uv: tuple[float, float],
    t_area_px: float,
    image_size: tuple[int, int],
    sheet_size: tuple[float, float] = g.SHEET_SIZE,
    margin_m: float = 0.03,
) -> tuple[tuple[int, int, int, int], float]:
    """Square crop ``(x0, y0, side, side)`` around the goal and the pixel scale, from the T itself.

    The scale comes from the T's pixel area at the goal, so no intrinsics are needed and the same
    code serves the real camera. It is the scale of the T's top face (30 mm closer to the camera
    than the sheet, about 4.5 % larger at 0.7 m), which makes the crop slightly generous. The goal
    centroid is the sheet centre by construction (:func:`geometry.goal_pose`). The side is capped
    by the frame; the caller checks it still contains the sheet.
    """
    px_per_m = math.sqrt(t_area_px / T_AREA_M2)
    side = int(math.ceil((max(sheet_size) + 2 * margin_m) * px_per_m))
    side = min(side, image_size[0], image_size[1])
    x0 = int(round(centroid_uv[0] - side / 2))
    y0 = int(round(centroid_uv[1] - side / 2))
    x0 = min(max(x0, 0), image_size[0] - side)
    y0 = min(max(y0, 0), image_size[1] - side)
    return (x0, y0, side, side), px_per_m


def polygon_px(pose, px_per_m: float, centre_uv: tuple[float, float]) -> np.ndarray:
    """The T outline for ``pose`` in crop pixels; ``centre_uv`` is where the sheet centre maps."""
    poly = g.block_polygon(pose)
    u = centre_uv[0] + poly[:, 0] * px_per_m
    v = centre_uv[1] - poly[:, 1] * px_per_m
    return np.stack([u, v], axis=1)


def draw_block(image: np.ndarray, pose, px_per_m: float, centre_uv, rgb, filled: bool = True):
    """Draw the T at ``pose`` into ``image`` (RGB, in place); used by tests and overlays."""
    pts = np.round(polygon_px(pose, px_per_m, centre_uv)).astype(np.int32)
    if filled:
        cv2.fillPoly(image, [pts], tuple(int(c) for c in rgb), lineType=cv2.LINE_AA)
    else:
        cv2.polylines(image, [pts], True, tuple(int(c) for c in rgb), 1, lineType=cv2.LINE_AA)


@dataclass
class Calibration:
    """Everything ``setup_camera`` stores for one camera pose (plan P2.3 and P4.3).

    The crop is ``(x0, y0, w, h)`` in the full frame; the goal mask is crop-sized. ``px_per_m``
    is the top-face scale from :func:`fit_crop`; ``centre_uv`` is the goal-mask centroid in crop
    pixels, which is where the sheet centre maps. ``meta`` carries provenance (mode, topic, date,
    camera_info if any) and is not used for scoring.
    """

    image_size: tuple[int, int]
    crop: tuple[int, int, int, int]
    px_per_m: float
    t_area_px: int
    goal_mask: np.ndarray
    threshold: HsvRange
    t_hsv_median: tuple[int, int, int]
    centre_uv: tuple[float, float]
    meta: dict = field(default_factory=dict)

    def apply_crop(self, frame: np.ndarray) -> np.ndarray:
        x0, y0, w, h = self.crop
        return frame[y0 : y0 + h, x0 : x0 + w]

    def check_frame(self, frame: np.ndarray) -> None:
        if frame.shape[1::-1] != tuple(self.image_size):
            raise ValueError(
                f"frame is {frame.shape[1]} x {frame.shape[0]}, calibration expects "
                f"{self.image_size[0]} x {self.image_size[1]}"
            )

    def save(self, directory: str | Path, reference_rgb: np.ndarray | None = None) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        doc = {
            "image_size": list(self.image_size),
            "crop": list(self.crop),
            "px_per_m": float(self.px_per_m),
            "t_area_px": int(self.t_area_px),
            "goal_area_px": int(np.count_nonzero(self.goal_mask)),
            "threshold": self.threshold.to_dict(),
            "t_hsv_median": list(int(x) for x in self.t_hsv_median),
            "centre_uv": [float(x) for x in self.centre_uv],
            "goal_mask": GOAL_MASK_PNG,
            "reference": REFERENCE_PNG if reference_rgb is not None else None,
            "meta": dict(self.meta),
        }
        (d / CALIBRATION_YAML).write_text(yaml.safe_dump(doc, sort_keys=False))
        cv2.imwrite(str(d / GOAL_MASK_PNG), self.goal_mask.astype(np.uint8) * 255)
        if reference_rgb is not None:
            cv2.imwrite(str(d / REFERENCE_PNG), cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2BGR))
            overlay = self.apply_crop(reference_rgb).copy()
            edge = cv2.Canny(self.goal_mask.astype(np.uint8) * 255, 50, 150) > 0
            overlay[edge] = (255, 0, 255)
            draw_block(overlay, g.goal_pose(), self.px_per_m, self.centre_uv, (255, 200, 0), False)
            cv2.imwrite(str(d / OVERLAY_PNG), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return d

    @classmethod
    def load(cls, directory: str | Path) -> Calibration:
        d = Path(directory)
        doc = yaml.safe_load((d / CALIBRATION_YAML).read_text())
        mask_img = cv2.imread(str(d / doc.get("goal_mask", GOAL_MASK_PNG)), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise FileNotFoundError(d / GOAL_MASK_PNG)
        return cls(
            tuple(int(x) for x in doc["image_size"]),
            tuple(int(x) for x in doc["crop"]),
            float(doc["px_per_m"]),
            int(doc["t_area_px"]),
            mask_img > 127,
            HsvRange.from_dict(doc["threshold"]),
            tuple(int(x) for x in doc["t_hsv_median"]),
            tuple(float(x) for x in doc["centre_uv"]),
            dict(doc.get("meta") or {}),
        )


def calibrate_from_frame(
    frame_rgb: np.ndarray,
    cfg: MaskConfig,
    sheet_size: tuple[float, float] = g.SHEET_SIZE,
    margin_m: float = 0.03,
    hue_margin: int = 8,
    sv_margin: int = 40,
    meta: dict | None = None,
) -> tuple[Calibration, MaskResult]:
    """Build a :class:`Calibration` from one frame with the T on the goal and nothing else blue.

    Finds the T with ``cfg.threshold`` over the whole frame, fits the crop around it, refines the
    threshold from the T's own pixels and re-segments inside the crop for the goal mask. Raises
    ``ValueError`` when no T is found or the crop cannot contain the sheet.
    """
    h, w = frame_rgb.shape[:2]
    first = segment_t(frame_rgb, cfg)
    if first.area == 0:
        raise ValueError("no T found with the initial threshold; check the colour range")
    crop, px_per_m = fit_crop(first.centroid, first.area, (w, h), sheet_size, margin_m)
    # The sheet lies 30 mm below the T's top face, so it is a little smaller than px_per_m says;
    # 0.95 leaves that slack but still rejects a crop that would cut the sheet.
    if crop[2] < 0.95 * max(sheet_size) * px_per_m:
        raise ValueError(
            f"a {crop[2]} px crop cannot contain the {max(sheet_size)} m sheet at "
            f"{px_per_m:.0f} px/m; move the camera up or reduce the sheet"
        )
    hsv_t = to_hsv(frame_rgb)[first.mask]
    refined = refine_threshold(hsv_t, hue_margin, sv_margin)
    cal_cfg = MaskConfig(refined, cfg.open_px, cfg.area_tolerance)
    x0, y0, cw, ch = crop
    crop_rgb = frame_rgb[y0 : y0 + ch, x0 : x0 + cw]
    final = segment_t(crop_rgb, cal_cfg)
    if final.area == 0:
        raise ValueError("refined threshold lost the T; widen the margins")
    median = tuple(int(x) for x in np.median(hsv_t, axis=0))
    info = {"created": datetime.datetime.now().isoformat(timespec="seconds")}
    info.update(meta or {})
    cal = Calibration(
        (w, h), crop, px_per_m, final.area, final.mask, refined, median, final.centroid, info
    )
    return cal, final


# --------------------------------------------------------------------------------- scorer


class PixelScorer:
    """Scores full frames with a :class:`Calibration`; keeps the last mask for publishing."""

    def __init__(self, calibration: Calibration, cfg: MaskConfig | None = None) -> None:
        self.cal = calibration
        base = cfg or MaskConfig()
        self.cfg = MaskConfig(calibration.threshold, base.open_px, base.area_tolerance)
        self.last: MaskResult | None = None

    def score(self, frame_rgb: np.ndarray) -> Score:
        self.cal.check_frame(frame_rgb)
        self.last = segment_t(self.cal.apply_crop(frame_rgb), self.cfg)
        return mask_score(
            self.last.mask, self.cal.goal_mask, self.cal.t_area_px, self.cfg.area_tolerance
        )


# ---------------------------------------------------------------------------- image decoding


def decode_image(data, width: int, height: int, step: int, encoding: str) -> np.ndarray:
    """``sensor_msgs/Image`` bytes to an RGB uint8 (H, W, 3) array with numpy only.

    cv_bridge is avoided on purpose: the apt build is tied to numpy 1.26 and LeRobot needs
    numpy 2 (plan Section 6). Supports rgb8, bgr8, rgba8 and bgra8.
    """
    rows = np.frombuffer(data, dtype=np.uint8).reshape(height, step)
    if encoding in ("rgb8", "bgr8"):
        arr = rows[:, : width * 3].reshape(height, width, 3)
        if encoding == "bgr8":
            arr = arr[..., ::-1]
    elif encoding in ("rgba8", "bgra8"):
        arr = rows[:, : width * 4].reshape(height, width, 4)[..., :3]
        if encoding == "bgra8":
            arr = arr[..., ::-1]
    else:
        raise ValueError(f"unsupported image encoding {encoding!r}")
    return np.ascontiguousarray(arr)


def score_as_dict(s: Score) -> dict:
    d = asdict(s)
    d["success"] = s.success
    return d
