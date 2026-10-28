"""
Episode recorder logic without ROS (plan P2.4).

:class:`EpisodeBuffer` turns one :class:`Sample` per 10 Hz tick into a LeRobot frame of the
recorder schema (``movensys_imitation.data.convert_gym_pusht.recorder_features``) and keeps the
timing statistics that decide whether the episode is trustworthy: LeRobot derives timestamps from
the frame index, so a missed tick silently shifts every later action against its observation.
:class:`EpisodeStore` writes finished buffers to a ``LeRobotDataset`` (created or resumed) and
appends one JSON line of episode metadata per saved episode to
``<root>/meta/movensys_episodes.jsonl`` (start seed, operator, timing, end score, versions:
the plan's episode metadata, which the LeRobot episode table cannot hold).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from movensys_imitation.data.convert_gym_pusht import recorder_features, TASK
from movensys_imitation.tasks.pusht import scoring as sc

SIDECAR = "movensys_episodes.jsonl"
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))


@dataclass
class RecorderConfig:
    repo_id: str = "movensys/pusht_cr3a_sim_v1"
    root: Path = Path("datasets/movensys/pusht_cr3a_sim_v1")
    fps: int = 10
    image_size: int = 256
    robot_type: str = "dobot_cr3a"
    task: str = TASK
    joint_names: tuple[str, ...] = JOINT_NAMES
    max_image_age: float = 0.15
    """Seconds between the tick and the image stamp beyond which the frame counts as stale."""
    tick_tolerance: float = 0.02
    """Seconds of slack on the 1 / fps tick interval before a tick counts as missed."""
    success_threshold: float = sc.SUCCESS_THRESHOLD


@dataclass
class Sample:
    """What the node has at one tick; ``t`` and ``image_t`` are seconds on the same clock."""

    t: float
    image: np.ndarray | None = None  # full overhead frame, RGB uint8 (H, W, 3)
    image_t: float = -math.inf
    eef_xy: tuple[float, float] | None = None
    action_xy: tuple[float, float] | None = None
    joints: tuple[float, ...] | None = None
    block_pose: tuple[float, float, float] | None = None  # None on the real robot: NaN
    coverage: float = math.nan
    iou: float = math.nan


REQUIRED = ("image", "eef_xy", "action_xy", "joints")


def missing_inputs(s: Sample) -> list[str]:
    """Names of the inputs a frame cannot be built without."""
    return [name for name in REQUIRED if getattr(s, name) is None]


def resize_crop(crop_rgb: np.ndarray, size: int) -> np.ndarray:
    """Square crop to ``size`` x ``size`` with area interpolation (downscale from about 628 px)."""
    import cv2

    if crop_rgb.shape[0] == size and crop_rgb.shape[1] == size:
        return np.ascontiguousarray(crop_rgb)
    return np.ascontiguousarray(cv2.resize(crop_rgb, (size, size), interpolation=cv2.INTER_AREA))


def build_frame(s: Sample, cal: sc.Calibration, cfg: RecorderConfig) -> dict:
    """One LeRobot frame of the recorder schema from a complete sample."""
    cal.check_frame(s.image)
    image = resize_crop(cal.apply_crop(s.image), cfg.image_size)
    block = s.block_pose if s.block_pose is not None else (math.nan, math.nan, math.nan)
    return {
        "observation.images.top": image,
        "observation.state": np.asarray(s.eef_xy, dtype=np.float32).reshape(2),
        "action": np.asarray(s.action_xy, dtype=np.float32).reshape(2),
        "observation.joints": np.asarray(s.joints, dtype=np.float32).reshape(6),
        "block_pose": np.asarray(block, dtype=np.float32).reshape(3),
        "coverage": np.array([s.coverage], dtype=np.float32),
        "iou": np.array([s.iou], dtype=np.float32),
        "task": cfg.task,
    }


@dataclass
class Timing:
    """Per-episode timing statistics; ``ok`` is the recorder's verdict on the sampling."""

    n_frames: int = 0
    missed_ticks: int = 0
    max_tick_dt: float = 0.0
    stale_frames: int = 0
    max_image_age: float = 0.0
    first_success_frame: int = -1

    def ok(self) -> bool:
        return self.missed_ticks == 0 and self.stale_frames == 0

    def to_dict(self, fps: int) -> dict:
        d = {
            "n_frames": self.n_frames,
            "missed_ticks": self.missed_ticks,
            "max_tick_dt": round(self.max_tick_dt, 4),
            "stale_frames": self.stale_frames,
            "max_image_age": round(self.max_image_age, 4),
            "timing_ok": self.ok(),
            "first_success_frame": self.first_success_frame,
        }
        d["time_to_success"] = (
            self.first_success_frame / fps if self.first_success_frame >= 0 else None
        )
        return d


class EpisodeBuffer:
    """Frames of the episode being recorded plus their timing statistics."""

    def __init__(self, cal: sc.Calibration, cfg: RecorderConfig) -> None:
        self.cal = cal
        self.cfg = cfg
        self.frames: list[dict] = []
        self.timing = Timing()
        self.last_t: float | None = None

    def add(self, s: Sample) -> dict:
        """Build and keep the frame for ``s``; raises ``ValueError`` if an input is missing."""
        missing = missing_inputs(s)
        if missing:
            raise ValueError("missing " + ", ".join(missing))
        cfg, tm = self.cfg, self.timing
        period = 1.0 / cfg.fps
        if self.last_t is not None:
            dt = s.t - self.last_t
            tm.max_tick_dt = max(tm.max_tick_dt, dt)
            if dt > period + cfg.tick_tolerance:
                tm.missed_ticks += max(1, int(round(dt / period)) - 1)
        self.last_t = s.t
        age = s.t - s.image_t
        tm.max_image_age = max(tm.max_image_age, age)
        if age > cfg.max_image_age:
            tm.stale_frames += 1
        frame = build_frame(s, self.cal, cfg)
        if tm.first_success_frame < 0 and s.coverage >= cfg.success_threshold:
            tm.first_success_frame = len(self.frames)
        self.frames.append(frame)
        tm.n_frames = len(self.frames)
        return frame

    def __len__(self) -> int:
        return len(self.frames)


def software_versions() -> dict:
    """Versions worth keeping with an episode (plan Section 5 metadata)."""
    out: dict = {"numpy": np.__version__}
    try:
        import lerobot

        out["lerobot"] = lerobot.__version__
    except Exception:  # noqa: B902  not installed in CI
        pass
    try:
        import subprocess

        here = Path(__file__).resolve().parent
        out["git"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True, text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:  # noqa: B902  not a checkout
        pass
    return out


class EpisodeStore:
    """A ``LeRobotDataset`` in write mode plus the metadata sidecar. LeRobot is imported lazily."""

    def __init__(self, cfg: RecorderConfig) -> None:
        self.cfg = cfg
        self.dataset = None
        self.sidecar = Path(cfg.root) / "meta" / SIDECAR

    def open_dataset(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = Path(self.cfg.root)
        if (root / "meta" / "info.json").exists():
            self.dataset = LeRobotDataset.resume(
                self.cfg.repo_id, root=root, image_writer_threads=4
            )
            if self.dataset.fps != self.cfg.fps:
                raise ValueError(f"{root} has fps {self.dataset.fps}, recorder fps {self.cfg.fps}")
        else:
            # LeRobotDataset.create insists the root does not exist. An empty leftover directory
            # (a smoke start, an interrupted create) is harmless and removed; anything else is
            # someone's data and the operator decides.
            if root.exists():
                leftovers = [q for q in root.rglob("*") if q.is_file()]
                if leftovers:
                    raise FileExistsError(
                        f"{root} exists without meta/info.json but holds {len(leftovers)} "
                        "file(s); remove it or record into another root"
                    )
                shutil.rmtree(root)
            self.dataset = LeRobotDataset.create(
                repo_id=self.cfg.repo_id,
                fps=self.cfg.fps,
                features=recorder_features(self.cfg.image_size),
                root=root,
                robot_type=self.cfg.robot_type,
                use_videos=True,
                image_writer_threads=4,
            )
        return self.dataset

    @property
    def next_episode_index(self) -> int:
        return 0 if self.dataset is None else int(self.dataset.meta.total_episodes)

    def save(self, buffer: EpisodeBuffer, meta: dict) -> dict:
        """Write the buffered frames as one episode and the sidecar line; returns the line."""
        if self.dataset is None:
            self.open_dataset()
        if len(buffer) == 0:
            raise ValueError("empty episode")
        index = self.next_episode_index
        for frame in buffer.frames:
            self.dataset.add_frame(dict(frame))
        self.dataset.save_episode()
        line = {
            "episode_index": index,
            "repo_id": self.cfg.repo_id,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "fps": self.cfg.fps,
            **buffer.timing.to_dict(self.cfg.fps),
            "versions": software_versions(),
            **meta,
        }
        self.sidecar.parent.mkdir(parents=True, exist_ok=True)
        with self.sidecar.open("a") as f:
            f.write(json.dumps(line) + "\n")
        return line

    def close(self) -> None:
        if self.dataset is not None:
            self.dataset.finalize()
            self.dataset = None


def read_sidecar(root: Path) -> list[dict]:
    """Read the metadata lines of a dataset, in episode order (empty if none were written)."""
    path = Path(root) / "meta" / SIDECAR
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
