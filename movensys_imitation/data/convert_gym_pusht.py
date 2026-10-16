"""Plan task P1.6: convert ``lerobot/pusht`` into the recorder's dataset schema (plan Section 5).

The output has the keys the sim and real recorders will write, so the policy feature
definitions (``observation.images.top``, ``observation.state``, ``action``) are proven on gym
data before any robot data exists. Values stay in gym-pusht pixel units (0 to 512).

| Key | Source |
|---|---|
| ``observation.images.top`` 256 x 256 x 3 video | ``observation.image`` (96 x 96) upscaled |
| ``observation.state`` (2,) | ``observation.state`` (agent x, y) |
| ``action`` (2,) | ``action`` |
| ``observation.joints`` (6,) | zeros: gym-pusht has no arm |
| ``block_pose`` (3,) | zeros: not stored in ``lerobot/pusht`` (``pusht_keypoints`` has it) |
| ``coverage`` (1,) | ``next.reward`` x 0.95 (exact below 0.95, saturated at 0.95) |

    movensys-convert-gym [--episodes N] [--size 256] [--force]
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPO = "lerobot/pusht"
TARGET_REPO = "movensys/pusht_gym"
TARGET_ROOT = ROOT / "datasets" / "movensys" / "pusht_gym"
TASK = "Push the T-shaped block onto the T-shaped target."
SUCCESS_THRESHOLD = 0.95  # gym-pusht: reward = clip(coverage / 0.95, 0, 1)

log = logging.getLogger("convert_gym_pusht")


def recorder_features(size: int) -> dict:
    """Dataset features of the recorder schema (plan Section 5), image side ``size``."""
    return {
        "observation.images.top": {
            "dtype": "video",
            "shape": (size, size, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
        "observation.joints": {
            "dtype": "float32",
            "shape": (6,),
            "names": [f"joint{i}" for i in range(1, 7)],
        },
        "block_pose": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "yaw"]},
        "coverage": {"dtype": "float32", "shape": (1,), "names": None},
    }


def to_uint8_hwc(image) -> np.ndarray:
    """LeRobot returns decoded video frames as float32 CHW in [0, 1]."""
    arr = image.numpy() if hasattr(image, "numpy") else np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def resize(image_hwc: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    if image_hwc.shape[0] == size and image_hwc.shape[1] == size:
        return image_hwc
    return np.asarray(Image.fromarray(image_hwc).resize((size, size), Image.BILINEAR))


def convert_frame(item: dict, size: int) -> dict:
    reward = float(np.asarray(item["next.reward"]).reshape(-1)[0])
    return {
        "observation.images.top": resize(to_uint8_hwc(item["observation.image"]), size),
        "observation.state": np.asarray(item["observation.state"], dtype=np.float32).reshape(2),
        "action": np.asarray(item["action"], dtype=np.float32).reshape(2),
        "observation.joints": np.zeros(6, dtype=np.float32),
        "block_pose": np.zeros(3, dtype=np.float32),
        "coverage": np.array([reward * SUCCESS_THRESHOLD], dtype=np.float32),
        "task": TASK,
    }


def convert(
    target_root: Path = TARGET_ROOT,
    repo_id: str = TARGET_REPO,
    size: int = 256,
    episodes: int | None = None,
    force: bool = False,
) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if target_root.exists():
        if not force:
            raise FileExistsError(f"{target_root} exists; pass --force to overwrite")
        shutil.rmtree(target_root)

    source = LeRobotDataset(SOURCE_REPO)
    n_episodes = (
        source.meta.total_episodes
        if episodes is None
        else min(episodes, source.meta.total_episodes)
    )
    log.info(
        "source %s: %d episodes, %d frames, fps %d",
        SOURCE_REPO,
        source.meta.total_episodes,
        len(source),
        source.fps,
    )

    target = LeRobotDataset.create(
        repo_id=repo_id,
        fps=source.fps,
        features=recorder_features(size),
        root=target_root,
        robot_type="gym_pusht",
        use_videos=True,
        image_writer_threads=4,
    )
    current = None
    saved = 0
    for i in range(len(source)):
        item = source[i]
        ep = int(np.asarray(item["episode_index"]).reshape(-1)[0])
        if current is not None and ep != current:
            target.save_episode()
            saved += 1
            log.info("saved episode %d (%d / %d)", current, saved, n_episodes)
            if saved >= n_episodes:
                current = None
                break
        current = ep
        target.add_frame(convert_frame(item, size))
    if current is not None and target.has_pending_frames():
        target.save_episode()
        saved += 1
        log.info("saved episode %d (%d / %d)", current, saved, n_episodes)
    target.finalize()
    log.info(
        "wrote %s: %d episodes, %d frames",
        target_root,
        target.meta.total_episodes,
        target.meta.total_frames,
    )
    return target_root


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", default=str(TARGET_ROOT))
    parser.add_argument("--repo-id", default=TARGET_REPO)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--episodes", type=int, help="convert only the first N episodes")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = parser.parse_args(argv)
    convert(Path(args.root), args.repo_id, args.size, args.episodes, args.force)


if __name__ == "__main__":
    main()
