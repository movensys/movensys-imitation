"""Recorder core (P2.4): frame schema, timing verdicts, dataset and sidecar writing."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from movensys_imitation.data.convert_gym_pusht import recorder_features
from movensys_imitation.tasks.pusht import scoring as sc
from movensys_imitation_ros import recorder_core as rc

W, H, SIDE = 1280, 720, 628


def calibration() -> sc.Calibration:
    return sc.Calibration(
        image_size=(W, H), crop=(315, 45, SIDE, SIDE), px_per_m=951.0, t_area_px=5717,
        goal_mask=np.zeros((SIDE, SIDE), dtype=bool), threshold=sc.HsvRange(),
        t_hsv_median=(100, 200, 100), centre_uv=(314.0, 314.0),
    )


def frame_image(value: int = 7) -> np.ndarray:
    img = np.full((H, W, 3), value, dtype=np.uint8)
    img[45:45 + SIDE, 315:315 + SIDE] = 200  # the crop region is brighter than the rest
    return img


def sample(t: float, **kw) -> rc.Sample:
    base = {
        "t": t, "image": frame_image(), "image_t": t - 0.03, "eef_xy": (0.1, -0.05),
        "action_xy": (0.11, -0.05), "joints": (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
        "block_pose": (0.0, 0.0, 0.785), "coverage": 0.5, "iou": 0.4,
    }
    base.update(kw)
    return rc.Sample(**base)


def test_build_frame_matches_schema():
    cfg = rc.RecorderConfig()
    frame = rc.build_frame(sample(0.0), calibration(), cfg)
    feats = recorder_features(256)
    assert set(frame) == set(feats) | {"task"}
    for key, ft in feats.items():
        assert tuple(frame[key].shape) == tuple(ft["shape"]), key
    assert frame["observation.images.top"].dtype == np.uint8
    assert frame["observation.images.top"].min() == 200  # only the crop region, resized
    assert frame["observation.state"].dtype == np.float32
    assert np.allclose(frame["action"], [0.11, -0.05])
    assert frame["coverage"][0] == pytest.approx(0.5) and frame["iou"][0] == pytest.approx(0.4)
    assert frame["task"] == cfg.task


def test_state_and_action_are_in_the_sheet_frame():
    cfg = rc.RecorderConfig(sheet_origin=(-0.03, -0.05))
    frame = rc.build_frame(sample(0.0), calibration(), cfg)
    assert np.allclose(frame["observation.state"], [0.13, 0.0])
    assert np.allclose(frame["action"], [0.14, 0.0])


def test_block_pose_nan_on_the_real_robot():
    frame = rc.build_frame(sample(0.0, block_pose=None), calibration(), rc.RecorderConfig())
    assert np.all(np.isnan(frame["block_pose"]))


def test_missing_inputs_named():
    s = sample(0.0, image=None, joints=None)
    assert rc.missing_inputs(s) == ["image", "joints"]
    with pytest.raises(ValueError, match="missing image, joints"):
        rc.EpisodeBuffer(calibration(), rc.RecorderConfig()).add(s)


def test_timing_clean_episode():
    buf = rc.EpisodeBuffer(calibration(), rc.RecorderConfig())
    for i in range(5):
        buf.add(sample(i * 0.1, coverage=0.3 if i < 3 else 0.96))
    t = buf.timing
    assert len(buf) == 5 and t.ok()
    assert t.missed_ticks == 0 and t.stale_frames == 0
    assert t.max_tick_dt == pytest.approx(0.1)
    assert t.max_image_age == pytest.approx(0.03)
    assert t.first_success_frame == 3
    assert t.to_dict(10)["time_to_success"] == pytest.approx(0.3)


def test_timing_missed_tick_and_stale_image():
    buf = rc.EpisodeBuffer(calibration(), rc.RecorderConfig())
    buf.add(sample(0.0))
    buf.add(sample(0.1))
    buf.add(sample(0.4))  # two ticks missing
    buf.add(sample(0.5, image_t=0.2))  # image 0.3 s old
    t = buf.timing
    assert t.missed_ticks == 2
    assert t.stale_frames == 1
    assert t.max_image_age == pytest.approx(0.3)
    assert not t.ok()
    d = t.to_dict(10)
    assert d["timing_ok"] is False and d["time_to_success"] is None


def test_tick_jitter_within_tolerance_is_not_missed():
    buf = rc.EpisodeBuffer(calibration(), rc.RecorderConfig(tick_tolerance=0.02))
    buf.add(sample(0.0))
    buf.add(sample(0.115))
    assert buf.timing.missed_ticks == 0


@pytest.mark.lerobot
def test_store_writes_dataset_and_sidecar(tmp_path: Path):
    pytest.importorskip("lerobot")
    cfg = rc.RecorderConfig(repo_id="test/pusht", root=tmp_path / "ds", image_size=64)
    buf = rc.EpisodeBuffer(calibration(), cfg)
    for i in range(4):
        buf.add(sample(i * 0.1))
    store = rc.EpisodeStore(cfg)
    line = store.save(buf, {"seed": 1000, "operator": "test"})
    assert line["episode_index"] == 0 and line["n_frames"] == 4 and line["seed"] == 1000
    assert line["timing_ok"] is True and line["sheet_origin_wm"] == [0.0, 0.0]
    # Finalized after the save: readable before the store closes, and the index carries on.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    assert LeRobotDataset(cfg.repo_id, root=cfg.root).meta.total_episodes == 1
    assert store.next_episode_index == 1
    store.close()
    lines = rc.read_sidecar(cfg.root)
    assert len(lines) == 1 and lines[0] == json.loads(json.dumps(line))

    # A second store resumes the same dataset and appends episode 1.
    store2 = rc.EpisodeStore(cfg)
    buf2 = rc.EpisodeBuffer(calibration(), cfg)
    for i in range(3):
        buf2.add(sample(i * 0.1))
    line2 = store2.save(buf2, {"seed": 1001})
    store2.close()
    assert line2["episode_index"] == 1
    assert len(rc.read_sidecar(cfg.root)) == 2

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(cfg.repo_id, root=cfg.root)
    assert ds.meta.total_episodes == 2 and ds.meta.total_frames == 7 and ds.fps == 10
    assert set(recorder_features(64)) <= set(ds.meta.features)
    item = ds[0]
    assert tuple(item["observation.images.top"].shape) == (3, 64, 64)
    assert math.isclose(float(item["coverage"]), 0.5, abs_tol=1e-6)


@pytest.mark.lerobot
def test_store_clears_an_empty_root_and_refuses_a_foreign_one(tmp_path: Path):
    pytest.importorskip("lerobot")
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)  # leftover of a smoke start: directories, no files
    cfg = rc.RecorderConfig(repo_id="test/pusht", root=root, image_size=64)
    buf = rc.EpisodeBuffer(calibration(), cfg)
    buf.add(sample(0.0))
    store = rc.EpisodeStore(cfg)
    assert store.save(buf, {"seed": 1})["episode_index"] == 0
    store.close()

    foreign = tmp_path / "other"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("not a dataset")
    cfg2 = rc.RecorderConfig(repo_id="test/pusht", root=foreign, image_size=64)
    with pytest.raises(FileExistsError, match="without meta/info.json"):
        rc.EpisodeStore(cfg2).open_dataset()
    assert (foreign / "notes.txt").exists()
