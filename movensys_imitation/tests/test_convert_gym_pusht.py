"""Unit tests for the gym conversion helpers (no torch / LeRobot needed)."""

import numpy as np

from movensys_imitation.data import convert_gym_pusht as cg


def test_recorder_features_match_plan_schema():
    f = cg.recorder_features(256)
    assert f["observation.images.top"]["shape"] == (256, 256, 3)
    assert f["observation.state"]["shape"] == (2,) and f["action"]["shape"] == (2,)
    assert f["observation.joints"]["shape"] == (6,) and f["block_pose"]["shape"] == (3,)
    assert f["coverage"]["shape"] == (1,) and f["iou"]["shape"] == (1,)
    # Plan Section 5 table, in order; the recorder (P2.4) writes exactly these keys.
    assert list(f) == [
        "observation.images.top", "observation.state", "action", "observation.joints",
        "block_pose", "coverage", "iou",
    ]


def test_convert_frame_units_and_dtypes():
    item = {
        "observation.image": np.random.rand(3, 96, 96).astype(np.float32),
        "observation.state": np.array([100.0, 200.0]),
        "action": np.array([110.0, 210.0]),
        "next.reward": np.array([0.5]),
    }
    frame = cg.convert_frame(item, 256)
    assert frame["observation.images.top"].shape == (256, 256, 3)
    assert frame["observation.images.top"].dtype == np.uint8
    assert frame["observation.state"].dtype == np.float32
    assert np.allclose(frame["observation.state"], [100.0, 200.0])  # pixel units kept
    assert np.isclose(frame["coverage"][0], 0.5 * 0.95)
    assert frame["iou"].shape == (1,) and frame["iou"][0] == 0.0
    assert frame["task"] == cg.TASK


def test_to_uint8_hwc_roundtrip():
    img = (np.arange(3 * 4 * 5).reshape(3, 4, 5) % 256).astype(np.float32) / 255.0
    out = cg.to_uint8_hwc(img)
    assert out.shape == (4, 5, 3) and out.dtype == np.uint8
    assert out[0, 0, 0] == 0 and out[0, 1, 0] == 1
