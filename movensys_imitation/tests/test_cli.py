"""CLI dispatch tests. The resolution test needs LeRobot and is skipped where it is absent."""

import pytest
import yaml

from movensys_imitation.cli import _lerobot


@pytest.mark.lerobot
@pytest.mark.parametrize("command", ["train", "eval"])
def test_lerobot_entry_points_resolve(command):
    pytest.importorskip("lerobot", reason="needs the full venv (plan task P0.2)")
    assert callable(_lerobot._resolve(command))


def test_merge_config_paths_deep_merges(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("policy:\n  type: diffusion\n  horizon: 16\nsteps: 100\n")
    b.write_text("policy:\n  horizon: 32\neval:\n  n_episodes: 5\n")
    argv = _lerobot.merge_config_paths([f"--config_path={a},{b}", "--seed=1"])
    assert argv[1] == "--seed=1"
    merged = yaml.safe_load(open(argv[0].removeprefix("--config_path=")))
    assert merged == {
        "policy": {"type": "diffusion", "horizon": 32},
        "steps": 100,
        "eval": {"n_episodes": 5},
    }


def test_merge_config_paths_passes_single_path_through():
    argv = ["--config_path=outputs/x/train_config.json", "--resume=true"]
    assert _lerobot.merge_config_paths(argv) == argv
