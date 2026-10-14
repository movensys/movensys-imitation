"""CLI dispatch tests. The resolution test needs LeRobot and is skipped where it is absent."""

import pytest

from movensys_imitation.cli import _lerobot

lerobot = pytest.importorskip("lerobot", reason="needs the full venv (plan task P0.2)")


@pytest.mark.lerobot
@pytest.mark.parametrize("command", ["train", "eval"])
def test_lerobot_entry_points_resolve(command):
    assert callable(_lerobot._resolve(command))
