import os

import pytest

ament_flake8 = pytest.importorskip("ament_flake8.main")  # only in a sourced Jazzy shell

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(os.path.dirname(PACKAGE_DIR), "ament_flake8.ini")


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    # Explicit paths: ament_flake8 otherwise lints the current directory, which from the repo
    # root would include .venv.
    rc, errors = ament_flake8.main_with_errors(argv=["--config", CONFIG, PACKAGE_DIR])
    assert rc == 0, "Found %d code style errors / warnings:\n" % len(errors) + "\n".join(errors)
