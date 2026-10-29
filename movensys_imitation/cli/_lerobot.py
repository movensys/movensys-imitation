"""Thin dispatch to LeRobot's train / eval scripts after registering the project policies.

Every LeRobot CLI flag keeps working; the only addition is that ``--policy.type`` can name a
policy from :mod:`movensys_imitation.policies`.
"""

from __future__ import annotations

import importlib
import importlib.util
import multiprocessing
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import yaml

from movensys_imitation.policies import register_all

# (module, attribute) candidates per command, newest LeRobot layout first.
_ENTRY_POINTS: dict[str, tuple[tuple[str, str], ...]] = {
    "train": (
        ("lerobot.scripts.lerobot_train", "main"),
        ("lerobot.scripts.train", "main"),
        ("lerobot.scripts.train", "train"),
    ),
    "eval": (
        ("lerobot.scripts.lerobot_eval", "main"),
        ("lerobot.scripts.eval", "main"),
        ("lerobot.scripts.eval", "eval_main"),
    ),
}


def _resolve(command: str) -> Callable[[], None]:
    errors: list[str] = []
    for module_name, attr in _ENTRY_POINTS[command]:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:  # pragma: no cover - depends on the installed LeRobot
            errors.append(f"{module_name}: {exc}")
            continue
        fn = getattr(module, attr, None)
        if callable(fn):
            return fn
        errors.append(f"{module_name} has no callable {attr!r}")
    raise ImportError(
        f"Could not find LeRobot's {command} entry point. Tried:\n  " + "\n  ".join(errors)
    )


# Gym packages whose environments register on import. LeRobot 0.6.1 imports them in the parent
# process only, but builds AsyncVectorEnv with the "forkserver" context, whose workers start from
# a clean interpreter and therefore raise NamespaceNotFound. Preloading them into the forkserver
# fixes that without touching LeRobot.
_FORKSERVER_PRELOAD = ("gym_pusht",)


def _preload_gym_packages() -> None:
    present = [name for name in _FORKSERVER_PRELOAD if importlib.util.find_spec(name)]
    if present:
        multiprocessing.set_forkserver_preload(present)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def merge_config_paths(argv: list[str]) -> list[str]:
    """Allow ``--config_path=a.yaml,b.yaml``: later files deep-merge over earlier ones.

    Draccus takes one config file. The project keeps task settings (``pusht_gym.yaml``) and
    policy settings (``diffusion.yaml`` ...) in separate fragments, so a comma-separated list is
    merged into one temporary YAML before LeRobot's parser sees it. A single path is passed
    through untouched (this keeps ``--config_path=.../train_config.json --resume=true`` working).
    """
    for index, arg in enumerate(argv):
        if arg.startswith("--config_path=") and "," in arg:
            paths = [Path(p) for p in arg.removeprefix("--config_path=").split(",") if p]
            merged: dict = {}
            for path in paths:
                with open(path) as f:
                    data = yaml.safe_load(f) or {}
                if not isinstance(data, dict):
                    raise ValueError(f"{path} must contain a mapping at the top level")
                merged = _deep_merge(merged, data)
            tmp = tempfile.NamedTemporaryFile(
                "w", suffix=".yaml", delete=False, prefix="movensys_cfg_"
            )
            yaml.safe_dump(merged, tmp, default_flow_style=False)
            tmp.close()
            argv = list(argv)
            argv[index] = f"--config_path={tmp.name}"
    return argv


def run(command: str) -> None:
    _preload_gym_packages()
    register_all()
    sys.argv[1:] = merge_config_paths(sys.argv[1:])
    _resolve(command)()
