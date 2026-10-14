"""Thin dispatch to LeRobot's train / eval scripts after registering the project policies.

Every LeRobot CLI flag keeps working; the only addition is that ``--policy.type`` can name a
policy from :mod:`movensys_imitation.policies`.
"""

from __future__ import annotations

import importlib
import importlib.util
import multiprocessing
from collections.abc import Callable

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


def run(command: str) -> None:
    _preload_gym_packages()
    register_all()
    _resolve(command)()
