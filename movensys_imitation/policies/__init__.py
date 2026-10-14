"""Project policy types for LeRobot.

``register_all()`` imports every policy package below so that its ``PreTrainedConfig`` subclass
registers itself with LeRobot's config registry. It is a no-op for policies that are not
implemented yet (plan task P1.1 adds ``flow_matching_unet``).
"""

from __future__ import annotations

import importlib

_POLICY_PACKAGES = ("movensys_imitation.policies.flow_matching_unet",)


def register_all() -> list[str]:
    """Import all project policy packages; return the names that registered."""
    registered: list[str] = []
    for name in _POLICY_PACKAGES:
        module = importlib.import_module(name)
        if getattr(module, "POLICY_TYPE", None):
            registered.append(module.POLICY_TYPE)
    return registered
