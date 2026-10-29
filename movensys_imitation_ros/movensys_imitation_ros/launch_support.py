"""
Helpers for launch files that start venv nodes with ``python -m`` (scorer, recorder, manager).

Once a wildcard params file (``/**``, task.yaml) and an exact-name one (scorer.yaml,
recorder.yaml) are both given, rcl keeps the exact-name section over any later ``-p
name:=value`` and over any later wildcard file (measured 2026-10-28: task.yaml + scorer.yaml +
``-p mode:=real`` keeps the yaml's ``mode``; a single file plus ``-p`` behaves). Only a later
file with the node's own name wins, which is what the Node action writes for its dict
parameters; this module does the same for ExecuteProcess.
"""

from __future__ import annotations

import os
import tempfile

from launch.actions import ExecuteProcess
from launch.launch_context import LaunchContext
from launch.substitutions import LaunchConfiguration
from launch.utilities import perform_substitutions
import yaml


def perform(context: LaunchContext, value):
    """Resolve a substitution (or list of them) to a string; plain values pass through."""
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        return "".join(str(perform(context, v)) for v in value)
    return perform_substitutions(context, [value])


def coerce(text):
    """Cast launch-argument text (``true``, ``10``, ``0.5``) to the type the node declares."""
    if not isinstance(text, str):
        return text
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def overrides_file(context: LaunchContext, node_name: str, overrides: dict) -> str:
    """Write ``overrides`` under ``node_name`` to a temporary params file; return its path."""
    params = {k: coerce(perform(context, v)) for k, v in overrides.items()}
    fd, path = tempfile.mkstemp(prefix=f"movensys_{node_name}_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump({node_name: {"ros__parameters": params}}, f)
    return path


def venv_process(
    context: LaunchContext,
    module: str,
    params_files: list[str],
    overrides: dict,
    node_name: str | None = None,
    **kwargs,
) -> ExecuteProcess:
    """``python -m movensys_imitation_ros.<module>`` with the params files, overrides last.

    ``node_name`` defaults to ``module`` (the nodes here are named after their module).
    """
    cmd = ["python", "-m", f"movensys_imitation_ros.{module}", "--ros-args"]
    for path in params_files:
        cmd += ["--params-file", path]
    if overrides:
        cmd += ["--params-file", overrides_file(context, node_name or module, overrides)]
    return ExecuteProcess(cmd=cmd, output="screen", **kwargs)


def sim_time_and_mode(context: LaunchContext) -> dict:
    """``use_sim_time`` and the node ``mode`` from the launch argument ``mode`` (sim|hil|real)."""
    mode = perform(context, LaunchConfiguration("mode"))
    return {"use_sim_time": mode != "real", "mode": "sim" if mode != "real" else "real"}
