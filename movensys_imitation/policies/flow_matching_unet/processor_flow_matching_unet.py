"""Pre / post processors for ``flow_matching_unet``: identical to Diffusion's (plan, Section 6).

LeRobot resolves ``make_<type>_pre_post_processors`` from this module by naming convention.
"""

from __future__ import annotations

from typing import Any

import torch
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    make_default_pre_post_processors,
)

from .configuration_flow_matching_unet import FlowMatchingUnetConfig


def make_flow_matching_unet_pre_post_processors(
    config: FlowMatchingUnetConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Same pipeline as Diffusion: rename, normalise (MIN_MAX state / action, MEAN_STD images),
    add batch dim, move to device; and the reverse on the action."""
    return make_default_pre_post_processors(config, dataset_stats)
