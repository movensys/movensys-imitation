"""Flow Matching policy sharing the Diffusion Policy encoder and UNet (plan, Section 6).

Importing this package registers ``flow_matching_unet`` with LeRobot's config registry; the
policy and processor classes are then resolved by LeRobot's naming convention
(``configuration_`` -> ``modeling_`` / ``processor_``).
"""

from .configuration_flow_matching_unet import FlowMatchingUnetConfig

POLICY_TYPE: str | None = "flow_matching_unet"

__all__ = ["POLICY_TYPE", "FlowMatchingUnetConfig"]
