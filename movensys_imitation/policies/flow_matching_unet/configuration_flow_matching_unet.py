"""Configuration for the Flow Matching policy (plan, Section 6).

The class subclasses LeRobot's ``DiffusionConfig`` so that the encoder, UNet and horizon
settings are shared field for field; only the generative objective differs. The defaults
are the Phase 1 fairness settings (the ``lerobot/diffusion_pusht`` model-card values), so
``--policy.type=flow_matching_unet`` alone gives the matched 16 / 8, 84 x 84 crop, GroupNorm,
no-pretrained-weights configuration. ``diffusion.yaml`` sets the same values for Diffusion.

Inherited Diffusion fields that are not used by the flow objective (``noise_scheduler_type``,
``num_train_timesteps``, ``beta_*``, ``prediction_type``, ``num_inference_steps``) are kept so
that the two config files diff cleanly. ``clip_sample`` / ``clip_sample_range`` are honoured: the
final sample is clamped the same way Diffusion clamps each denoising step.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

FM_SOLVERS = ("euler", "midpoint")
FM_TIME_SAMPLINGS = ("uniform", "beta")


@PreTrainedConfig.register_subclass("flow_matching_unet")
@dataclass
class FlowMatchingUnetConfig(DiffusionConfig):
    """Flow Matching with the Diffusion Policy encoder and 1-D UNet.

    Conditional optimal-transport path ``x_t = (1 - t) * noise + t * action`` with ``t`` in [0, 1],
    velocity target ``action - noise``, MSE loss. Inference integrates the learned velocity from
    ``t = 0`` to ``t = 1`` with a fixed-step solver.

    Args (in addition to ``DiffusionConfig``):
        fm_num_steps: Number of solver steps at inference (K in the plan: 1, 2, 4, 8, 16).
        fm_solver: ``"euler"`` (one velocity call per step) or ``"midpoint"`` (two calls per step).
        fm_time_sampling: Training-time distribution of ``t``. ``"uniform"`` or ``"beta"``, the
            pi0 schedule: the noise level ``1 - t`` is drawn from Beta(1.5, 1), which emphasises
            noisier samples.
        fm_time_embed_scale: ``t`` is multiplied by this before the UNet's sinusoidal step
            embedding so that [0, 1] spans the same embedding range as Diffusion's 100 integer
            timesteps.
    """

    # Phase 1 fairness defaults (plan Section 6).
    horizon: int = 16
    n_action_steps: int = 8
    crop_shape: tuple[int, int] | None = (84, 84)
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    use_separate_rgb_encoder_per_camera: bool = False

    # Flow matching.
    fm_num_steps: int = 8
    fm_solver: str = "euler"
    fm_time_sampling: str = "uniform"
    fm_time_embed_scale: float = 100.0

    def __post_init__(self):
        super().__post_init__()
        if self.fm_num_steps < 1:
            raise ValueError(f"`fm_num_steps` must be >= 1. Got {self.fm_num_steps}.")
        if self.fm_solver not in FM_SOLVERS:
            raise ValueError(f"`fm_solver` must be one of {FM_SOLVERS}. Got {self.fm_solver!r}.")
        if self.fm_time_sampling not in FM_TIME_SAMPLINGS:
            raise ValueError(
                f"`fm_time_sampling` must be one of {FM_TIME_SAMPLINGS}. "
                f"Got {self.fm_time_sampling!r}."
            )
        if self.fm_time_embed_scale <= 0:
            raise ValueError(f"`fm_time_embed_scale` must be > 0. Got {self.fm_time_embed_scale}.")
