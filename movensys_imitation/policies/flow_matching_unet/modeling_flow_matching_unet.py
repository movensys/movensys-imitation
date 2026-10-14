"""Flow Matching policy sharing the Diffusion Policy encoder and UNet (plan, Section 6).

``DiffusionRgbEncoder`` and ``DiffusionConditionalUnet1d`` are imported from LeRobot, not copied,
so the only difference to ``DiffusionPolicy`` is the generative objective:

* training:  ``x_t = (1 - t) * x0 + t * x1`` with ``x0 ~ N(0, I)`` and ``x1`` the normalised action
  chunk; the UNet predicts the velocity ``x1 - x0`` and is trained with MSE;
* inference: start from ``x0`` and integrate the predicted velocity from ``t = 0`` to ``t = 1`` with
  ``fm_num_steps`` fixed Euler (or midpoint) steps.

The path and solver are module-level functions (``ot_path``, ``integrate``) so they can be tested
against Meta's ``flow_matching`` library independently of the network.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionModel,
    DiffusionPolicy,
    DiffusionRgbEncoder,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn

from .configuration_flow_matching_unet import FlowMatchingUnetConfig

VelocityFn = Callable[[Tensor, Tensor], Tensor]
"""``v(x, t) -> dx/dt`` with ``x`` of shape (B, ...) and ``t`` of shape (B,) in [0, 1]."""


def ot_path(x0: Tensor, x1: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """Conditional optimal-transport (linear) probability path.

    Args:
        x0: noise sample, (B, ...).
        x1: data sample, same shape as ``x0``.
        t: times in [0, 1], shape (B,).
    Returns:
        ``(x_t, dx_t)``: the interpolant ``(1 - t) * x0 + t * x1`` and its time derivative
        ``x1 - x0`` (the regression target).
    """
    t_ = t.reshape(-1, *([1] * (x0.ndim - 1))).to(x0.dtype)
    return (1 - t_) * x0 + t_ * x1, x1 - x0


def sample_time(batch_size: int, method: str, device: torch.device, generator=None) -> Tensor:
    """Training-time distribution of ``t`` (see ``FlowMatchingUnetConfig.fm_time_sampling``)."""
    if method == "uniform":
        return torch.rand(batch_size, device=device, generator=generator)
    if method == "beta":
        # pi0: noise level s ~ Beta(1.5, 1) scaled into [0.001, 1]; our t is 1 - s.
        beta = torch.distributions.Beta(
            torch.tensor(1.5, device=device), torch.tensor(1.0, device=device)
        )
        s = beta.sample((batch_size,)) * 0.999 + 0.001
        return 1.0 - s
    raise ValueError(f"Unknown time sampling {method!r}")


def integrate(velocity: VelocityFn, x0: Tensor, num_steps: int, solver: str = "euler") -> Tensor:
    """Integrate ``dx/dt = velocity(x, t)`` from ``t = 0`` to ``t = 1`` on a uniform grid.

    ``"euler"``: one velocity evaluation per step. ``"midpoint"`` (explicit RK2): two per step.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    x = x0
    dt = 1.0 / num_steps
    batch = x0.shape[0]
    for k in range(num_steps):
        t = torch.full((batch,), k * dt, device=x0.device, dtype=x0.dtype)
        v = velocity(x, t)
        if solver == "euler":
            x = x + dt * v
        elif solver == "midpoint":
            v_mid = velocity(x + 0.5 * dt * v, t + 0.5 * dt)
            x = x + dt * v_mid
        else:
            raise ValueError(f"Unknown solver {solver!r}")
    return x


class FlowMatchingModel(DiffusionModel):
    """Encoder + UNet from ``DiffusionModel``; flow-matching loss and sampler on top.

    ``_prepare_global_conditioning`` and ``generate_actions`` are inherited unchanged.
    """

    def __init__(self, config: FlowMatchingUnetConfig):
        nn.Module.__init__(self)  # skip DiffusionModel.__init__: no noise scheduler here
        self.config = config

        global_cond_dim = config.robot_state_feature.shape[0]
        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if config.env_state_feature:
            global_cond_dim += config.env_state_feature.shape[0]

        self.unet = DiffusionConditionalUnet1d(
            config, global_cond_dim=global_cond_dim * config.n_obs_steps
        )
        if config.compile_model:
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

        self.num_inference_steps = config.fm_num_steps

    # ========= velocity field ============
    def velocity(self, x: Tensor, t: Tensor, global_cond: Tensor | None) -> Tensor:
        """UNet as a velocity field; ``t`` in [0, 1] is rescaled for the sinusoidal embedding."""
        return self.unet(x, t * self.config.fm_time_embed_scale, global_cond=global_cond)

    # ========= inference ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        x0 = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )
        sample = integrate(
            lambda x, t: self.velocity(x, t, global_cond),
            x0,
            num_steps=self.config.fm_num_steps,
            solver=self.config.fm_solver,
        )
        if self.config.clip_sample:
            sample = sample.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)
        return sample

    # ========= training ============
    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Same batch contract as ``DiffusionModel.compute_loss``."""
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        x1 = batch[ACTION]
        x0 = torch.randn_like(x1)
        t = sample_time(x1.shape[0], self.config.fm_time_sampling, x1.device)
        xt, target = ot_path(x0, x1, t)
        pred = self.velocity(xt, t, global_cond)

        loss = F.mse_loss(pred, target, reduction="none")

        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            mask = (~batch["action_is_pad"]).unsqueeze(-1)
            num_valid = mask.sum() * loss.shape[-1]
            return (loss * mask).sum() / num_valid.clamp_min(1)

        return loss.mean()


class FlowMatchingUnetPolicy(DiffusionPolicy):
    """Flow Matching policy. ``select_action`` / ``predict_action_chunk`` / ``forward`` and the
    observation and action queues are inherited from ``DiffusionPolicy`` so that chunking and
    history handling are identical by construction (plan Section 6, fairness)."""

    config_class = FlowMatchingUnetConfig
    name = "flow_matching_unet"

    def __init__(self, config: FlowMatchingUnetConfig, **kwargs):
        PreTrainedPolicy.__init__(self, config)  # not DiffusionPolicy.__init__ (needs diffusers)
        config.validate_features()
        self.config = config
        self._queues = None
        # Attribute name kept as in DiffusionPolicy: the inherited methods call `self.diffusion`.
        self.diffusion = FlowMatchingModel(config)
        self.reset()

    @property
    def flow(self) -> FlowMatchingModel:
        return self.diffusion
