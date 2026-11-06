"""Closed-loop inference helper shared by the sim runner and the offline check (plan P2.8).

:class:`PolicyRunner` loads a LeRobot checkpoint with the project policies registered and the
checkpoint's own pre/post-processors (normalisation, device placement), and turns one recorder
frame (the 256 x 256 RGB crop plus the tip position in the sheet frame) into the next tip target
in the sheet frame. Everything ROS-specific stays in ``movensys_imitation_ros.policy_runner``.

The policy keeps its observation history (``n_obs_steps``) and its action chunk internally:
call :meth:`reset` at every episode start and :meth:`act` once per 10 Hz tick in order.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

OBS_IMAGE = "observation.images.top"
OBS_STATE = "observation.state"
ACTION = "action"


def load_config(pretrained: Path, overrides: tuple[str, ...] = ()):
    """``PreTrainedConfig`` of a checkpoint with policy-level overrides (``--fm_num_steps=4``).

    ``--policy.`` prefixes are stripped as the benchmark does; every override is checked
    against the loaded config because draccus ignores unknown keywords silently.
    """
    from lerobot.configs import PreTrainedConfig

    from movensys_imitation.policies import register_all

    register_all()
    cfg_args = [arg.replace("--policy.", "--", 1) for arg in overrides]
    cfg = PreTrainedConfig.from_pretrained(str(pretrained), cli_overrides=cfg_args)
    for arg in cfg_args:
        key, value = arg.removeprefix("--").split("=", 1)
        actual = getattr(cfg, key)
        if str(actual).lower() != value.lower() and str(actual) != value:
            try:
                assert float(actual) == float(value)
            except (TypeError, ValueError, AssertionError) as exc:
                raise RuntimeError(f"override {arg} not applied: {key}={actual!r}") from exc
    cfg.pretrained_path = pretrained
    return cfg


class PolicyRunner:
    """One checkpoint, one episode at a time."""

    def __init__(
        self, pretrained: str | Path, overrides: tuple[str, ...] = (), device: str | None = None
    ):
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        pretrained = Path(pretrained)
        self.cfg = load_config(pretrained, overrides)
        if device:
            self.cfg.device = device
        policy_cls = get_policy_class(self.cfg.type)
        self.policy = policy_cls.from_pretrained(str(pretrained), config=self.cfg)
        self.policy = self.policy.to(self.cfg.device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.cfg,
            pretrained_path=str(pretrained),
            preprocessor_overrides={"device_processor": {"device": str(self.cfg.device)}},
        )
        if OBS_IMAGE not in self.cfg.input_features or OBS_STATE not in self.cfg.input_features:
            raise RuntimeError(
                f"checkpoint inputs {sorted(self.cfg.input_features)} lack "
                f"{OBS_IMAGE} / {OBS_STATE}"
            )
        self.image_shape = tuple(self.cfg.input_features[OBS_IMAGE].shape)  # (C, H, W)
        self.last_latency_ms = float("nan")
        self.reset()
        self.warmup()

    def warmup(self, n: int = 3) -> None:
        """Run a few dummy calls so the first real tick does not pay for CUDA start-up.

        The first call took about 1 s on 2026-11-06, a real one 6 to 100 ms. Leaves the policy
        reset.
        """
        c, h, w = self.image_shape
        blank = np.zeros((h, w, c), dtype=np.uint8)
        for _ in range(n):
            self.act(blank, (0.0, 0.0))
        self.reset()

    @property
    def n_obs_steps(self) -> int:
        return int(getattr(self.cfg, "n_obs_steps", 1))

    def reset(self) -> None:
        self.policy.reset()
        self.n_calls = 0

    def observation(self, image_rgb: np.ndarray, state_xy) -> dict[str, torch.Tensor]:
        """Batch of one from a uint8 HWC frame (recorder crop) and the sheet-frame tip xy."""
        if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(
                f"image must be uint8 HWC RGB, got {image_rgb.dtype} {image_rgb.shape}"
            )
        c, h, w = self.image_shape
        if image_rgb.shape[:2] != (h, w):
            raise ValueError(f"image is {image_rgb.shape[:2]}, the checkpoint expects {(h, w)}")
        img = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float().div_(255.0)
        state = torch.as_tensor(np.asarray(state_xy, dtype=np.float32).reshape(2))
        return {OBS_IMAGE: img.unsqueeze(0), OBS_STATE: state.unsqueeze(0), "task": [""]}

    @torch.no_grad()
    def act(self, image_rgb: np.ndarray, state_xy) -> np.ndarray:
        """Next tip target (x, y) in the sheet frame for this tick's observation."""
        obs = self.observation(image_rgb, state_xy)
        t0 = time.perf_counter()
        batch = self.preprocessor(obs)
        action = self.policy.select_action(batch)
        action = self.postprocessor(action)
        if torch.is_tensor(action) and action.is_cuda:
            torch.cuda.synchronize()
        self.last_latency_ms = (time.perf_counter() - t0) * 1e3
        self.n_calls += 1
        return np.asarray(action.detach().cpu().numpy(), dtype=np.float32).reshape(-1)[:2]
