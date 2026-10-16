"""Plan task P1.1 acceptance tests for ``flow_matching_unet`` (need torch + LeRobot)."""

import pytest

torch = pytest.importorskip("torch", reason="needs the full venv (plan task P0.2)")
pytest.importorskip("lerobot", reason="needs the full venv (plan task P0.2)")

from lerobot.configs import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_policy_config  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE  # noqa: E402

from movensys_imitation.policies import register_all  # noqa: E402
from movensys_imitation.policies.flow_matching_unet.modeling_flow_matching_unet import (  # noqa: E402
    FlowMatchingUnetPolicy,
    integrate,
    ot_path,
    sample_time,
)

pytestmark = pytest.mark.lerobot

STATE_DIM, ACTION_DIM, IMG = 2, 2, 96


def _tiny_config(**overrides):
    """A small flow_matching_unet config (narrow UNet, horizon 16 / 8, 84 x 84 crop)."""
    register_all()
    cfg = make_policy_config(
        "flow_matching_unet",
        down_dims=(32, 64, 128),
        diffusion_step_embed_dim=32,
        spatial_softmax_num_keypoints=8,
        device="cpu",
        **overrides,
    )
    cfg.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMG, IMG)),
    }
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    return cfg


def _batch(cfg, batch_size=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        OBS_STATE: torch.rand(batch_size, cfg.n_obs_steps, STATE_DIM, generator=g) * 2 - 1,
        "observation.image": torch.rand(batch_size, cfg.n_obs_steps, 3, IMG, IMG, generator=g),
        ACTION: torch.rand(batch_size, cfg.horizon, ACTION_DIM, generator=g) * 2 - 1,
        "action_is_pad": torch.zeros(batch_size, cfg.horizon, dtype=torch.bool),
    }


def test_registers_and_resolves_by_type():
    assert "flow_matching_unet" in register_all()
    assert get_policy_class("flow_matching_unet") is FlowMatchingUnetPolicy
    cfg = make_policy_config("flow_matching_unet")
    assert (cfg.horizon, cfg.n_action_steps, cfg.crop_shape) == (16, 8, (84, 84))


@pytest.mark.parametrize(
    "bad", [{"fm_solver": "rk4"}, {"fm_time_sampling": "cosine"}, {"fm_num_steps": 0}]
)
def test_config_validation(bad):
    with pytest.raises(ValueError):
        _tiny_config(**bad)


def test_shapes():
    cfg = _tiny_config()
    policy = FlowMatchingUnetPolicy(cfg).eval()
    batch = _batch(cfg)
    loss, _ = policy.forward(dict(batch))
    assert loss.ndim == 0 and torch.isfinite(loss)
    obs = {k: v for k, v in batch.items() if k not in (ACTION, "action_is_pad")}
    chunk = policy.predict_action_chunk(obs)
    assert chunk.shape == (4, cfg.n_action_steps, ACTION_DIM)
    policy.reset()
    single = {
        OBS_STATE: obs[OBS_STATE][:, -1],
        "observation.image": obs["observation.image"][:, -1],
    }
    action = policy.select_action(single)
    assert action.shape == (4, ACTION_DIM)
    assert len(policy._queues[ACTION]) == cfg.n_action_steps - 1
    assert policy._queues[OBS_IMAGES].maxlen == cfg.n_obs_steps


def test_loss_decreases_on_synthetic_data():
    torch.manual_seed(0)
    cfg = _tiny_config()
    policy = FlowMatchingUnetPolicy(cfg).train()
    batch = _batch(cfg, batch_size=8)
    opt = torch.optim.Adam(policy.get_optim_params(), lr=1e-3)
    losses = []
    for _ in range(40):
        loss, _ = policy.forward(dict(batch))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert last < 0.7 * first, f"loss did not decrease: {first:.3f} -> {last:.3f}"


@pytest.mark.parametrize("num_steps", [1, 16])
@pytest.mark.parametrize("solver", ["euler", "midpoint"])
def test_sampler_finite_and_seed_deterministic(num_steps, solver):
    torch.manual_seed(0)
    cfg = _tiny_config(fm_num_steps=num_steps, fm_solver=solver)
    policy = FlowMatchingUnetPolicy(cfg).eval()
    obs = {k: v for k, v in _batch(cfg).items() if k not in (ACTION, "action_is_pad")}

    def run(seed):
        torch.manual_seed(seed)
        policy.reset()
        return policy.predict_action_chunk(dict(obs))

    a, b, c = run(1), run(1), run(2)
    assert torch.isfinite(a).all()
    assert a.abs().max() <= cfg.clip_sample_range
    assert torch.equal(a, b), "same seed must give the same chunk"
    assert not torch.allclose(a, c), "different seeds must give different chunks"


def test_sample_time_ranges():
    for method in ("uniform", "beta"):
        t = sample_time(10_000, method, torch.device("cpu"))
        assert t.shape == (10_000,) and (t >= 0).all() and (t <= 1).all()
    # The beta schedule puts more mass at noisy times (small t) than uniform does.
    torch.manual_seed(0)
    assert sample_time(10_000, "beta", torch.device("cpu")).mean() < 0.45


# ---- 2-D toy problem against Meta's flow_matching library ---------------------------------


def test_ot_path_matches_meta_flow_matching():
    fm_path = pytest.importorskip("flow_matching.path")
    fm_sched = pytest.importorskip("flow_matching.path.scheduler")
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(64, 2, generator=g, dtype=torch.float64)
    x1 = torch.randn(64, 2, generator=g, dtype=torch.float64)
    t = torch.rand(64, generator=g, dtype=torch.float64)
    ours_xt, ours_dxt = ot_path(x0, x1, t)
    ref = fm_path.AffineProbPath(scheduler=fm_sched.CondOTScheduler()).sample(x_0=x0, x_1=x1, t=t)
    assert torch.allclose(ours_xt, ref.x_t, atol=1e-12)
    assert torch.allclose(ours_dxt, ref.dx_t, atol=1e-12)


class _ToyField(torch.nn.Module):
    """Analytic 2-D velocity field: a rotation plus a time-dependent drift."""

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kw) -> torch.Tensor:
        t_ = t.reshape(-1, 1).to(x.dtype)  # (B, 1) or (1, 1): Meta's solver passes a scalar t
        rot = torch.stack([-x[:, 1], x[:, 0]], dim=1)
        return rot + torch.cat([torch.sin(3.0 * t_), torch.cos(2.0 * t_)], dim=1)


@pytest.mark.parametrize("solver,num_steps", [("euler", 1), ("euler", 16), ("midpoint", 8)])
def test_integrator_matches_meta_ode_solver(solver, num_steps):
    fm_solver = pytest.importorskip("flow_matching.solver")
    field = _ToyField()
    x0 = torch.randn(32, 2, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    ours = integrate(lambda x, t: field(x, t), x0, num_steps=num_steps, solver=solver)
    ref = fm_solver.ODESolver(velocity_model=field).sample(
        x_init=x0, step_size=1.0 / num_steps, method=solver, time_grid=torch.tensor([0.0, 1.0])
    )
    assert torch.allclose(ours, ref, atol=1e-10), (ours - ref).abs().max()
