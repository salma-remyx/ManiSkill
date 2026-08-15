"""Tests for the clockwork (resolution-stratified) denoising loop.

These exercise the wiring in examples/baselines/diffusion_policy/train.py --
the Agent.get_action call site -- not just the cache module in isolation.
``diffusers`` is an optional dependency of that example, so the whole module
skips cleanly when it is absent.
"""

import pathlib
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("gymnasium")
pytest.importorskip("diffusers")
pytest.importorskip("tyro")

EXAMPLES_DIR = (
    pathlib.Path(__file__).parents[3] / "examples" / "baselines" / "diffusion_policy"
).resolve()
sys.path.insert(0, str(EXAMPLES_DIR))

from diffusion_policy.conditional_unet1d import ConditionalUnet1D  # noqa: E402
from diffusion_policy.clockwork_denoise import (  # noqa: E402
    ClockworkUnet1D,
    clockwork_denoise,
    stage_timesteps,
)

OBS_DIM = 32
ACT_DIM = 7
OBS_HORIZON = 2
PRED_HORIZON = 16
UNET_DIMS = [64, 128, 256]


def _make_args(**overrides):
    args = SimpleNamespace(
        obs_horizon=OBS_HORIZON,
        act_horizon=8,
        pred_horizon=PRED_HORIZON,
        diffusion_step_embed_dim=64,
        unet_dims=UNET_DIMS,
        n_groups=8,
        clockwork_intervals=None,
    )
    args.__dict__.update(overrides)
    return args


def _make_agent(intervals=None):
    """Build the Agent from the training script with a stubbed observation space.

    Agent.__init__ only reads shapes off the environment, so a lightweight
    stand-in keeps this test free of any simulation dependency.
    """
    import train as train_module

    class _Space:
        def __init__(self, shape):
            self.shape = shape

    class _ObsSpace(_Space):
        high = None
        low = None

    class _ActionSpace(_Space):
        high = torch.ones(ACT_DIM)
        low = -torch.ones(ACT_DIM)

    env = SimpleNamespace(
        single_observation_space=_ObsSpace((OBS_HORIZON, OBS_DIM)),
        single_action_space=_ActionSpace((ACT_DIM,)),
    )
    return train_module.Agent(env, _make_args(clockwork_intervals=intervals))


@pytest.fixture(scope="module")
def unet():
    torch.manual_seed(0)
    return ConditionalUnet1D(
        input_dim=ACT_DIM,
        global_cond_dim=OBS_HORIZON * OBS_DIM,
        diffusion_step_embed_dim=64,
        down_dims=UNET_DIMS,
        n_groups=8,
    )


@pytest.fixture(scope="module")
def scheduler():
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    return DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )


def test_stage_timesteps_partitions_the_schedule():
    points = stage_timesteps(range(99, -1, -1), [4, 2, 1])
    assert [len(p) for p in points] == [25, 50, 100]
    assert points[2] == list(range(100))  # interval 1 refreshes every step
    assert all(j % 4 == 0 for j in points[0])
    assert all(j % 2 == 0 for j in points[1])


def test_stage_timesteps_rejects_non_positive_intervals():
    with pytest.raises(ValueError):
        stage_timesteps(range(10), [4, 0, 1])


def test_interval_of_one_reproduces_the_vanilla_forward(unet, scheduler):
    """No caching => identical output to calling the UNet directly."""
    torch.manual_seed(1)
    sample = torch.randn(2, PRED_HORIZON, ACT_DIM)
    cond = torch.randn(2, OBS_HORIZON * OBS_DIM)

    vanilla = unet(sample=sample, timestep=3, global_cond=cond)
    cached = ClockworkUnet1D(unet, [1, 1, 1])
    cached.set_schedule(scheduler.timesteps)
    cached.step_index = 3
    out = cached(sample=sample, timestep=3, global_cond=cond)

    assert torch.allclose(vanilla, out, atol=1e-6)
    assert [r for r, _ in cached.cache_stats()] == [1, 1, 1]
    assert [u for _, u in cached.cache_stats()] == [0, 0, 0]


def test_clockwork_denoise_matches_vanilla_loop_when_uncached(unet, scheduler):
    """clockwork_denoise with all-ones intervals is the vanilla loop."""
    torch.manual_seed(2)
    sample = torch.randn(2, PRED_HORIZON, ACT_DIM)
    cond = torch.randn(2, OBS_HORIZON * OBS_DIM)

    expected = sample
    for k in scheduler.timesteps:
        noise_pred = unet(sample=expected, timestep=k, global_cond=cond)
        expected = scheduler.step(
            model_output=noise_pred, timestep=k, sample=expected
        ).prev_sample

    actual = clockwork_denoise(
        unet, scheduler, sample, global_cond=cond, refresh_intervals=[1, 1, 1]
    )
    assert torch.allclose(expected, actual, atol=1e-5)


def test_clockwork_denoise_reuses_low_resolution_stages(unet, scheduler):
    """Coarse stages refresh on their clocks; the finest stage never caches."""
    torch.manual_seed(3)
    sample = torch.randn(2, PRED_HORIZON, ACT_DIM)
    cond = torch.randn(2, OBS_HORIZON * OBS_DIM)

    calls = []
    clockwork_denoise(
        unet,
        scheduler,
        sample,
        global_cond=cond,
        refresh_intervals=[4, 2, 1],
        on_step=lambda i, k: calls.append((i, int(k))),
    )
    assert len(calls) == len(scheduler.timesteps)

    cached = ClockworkUnet1D(unet, [4, 2, 1])
    cached.set_schedule(scheduler.timesteps)
    for step_index, k in enumerate(scheduler.timesteps):
        cached.step_index = step_index
        cached(
            sample=torch.randn_like(sample), timestep=k, global_cond=cond
        )
    refreshes, reuses = zip(*cached.cache_stats())
    assert refreshes == (25, 50, 100)
    assert reuses == (75, 50, 0)


def test_cached_stages_are_bounded(unet, scheduler):
    """A cached stage's output keeps its own shape, not the fresh input's."""
    torch.manual_seed(4)
    cond = torch.randn(2, OBS_HORIZON * OBS_DIM)
    cached = ClockworkUnet1D(unet, [8, 1, 1])
    cached.set_schedule(scheduler.timesteps)

    shapes = []
    for step_index, k in enumerate(scheduler.timesteps):
        cached.step_index = step_index
        out = cached(sample=torch.randn(2, PRED_HORIZON, ACT_DIM), timestep=k, global_cond=cond)
        shapes.append(tuple(out.shape))
        assert cached.cached_features[0].shape == (2, UNET_DIMS[0], PRED_HORIZON)
    assert set(shapes) == {(2, PRED_HORIZON, ACT_DIM)}


def test_agent_get_action_runs_clockwork(scheduler):
    """The call site: Agent.get_action dispatches to clockwork_denoise."""
    agent = _make_agent(intervals=[4, 2, 1])
    agent.noise_scheduler = scheduler
    obs_seq = torch.randn(3, OBS_HORIZON, OBS_DIM)

    torch.manual_seed(5)
    actions = agent.get_action(obs_seq)
    assert actions.shape == (3, agent.act_horizon, ACT_DIM)
    assert agent.clockwork_intervals == [4, 2, 1]


def test_agent_get_action_default_is_vanilla(scheduler):
    """Without the flag, get_action keeps the original denoising loop."""
    agent = _make_agent()
    agent.noise_scheduler = scheduler
    obs_seq = torch.randn(3, OBS_HORIZON, OBS_DIM)

    torch.manual_seed(6)
    actions = agent.get_action(obs_seq)
    assert actions.shape == (3, agent.act_horizon, ACT_DIM)
    assert agent.clockwork_intervals is None


def test_clockwork_and_vanilla_actions_stay_close(scheduler):
    """Moderate clocks should perturb the policy's output, not scramble it."""
    torch.manual_seed(7)
    obs_seq = torch.randn(3, OBS_HORIZON, OBS_DIM)

    vanilla_agent = _make_agent()
    vanilla_agent.noise_scheduler = scheduler
    vanilla_actions = vanilla_agent.get_action(obs_seq)

    clockwork_agent = _make_agent(intervals=[2, 1, 1])
    clockwork_agent.noise_scheduler = scheduler
    clockwork_agent.load_state_dict(vanilla_agent.state_dict())
    clockwork_actions = clockwork_agent.get_action(obs_seq)

    # the finest-resolution stage is still recomputed every step, so the two
    # loops should agree to within the coarse stages' staleness
    assert torch.isclose(vanilla_actions, clockwork_actions, atol=0.5).all()
