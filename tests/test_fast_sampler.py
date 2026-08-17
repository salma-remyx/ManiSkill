"""Integration test for fast ODE-based (DEIS) sampling in the Diffusion Policy baseline.

Exercises the wiring in ``examples/baselines/diffusion_policy/train.py``: the
``Agent`` built with ``use_deis=True`` must sample its actions through the DEIS
scheduler instead of the DDPM loop, with the number of noise-prediction
evaluations set by ``num_inference_steps``.
"""

import importlib.util
import pathlib
import sys

import pytest
import torch

EXAMPLE_DIR = (
    pathlib.Path(__file__).parent.parent
    / "examples"
    / "baselines"
    / "diffusion_policy"
)

pytest.importorskip("diffusers")

# train.py and the diffusion_policy package live side by side in the example
# directory, and train.py imports that package by name.
sys.path.insert(0, str(EXAMPLE_DIR))


def _load_train_module():
    """Import ``train.py`` as a module without triggering its ``__main__`` block."""
    spec = importlib.util.spec_from_file_location("diffusion_policy_train", EXAMPLE_DIR / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    train_module = _load_train_module()
except ImportError as e:  # pragma: no cover - depends on optional example deps
    pytest.skip(f"diffusion policy example deps unavailable: {e}", allow_module_level=True)


class _Box:
    """Minimal stand-in for a gymnasium Box action/observation space."""

    def __init__(self, shape):
        self.shape = shape


class _FakeEnv:
    """Just the attributes ``Agent.__init__`` reads off the vector env."""

    def __init__(self, obs_dim, act_dim, obs_horizon):
        self.single_observation_space = _Box((obs_horizon, obs_dim))
        self.single_action_space = _Box((act_dim,))
        self.single_action_space.high = torch.ones(act_dim)
        self.single_action_space.low = -torch.ones(act_dim)


class _CountingUNet(torch.nn.Module):
    """Records every timestep it is conditioned on, returns a fixed noise estimate."""

    def __init__(self):
        super().__init__()
        self.seen_timesteps = []

    def forward(self, sample, timestep, global_cond=None):
        self.seen_timesteps.append(int(timestep))
        return torch.zeros_like(sample)


def _make_args(num_inference_steps, use_deis):
    args = train_module.Args(exp_name="test", obs_horizon=2, act_horizon=4, pred_horizon=8)
    args.num_inference_steps = num_inference_steps
    args.use_deis = use_deis
    return args


def _build_agent(num_inference_steps, use_deis):
    agent = train_module.Agent(_FakeEnv(obs_dim=5, act_dim=3, obs_horizon=2),
                               _make_args(num_inference_steps, use_deis))
    # swap in a tiny counting network so no real UNet weights are needed
    agent.noise_pred_net = _CountingUNet()
    return agent


def test_deis_get_action_uses_fast_timestep_grid():
    """use_deis=True makes get_action call the network once per inference step."""
    agent = _build_agent(num_inference_steps=5, use_deis=True)
    assert agent.inference_scheduler is not None

    obs_seq = torch.randn(2, agent.obs_horizon, 5)
    action = agent.get_action(obs_seq)

    assert action.shape == (2, agent.act_horizon, 3)
    # NFE decoupled from the 100 training steps
    assert len(agent.noise_pred_net.seen_timesteps) == 5
    # timesteps come from the fast solver, not the full DDPM grid
    assert set(agent.noise_pred_net.seen_timesteps) <= set(
        agent.inference_scheduler.timesteps.tolist()
    )
    # every conditioning timestep must lie on the 0..99 training grid
    assert all(
        0 <= t < agent.num_diffusion_iters
        for t in agent.noise_pred_net.seen_timesteps
    )


def test_default_get_action_still_uses_ddpm():
    """Without use_deis the baseline behaves exactly as before."""
    agent = _build_agent(num_inference_steps=None, use_deis=False)
    assert agent.inference_scheduler is None

    obs_seq = torch.randn(2, agent.obs_horizon, 5)
    action = agent.get_action(obs_seq)

    assert action.shape == (2, agent.act_horizon, 3)
    assert len(agent.noise_pred_net.seen_timesteps) == agent.num_diffusion_iters


def test_deis_tracks_training_noise_schedule():
    """The fast sampler shares the training schedule, so the alpha/sigma curves agree."""
    from diffusion_policy.fast_sampler import build_deis_scheduler

    train_sched = _build_agent(None, False).noise_scheduler
    fast_sched = build_deis_scheduler(train_sched, 5)

    assert fast_sched.config.num_train_timesteps == train_sched.config.num_train_timesteps
    assert fast_sched.config.beta_schedule == train_sched.config.beta_schedule
    assert fast_sched.config.prediction_type == train_sched.config.prediction_type
    assert len(fast_sched.timesteps) == 5
    torch.testing.assert_close(fast_sched.alphas_cumprod, train_sched.alphas_cumprod)


def test_deis_denoise_recovers_conditional_target():
    """End to end sanity: with an epsilon predictor that is exact for a single
    target, 5-step DEIS lands close to that target (not exact -- it is a 5-step
    approximation -- but far closer than the [-1, 1] action range)."""
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    from diffusion_policy.fast_sampler import build_deis_scheduler, deis_denoise

    train_sched = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    fast_sched = build_deis_scheduler(train_sched, 5)

    target = torch.tensor([[0.5, -0.5], [0.25, 0.75]])

    def eps_net(sample, timestep, global_cond=None):
        # exact epsilon for x0 = target: eps = (x_t - alpha_t * x0) / sigma_t
        alpha = train_sched.alphas_cumprod[int(timestep)].sqrt()
        sigma = (1 - train_sched.alphas_cumprod[int(timestep)]).sqrt()
        return (sample - alpha * target) / sigma

    out = deis_denoise(eps_net, fast_sched, None, target.shape, target.device)
    assert out.shape == target.shape
    assert (out - target).abs().max() < 0.2


def test_deis_get_action_is_reentrant():
    """A policy is queried once per control step; each call must be a full, clean
    rollout rather than continuing the previous one's multistep state."""
    agent = _build_agent(num_inference_steps=5, use_deis=True)

    for _ in range(3):
        agent.noise_pred_net.seen_timesteps.clear()
        action = agent.get_action(torch.randn(2, agent.obs_horizon, 5))
        assert action.shape == (2, agent.act_horizon, 3)
        assert agent.noise_pred_net.seen_timesteps == [99, 79, 59, 40, 20]
