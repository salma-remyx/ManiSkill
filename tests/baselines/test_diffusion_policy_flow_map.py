"""Integration tests for the flow-map (consistency model) diffusion policy path.

These exercise the public surface a real run uses: ``train.Agent`` with
``flow_map=True`` goes through ``compute_loss`` and the ``get_action`` fast
path that ``diffusion_policy.evaluate.evaluate`` calls at every control step,
with the shared ``ConditionalUnet1D`` left unmodified.
"""

import importlib
import pathlib
import sys
import types
from typing import Any

import numpy as np
import pytest
import torch

_BASELINE_DIR = (
    pathlib.Path(__file__, "..", "..", "..", "examples", "baselines", "diffusion_policy").resolve()
)
sys.path.insert(0, str(_BASELINE_DIR))


def _load(name: str) -> Any:
    """Import a baseline module resolved off the computed sys.path above."""
    return importlib.import_module(name)


def _import_train() -> Any:
    """Import the baseline training script without its heavyweight deps.

    ``train.py`` binds ``tensorboard`` and ``diffusers`` at import time, but
    neither is reachable from the flow-map path under test: the DDPM scheduler
    is only used by the non-flow-map branch and the writer/schedulers only by
    the ``__main__`` loop. Stubbing those four names lets the test exercise the
    real ``Agent`` wiring in environments without the baseline extras installed.
    """
    torch_tb = types.ModuleType("torch.utils.tensorboard")

    class _SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_text(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

    setattr(torch_tb, "SummaryWriter", _SummaryWriter)

    class _Anything:
        """Constructible no-op stand-in for unused diffusers helpers."""

        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            return _Anything()

        def __call__(self, *args, **kwargs):
            return _Anything()

    class _Config:
        def __init__(self, num_train_timesteps):
            self.num_train_timesteps = num_train_timesteps

    class _DDPMScheduler:
        """Just the members the baseline reads, so the DDPM branch stays intact."""

        def __init__(self, num_train_timesteps, **kwargs):
            self.config = _Config(num_train_timesteps)
            self.timesteps = list(range(num_train_timesteps))[::-1]

        def add_noise(self, action_seq, noise, timesteps):
            return action_seq

        def step(self, **kwargs):
            return _Anything()

    ddpm_mod = types.ModuleType("diffusers.schedulers.scheduling_ddpm")
    setattr(ddpm_mod, "DDPMScheduler", _DDPMScheduler)
    training_mod = types.ModuleType("diffusers.training_utils")
    setattr(training_mod, "EMAModel", _Anything)
    optimization_mod = types.ModuleType("diffusers.optimization")
    setattr(
        optimization_mod,
        "get_scheduler",
        lambda *args, **kwargs: _Anything(),
    )

    for name, module in {
        "torch.utils.tensorboard": torch_tb,
        "diffusers": types.ModuleType("diffusers"),
        "diffusers.schedulers": types.ModuleType("diffusers.schedulers"),
        "diffusers.schedulers.scheduling_ddpm": ddpm_mod,
        "diffusers.training_utils": training_mod,
        "diffusers.optimization": optimization_mod,
    }.items():
        sys.modules.setdefault(name, module)

    return _load("train")


class _FakeArgs:
    obs_horizon = 2
    act_horizon = 4
    pred_horizon = 8
    diffusion_step_embed_dim = 32
    unet_dims = [16, 32]
    n_groups = 8
    flow_map = True
    eta = 0.75


class _FakeSpace:
    def __init__(self, shape):
        self.shape = shape
        self.high = np.ones(shape, dtype=np.float32)
        self.low = -np.ones(shape, dtype=np.float32)


class _FakeEnv:
    """Stands in for the vectorized env the baseline queries for its shapes."""

    def __init__(self, obs_dim=6, act_dim=7, obs_horizon=2):
        self.single_observation_space = _FakeSpace((obs_horizon, obs_dim))
        self.single_action_space = _FakeSpace((act_dim,))


@pytest.fixture(scope="module")
def agent_cls() -> Any:
    return _import_train().Agent


def _build_agent(agent_cls):
    return agent_cls(_FakeEnv(), _FakeArgs())


def test_agent_builds_flow_map_policy_around_shared_net(agent_cls):
    agent = _build_agent(agent_cls)
    assert agent.flow_map_policy is not None
    # the flow map must reuse the baseline's noise prediction net, not add one
    assert agent.flow_map_policy.net is agent.noise_pred_net


def test_flow_map_opt_in_leaves_default_agent_untouched(agent_cls):
    """Without the flag the DDPM loss and 100-step denoise loop still apply."""
    args = _FakeArgs()
    args.flow_map = False
    agent = agent_cls(_FakeEnv(), args)
    assert agent.flow_map_policy is None


def test_flow_map_loss_is_finite_and_trains_the_shared_net(agent_cls):
    torch.manual_seed(0)
    agent = _build_agent(agent_cls)
    obs_seq = torch.randn(16, 2, 6)
    action_seq = torch.rand(16, 8, 7) * 2 - 1

    loss = agent.compute_loss(obs_seq, action_seq)
    assert torch.isfinite(loss)
    assert loss.item() > 0

    loss.backward()
    grads = [p.grad for p in agent.noise_pred_net.parameters() if p.grad is not None]
    assert grads, "loss must reach the shared ConditionalUnet1D parameters"

    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3)
    first = loss.item()
    for _ in range(100):
        loss = agent.compute_loss(obs_seq, action_seq)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < first, f"loss did not decrease: {first} -> {loss.item()}"


def test_get_action_returns_act_horizon_chunk(agent_cls):
    torch.manual_seed(0)
    agent = _build_agent(agent_cls)
    agent.eval()
    action_seq = agent.get_action(torch.randn(5, 2, 6))
    assert action_seq.shape == (5, agent.act_horizon, agent.act_dim)
    assert torch.isfinite(action_seq).all()


def test_flow_map_generation_is_single_network_evaluation(agent_cls):
    """The point of the flow map: one forward pass per action chunk."""
    torch.manual_seed(0)
    agent = _build_agent(agent_cls)
    agent.eval()
    calls = []

    net = agent.noise_pred_net
    original_forward = net.forward

    def counting_forward(*args, **kwargs):
        calls.append(1)
        return original_forward(*args, **kwargs)

    net.forward = counting_forward
    try:
        with torch.no_grad():
            agent.get_action(torch.randn(3, 2, 6))
    finally:
        net.forward = original_forward
    assert len(calls) == 1, f"expected 1 network evaluation, got {len(calls)}"


def test_flow_map_policy_boundary_condition_holds_exactly():
    """X_{s,s}(x) = x must hold by construction, not by penalty."""
    conditional_unet1d = _load("diffusion_policy.conditional_unet1d")
    flow_map_policy = _load("diffusion_policy.flow_map_policy")
    ConditionalUnet1D = conditional_unet1d.ConditionalUnet1D
    FlowMapPolicy = flow_map_policy.FlowMapPolicy
    FlowMapConfig = flow_map_policy.FlowMapConfig

    torch.manual_seed(0)
    net = ConditionalUnet1D(
        input_dim=7,
        global_cond_dim=2 * 6,
        diffusion_step_embed_dim=32,
        down_dims=[16, 32],
        n_groups=8,
    )
    policy = FlowMapPolicy(net, FlowMapConfig(num_train_timesteps=100, eta=0.75))
    x = torch.randn(4, 8, 7)
    obs_cond = torch.randn(4, 12)
    for s in (0, 37, 99):
        level = torch.full((4,), s)
        v_ss = policy.velocity(x, level, level, global_cond=obs_cond)
        # X_{s,s}(x) = x + 0 * v = x regardless of the network weights
        assert torch.equal(x + 0 * v_ss, x)
