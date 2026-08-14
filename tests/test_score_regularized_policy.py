"""Tests for the score-regularized deterministic policy (SRPO) baseline.

Exercises the integration between ``train.py``'s ``ScoreRegularizedAgent``
(the call site that wires the distilled policy into the diffusion policy
baseline's training/eval loop) and the
``diffusion_policy.score_regularized_policy`` module it drives.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

# train.py imports diffusers/tensorboard at module scope; stub the pieces it
# only references at runtime so the agent definitions can be imported without
# the full training stack installed.
for _name, _attrs in [
    ("diffusers", {}),
    ("diffusers.schedulers", {}),
    ("diffusers.schedulers.scheduling_ddpm", {"DDPMScheduler": object}),
    ("diffusers.training_utils", {"EMAModel": object}),
    ("diffusers.optimization", {"get_scheduler": object}),
    ("torch.utils.tensorboard", {"SummaryWriter": object}),
]:
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        for _k, _v in _attrs.items():
            setattr(_mod, _k, _v)
        sys.modules[_name] = _mod

_TRAIN_DIR = Path(__file__, "..", "..", "examples", "baselines", "diffusion_policy")
sys.path.insert(0, str(_TRAIN_DIR.resolve()))

from diffusion_policy.score_regularized_policy import (  # noqa: E402
    ScoreRegularizedPolicy,
    score_weight,
    vp_marginal_prob_std,
    vp_to_ddpm_timestep,
)
from train import Args, ScoreRegularizedAgent  # noqa: E402


class _FakeNoisePredNet(torch.nn.Module):
    """Stand-in for a trained ConditionalUnet1D epsilon-prediction network."""

    def __init__(self, act_dim):
        super().__init__()
        self.lin = torch.nn.Linear(act_dim, act_dim)

    def forward(self, sample, timestep, global_cond):
        return self.lin(sample)


class _FakeScheduler:
    """Only ``alphas_cumprod`` is needed by the score-regularized policy."""

    def __init__(self, num_train_timesteps=100):
        betas = torch.linspace(0.0001, 0.02, num_train_timesteps)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)


class _FakeDiffusionAgent:
    def __init__(self, act_dim, pred_horizon):
        self.act_dim = act_dim
        self.pred_horizon = pred_horizon
        self.obs_horizon = 2
        self.act_horizon = 8
        self.noise_pred_net = _FakeNoisePredNet(act_dim)
        self.noise_scheduler = _FakeScheduler()


def _make_agent(act_dim=7, pred_horizon=16, obs_dim=32, B=4):
    diffusion_agent = _FakeDiffusionAgent(act_dim, pred_horizon)
    args = Args(total_iters=10, srpo_iters=10)
    agent = ScoreRegularizedAgent(diffusion_agent, args, global_cond_dim=2 * obs_dim)
    return agent, act_dim, pred_horizon, obs_dim, B


def test_score_regularized_agent_action_shape_and_range():
    """The distilled agent serves the diffusion policy's get_action contract."""
    agent, act_dim, pred_horizon, obs_dim, B = _make_agent()
    obs_seq = torch.randn(B, agent.obs_horizon, obs_dim)
    with torch.no_grad():
        actions = agent.get_action(obs_seq)
    assert actions.shape == (B, agent.act_horizon, act_dim)
    # policy ends in tanh, so actions live in [-1, 1] like the denoised chunks
    assert actions.max() <= 1.0 and actions.min() >= -1.0


def test_score_regularized_agent_loss_finite_and_trainable():
    """The SRPO loss is finite and only the policy head receives gradients."""
    agent, act_dim, pred_horizon, obs_dim, B = _make_agent()
    obs_seq = torch.randn(B, agent.obs_horizon, obs_dim)
    act_seq = torch.randn(B, pred_horizon, act_dim)
    loss = agent.compute_loss(obs_seq, act_seq)
    assert torch.isfinite(loss)

    loss.backward()
    policy_grads = [
        p.grad for p in agent.policy.policy.parameters() if p.grad is not None
    ]
    assert len(policy_grads) > 0
    # the diffusion behavior model stays frozen during distillation
    for p in agent.policy.diffusion_behavior.parameters():
        assert not p.requires_grad
        assert p.grad is None


def test_score_regularization_trains_against_frozen_teacher():
    """A few distillation steps lower the SRPO loss on a fixed batch.

    This is the paper's core claim at the smallest scale: regressing a
    deterministic policy onto a frozen diffusion behavior model's noise
    prediction is a trainable signal on its own, with no reward needed.
    """
    torch.manual_seed(0)
    act_dim, pred_horizon, obs_dim, B = 4, 8, 16, 6
    policy = ScoreRegularizedPolicy(
        noise_pred_net=_FakeNoisePredNet(act_dim),
        scheduler=_FakeScheduler(),
        act_dim=act_dim,
        pred_horizon=pred_horizon,
        global_cond_dim=obs_dim,
        hidden_dim=64,
    )
    obs_cond = torch.randn(B, obs_dim)
    opt = torch.optim.Adam(policy.policy.parameters(), lr=1e-2)

    torch.manual_seed(1)  # fix the sampled timesteps/noise to compare fairly
    first = policy.loss(obs_cond)
    first.backward()
    opt.step()
    opt.zero_grad()
    for _ in range(30):
        policy.loss(obs_cond).backward()
        opt.step()
        opt.zero_grad()
    torch.manual_seed(1)
    last = policy.loss(obs_cond)
    assert last < first


def test_critic_guidance_pulls_policy_towards_high_q_actions():
    """With beta > 0 the optional Q-guidance term shapes the objective."""
    torch.manual_seed(0)
    act_dim, pred_horizon, obs_dim, B = 4, 8, 16, 6
    policy = ScoreRegularizedPolicy(
        noise_pred_net=_FakeNoisePredNet(act_dim),
        scheduler=_FakeScheduler(),
        act_dim=act_dim,
        pred_horizon=pred_horizon,
        global_cond_dim=obs_dim,
        hidden_dim=64,
        beta=1.0,
    )
    obs_cond = torch.randn(B, obs_dim)

    def critic(cond, actions):
        # Q increases along the first action dimension
        return actions[..., 0].sum(dim=1)

    torch.manual_seed(2)
    base = policy.loss(obs_cond)
    guided = policy.loss(obs_cond, critic=critic)
    assert torch.isfinite(guided) and guided is not base
    guided.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in policy.policy.parameters()
    )


def test_vp_schedule_endpoints_and_weight_modes():
    t = torch.tensor([0.0, 1.0])
    alpha, std = vp_marginal_prob_std(t)
    # t=0 leaves the data untouched, t=1 is (nearly) pure noise
    assert torch.isclose(alpha[0], torch.tensor(1.0))
    assert alpha[1] < 0.01 and std[1] > 0.99

    assert torch.allclose(
        score_weight(torch.ones(2), "stable"), torch.ones(2)
    )
    assert torch.allclose(score_weight(torch.ones(2), "vds"), (std[1] ** 2) * torch.ones(2))
    with pytest.raises(ValueError):
        score_weight(torch.ones(2), "nope")


def test_vp_to_ddpm_timestep_mapping():
    assert vp_to_ddpm_timestep(torch.tensor([0.0]), 100).item() == 0
    assert vp_to_ddpm_timestep(torch.tensor([1.0]), 100).item() == 99
    # out-of-range times are clamped into the network's timestep vocabulary
    assert vp_to_ddpm_timestep(torch.tensor([2.0]), 100).item() == 99


def test_agent_integrates_with_mani_skill_observation_pipeline():
    """The call site consumes observations the way the baseline's eval loop does."""
    from mani_skill.utils import common  # non-new module under test

    agent, act_dim, pred_horizon, obs_dim, B = _make_agent()
    raw = np.random.randn(B, agent.obs_horizon, obs_dim).astype(np.float32)
    obs = common.to_tensor(raw, agent.policy.policy[0].weight.device)
    with torch.no_grad():
        actions = agent.get_action(obs)
    assert actions.shape == (B, agent.act_horizon, act_dim)
