"""Tests for the few-step ODE (AMED-Solver) sampling path in the Diffusion
Policy baseline.

These exercise the integration through the baseline's real
``ConditionalUnet1D`` and a scheduler exposing the same surface as the
DDPMScheduler that ``train.py`` builds and ``diffusion_policy/evaluate.py``
drives at evaluation time. If ``diffusers`` is installed, the real
``DDPMScheduler`` is used.
"""

import math
import sys
import types
from pathlib import Path

import pytest
import torch

BASELINE_DIR = (
    Path(__file__).resolve().parents[2] / "examples" / "baselines" / "diffusion_policy"
)
sys.path.insert(0, str(BASELINE_DIR))

from diffusion_policy.amed_sampler import (  # noqa: E402
    AmedPredictor,
    amed_distill_loss,
    amed_sample,
    amed_sigmas,
)
from diffusion_policy.conditional_unet1d import ConditionalUnet1D  # noqa: E402

try:  # the baseline's scheduler, when its deps are installed
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    def make_scheduler(num_train_timesteps=100):
        return DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

except ImportError:  # pragma: no cover - exercised only without diffusers

    def make_scheduler(num_train_timesteps=100):
        return _TinyScheduler(num_train_timesteps)


class _TinyScheduler:
    """Minimal stand-in for DDPMScheduler's alpha-bar schedule.

    Uses the same squaredcos_cap_v2 betas the baseline trains with, so
    ``alphas_cumprod`` matches the real scheduler's values.
    """

    def __init__(self, num_train_timesteps=100):
        import types

        t = torch.arange(num_train_timesteps + 1, dtype=torch.float64)
        f = torch.cos(((t / num_train_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = torch.clip(1 - f[1:] / f[:-1], 0.0001, 0.9999)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).float()
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)
        self.config = types.SimpleNamespace(num_train_timesteps=num_train_timesteps)

    def step(self, model_output, timestep, sample):
        """Epsilon-prediction DDPM posterior step, as DDPMScheduler computes it."""
        t = int(timestep) if not torch.is_tensor(timestep) else int(timestep.reshape(-1)[0])
        prev_t = max(t - 1, 0)
        alpha_prod = self.alphas_cumprod[t]
        alpha_prod_prev = (
            self.alphas_cumprod[prev_t] if prev_t != t else torch.tensor(1.0)
        )
        beta_prod = 1 - alpha_prod
        pred_original = (sample - beta_prod.sqrt() * model_output) / alpha_prod.sqrt()
        pred_original = pred_original.clamp(-1, 1)
        # posterior variance: beta~_t = (1-abar_{t-1})/(1-abar_t) * beta_t
        variance = (1 - alpha_prod_prev) / (1 - alpha_prod) * beta_prod
        std = variance.clamp_min(1e-20).sqrt()
        prev = (
            alpha_prod_prev.sqrt() * pred_original
            + (1 - alpha_prod_prev - variance).clamp_min(0).sqrt() * model_output
            + std * torch.randn_like(sample)
        )
        return types.SimpleNamespace(prev_sample=prev)


class _ScriptAgent(torch.nn.Module):
    """Stand-in for train.py's Agent, built the way the script builds it."""

    def __init__(self, act_dim=4, pred_horizon=8, obs_dim=12, amed_steps=5):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.amed_steps = amed_steps
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=act_dim,
            global_cond_dim=obs_dim,
            diffusion_step_embed_dim=16,
            down_dims=[8, 16],
            n_groups=8,
        )
        self.noise_scheduler = make_scheduler()
        self.amed_predictor = AmedPredictor(
            self.noise_scheduler, embed_dim=16, hidden_dim=32
        )

    def get_action(self, obs_seq):
        """Mirrors the amed branch of Agent.get_action in train.py."""
        B = obs_seq.shape[0]
        with torch.no_grad():
            obs_cond = obs_seq.flatten(start_dim=1)
            if self.amed_steps > 0:
                return amed_sample(
                    self.noise_pred_net,
                    self.noise_scheduler,
                    self.amed_predictor,
                    (B, self.pred_horizon, self.act_dim),
                    obs_cond,
                    num_inference_steps=self.amed_steps,
                )
        raise AssertionError("unreachable")


def test_amed_sigma_grid():
    scheduler = make_scheduler()
    sigmas, t_idx = amed_sigmas(scheduler, 5, torch.device("cpu"))
    assert len(sigmas) == 5  # one grid point per function evaluation
    assert (sigmas[:-1] > sigmas[1:]).all()  # strictly decreasing
    assert float(sigmas[-1]) == 0.0  # ends at pure data
    assert int(t_idx[-1]) == 0
    # starts from the noisiest usable timestep; squaredcos_cap_v2's final
    # training steps are extreme sigma outliers and get trimmed
    assert int(t_idx[0]) >= scheduler.config.num_train_timesteps - 5
    assert int(t_idx[0]) < scheduler.config.num_train_timesteps - 1
    # geometric spacing: constant ratio between consecutive sigmas
    ratios = sigmas[:-2] / sigmas[1:-1]
    assert torch.allclose(ratios, ratios[0].expand_as(ratios), rtol=1e-4)
    # grid points spread across the schedule instead of collapsing on t~99
    assert int(t_idx[1]) <= int(t_idx[0]) - 2


def test_amed_sample_runs_in_five_steps():
    scheduler = make_scheduler()
    net = _ScriptAgent(amed_steps=5).noise_pred_net
    predictor = AmedPredictor(scheduler, embed_dim=16, hidden_dim=32)
    calls = []
    original = net.forward

    def counting_forward(sample, timestep, global_cond):
        t = timestep if torch.is_tensor(timestep) else torch.tensor(timestep)
        calls.append(int(t.flatten()[0]))
        return original(sample, timestep, global_cond=global_cond)

    net.forward = counting_forward
    obs_cond = torch.randn(3, 12)
    out = amed_sample(
        net, scheduler, predictor, (3, 8, 4), obs_cond, num_inference_steps=5
    )
    assert out.shape == (3, 8, 4)
    assert len(calls) == 5  # NFE == num_inference_steps
    assert calls == sorted(calls, reverse=True)  # timesteps go T -> 0


def test_amed_distill_loss_trains_only_predictor():
    agent = _ScriptAgent(amed_steps=5)
    obs_seq = torch.randn(8, 1, agent.obs_dim)
    action_seq = torch.randn(8, agent.pred_horizon, agent.act_dim)

    loss = amed_distill_loss(agent, obs_seq, action_seq, batch_size=4)
    assert torch.isfinite(loss)
    loss.backward()

    predictor_grads = [p.grad for p in agent.amed_predictor.parameters()]
    assert all(g is not None for g in predictor_grads)
    assert any(g.abs().sum() > 0 for g in predictor_grads)
    # the teacher trajectory is computed under no_grad, so the noise network
    # only receives gradients through the student pass (they are never applied
    # by the predictor's optimizer)


def test_distill_loss_decreases_when_training_predictor():
    torch.manual_seed(0)
    agent = _ScriptAgent(amed_steps=5)
    obs_seq = torch.randn(8, 1, agent.obs_dim)
    action_seq = torch.randn(8, agent.pred_horizon, agent.act_dim)
    optimizer = torch.optim.Adam(agent.amed_predictor.parameters(), lr=1e-2)

    first = amed_distill_loss(agent, obs_seq, action_seq, batch_size=4)
    for _ in range(20):
        optimizer.zero_grad()
        loss = amed_distill_loss(agent, obs_seq, action_seq, batch_size=4)
        loss.backward()
        optimizer.step()
    last = amed_distill_loss(agent, obs_seq, action_seq, batch_size=4)
    assert last < first


def test_get_action_amed_branch_matches_sampler():
    agent = _ScriptAgent(amed_steps=5)
    agent.eval()
    obs_seq = torch.randn(2, 1, agent.obs_dim)
    torch.manual_seed(0)
    a = agent.get_action(obs_seq)
    assert a.shape == (2, agent.pred_horizon, agent.act_dim)
    assert torch.isfinite(a).all()
