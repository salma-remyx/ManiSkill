"""Few-step ODE sampling for the Diffusion Policy baseline.

Adapted from "Fast ODE-based Sampling for Diffusion Models in Around 5
Steps" (https://arxiv.org/abs/2312.00094, AMED-Solver).

The paper's observation is that a sampling trajectory almost lies in a
two-dimensional subspace, so instead of a higher-order solver it learns,
per step, (a) an *intermediate time* ``s_n`` at which to evaluate the
model and (b) a *scale* ``c_n`` on the velocity, recovering the mean-value
integral exactly rather than approximating it:

    x_{t_n} ~= x_{t_{n+1}} + c_n * (t_n - t_{n+1}) * eps(x_{s_n}, s_n)

with ``x_{s_n}`` reached by a Euler step. Each evaluation is shared
between adjacent intervals, so ``N`` function evaluations integrate
``N - 1`` intervals -- 5 NFE instead of the baseline's 100.

Target-native substitutions (Mode 2):

- The paper's U-Net bottleneck feature is not exposed by
  ``ConditionalUnet1D``, so the predictor is conditioned on a mean-pooled
  summary of the model's own noise prediction at the current point.
- The paper's FID-based evaluation objective is dropped; the predictor is
  trained by regressing the few-step trajectory onto the full DDPM
  teacher's sample, which is the part that matters for policy inference.

The DDPM schedule is handled by carrying the trajectory in the sample
coordinates the policy was trained in (``x = sqrt(abar_t) * y``) and
stepping in the unit-data-scale coordinates ``y = x / sqrt(abar_t)``,
``sigma_t = sqrt(1 - abar_t) / sqrt(abar_t)``, where the probability-flow
ODE is ``dy/dsigma = eps``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Half sine / half cosine positional embedding, matching the policy's."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        freq = torch.exp(
            torch.arange(half_dim, device=x.device, dtype=torch.float32)
            * (-math.log(10000.0) / (half_dim - 1))
        )
        emb = x.float().unsqueeze(-1) * freq  # (..., half_dim)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class AmedPredictor(nn.Module):
    """Predicts the intermediate time and velocity scale for one ODE step.

    Follows the paper's ``AmedPredictor``: a small MLP over sinusoidal
    embeddings of the sigma pair ``(sigma_n, sigma_{n+1})`` plus a
    conditioning scalar, emitting the interpolation ratio ``r`` (sigmoid)
    and the scale offset applied to the velocity.

    ``scheduler`` is kept only for its alpha-bar schedule; any DDPM-family
    scheduler exposing ``config.num_train_timesteps`` and ``alphas_cumprod``
    works.
    """

    def __init__(self, scheduler, embed_dim: int = 64, hidden_dim: int = 512):
        super().__init__()
        self.scheduler = scheduler
        self.pos_emb = SinusoidalPosEmb(embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.cond_proj = nn.Linear(1, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 2)
        # zero-init so an untrained predictor reproduces plain Euler (r=1/2, c=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, sigma_pair, cond):
        """Return ``(r, c)`` for a batch of sigma pairs.

        Args:
            sigma_pair: (B, 2) tensor of ``(sigma_n, sigma_{n+1})``.
            cond: (B, 1) conditioning signal summarizing the current point.

        Returns:
            (B, 2) tensor whose first column is the interpolation ratio in
            ``(0, 1)`` and whose second column is the velocity scale.
        """
        t_emb = self.time_mlp(self.pos_emb(sigma_pair).flatten(start_dim=1))
        out = self.out_proj(t_emb + self.cond_proj(cond))
        r = torch.sigmoid(out[:, 0])
        c = 1.0 + out[:, 1]
        return torch.stack([r, c], dim=1)


def _sigma_table(scheduler):
    """Ascending ``sigma(t)`` table for each training timestep."""
    alphas_cumprod = scheduler.alphas_cumprod.to(torch.float32)
    return torch.sqrt(1 - alphas_cumprod) / torch.sqrt(alphas_cumprod)


def amed_sigmas(scheduler, num_inference_steps: int, device):
    """Timestep grid for few-step sampling, denser at high noise.

    Spacings are geometric in ``sigma(t) = sqrt(1-abar_t)/sqrt(abar_t)``,
    the natural ODE coordinate for a DDPM schedule. Returns the grid
    points' training-timestep indices, length ``num_inference_steps``,
    strictly decreasing from the noisiest usable timestep to 0.
    """
    sigmas = _sigma_table(scheduler)
    # squaredcos_cap_v2's final training steps are extreme outliers in sigma
    # space (abar ~ 1e-8); trim that tail so the few-step grid does not spend
    # an evaluation on a near-identical noise level. The baseline's own init
    # x ~ randn already sits at that noise level, so nothing is lost.
    while sigmas.numel() > 2 and sigmas[-1] > 10 * sigmas[-2]:
        sigmas = sigmas[:-1]
    n_intervals = num_inference_steps - 1
    sigma_max = sigmas[-1]
    i = torch.arange(num_inference_steps, device=device, dtype=torch.float32)
    grid = sigma_max ** (1 - i / n_intervals)
    grid[-1] = 0.0
    # map each grid sigma back to the nearest training timestep
    t_idx = torch.argmin((sigmas.to(device)[:, None] - grid[None, :]).abs(), dim=0)
    # pin the endpoints: noisiest kept timestep, and pure data (t = 0)
    t_idx[0] = sigmas.numel() - 1
    t_idx[-1] = 0
    return grid, t_idx


def _sigma_min(scheduler):
    """Smallest sigma on the schedule, used to floor intermediate sigmas."""
    return float(_sigma_table(scheduler)[0])


def amed_sample(
    noise_pred_net,
    scheduler,
    amed_predictor,
    shape,
    global_cond,
    num_inference_steps: int = 5,
    generator=None,
):
    """Run the few-step AMED ODE sampler and return the denoised action chunk.

    The caller owns the gradient mode: ``Agent.get_action`` wraps inference
    in ``torch.no_grad()``, while ``amed_distill_loss`` relies on gradients
    flowing through to the predictor.

    Args:
        noise_pred_net: the policy's noise prediction network, called as
            ``noise_pred_net(sample=..., timestep=..., global_cond=...)``.
        scheduler: the DDPM scheduler the policy was trained with; only its
            alpha-bar schedule is read, never its ``step()``.
        amed_predictor: a trained (or freshly initialized) AmedPredictor.
        shape: shape of the action sequence to sample, ``(B, T, act_dim)``.
        global_cond: (B, cond_dim) FiLM conditioning from the observations.
        num_inference_steps: total number of function evaluations (paper: ~5).
        generator: optional torch generator for reproducible initial noise.

    Returns:
        Tensor of the sampled action sequence, shape ``shape``.
    """
    device = global_cond.device
    sigmas, t_idx = amed_sigmas(scheduler, num_inference_steps, device)
    alphas_cumprod = scheduler.alphas_cumprod.to(device).to(torch.float32)

    # sqrt(abar) at each grid point, for mapping between the sample
    # coordinates the policy was trained in and the ODE coordinates
    a_grid = torch.sqrt(alphas_cumprod[t_idx])

    # initial sample at the noisiest grid timestep, in the same form the
    # policy was trained on: x = sqrt(abar) * x0 + sqrt(1-abar) * eps with
    # x0 unknown, so x = sqrt(abar_t) * (x0 + sigma_t * eps) ~ scaled noise
    y = torch.randn(shape, device=device, generator=generator)
    x = a_grid[0] * y
    t = int(t_idx[0])
    eps = noise_pred_net(sample=x, timestep=t, global_cond=global_cond)

    for n in range(num_inference_steps - 1):
        cond = eps.flatten(1).abs().mean(dim=1, keepdim=True)
        sigma_pair = torch.stack(
            [sigmas[n].expand(shape[0]), sigmas[n + 1].expand(shape[0])], dim=1
        )
        d = amed_predictor(sigma_pair, cond)
        r, c = d[:, 0], d[:, 1]
        # intermediate sigma, geometric interpolation between the endpoints,
        # floored at the schedule's smallest sigma so the final interval (whose
        # far endpoint is sigma = 0) still evaluates at a real noise level
        s_raw = sigmas[n].pow(r) * sigmas[n + 1].clamp_min(_sigma_min(scheduler)).pow(1 - r)
        s_n = s_raw.reshape(-1, 1, 1)
        # keep the evaluation strictly between the two grid points
        t_s = int(
            _nearest_timestep(
                scheduler,
                s_n.mean(),
                device,
                t_lo=int(t_idx[n + 1]) + 1,
                t_hi=int(t_idx[n]) - 1,
            )
        )

        # x0 prediction at the current point, then a Euler half-step in the
        # y = x / sqrt(abar) coordinate where the ODE is dy/dsigma = eps
        y = x / a_grid[n]
        y_s = y + (s_n - sigmas[n]) * eps
        # evaluate the model at the learned intermediate time (in x coords)
        x_s = torch.sqrt(alphas_cumprod[t_s]) * y_s
        eps = noise_pred_net(sample=x_s, timestep=t_s, global_cond=global_cond)
        # scaled step out of the interval; the evaluation just made is reused
        # as the next interval's incoming direction, keeping NFE at N
        y = y + c.reshape(-1, 1, 1) * (sigmas[n + 1] - sigmas[n]) * eps
        x = a_grid[n + 1] * y

    return x


def _nearest_timestep(scheduler, sigma, device, t_lo=None, t_hi=None):
    """Training-timestep index whose sigma is closest to ``sigma``.

    ``t_lo``/``t_hi`` bound the result to the open interval between two grid
    points, so an intermediate evaluation never lands on a grid endpoint
    (where it would duplicate a neighbouring evaluation, or in the final
    interval collapse onto the pure-data point the model never saw).
    """
    sigmas = _sigma_table(scheduler).to(device)
    if sigma.ndim != 0:
        # per-sample sigmas: evaluate at the batch's mean so one NFE serves all
        sigma = sigma.mean()
    t = torch.argmin((sigmas - sigma).abs())
    if t_lo is not None and t < t_lo:
        return t_lo
    if t_hi is not None and t > t_hi:
        return t_hi
    return t


def ddpm_teacher_sample(agent, obs_cond, shape):
    """Run the policy's existing full DDPM loop (the distillation teacher)."""
    x = torch.randn(shape, device=obs_cond.device)
    for k in agent.noise_scheduler.timesteps:
        noise_pred = agent.noise_pred_net(sample=x, timestep=k, global_cond=obs_cond)
        x = agent.noise_scheduler.step(
            model_output=noise_pred, timestep=k, sample=x
        ).prev_sample
    return x


def amed_distill_loss(agent, obs_seq, action_seq, batch_size: int = 8):
    """Train the AMED predictor by regressing onto the DDPM teacher sample.

    This is the paper's distillation objective reduced to its core: the
    few-step AMED trajectory is fit to the full DDPM teacher's output under
    the same conditioning. Only the predictor receives gradients; the noise
    prediction network stays frozen.

    Args:
        agent: the diffusion policy Agent (provides the frozen networks).
        obs_seq: (B, obs_horizon, obs_dim) observation batch.
        action_seq: (B, pred_horizon, act_dim) action batch; only its shape
            and batch size are used.
        batch_size: how many samples of the batch to distill on per step.
    """
    obs_cond = obs_seq[:batch_size].flatten(start_dim=1)
    shape = action_seq[:batch_size].shape
    with torch.no_grad():
        teacher = ddpm_teacher_sample(agent, obs_cond, shape)
    student = amed_sample(
        agent.noise_pred_net,
        agent.noise_scheduler,
        agent.amed_predictor,
        shape,
        obs_cond,
        num_inference_steps=agent.amed_steps,
    )
    return nn.functional.mse_loss(student, teacher)
