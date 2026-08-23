"""Flow-map (consistency-model) action generation for the diffusion policy baseline.

Adapted from "How to build a consistency model: Learning flow maps via
self-distillation" (Boffi, Albergo & Vanden-Eijnden, arXiv:2505.18825).

Instead of learning a one-step denoiser and paying a 100-step DDPM solve per
action chunk at inference, this learns the *solution operator* of that solve:
a two-time flow map ``X_{s,t}(x) = x + (t - s) v_{s,t}(x)`` that jumps
straight from noise (``s = 0``) to actions (``t = 1``). The paper's
Lagrangian self-distillation (LSD) loss trains it directly, with no
pre-trained teacher and no bootstrapping off small steps:

    |d/dt X_{s,t}(I_s) - v_{t,t}(sg(X_{s,t}(I_s)))|^2
      = |v_{s,t}(I_s) - v_{t,t}(sg(X_{s,t}(I_s)))|^2

where ``I_s = (1-s) x_0 + s a_1`` is the linear (flow-matching) interpolant
between noise and demonstrated actions. The boundary condition ``X_{s,s} = x``
holds exactly by construction, and the diagonal ``v_{t,t}`` is pinned to the
true velocity by the ordinary flow-matching loss on ``t = s``. Information
flows from the diagonal (which sees the demonstrations) out to the
off-diagonal, which is what stops the map collapsing to a constant.

Port notes (target-native substitutions): the paper's bespoke two-time
weighting network and EMA-teacher variants are dropped; (s, t) pairs come
from the paper's mixture ``eta * delta_diag + (1 - eta) * upper_triangle``
and the diagonal share doubles as the flow-matching term, so the whole
objective is a single weighted MSE at repo learning-rate scale. Evaluation
belongs to the existing baseline harness.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FlowMapConfig:
    """Hyperparameters for Lagrangian self-distilled flow-map training."""

    num_train_timesteps: int = 100
    """number of discrete time levels v_{s,t} is evaluated on"""
    eta: float = 0.75
    """probability of sampling the diagonal s == t, which trains v_{t,t} with
    the flow-matching target; the remaining samples land on s < t and carry the
    self-distillation target (paper default is 0.75)"""
    time_condition_bias: float = 1.0
    """offset added to (s, t) before rounding to integer timesteps, so that
    e.g. t == 0 still reaches a non-degenerate timestep embedding"""


class FlowMapPolicy(nn.Module):
    """Two-time velocity network wrapping the baseline's noise prediction net.

    The wrapped net keeps its usual signature ``(sample, timestep, global_cond)``
    and is called with *two* timesteps packed into one by summing
    ``timestep * time_levels + s``. Packing (rather than a second input branch)
    means an unmodified ``ConditionalUnet1D`` can represent the map, and a
    checkpoint trained the standard way remains loadable as the ``s == t``
    diagonal.
    """

    def __init__(self, noise_pred_net: nn.Module, config: FlowMapConfig):
        super().__init__()
        self.net = noise_pred_net
        self.config = config
        self.time_levels = config.num_train_timesteps

    def velocity(self, sample, s, t, global_cond=None):
        """v_{s,t}(x): (B, T, act_dim), s and t are (B,) integer time levels."""
        packed_timesteps = t * self.time_levels + s
        return self.net(sample, packed_timesteps, global_cond=global_cond)

    def compute_lagrangian_self_distillation_loss(
        self, action_seq, global_cond, generator=None
    ):
        """L_SD = L_b + L_D, the paper's Lagrangian self-distillation objective.

        Each batch element draws ``(s, t)`` from the mixture
        ``eta * diagonal + (1 - eta) * upper triangle`` and contributes the
        residual for the region it landed in, so a single mean realizes
        ``eta * L_b + (1 - eta) * L_D``:

        * diagonal (``s == t``): the flow-matching target
          ``|v_{t,t}(I_t) - dI_t/dt|^2`` with ``dI_t/dt = a_1 - x_0``. This is
          the only term that sees the demonstrations, so it is what keeps the
          map from collapsing to a constant.
        * off-diagonal (``s < t``): because
          ``X_{s,t}(x) = x + (t - s) v_{s,t}(x)``, the Lagrangian
          characterization ``d/dt X_{s,t} = v_{t,t}(X_{s,t})`` reduces to
          ``|v_{s,t}(I_s) - v_{t,t}(sg(X_{s,t}(I_s)))|^2``. The semigradient
          keeps information flowing from the diagonal out to the
          off-diagonal, not the reverse.
        """
        B = action_seq.shape[0]
        device = action_seq.device
        cfg = self.config

        s, t, on_diagonal = _sample_time_pair(
            B, cfg.eta, self.time_levels, device, generator
        )

        noise = torch.randn(action_seq.shape, device=device, generator=generator)
        # I_u = (1 - u) x_0 + u a_1 is the linear interpolant between noise and
        # demonstrated actions, so dI_u/du = a_1 - x_0 at every u.
        velocity_target = action_seq - noise
        on_diagonal = on_diagonal.reshape(-1, 1, 1)

        def interpolate(u):
            return (1 - u).reshape(-1, 1, 1) * noise + u.reshape(-1, 1, 1) * action_seq

        # On the diagonal s == t, so I_s is I_t and this covers both branches.
        x_s = interpolate(_to_unit(s, self.time_levels, cfg.time_condition_bias))

        v_st = self.velocity(x_s, s, t, global_cond=global_cond)
        with torch.no_grad():
            x_t = x_s + _delta(t, s, self.time_levels) * v_st
            v_tt = self.velocity(x_t, t, t, global_cond=global_cond)
        target = torch.where(on_diagonal, velocity_target, v_tt)
        return F.mse_loss(v_st, target)

    def generate_actions(self, obs_cond, pred_horizon, act_dim):
        """One-step action chunk: X_{0,1}(x_0) = x_0 + v_{0,1}(x_0).

        A single network evaluation replaces the baseline's 100-step DDPM loop.
        """
        device = obs_cond.device
        x_0 = torch.randn(
            (obs_cond.shape[0], pred_horizon, act_dim), device=device
        )
        v_01 = self.velocity(x_0, 0, self.time_levels - 1, global_cond=obs_cond)
        return x_0 + v_01


def _sample_time_pair(B, eta, time_levels, device, generator=None):
    """Draw (s, t) from eta * diagonal + (1 - eta) * upper triangle s < t.

    Returns ``(s, t, on_diagonal)``; ``s`` and ``t`` are integer time levels.
    """
    on_diagonal = torch.rand((B,), device=device, generator=generator) < eta
    diag = torch.randint(0, time_levels, (B,), device=device, generator=generator)
    # Sample s uniformly, then rejection-sample t from the levels strictly above
    # it, which keeps the off-diagonal pairs marginally uniform on the upper
    # triangle.
    s = torch.randint(0, time_levels, (B,), device=device, generator=generator)
    t = torch.randint(0, time_levels, (B,), device=device, generator=generator)
    too_low = t <= s
    for _ in range(32):
        if not torch.any(too_low):
            break
        resample = torch.randint(
            0, time_levels, (B,), device=device, generator=generator
        )
        t = torch.where(too_low, resample, t)
        too_low = t <= s
    # Levels where no strictly-greater resample succeeded fall back to s == t,
    # which is still a valid draw from the diagonal component of the mixture.
    t = torch.where(too_low, s, t)
    return (
        torch.where(on_diagonal, diag, s),
        torch.where(on_diagonal, diag, t),
        on_diagonal,
    )


def _to_unit(level, time_levels, bias):
    """Map an integer time level to [0, 1], avoiding an exactly-zero endpoint."""
    return (level.float() + bias) / (time_levels - 1 + bias)


def _delta(t, s, time_levels):
    """(t - s) in continuous units, as a (B, 1, 1) tensor for broadcasting."""
    return (t - s).float().reshape(-1, 1, 1) / (time_levels - 1)
