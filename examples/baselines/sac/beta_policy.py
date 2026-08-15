"""Bounded-support Beta policy head for SAC via implicit reparameterization.

Adapted from "Soft Actor-Critic with Beta Policy via Implicit Reparameterization
Gradients" (arXiv:2409.04971; reference implementation github.com/lucadellalib/sac-beta,
Apache-2.0).

The stock SAC policy (``sac.py``) samples a Gaussian and squashes it through
tanh. The squash saturates at the action bounds, and the log-probability
correction ``-log(scale * (1 - y^2))`` it requires blows up as ``y`` approaches
+-1. The paper instead samples the pre-action from a Beta distribution whose
support *is* the action range, so no squashing (and no correction) is needed.

Beta samples admit no explicit reparameterization ``x = f(eps; phi)``, so
gradients flow with the *implicit* reparameterization trick: for the CDF
``F(x; phi)`` of a sampled ``x``,

    dx/dphi = -(dF/dphi) / (dF/dx)

``torch.distributions.Beta.rsample`` already implements exactly this gradient
(via ``torch._dirichlet_grad``), so the policy head can sample through the
stock distribution. ``beta_cdf`` below supplies the differentiable CDF that
the formula is built on; ``implicit_reparam_grads`` evaluates the formula
directly and is used by the tests to pin the mechanism down.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "beta_cdf",
    "implicit_reparam_grads",
    "BetaPolicyHead",
]

# concentrations are softplus(z) + this floor, which keeps the distribution
# numerically well behaved (below ~0.1 the CDF mass collapses onto the support ends)
MIN_CONCENTRATION = 0.1


def _gauss_legendre(n: int, dtype: torch.dtype, device: torch.device):
    """Gauss-Legendre nodes on [-1, 1] and matching weights."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return (
        torch.tensor(nodes, dtype=dtype, device=device),
        torch.tensor(weights, dtype=dtype, device=device),
    )


def beta_cdf(a: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Regularized incomplete beta function ``I_x(a, b)``, i.e. the CDF of
    ``Beta(a, b)`` at ``x``, differentiable with respect to all three inputs.

    PyTorch does not ship a differentiable Beta CDF (``Beta.cdf`` raises
    ``NotImplementedError``), so this evaluates

        I_x(a, b) = x^a / B(a, b) * int_0^1 u^(a-1) (1 - x u)^(b-1) du

    by Gauss-Legendre quadrature. For ``a >= 1`` the integrand is smooth on
    (0, 1) and a single composite rule suffices. For ``a < 1`` the factor
    ``u^(a-1)`` is singular at 0; there the change of variable ``u = v^(1/a)``
    absorbs it, leaving a smooth integrand (up to 256-node rule).

    Accuracy is ~1e-5 in float32 over concentrations in [0.05, 30] and x in
    (1e-3, 1 - 1e-3), verified against ``scipy.special.betainc`` in the tests.
    """
    nodes, weights = _gauss_legendre(64, x.dtype, x.device)
    x = x.clamp(1e-6, 1 - 1e-6)
    a_, b_, x_ = a.unsqueeze(-1), b.unsqueeze(-1), x.unsqueeze(-1)
    log_beta = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

    # branch 1 (a >= 1): composite Gauss-Legendre over [0, 1] with panels
    # refined toward 0 (harmless when there is no singularity)
    panel_edges = torch.cat(
        [
            torch.zeros(1, dtype=x.dtype, device=x.device),
            0.5 ** torch.arange(10, 0, -1, dtype=x.dtype, device=x.device),
            torch.ones(1, dtype=x.dtype, device=x.device),
        ]
    )
    mid = (panel_edges[:-1] + panel_edges[1:]) / 2
    half = (panel_edges[1:] - panel_edges[:-1]) / 2
    u = (mid + nodes[:, None] * half).unsqueeze(0)  # (1, nodes, panels)
    integrand = (1 - x_.unsqueeze(-1) * u).clamp(min=0).pow(
        b_.unsqueeze(-1) - 1
    ) * u.pow(a_.unsqueeze(-1) - 1)
    integral = ((weights[:, None] * half) * integrand).sum(dim=(-1, -2))
    cdf_a_ge_1 = torch.exp(a * torch.log(x) - log_beta) * integral

    # branch 2 (a < 1): substitution u = v^(1/a) cancels the u^(a-1) singularity
    nodes2, weights2 = _gauss_legendre(256, x.dtype, x.device)
    v = ((nodes2 + 1) / 2).unsqueeze(0)  # (1, nodes2)
    integrand2 = (1 - x_ * v.pow(1 / a_)).clamp(min=0).pow(b_ - 1)
    integral2 = (weights2 / 2 * integrand2).sum(dim=-1)
    cdf_a_lt_1 = (
        torch.exp(a * torch.log(x) - torch.log(a) - log_beta) * integral2
    )

    return torch.where(a >= 1, cdf_a_ge_1, cdf_a_lt_1)


def implicit_reparam_grads(
    a: torch.Tensor, b: torch.Tensor, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Implicit reparameterization gradients ``dx/da`` and ``dx/db`` of a
    Beta sample ``x`` at fixed CDF value (fixed underlying noise).

    Implements dx/dphi = -(dF/dphi) / (dF/dx) from the paper, with ``F`` the
    differentiable CDF from :func:`beta_cdf`.
    """
    ones = torch.ones_like(x)
    x = x.detach().requires_grad_(True)
    a = a.detach().requires_grad_(True)
    b = b.detach().requires_grad_(True)
    cdf = beta_cdf(a, b, x)
    d_cdf_d_x = torch.autograd.grad(cdf, x, ones, create_graph=True)[0]
    d_cdf_d_a = torch.autograd.grad(cdf, a, ones, create_graph=True)[0]
    d_cdf_d_b = torch.autograd.grad(cdf, b, ones, create_graph=True)[0]
    return -d_cdf_d_a / d_cdf_d_x, -d_cdf_d_b / d_cdf_d_x


class BetaPolicyHead(nn.Module):
    """Drop-in replacement for the squashed-Gaussian SAC actor head.

    Maps a 256-dim backbone feature to Beta concentrations and samples a
    pre-action in (0, 1) that is affinely rescaled onto the environment action
    space. Because the Beta support is bounded, the sampled action already
    respects the bounds and no tanh correction of the log-probability is
    needed. Sampling goes through ``torch.distributions.Beta.rsample``, which
    carries the implicit reparameterization gradients, so the SAC losses need
    no changes beyond swapping the head.
    """

    def __init__(self, action_space):
        super().__init__()
        self.action_dim = int(np.prod(action_space.shape))
        self.fc_alpha = nn.Linear(256, self.action_dim)
        self.fc_beta = nn.Linear(256, self.action_dim)
        # action rescaling from [0, 1] to [low, high]
        h, l = action_space.high, action_space.low
        self.register_buffer(
            "action_scale", torch.tensor(h - l, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor(l, dtype=torch.float32)
        )
        # will be saved in the state_dict, matching the Gaussian head

    def forward(self, x):
        alpha = F.softplus(self.fc_alpha(x)) + MIN_CONCENTRATION
        beta = F.softplus(self.fc_beta(x)) + MIN_CONCENTRATION
        return alpha, beta

    def get_action(self, x):
        """Sample an action; returns ``(action, log_prob, mean)`` with the same
        shapes as the Gaussian head in ``sac.py``."""
        alpha, beta = self(x)
        dist = torch.distributions.Beta(alpha, beta)
        u = dist.rsample()  # implicit reparameterization gradients flow here
        action = u * self.action_scale + self.action_bias
        log_prob = dist.log_prob(u).sum(dim=1, keepdim=True)
        mean = dist.mean * self.action_scale + self.action_bias
        return action, log_prob, mean

    def get_eval_action(self, x):
        alpha, beta = self(x)
        dist = torch.distributions.Beta(alpha, beta)
        return dist.mean * self.action_scale + self.action_bias
