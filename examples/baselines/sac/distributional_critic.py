"""Distributional critic for the SAC baseline.

Implements the DSAC-T value distribution from Duan et al., "Distributional
Soft Actor-Critic with Three Refinements" (TPAMI 2025, arXiv:2310.05858):

  1. Expected value substituting -- the mean-related critic gradient is
     driven by the expected TD target y_q (eq. 18) instead of a random return
     sampled from the target distribution (eq. 13), removing the dominant
     source of critic-gradient randomness in DSACv1.
  2. Twin value-distribution learning -- a distributional analogue of clipped
     double Q-learning. The target network with the smaller mean (eq. 19)
     supplies both the critic target (eq. 20) and the actor target (eq. 22).
  3. Variance-based critic gradient adjustment -- the critic gradient is
     rescaled by a moving-average weight omega (eq. 24-26) and the random
     return is clipped to an adaptive boundary b = xi * E[sigma] (eq. 12, 23,
     27), so no per-task clipping boundary has to be hand-tuned.

The critic head parameterizes a diagonal Gaussian value distribution by its
mean and standard deviation. Because the mean is the expected value Q(s, a),
the scalar contract the SAC actor loss relies on is preserved, and the twin
critics are a drop-in replacement for ``sac.SoftQNetwork``.

The critic loss below is a surrogate whose autograd gradient is exactly the
gradient of eq. (26); see the comment on ``critic_loss``.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# xi in eq. (23): the clipping boundary is xi times the mean predicted
# standard deviation -- the three-sigma rule.
XI = 3.0
# eps in eq. (26): guards the variance-related gradient as sigma -> 0.
EPS = 0.1
# eps_omega in eq. (26): guards the mean-related gradient as omega -> 0.
EPS_OMEGA = 0.1


class DistributionalQNetwork(nn.Module):
    """Q network whose head outputs a Gaussian value distribution.

    Mirrors the ``SoftQNetwork`` layout in ``sac.py`` apart from the head, so
    the trained weights can be checkpointed under the same ``qf1``/``qf2``
    keys the rest of the baseline already reads and writes.
    """

    def __init__(self, env, hidden_dim: int = 256):
        super().__init__()
        obs_dim = int(np.array(env.single_observation_space.shape).prod())
        act_dim = int(np.prod(env.single_action_space.shape))
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(hidden_dim, 1)
        self.fc_log_std = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        x = self.net(torch.cat([x, a], 1))
        # softplus keeps sigma strictly positive.
        return self.fc_mean(x), F.softplus(self.fc_log_std(x))

    def expected_value(self, x, a):
        """Scalar Q(s, a): the mean of the value distribution."""
        return self(x, a)[0]


def critic_loss(qf, obs, actions, y_q, y_z, b, omega):
    """Surrogate loss for one value distribution, following eq. (26).

    Differentiating this reproduces the two gradient terms of eq. (26):

        dL/dQ     = -(omega + eps_omega) * (y_q - Q) / (sigma^2 + eps)
        dL/dsigma = ((y_z_clipped - Q)^2 - sigma^2) / (sigma^3 + eps)

    The first term is the expected-value-substituted mean gradient scaled by
    the variance-based weight omega; the second is the variance-related
    gradient under the adaptive clipping boundary b (eq. 12). ``sigma`` is
    detached in the first term and ``Q`` in the second so each head only
    receives its own gradient, and ``y_z_clipped`` is held constant so the
    clip bounds do not inject an extra gradient into the mean head.

    Both gradients match eq. (26) exactly as eps -> 0; the residual at the
    default eps is the paper's own guard against sigma -> 0.
    """
    mean, sigma = qf(obs, actions)
    mean = mean.view(-1)
    sigma = sigma.view(-1)
    denom = sigma.pow(2) + EPS
    # Eq. (12): clip the random return to Q(s,a) +/- b. Held constant here, as
    # in the analytic gradient of eq. (13), so the bounds do not inject an
    # extra gradient into the mean head.
    with torch.no_grad():
        y_z_clipped = mean + (y_z - mean).clamp(-b, b)
    mean_term = 0.5 * (omega + EPS_OMEGA) * (y_q - mean).pow(2) / denom.detach()
    std_term = (
        0.5 * (y_z_clipped - mean.detach()).pow(2) / denom + 0.5 * torch.log(denom)
    )
    return (mean_term + std_term).mean()


class DistributionalSACCritic:
    """Twin value distributions plus the three DSAC-T refinements.

    Owns the online/target critic pair, the adaptive clipping boundaries
    ``b1``/``b2`` and the gradient rescaling weights ``omega1``/``omega2``,
    and exposes the target/loss/target-update calls the SAC loop needs.
    """

    def __init__(self, env, tau: float, hidden_dim: int = 256):
        self.tau = tau
        self.qf1 = DistributionalQNetwork(env, hidden_dim)
        self.qf2 = DistributionalQNetwork(env, hidden_dim)
        self.qf1_target = DistributionalQNetwork(env, hidden_dim)
        self.qf2_target = DistributionalQNetwork(env, hidden_dim)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.b1 = torch.zeros(1)
        self.b2 = torch.zeros(1)
        self.omega1 = torch.ones(1)
        self.omega2 = torch.ones(1)

    def parameters(self):
        return list(self.qf1.parameters()) + list(self.qf2.parameters())

    def to(self, device):
        self.qf1 = self.qf1.to(device)
        self.qf2 = self.qf2.to(device)
        self.qf1_target = self.qf1_target.to(device)
        self.qf2_target = self.qf2_target.to(device)
        self.b1 = self.b1.to(device)
        self.b2 = self.b2.to(device)
        self.omega1 = self.omega1.to(device)
        self.omega2 = self.omega2.to(device)
        return self

    @torch.no_grad()
    def targets(
        self,
        next_obs,
        next_state_actions,
        next_state_log_pi,
        rewards,
        dones,
        gamma,
        alpha,
    ):
        """Expected-value and random-return targets, eq. (20).

        Both targets are built from the target network with the smaller mean
        (eq. 19), giving twin value-distribution learning. Returns
        ``(y_q_min, y_z_min)``.
        """
        q1_next, s1_next = self.qf1_target(next_obs, next_state_actions)
        q2_next, s2_next = self.qf2_target(next_obs, next_state_actions)
        q1_next, q2_next = q1_next.view(-1), q2_next.view(-1)
        # ibar (eq. 19): index of the target distribution with the smaller mean.
        ibar = (q1_next > q2_next).long()
        min_q_next = torch.minimum(q1_next, q2_next)
        min_s_next = torch.where(ibar == 0, s1_next.view(-1), s2_next.view(-1))
        bootstrap = (1 - dones.flatten()) * gamma
        soft_value = min_q_next - alpha * next_state_log_pi.view(-1)
        y_q_min = rewards.flatten() + bootstrap * soft_value
        # y_z_min draws a random return from that same winning distribution.
        y_z_min = y_q_min + bootstrap * torch.randn_like(min_q_next) * min_s_next
        return y_q_min, y_z_min

    def loss(self, obs, actions, y_q_min, y_z_min):
        """Sum of the two per-distribution losses (eq. 26), plus b/omega updates."""
        sigma1 = self.qf1(obs, actions)[1].view(-1)
        sigma2 = self.qf2(obs, actions)[1].view(-1)
        total = critic_loss(
            self.qf1, obs, actions, y_q_min, y_z_min, self.b1, self.omega1
        )
        total = total + critic_loss(
            self.qf2, obs, actions, y_q_min, y_z_min, self.b2, self.omega2
        )
        self._update_adaptive_terms(sigma1, sigma2)
        return total

    @torch.no_grad()
    def _update_adaptive_terms(self, sigma1, sigma2):
        """Slow-moving b (eq. 27) and omega (eq. 25), one per distribution."""
        for i, sigma in enumerate([sigma1, sigma2]):
            b_new = XI * sigma.mean()
            omega_new = sigma.pow(2).mean()
            if i == 0:
                self.b1 = self.tau * b_new + (1 - self.tau) * self.b1
                self.omega1 = self.tau * omega_new + (1 - self.tau) * self.omega1
            else:
                self.b2 = self.tau * b_new + (1 - self.tau) * self.b2
                self.omega2 = self.tau * omega_new + (1 - self.tau) * self.omega2

    def min_expected_value(self, obs, actions):
        """min over the twin distributions, for the actor loss (eq. 22)."""
        return torch.minimum(self.qf1(obs, actions)[0], self.qf2(obs, actions)[0])

    @torch.no_grad()
    def update_targets(self):
        """Polyak update of both target distributions."""
        for online, target in [
            (self.qf1, self.qf1_target),
            (self.qf2, self.qf2_target),
        ]:
            for param, target_param in zip(online.parameters(), target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
