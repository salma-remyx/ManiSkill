"""Score-regularized deterministic policy distilled from a diffusion policy.

Diffusion policies need tens to hundreds of denoising steps per control step,
which is slow at inference. Following *Score Regularized Policy Optimization
through Diffusion Behavior* (SRPO, https://arxiv.org/abs/2310.07297), this
module extracts a **fast deterministic policy** from a pretrained diffusion
policy by regressing the policy's action onto the diffusion behavior model's
noise prediction ("score regularization"), optionally combined with critic
guidance. The result is a single forward pass per control step.

The paper's objective for the deterministic policy ``a = pi(s)`` is

    L(s) = wt * <epsilon(pi_t(s), t, s), a> - beta * <grad_a Q(s, a)|_{a=pi(s)}, a>

where ``pi_t(s)`` is the policy action perturbed by the diffusion forward
process at a random time ``t``, ``epsilon`` is the **frozen** pretrained
diffusion behavior model's noise prediction, ``wt`` is a time-dependent
weighting (``std**2``, ``1``, or ``alpha/std``), and the second term steers
the policy up the critic's gradient (SRPO drops it with ``beta = 0`` to run
pure imitation, which is what ManiSkill's demo-only pipeline uses).

This is an adapted port (Mode 2):

- the paper's VP-SDE score network over flat actions is replaced by ManiSkill's
  trained ``ConditionalUnet1D`` epsilon-prediction network over action chunks,
  kept frozen exactly as SRPO keeps its behavior model frozen;
- the paper's learned critic guidance (``-beta * <grad_a Q, a>``) is optional
  here and defaults to off, since ManiSkill's demonstration datasets are
  reward-annotated but the imitation baselines are trained without critics.
  When a critic-like scoring function is supplied, the same autograd-gradient
  guidance is applied.

Adapted from Huayu Chen et al., "Score Regularized Policy Optimization
through Diffusion Behavior" (ICLR 2024), https://arxiv.org/abs/2310.07297v3.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn


def vp_marginal_prob_std(
    t: torch.Tensor, beta_0: float = 0.1, beta_1: float = 20.0
):
    """Mean/std of the VP-SDE forward kernel ``p_0t(x_t | x_0)``.

    Matches SRPO's schedule: ``alpha_t = exp(-0.25 t^2 (beta_1 - beta_0)
    - 0.5 t beta_0)``, ``std_t = sqrt(1 - alpha_t^2)``.

    Args:
        t: (B,) continuous times in [0, 1].
        beta_0: smallest VP beta.
        beta_1: largest VP beta.

    Returns:
        ``(alpha, std)`` each (B,).
    """
    log_mean_coeff = -0.25 * t**2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    alpha = torch.exp(log_mean_coeff)
    std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
    return alpha, std


def vp_to_ddpm_timestep(t: torch.Tensor, num_train_timesteps: int) -> torch.Tensor:
    """Map SRPO's continuous VP time in [0, 1] to a DDPM training timestep.

    Both parameterizations share the endpoint semantics (``t -> 1`` is the
    noisiest end), so a linear map onto the DDPM index range is used to query
    a network trained with a discrete DDPM schedule.
    """
    return (t.clamp(0, 1) * (num_train_timesteps - 1)).round().long()


def score_weight(t: torch.Tensor, mode: str = "stable") -> torch.Tensor:
    """SRPO's time-dependent weighting ``wt`` for the score-alignment term.

    Args:
        t: (B,) continuous times in [0, 1].
        mode: one of ``"vds"`` (``std**2``), ``"stable"`` (``1``) or
            ``"score"`` (``alpha / std``).

    Returns:
        (B,) weights.
    """
    alpha, std = vp_marginal_prob_std(t)
    if mode == "vds":
        return std**2
    if mode == "stable":
        return torch.ones_like(alpha)
    if mode == "score":
        return alpha / std
    raise ValueError(f"unknown score weight mode: {mode}")


class ScoreRegularizedPolicy(nn.Module):
    """Deterministic policy distilled from a frozen diffusion behavior policy.

    The policy network maps an observation condition directly to an action
    chunk (a plain MLP over the flattened observation stack, matching the
    baseline's FiLM conditioning layout). It is trained with SRPO's
    score-regularized objective so that its outputs lie where the pretrained
    diffusion policy would denoise to, while inference costs one forward pass
    instead of ``num_diffusion_iters`` denoising steps.
    """

    def __init__(
        self,
        noise_pred_net: nn.Module,
        scheduler,
        act_dim: int,
        pred_horizon: int,
        global_cond_dim: int,
        hidden_dim: int = 512,
        weight_mode: str = "stable",
        beta: float = 0.0,
        t_min: float = 0.02,
        t_max: float = 0.98,
    ):
        """
        Args:
            noise_pred_net: the **pretrained** epsilon-prediction network of a
                diffusion policy (e.g. the baseline's ``ConditionalUnet1D``);
                frozen during policy training.
            scheduler: the diffusion policy's training-time scheduler; only
                ``alphas_cumprod`` is read, to map continuous VP times to the
                network's discrete timestep vocabulary.
            act_dim: dimensionality of one action.
            pred_horizon: number of actions per chunk.
            global_cond_dim: size of the flattened observation conditioning.
            hidden_dim: hidden width of the policy MLP.
            weight_mode: SRPO score weighting mode (see :func:`score_weight`).
            beta: critic-guidance coefficient; ``0`` disables guidance and
                gives pure score-regularized imitation.
            t_min/t_max: range of sampled forward-process times.
        """
        super().__init__()
        self.diffusion_behavior = noise_pred_net
        for p in self.diffusion_behavior.parameters():
            p.requires_grad_(False)
        self.num_train_timesteps = len(scheduler.alphas_cumprod)
        self.weight_mode = weight_mode
        self.beta = beta
        self.t_min = t_min
        self.t_max = t_max

        self.policy = nn.Sequential(
            nn.Linear(global_cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim * pred_horizon),
            nn.Tanh(),
        )
        self.act_dim = act_dim
        self.pred_horizon = pred_horizon

    def forward(self, obs_cond: torch.Tensor) -> torch.Tensor:
        """(B, global_cond_dim) -> (B, pred_horizon, act_dim) action chunk."""
        a = self.policy(obs_cond)
        return a.reshape(a.shape[0], self.pred_horizon, self.act_dim)

    def loss(
        self,
        obs_cond: torch.Tensor,
        critic: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """SRPO's score-regularized policy loss.

        Args:
            obs_cond: (B, global_cond_dim) flattened observation conditioning.
            critic: optional ``Q(obs_cond, actions) -> (B,)`` scorer. When
                given (and ``beta > 0``), its action-gradient steers the
                policy like the paper's Q-guidance term.

        Returns:
            Scalar loss to minimize.
        """
        a = self(obs_cond)

        t = torch.rand(a.shape[0], device=a.device) * (self.t_max - self.t_min) + self.t_min
        alpha, std = vp_marginal_prob_std(t)
        # broadcast (B,) schedule terms over the (B, pred_horizon, act_dim) chunk
        alpha, std = alpha[:, None, None], std[:, None, None]
        z = torch.randn_like(a)
        perturbed_a = a * alpha + z * std

        epsilon = self.diffusion_behavior(
            sample=perturbed_a,
            timestep=vp_to_ddpm_timestep(t, self.num_train_timesteps),
            global_cond=obs_cond,
        ).detach()

        wt = score_weight(t, self.weight_mode)
        loss = (epsilon * a).sum(-1) * wt[:, None]

        if critic is not None and self.beta != 0.0:
            detach_a = a.detach().requires_grad_(True)
            q = critic(obs_cond, detach_a).sum()
            guidance = torch.autograd.grad(q, detach_a)[0].detach()
            guidance = guidance / guidance.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            loss = loss - (guidance * a).sum(-1) * self.beta

        return loss.mean()

    @torch.no_grad()
    def get_action(self, obs_cond: torch.Tensor, act_horizon: int, obs_horizon: int):
        """One-forward-pass action chunk, mirroring the diffusion policy's API.

        Args:
            obs_cond: (B, global_cond_dim) flattened observation conditioning.
            act_horizon: number of actions to execute.
            obs_horizon: observation horizon, for the same offset the diffusion
                policy uses when slicing its predicted chunk.

        Returns:
            (B, act_horizon, act_dim) actions in [-1, 1].
        """
        a = self(obs_cond)
        start = obs_horizon - 1
        return a[:, start : start + act_horizon]
