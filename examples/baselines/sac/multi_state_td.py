"""Multi-state TD targets for the SAC baselines.

The standard SAC critic regresses onto a one-step target that bootstraps from a
single successor state. *Multi-State TD Target for Model-Free Reinforcement
Learning* (https://arxiv.org/abs/2405.16522) replaces that single bootstrap with
the truncated average over L targets, each looking a different number of steps
ahead (the soft-entropy variant, Eq. 14/15):

    y_t = (1/L) * sum_{l=1..L} [ sum_{i=0..l-1} gamma^i * r_{t+i}
                                 + gamma^l * V(s_{t+l}) ]

Averaging over horizons rather than committing to one keeps the variance
reduction of multi-step returns without picking a single bias point, which is
the paper's core result.

Adaptations (auxiliary machinery is deliberately not ported):

- The bootstrap values V(s_{t+l}) are the baseline's own double target critics
  and tanh-squashed actor, passed in by the caller as
  ``min(Q1_t, Q2_t) - alpha * log_pi`` over a sampled next action. The paper's
  value estimator is not re-implemented.
- Lookahead windows are read out of the on-policy ring replay buffer the script
  already fills, instead of a dedicated trajectory store. Episode boundaries
  reuse the stored ``stop_bootstrap`` flag the one-step target already uses.
- Setting ``horizon=1`` reproduces the unmodified one-step target exactly.
"""

from dataclasses import dataclass
from typing import Dict, Union

import torch

Obs = Union[torch.Tensor, Dict[str, torch.Tensor]]


@dataclass
class MultiStepSample:
    """A batch of L-step windows, field-compatible with ``ReplayBufferSample``.

    ``obs``/``actions`` describe the first transition (s_t, a_t) so the critic
    regression ``Q(s_t, a_t) -> y_t`` and the actor update are unchanged.
    ``rewards``/``dones`` cover the L transitions of the window, and
    ``next_obs`` stacks the L bootstrap states s_{t+1}..s_{t+L} so a single
    actor/critic pass evaluates every one of them.
    """

    obs: Obs
    next_obs: Obs
    actions: torch.Tensor
    rewards: torch.Tensor  # [L, batch]
    dones: torch.Tensor  # [L, batch], the stored stop_bootstrap flags


def _gather_obs(obs, index: torch.Tensor, env_inds: torch.Tensor) -> Obs:
    """Index ``obs`` (tensor or dict of tensors) by [batch, L] steps."""
    if isinstance(obs, dict):
        return {k: _gather_obs(v, index, env_inds) for k, v in obs.items()}
    return obs[index, env_inds.unsqueeze(1)]


def _concat_obs(parts, dim: int) -> Obs:
    if isinstance(parts[0], dict):
        return {k: torch.cat([p[k] for p in parts], dim=dim) for k in parts[0]}
    return torch.cat(parts, dim=dim)


def _squeeze_obs(obs: Obs) -> Obs:
    """Drop the length-1 step axis a single-step index leaves behind."""
    if isinstance(obs, dict):
        return {k: _squeeze_obs(v) for k, v in obs.items()}
    return obs.squeeze(1)


def _to_obs(obs: Obs, device: torch.device) -> Obs:
    """Move a tensor or dict-of-tensors observation onto the sampling device."""
    if isinstance(obs, dict):
        return {k: _to_obs(v, device) for k, v in obs.items()}
    return obs.to(device)


class MultiStateTD:
    """Samples contiguous L-step windows and builds their averaged TD target."""

    def __init__(self, horizon: int):
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.horizon = horizon

    def sample(self, buffer, batch_size: int) -> MultiStepSample:
        """Draw ``batch_size`` windows of L consecutive transitions per env."""
        n = buffer.per_env_buffer_size
        filled = n if buffer.full else buffer.pos
        # Clamp the horizon so a window never reaches past written data, and
        # bound the start so it never wraps the ring into stale/overwritten
        # transitions -- windows stay contiguous within one env's stream.
        horizon = max(1, min(self.horizon, filled))
        device = buffer.rewards.device
        starts = torch.randint(0, filled - horizon + 1, (batch_size,), device=device)
        env_inds = torch.randint(0, buffer.num_envs, (batch_size,), device=device)
        steps = starts.unsqueeze(1) + torch.arange(horizon, device=device)
        # [batch, L] -> the [L, batch] layout the target recursion expects
        rewards = buffer.rewards[steps, env_inds.unsqueeze(1)].transpose(0, 1)
        dones = buffer.dones[steps, env_inds.unsqueeze(1)].transpose(0, 1)

        sample_device = buffer.sample_device
        bootstraps = [
            _squeeze_obs(_gather_obs(buffer.next_obs, steps[:, l - 1 : l], env_inds))
            for l in range(1, horizon + 1)
        ]
        first_obs = _squeeze_obs(_gather_obs(buffer.obs, steps[:, :1], env_inds))
        return MultiStepSample(
            obs=_to_obs(first_obs, sample_device),
            next_obs=_to_obs(_concat_obs(bootstraps, dim=0), sample_device),
            actions=buffer.actions[steps[:, 0], env_inds].to(sample_device),
            rewards=rewards.to(sample_device),
            dones=dones.to(sample_device),
        )

    def td_target(
        self, sample: MultiStepSample, gamma: float, bootstrap_values: torch.Tensor
    ) -> torch.Tensor:
        """Average of the L truncated l-step targets.

        ``bootstrap_values`` holds V(s_{t+l}) = min(Q1_t, Q2_t) - alpha*log_pi
        for the sampled next action at every state in ``sample.next_obs``,
        flattened to [L * batch] in window order.
        """
        rewards, dones = sample.rewards, sample.dones
        num_steps, batch = rewards.shape
        values = bootstrap_values.view(num_steps, batch)

        # alive[l] is 1 while transition l is still inside the episode that
        # started the window: no stop_bootstrap flag was raised at steps 0..l-1.
        # A flag at step i keeps the realized reward r_{t+i} but cuts the
        # bootstrap across that transition and everything after it, which is the
        # same cut the one-step target makes with (1 - done).
        alive = torch.cat(
            [
                torch.ones(1, batch, device=rewards.device),
                (1 - dones).cumprod(dim=0)[:-1],
            ],
            dim=0,
        )
        bootstrappable = alive * (1 - dones)
        steps = torch.arange(num_steps, device=rewards.device)
        # returns[l] = sum_{i<l} gamma^i * r_{t+i} * alive[i]
        returns = (rewards * alive * gamma**steps.view(-1, 1)).cumsum(dim=0)
        # bootstrap[l] = gamma^(l+1) * bootstrappable[l] * V(s_{t+l+1})
        bootstrap = bootstrappable * values * gamma ** (steps + 1).view(-1, 1)
        return (returns + bootstrap).sum(dim=0) / num_steps
