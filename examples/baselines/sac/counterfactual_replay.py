"""Counterfactual experience augmentation for the state-based SAC replay buffer.

Off-policy replay stores exactly one outcome per visited state-action pair, so
the update never sees what the environment *would* have done under the actions
that were not taken. This module relaxes that property: it samples unexecuted
actions, predicts their outcomes with a residual dynamics model, grounds the
synthetic rewards in real data, and writes the finished transitions back into
the same replay buffer the update samples from.

Adapted from "Counterfactual Experience Augmented Off-Policy Reinforcement
Learning" (Lee, Gong & Deng, 2025, arXiv:2503.13842). The paper's data-path
contract is kept intact (real transitions in, same-shape counterfactual
transitions out, added to the pool before the update) and so is its reward
grounding: rewards are copied from the closest real next state instead of
being predicted, which the paper identifies as the failure mode of model-based
alternatives such as MBPO. Auxiliary components are substituted with
parameter-free equivalents fitted in closed form:

- the state-transition (C)VAE over ``next_state - state`` conditioned on the
  action is replaced by a ridge-regressed linear residual model plus matched
  Gaussian noise; the regression carries the conditional mean and the noise
  stands in for the latent randomness that captures non-stationarity.
- the paper's grid-based KDE entropy estimate is replaced by a Monte-Carlo
  cross-entropy surrogate of the same KDE, since grids are impractical for the
  7-D action vectors of the joint-position control modes.
- the paper's absolute ``threshold_ratio = 0.1`` match threshold is
  recalibrated to be relative to the reference set's median nearest-neighbour
  distance, so it transfers across observation scales and dimensions.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class CounterfactualBatch:
    """A block of synthetic transitions, shaped like a replay buffer sample."""

    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


def _scott_bandwidth(history: torch.Tensor, min_bandwidth: float) -> float:
    """Scott's rule bandwidth for a Gaussian KDE over replayed actions.

    Floored so that a degenerate history (e.g. the same action repeated early
    in training) does not collapse the kernel and blow up the ascent gradient.
    """
    n, dim = history.shape
    std = history.std(dim=0)
    std = torch.where(std > 0, std, torch.full_like(std, 1.0 / math.sqrt(n)))
    bandwidth = float((n ** (-1.0 / (dim + 4))) * std.mean())
    return max(bandwidth, min_bandwidth)


class CounterfactualAugmenter:
    """Generates counterfactual transitions for a replay buffer of real ones.

    ``step`` is meant to be called every few gradient updates, before the
    update samples its batch: it fits the residual dynamics model on a fresh
    replay sample, proposes unexecuted actions, keeps only the predictions
    that stay groundable in real data, and writes the survivors into the
    buffer. The distance filter doubles as the paper's annealing schedule:
    while the model is poor (or the action coverage narrow) few predictions
    find a close real neighbour, so the supplementation rate grows on its own
    as real experience accumulates.
    """

    def __init__(
        self,
        env,
        device,
        k_actions: int = 4,
        noise_scale: float = 1.0,
        ridge: float = 1e-2,
        match_threshold: float = 1.0,
        fit_size: int = 2048,
        kde_steps: int = 15,
        kde_lr: float = 0.3,
    ):
        self.device = torch.device(device)
        self.obs_shape = tuple(env.single_observation_space.shape)
        self.action_shape = tuple(env.single_action_space.shape)
        self.obs_dim = int(np.array(self.obs_shape).prod())
        self.action_dim = int(np.array(self.action_shape).prod())
        self.k_actions = k_actions
        self.noise_scale = noise_scale
        self.ridge = ridge
        self.match_threshold = match_threshold
        self.fit_size = fit_size
        self.kde_steps = kde_steps
        self.kde_lr = kde_lr
        self.action_low = torch.as_tensor(
            env.single_action_space.low, dtype=torch.float32, device=self.device
        ).flatten()
        self.action_high = torch.as_tensor(
            env.single_action_space.high, dtype=torch.float32, device=self.device
        ).flatten()
        # Residual dynamics: next_obs - obs ~= [action, 1] @ weights.
        self.weights = torch.zeros(self.action_dim + 1, self.obs_dim, device=self.device)
        self.residual_std = torch.zeros(self.obs_dim, device=self.device)

    def fit(self, obs: torch.Tensor, actions: torch.Tensor, next_obs: torch.Tensor):
        """Fit ``next_obs - obs ~ [action, 1]`` in closed form on real transitions.

        CEA's state-transition autoencoder models p(delta | action) without
        conditioning on the state; the ridge solution carries that same
        conditional mean, and the matched residual std plays the role of its
        latent randomness.
        """
        with torch.no_grad():
            delta = next_obs - obs
            design = torch.cat([actions, torch.ones_like(actions[:, :1])], dim=1)
            gram = design.T @ design
            gram += self.ridge * torch.eye(
                gram.shape[0], device=gram.device, dtype=gram.dtype
            )
            self.weights = torch.linalg.solve(gram, design.T @ delta)
            residual = delta - design @ self.weights
            self.residual_std = torch.nan_to_num(residual.std(dim=0))

    def explore_actions(self, actions: torch.Tensor, num: int) -> torch.Tensor:
        """Propose ``num`` unexecuted actions by maximizing KDE entropy.

        A Gaussian KDE is fitted over the replayed actions and the proposals
        are updated by gradient ascent on the entropy of the density over
        (replayed + proposed) actions, so they migrate towards underexplored
        regions of the action space. The entropy of the union density is
        estimated by the cross-entropy of the proposals under their own
        leave-one-out KDE, the Monte-Carlo stand-in for the paper's grid
        integration.
        """
        history = actions[: self.fit_size]
        n_hist = history.shape[0]
        bandwidth = _scott_bandwidth(
            history, 0.05 * float((self.action_high - self.action_low).mean())
        )
        init = history[torch.randint(0, n_hist, size=(num,), device=history.device)]
        x = (init + 0.05 * torch.randn_like(init)).clamp(
            self.action_low, self.action_high
        )
        x = x.detach().requires_grad_(True)
        # Leave each proposal out of its own density estimate.
        self_mask = torch.zeros(num, n_hist + num, dtype=torch.bool, device=x.device)
        indices = torch.arange(num, device=x.device)
        self_mask[indices, n_hist + indices] = True
        for _ in range(self.kde_steps):
            with torch.enable_grad():
                support = torch.cat([history, x], dim=0)
                sq_dist = torch.cdist(x, support).pow(2)
                logits = sq_dist.div(-2 * bandwidth * bandwidth).masked_fill(
                    self_mask, float("-inf")
                )
                log_p = torch.logsumexp(logits, dim=1) - math.log(n_hist + num - 1)
                grad = torch.autograd.grad(-log_p.mean(), x)[0]
            # Normalizing the step by its own norm makes the ascent move at a
            # fixed speed in action-space units regardless of the bandwidth:
            # the raw gradient scales like 1/h^2 and would stall for wide
            # kernels and explode for narrow ones. Direction is unchanged.
            step = grad / (grad.norm(dim=1, keepdim=True) + 1e-8)
            x = (
                x.detach() + self.kde_lr * step
            ).clamp(self.action_low, self.action_high).requires_grad_(True)
        return x.detach()

    def generate(self, data, num_states: int) -> CounterfactualBatch:
        """Turn one real batch into at most ``num_states * k_actions`` synthetic ones.

        ``data`` is a replay buffer sample of real transitions. Rewards and
        dones are copied from the real transition whose next state is closest
        to the predicted one (CEA's closest-transition-pair grounding), and
        predictions whose nearest real next state is further away than real
        next states typically are from each other are dropped.
        """
        obs = data.obs.reshape(data.obs.shape[0], -1)
        next_obs = data.next_obs.reshape(data.next_obs.shape[0], -1)
        actions = data.actions.reshape(data.actions.shape[0], -1)
        num = num_states * self.k_actions
        self.fit(obs, actions, next_obs)
        cf_actions = self.explore_actions(actions, num)
        with torch.no_grad():
            state_inds = torch.randint(0, obs.shape[0], size=(num,), device=obs.device)
            design = torch.cat([cf_actions, torch.ones_like(cf_actions[:, :1])], dim=1)
            noise = (
                torch.randn(num, self.obs_dim, device=obs.device)
                * self.residual_std
                * self.noise_scale
            )
            cf_next_obs = obs[state_inds] + design @ self.weights + noise
            # Reward grounding: match against the real next states, and drop
            # predictions with no real neighbour within the typical spacing.
            dists = torch.cdist(cf_next_obs, next_obs)
            nearest = dists.argmin(dim=1)
            nearest_dist = dists.gather(1, nearest[:, None]).flatten()
            ref_dists = torch.cdist(next_obs, next_obs)
            ref_dists.fill_diagonal_(float("inf"))
            nn_dists = ref_dists.min(dim=1).values
            # Fall back to the overall spread when the reference set has
            # exact duplicates (or a single sample), which would otherwise
            # collapse the threshold to zero and reject every prediction.
            scale = nn_dists[nn_dists.isfinite()].median()
            if not torch.isfinite(scale) or scale == 0:
                scale = ref_dists.isfinite().float().median()
            if not torch.isfinite(scale) or scale == 0:
                scale = torch.ones((), device=obs.device)
            # Gate on model competence too: if the fitted dynamics explain
            # only a small share of the transition, the counterfactual
            # predictions are mostly noise and are all dropped however dense
            # the reference set happens to be. The share is measured against
            # the identity predictor (next = obs), the null model for
            # near-constant state vectors.
            model_err = (obs + torch.cat(
                [actions, torch.ones_like(actions[:, :1])], dim=1
            ) @ self.weights - next_obs).norm(dim=1).median()
            identity_err = (next_obs - obs).norm(dim=1).median()
            competent = model_err < 0.9 * identity_err
            keep = (nearest_dist <= self.match_threshold * scale) & competent
        return CounterfactualBatch(
            obs=obs[state_inds][keep],
            next_obs=cf_next_obs[keep],
            actions=cf_actions[keep],
            rewards=data.rewards.flatten()[nearest[keep]],
            dones=data.dones.flatten()[nearest[keep]],
        )

    def step(self, rb, num_states: int) -> int:
        """Augment ``rb`` in place with one batch of counterfactual transitions.

        Returns the number of transitions written; the buffer itself advances
        ``written // rb.num_envs`` rows, since a row holds one step per env.
        """
        data = rb.sample(self.fit_size)
        batch = self.generate(data, num_states)
        rows = batch.obs.shape[0] // rb.num_envs
        if rows == 0:
            return 0
        # Discard the ragged tail so whole buffer rows are filled; the count
        # reported is the number of transitions actually stored.
        rb.add(
            batch.obs[: rows * rb.num_envs].reshape(rows, rb.num_envs, *self.obs_shape),
            batch.next_obs[: rows * rb.num_envs].reshape(rows, rb.num_envs, *self.obs_shape),
            batch.actions[: rows * rb.num_envs].reshape(rows, rb.num_envs, *self.action_shape),
            batch.rewards[: rows * rb.num_envs].reshape(rows, rb.num_envs),
            batch.dones[: rows * rb.num_envs].reshape(rows, rb.num_envs),
        )
        return rows * rb.num_envs
