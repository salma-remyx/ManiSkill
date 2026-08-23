"""
Distributional critic support and random-shift image augmentation for the
pixel-based SAC baseline.

Adapted from "Continuous Control Reinforcement Learning: Distributed
Distributional DrQ Algorithms" (https://arxiv.org/abs/2404.10645), which shows
that combining a *distributional* critic with DrQ-style *random-shift* image
augmentation is what makes actor-critic-from-pixels work on high-dimensional
continuous control tasks. Both pieces are provided here and are wired into
``sac_rgbd.py`` behind the ``--distributional`` / ``--shift_aug`` flags:

- ``DistributionalSoftQNetwork`` predicts a distribution over returns as
  ``n_quantiles`` quantile atoms (QR-DQN style) instead of a scalar Q value, so
  the critic loss becomes a quantile-regression (huber) loss against the TD
  target distribution rather than an MSE against a scalar target.
- ``random_shift`` is the DrQ-v2 random-shift augmentation (same formulation as
  the ``ShiftAug`` layer used by the TD-MPC2 baseline) applied to the raw pixel
  batch before encoding.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    """Same MLP builder the baseline uses, repeated here so this module does not
    import back out of the single-file training script (which would be circular)."""
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)


def quantile_regression_loss(
    pred_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    huber_kappa: float = 1.0,
) -> torch.Tensor:
    """Quantile-regression (Huber) loss between two sets of quantile estimates.

    Follows the QR-DQN formulation used by DSAC/D3D: pairwise TD errors between
    the ``N`` predicted atoms and the ``N`` target atoms are weighted by the
    quantile levels of the *predicted* atoms, which makes the loss minimize the
    quantile of the TD-error distribution rather than its mean.

    Args:
        pred_quantiles: (batch_size, n_quantiles) predicted return quantiles.
        target_quantiles: (batch_size, n_quantiles) target return quantiles.
        huber_kappa: Huber threshold; quantile Huber loss degenerates to
            pinball loss as kappa -> 0.

    Returns:
        Scalar loss.
    """
    n_quantiles = pred_quantiles.shape[1]
    if n_quantiles == 1:
        # A single atom carries no distributional signal; plain MSE keeps the
        # critic equivalent to the scalar baseline instead of producing a
        # degenerate quantile loss.
        return F.mse_loss(pred_quantiles, target_quantiles)
    tau = (
        torch.arange(n_quantiles, device=pred_quantiles.device, dtype=torch.float32)
        + 0.5
    ) / n_quantiles
    # (B, N, 1) - (B, 1, N) -> pairwise TD errors between all atom pairs
    pairwise_td = target_quantiles.unsqueeze(-1) - pred_quantiles.unsqueeze(1)
    abs_td = pairwise_td.abs()
    huber = torch.where(
        abs_td <= huber_kappa,
        0.5 * pairwise_td.pow(2),
        huber_kappa * (abs_td - 0.5 * huber_kappa),
    )
    tau_hat = tau.view(1, 1, n_quantiles)
    # stop gradient on the target side, as in QR-DQN
    loss = (tau_hat - (pairwise_td.detach() < 0).float()).abs() * huber
    return loss.sum(dim=2).mean(dim=1).sum() / (n_quantiles * pred_quantiles.shape[0])


def distributional_q_target(
    qf1_next_target: torch.Tensor,
    qf2_next_target: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Build the distributional TD target from atom-wise double-Q estimates.

    Both target critics predict ``n_quantiles`` atoms; the atom-wise elementwise
    minimum is the distributional analogue of the clipped double Q trick used by
    the scalar SAC baseline. Rewards and dones are broadcast onto every atom and
    the target is returned detached from the graph.

    Args:
        qf1_next_target: (batch_size, n_quantiles) atoms of target critic 1.
        qf2_next_target: (batch_size, n_quantiles) atoms of target critic 2.
        rewards: (batch_size,) or (batch_size, 1) rewards.
        dones: (batch_size,) or (batch_size, 1) stop-bootstrap flags.
        gamma: discount factor.

    Returns:
        (batch_size, n_quantiles) detached target quantiles.
    """
    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
    rewards = rewards.flatten().unsqueeze(-1)
    dones = dones.flatten().unsqueeze(-1)
    next_q_value = rewards + (1 - dones) * gamma * min_qf_next_target
    return next_q_value.detach()


def random_shift(images: torch.Tensor, pad: int = 3) -> torch.Tensor:
    """DrQ-v2 random-shift augmentation on a batch of images.

    Same formulation as the ``ShiftAug`` layer of the TD-MPC2 baseline (adapted
    from https://github.com/facebookresearch/drqv2): pad with the edge pixels,
    then sample a per-sample translation via an affine grid.

    Args:
        images: (B, C, H, W) batch of images, H must equal W.
        pad: maximum shift in pixels in each direction.

    Returns:
        (B, C, H, W) shifted images.
    """
    x = images.float()
    n, _, h, w = x.size()
    assert h == w
    padding = tuple([pad] * 4)
    x = F.pad(x, padding, "replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device, dtype=x.dtype)[
        :h
    ]
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
    shift *= 2.0 / (h + 2 * pad)
    grid = base_grid + shift
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class DistributionalSoftQNetwork(nn.Module):
    """SAC critic that predicts a distribution over returns instead of a scalar.

    The encoder is shared with the actor exactly as in ``SoftQNetwork``; the
    head is widened to emit ``n_quantiles`` atoms whose mean is the scalar Q
    estimate. Mirroring the scalar baseline, the encoder gradients are only
    applied through the Q-network optimizer (the actor passes
    ``detach_encoder=True``).
    """

    def __init__(self, envs, encoder, n_quantiles: int = 32, hidden_dims=(512, 256)):
        super().__init__()
        self.encoder = encoder
        self.n_quantiles = n_quantiles
        action_dim = np.prod(envs.single_action_space.shape)
        state_dim = envs.single_observation_space["state"].shape[0]
        self.mlp = make_mlp(
            encoder.encoder.out_dim + action_dim + state_dim,
            list(hidden_dims) + [n_quantiles],
            last_act=False,
        )

    def forward(self, obs, action, visual_feature=None, detach_encoder=False):
        if visual_feature is None:
            visual_feature = self.encoder(obs)
        if detach_encoder:
            visual_feature = visual_feature.detach()
        x = torch.cat([visual_feature, obs["state"], action], dim=1)
        return self.mlp(x)
