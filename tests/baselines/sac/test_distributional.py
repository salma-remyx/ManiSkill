"""Tests for the distributional-critic / random-shift support wired into the SAC RGBD baseline.

Exercises the call-site edits in ``examples/baselines/sac/sac_rgbd.py`` together with the
``distributional`` capability module, so the integration (not just the helpers) is covered.
"""

import pathlib
import sys
import types
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

# sac_rgbd.py imports SummaryWriter at module scope; tensorboard is an optional dependency of
# the baseline (it is only needed once a run actually starts) so stub it if it is absent.
try:
    import tensorboard  # noqa: F401
except ModuleNotFoundError:
    sys.modules["torch.utils.tensorboard"] = types.ModuleType("torch.utils.tensorboard")
    sys.modules["torch.utils.tensorboard"].SummaryWriter = object

SAC_DIR = pathlib.Path(__file__).resolve().parents[3] / "examples" / "baselines" / "sac"
sys.path.insert(0, str(SAC_DIR))

import sac_rgbd  # noqa: E402

from distributional import (  # noqa: E402
    DistributionalSoftQNetwork,
    distributional_q_target,
    quantile_regression_loss,
    random_shift,
)


class FakeEnvs:
    """Minimal stand-in for the SyncVectorEnv the baseline passes to the networks."""

    def __init__(self, state_dim=8, action_dim=4):
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        self.single_observation_space = {"state": SimpleNamespace(shape=(state_dim,))}


def make_obs(batch_size=6, state_dim=8, image_size=64):
    return {
        "rgb": torch.randint(0, 255, (batch_size, image_size, image_size, 3), dtype=torch.uint8),
        "state": torch.randn(batch_size, state_dim),
    }


def test_sac_rgbd_wires_distributional_support():
    """The baseline must re-export the capability and expose the toggles used at the call site."""
    assert sac_rgbd.DistributionalSoftQNetwork is DistributionalSoftQNetwork
    assert sac_rgbd.quantile_regression_loss is quantile_regression_loss
    assert sac_rgbd.distributional_q_target is distributional_q_target
    assert sac_rgbd.random_shift is random_shift
    args = sac_rgbd.Args()
    assert args.distributional is False
    assert args.n_quantiles > 1
    assert args.shift_aug is False


def test_quantile_regression_loss_single_atom_is_mse():
    pred = torch.randn(5, 1)
    target = torch.randn(5, 1)
    assert torch.isclose(
        quantile_regression_loss(pred, target), F.mse_loss(pred, target)
    )


def test_quantile_regression_loss_prefers_true_quantiles_over_mean():
    """A predictor matching the target quantiles must beat one predicting the distribution mean."""
    generator = torch.Generator().manual_seed(0)
    target = torch.normal(0.0, 1.0, size=(256, 8), generator=generator)
    true_quantiles = torch.quantile(target, (torch.arange(8) + 0.5) / 8, dim=1).T
    mean_prediction = target.mean(dim=1, keepdim=True).expand(-1, 8)
    assert (
        quantile_regression_loss(true_quantiles, target)
        < quantile_regression_loss(mean_prediction, target)
    )


def test_quantile_regression_loss_is_non_negative_and_finite():
    pred = torch.randn(7, 32, requires_grad=True)
    target = torch.randn(7, 32)
    loss = quantile_regression_loss(pred, target)
    assert torch.isfinite(loss) and loss >= 0
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_distributional_q_target_clips_atoms_and_masks_bootstrap():
    n_quantiles = 4
    qf1 = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, 10.0, 10.0, 10.0]])
    qf2 = torch.tensor([[0.0, 5.0, 2.0, 8.0], [4.0, 4.0, 4.0, 4.0]])
    rewards = torch.tensor([1.0, 2.0])
    dones = torch.tensor([0.0, 1.0])
    target = distributional_q_target(qf1, qf2, rewards, dones, gamma=0.5)
    # atom-wise minimum of the two critics, reward broadcast, discount masked where done
    # min([1,2,3,4],[0,5,2,8]) = [0,2,2,4] -> 1 + 0.5 * that
    expected = torch.tensor([[1.0, 2.0, 2.0, 3.0], [2.0, 2.0, 2.0, 2.0]])
    assert target.shape == (2, n_quantiles)
    assert torch.allclose(target, expected)
    # the target must be detached from the graph so it acts as a regression target
    assert not target.requires_grad


def test_random_shift_preserves_shape_and_support():
    images = torch.zeros(3, 2, 32, 32)
    images[:, :, :, -1] = 1.0  # a vertical stripe on the right edge
    shifted = random_shift(images, pad=2)
    assert shifted.shape == images.shape
    assert torch.isfinite(shifted).all()
    # replicate padding means shifting right still leaves signal in the last column
    assert shifted[:, 0, :, -1].abs().sum() > 0
    # shifting never invents mass outside the original value range
    assert shifted.min() >= images.min() - 1e-6 and shifted.max() <= images.max() + 1e-6


def test_shift_aug_obs_only_touches_images():
    obs = make_obs(batch_size=4, image_size=64)
    obs["depth"] = torch.rand(4, 64, 64, 1)
    augmented = sac_rgbd.shift_aug_obs(obs)
    assert augmented["rgb"].shape == obs["rgb"].shape
    assert augmented["rgb"].dtype == obs["rgb"].dtype
    assert augmented["depth"].shape == obs["depth"].shape
    assert torch.equal(augmented["state"], obs["state"])
    # a non-trivial batch of random shifts is overwhelmingly unlikely to be a no-op
    assert not torch.equal(augmented["rgb"], obs["rgb"])


@pytest.mark.parametrize("n_quantiles", [8, 32])
def test_distributional_critic_replaces_scalar_head(n_quantiles):
    """The critic built at the call site emits a return distribution the actor can consume."""
    envs = FakeEnvs()
    sample_obs = make_obs(batch_size=3, image_size=64)
    actor = sac_rgbd.Actor(envs, sample_obs=sample_obs)
    qf = DistributionalSoftQNetwork(envs, actor.encoder, n_quantiles=n_quantiles)

    actions = torch.rand(3, 4)
    quantiles = qf(sample_obs, actions)
    assert quantiles.shape == (3, n_quantiles)

    # the actor loss uses the mean over atoms, exactly as wired in sac_rgbd.py
    qf2 = DistributionalSoftQNetwork(envs, actor.encoder, n_quantiles=n_quantiles)
    min_qf_pi = torch.min(quantiles, qf2(sample_obs, actions)).mean(dim=-1).view(-1)
    assert min_qf_pi.shape == (3,)

    # the critic trains on the distributional loss end to end
    with torch.no_grad():
        next_actions, _, _, visual_feature = actor.get_action(sample_obs)
        target = distributional_q_target(
            qf(sample_obs, next_actions, visual_feature),
            qf2(sample_obs, next_actions, visual_feature),
            torch.tensor([1.0, 0.0, 0.5]),
            torch.tensor([0.0, 0.0, 1.0]),
            gamma=0.8,
        )
    loss = quantile_regression_loss(qf(sample_obs, actions), target)
    assert torch.isfinite(loss)
    loss.backward()
    assert qf.mlp[0].weight.grad is not None


def test_shift_aug_feeds_the_shared_encoder():
    """Augmented observations must remain valid input to the baseline encoder."""
    obs = make_obs(batch_size=2, image_size=64)
    encoder = sac_rgbd.EncoderObsWrapper(sac_rgbd.PlainConv(in_channels=3, image_size=[64, 64]))
    features = encoder(sac_rgbd.shift_aug_obs(obs))
    assert features.shape == (2, encoder.encoder.out_dim)
    assert torch.isfinite(features).all()
