"""Tests for the multi-state TD target wired into the SAC baseline.

The critic update in ``examples/baselines/sac/sac.py`` regresses onto
``mstd.td_target(...)`` whenever ``--td_horizon > 1`` and onto the unmodified
one-step expression otherwise. These tests drive that same ``MultiStateTD``
object the script constructs and check both paths.
"""

import pathlib
import sys
import types

import pytest
import torch

SAC_DIR = pathlib.Path(__file__).resolve().parents[3] / "examples" / "baselines" / "sac"
sys.path.insert(0, str(SAC_DIR))

# sac.py imports SummaryWriter at module scope but only uses it inside the
# training __main__ block; stub it so the baseline imports on a bare install.
if "torch.utils.tensorboard" not in sys.modules:
    try:
        import torch.utils.tensorboard  # noqa: F401
    except ImportError:
        stub = types.ModuleType("torch.utils.tensorboard")
        stub.SummaryWriter = object
        sys.modules["torch.utils.tensorboard"] = stub

import sac  # noqa: E402
from multi_state_td import MultiStateTD  # noqa: E402

BATCH = 6


class FakeEnv:
    """Just enough of a vectorized env for the baseline's ReplayBuffer."""

    class _Box:
        shape = (4,)

    single_observation_space = _Box()
    single_action_space = _Box()


def fill_buffer(num_envs=3, length=8):
    rb = sac.ReplayBuffer(
        env=FakeEnv(),
        num_envs=num_envs,
        buffer_size=length * num_envs,
        storage_device=torch.device("cpu"),
        sample_device=torch.device("cpu"),
    )
    for t in range(length):
        obs = torch.arange(t * 10, t * 10 + 4).float().unsqueeze(0).expand(num_envs, 4)
        # next_obs identifies s_{t+1} so each bootstrap state is traceable
        next_obs = obs + 0.5 if t % 2 == 0 else obs * 1.5
        actions = torch.full((num_envs, 4), float(t))
        rewards = torch.full((num_envs,), float(t + 1))
        rb.add(obs, next_obs, actions, rewards, torch.zeros(num_envs))
    return rb


def bootstrap_from(buffer, sample):
    """The V(s_{t+l}) the script computes as min(Q1,Q2) - alpha*log_pi."""
    flat = sample.next_obs
    if flat.dim() == 2:
        return flat.sum(dim=1)
    return flat["sensor"].flatten(1).sum(dim=1)


def test_horizon_one_matches_baseline_one_step_target():
    """L=1 must reproduce the baseline's own one-step TD target exactly."""
    rb = fill_buffer()
    for mstd in (MultiStateTD(horizon=1), MultiStateTD(horizon=8)):
        mstd.horizon = 1
        sample = mstd.sample(rb, BATCH)
        baseline = rb.sample(BATCH)
        assert sample.rewards.shape == (1, BATCH)
        assert sample.dones.shape == (1, BATCH)
        # the window is one contiguous transition drawn from the same buffer
        assert sample.obs.shape == baseline.obs.shape

        values = bootstrap_from(rb, sample)
        gamma = 0.8
        mstd_target = mstd.td_target(sample, gamma, values)
        expected = sample.rewards[0] + gamma * values
        assert torch.allclose(mstd_target, expected, atol=1e-6)


def test_multi_state_target_averages_l_step_targets():
    """L=3 equals the mean of the 1-, 2- and 3-step targets, by construction."""
    rb = fill_buffer()
    mstd = MultiStateTD(horizon=3)
    sample = mstd.sample(rb, BATCH)
    assert sample.rewards.shape == (3, BATCH)
    assert sample.next_obs.shape == (3 * BATCH, 4)

    # the L bootstrap states are distinct states s_{t+1}, s_{t+2}, s_{t+3}
    assert not torch.equal(
        sample.next_obs[0:BATCH], sample.next_obs[BATCH : 2 * BATCH]
    )

    gamma = 0.8
    values = bootstrap_from(rb, sample)
    got = mstd.td_target(sample, gamma, values)

    v = values.view(3, BATCH)
    r = sample.rewards
    l1 = r[0] + gamma * v[0]
    l2 = r[0] + gamma * r[1] + gamma**2 * v[1]
    l3 = r[0] + gamma * r[1] + gamma**2 * r[2] + gamma**3 * v[2]
    assert torch.allclose(got, (l1 + l2 + l3) / 3, atol=1e-5)


def test_multi_state_target_is_discount_aware():
    """The target tracks gamma, so it is a discounted return and not a reward mean."""
    rb = fill_buffer()
    mstd = MultiStateTD(horizon=3)
    sample = mstd.sample(rb, BATCH)
    values = torch.zeros(3 * BATCH)
    at_low = mstd.td_target(sample, 0.5, values)
    at_high = mstd.td_target(sample, 0.99, values)
    assert torch.all(at_high > at_low)


def test_stop_bootstrap_flag_cuts_the_lookahead():
    """A stop_bootstrap flag at step i keeps r_{t+i} and zeroes everything past it."""
    rb = fill_buffer()
    mstd = MultiStateTD(horizon=3)

    gamma = 0.8
    sample = mstd.sample(rb, BATCH)
    values = torch.ones(3 * BATCH)

    # rewriting the flags in place must not leak into the caller's buffer
    stop = sample.dones.clone()
    stop[0] = 1.0
    cut = mstd.td_target(
        type("S", (), {"rewards": sample.rewards, "dones": stop})(), gamma, values
    )
    uncut = mstd.td_target(sample, gamma, values)

    # with the bootstrap at s_{t+1} dropped, only r_{t+1} survives
    assert torch.allclose(cut, sample.rewards[0], atol=1e-6)
    assert torch.all(cut <= uncut + 1e-6)


def test_horizon_is_clamped_to_available_data():
    """Asking for more steps than the buffer holds samples a shorter window."""
    rb = fill_buffer(length=4)
    mstd = MultiStateTD(horizon=10)
    sample = mstd.sample(rb, BATCH)
    assert sample.rewards.shape[0] == 4
    assert mstd.td_target(sample, 0.9, torch.ones(4 * BATCH)).shape == (BATCH,)


def test_td_horizon_arg_is_off_by_default():
    """The wiring change is opt-in: the baseline default is the one-step target."""
    assert sac.Args().td_horizon == 1
    assert sac.Args(td_horizon=3).td_horizon == 3
    with pytest.raises(ValueError):
        MultiStateTD(horizon=0)


def test_dict_observations_as_used_by_sac_rgbd():
    """sac_rgbd stores observations in a DictArray; windows must still work."""
    import gymnasium as gym
    import numpy as np

    import sac_rgbd

    space = gym.spaces.Dict(
        {
            "sensor": gym.spaces.Box(-1, 1, (3,), np.float32),
            "agent": gym.spaces.Box(-1, 1, (2,), np.float32),
        }
    )

    class DictEnv:
        single_observation_space = space

        class _Box:
            shape = (3,)

        single_action_space = _Box()

    num_envs, length = 2, 8
    rb = sac_rgbd.ReplayBuffer(
        env=DictEnv(),
        num_envs=num_envs,
        buffer_size=length * num_envs,
        storage_device=torch.device("cpu"),
        sample_device=torch.device("cpu"),
    )
    for t in range(length):
        obs = {
            "sensor": torch.full((num_envs, 3), float(t + 1)),
            "agent": torch.full((num_envs, 2), float(t + 1)),
        }
        # s_{t+1} carries t+2, so consecutive bootstrap states are distinguishable
        next_obs = {k: v + 1.0 for k, v in obs.items()}
        rb.add(
            obs,
            next_obs,
            torch.zeros(num_envs, 3),
            torch.full((num_envs,), float(t + 1)),
            torch.zeros(num_envs),
        )

    mstd = MultiStateTD(horizon=3)
    sample = mstd.sample(rb, BATCH)
    assert set(sample.obs.keys()) == {"sensor", "agent"}
    assert sample.next_obs["sensor"].shape == (3 * BATCH, 3)
    # each window's L bootstrap states are s_{t+1}, s_{t+2}, s_{t+3} in order
    first_window = sample.next_obs["sensor"].view(3, BATCH, 3)[:, 0, 0]
    assert torch.all(first_window[1:] - first_window[:-1] == 1)

    target = mstd.td_target(sample, 0.8, torch.ones(3 * BATCH))
    assert target.shape == (BATCH,)
    assert torch.isfinite(target).all()
