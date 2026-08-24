"""Integration test for the counterfactual replay augmentation wired into sac.py.

Builds the state-based SAC ReplayBuffer from the baseline script itself and
checks the CEA data-path contract end to end: real transitions go in, the
augmenter writes same-shape counterfactual transitions into the same buffer,
and those transitions are reachable from `sample`.
"""

import importlib.util
import pathlib
import sys
import types

import gymnasium as gym
import numpy as np
import pytest
import torch

SAC_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "examples"
    / "baselines"
    / "sac"
    / "sac.py"
)
CF_PATH = SAC_PATH.parent / "counterfactual_replay.py"


def _load(name, path):
    if "torch.utils.tensorboard" not in sys.modules:
        try:
            importlib.import_module("torch.utils.tensorboard")
        except ImportError:
            # The baseline script only needs SummaryWriter for live logging;
            # stub the whole module so the ReplayBuffer can be exercised
            # without the extra dependency installed.
            stub = types.ModuleType("torch.utils.tensorboard")
            stub.SummaryWriter = object
            sys.modules["torch.utils.tensorboard"] = stub
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sac():
    return _load("sac_baseline", SAC_PATH)


@pytest.fixture
def cf():
    return _load("counterfactual_replay", CF_PATH)


class FakeVectorEnv:
    """Just the space attributes the ReplayBuffer and augmenter read."""

    def __init__(self, num_envs=4, obs_dim=6, act_dim=3):
        self.num_envs = num_envs
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, (obs_dim,), np.float32
        )
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (act_dim,), np.float32)


def _fill(rb, env, n, generator):
    """Write `n` real steps of a noisy chain into the buffer."""
    obs = torch.randn(env.num_envs, *env.single_observation_space.shape, generator=generator)
    for _ in range(n):
        actions = torch.rand(
            env.num_envs, *env.single_action_space.shape, generator=generator
        ) * 2 - 1
        # next_obs depends on the action, so the dynamics model has signal.
        next_obs = obs + actions @ torch.eye(
            env.single_action_space.shape[0],
            env.single_observation_space.shape[0],
        )
        rb.add(obs, next_obs, actions, torch.rand(env.num_envs, generator=generator), torch.zeros(env.num_envs))
        # Nudge the state so no two replayed next states coincide exactly.
        obs = next_obs + 0.05 * torch.randn_like(next_obs)


def test_replay_buffer_roundtrip(sac):
    env = FakeVectorEnv()
    rb = sac.ReplayBuffer(env, env.num_envs, 128, torch.device("cpu"), torch.device("cpu"))
    _fill(rb, env, 10, torch.Generator().manual_seed(0))
    data = rb.sample(64)
    assert data.obs.shape == (64, 6)
    assert data.actions.shape == (64, 3)
    assert data.rewards.shape == (64,)


def test_counterfactual_step_grows_buffer_with_matching_shapes(sac, cf):
    env = FakeVectorEnv()
    rb = sac.ReplayBuffer(env, env.num_envs, 1024, torch.device("cpu"), torch.device("cpu"))
    _fill(rb, env, 64, torch.Generator().manual_seed(0))
    rb_before = rb.pos

    augmenter = cf.CounterfactualAugmenter(env, torch.device("cpu"), k_actions=4)
    written = augmenter.step(rb, num_states=16)

    assert written > 0
    assert written % rb.num_envs == 0
    # `written` counts transitions; the buffer advances one row per
    # num_envs of them.
    assert rb.pos == (rb_before + written // rb.num_envs) % rb.per_env_buffer_size
    assert not rb.full

    # Every synthetic transition must respect the action bounds and land
    # within the buffer's reward range: rewards are copied from real data,
    # never predicted.
    data = rb.sample(512)
    assert data.obs.shape == (512, 6)
    assert data.actions.abs().max() <= 1.0 + 1e-5
    assert data.rewards.min() >= 0.0 - 1e-6
    assert data.rewards.max() <= 1.0 + 1e-6


def test_generate_grounds_rewards_in_real_data(sac, cf):
    env = FakeVectorEnv()
    rb = sac.ReplayBuffer(env, env.num_envs, 1024, torch.device("cpu"), torch.device("cpu"))
    _fill(rb, env, 64, torch.Generator().manual_seed(1))

    augmenter = cf.CounterfactualAugmenter(env, torch.device("cpu"), k_actions=4)
    data = rb.sample(512)
    batch = augmenter.generate(data, num_states=32)

    assert batch.obs.shape[0] > 0
    assert batch.obs.shape[1:] == (6,)
    assert batch.next_obs.shape == batch.obs.shape
    assert batch.actions.shape[1:] == (3,)
    assert batch.rewards.shape == (batch.obs.shape[0],)

    # Closest-transition-pair grounding: each synthetic reward equals the
    # reward of some real transition in the reference batch.
    real_rewards = data.rewards.flatten()
    for reward in batch.rewards.tolist():
        assert torch.isclose(
            torch.as_tensor(reward), real_rewards, atol=1e-6
        ).any()


def test_generated_actions_are_not_copies_of_replayed_ones(sac, cf):
    env = FakeVectorEnv()
    rb = sac.ReplayBuffer(env, env.num_envs, 1024, torch.device("cpu"), torch.device("cpu"))
    _fill(rb, env, 64, torch.Generator().manual_seed(2))

    augmenter = cf.CounterfactualAugmenter(env, torch.device("cpu"), k_actions=8)
    data = rb.sample(512)
    batch = augmenter.generate(data, num_states=32)

    # The proposals live in underexplored regions rather than duplicating the
    # replayed actions they were seeded from.
    dists = torch.cdist(batch.actions, data.actions)
    assert dists.min(dim=1).values.max() > 1e-3


def test_explore_actions_spreads_coverage(cf):
    env = FakeVectorEnv()
    augmenter = cf.CounterfactualAugmenter(env, torch.device("cpu"), kde_lr=0.5)

    # A deliberately narrow action history: entropy maximization has to push
    # the proposals away from that cluster (here, towards the upper bound).
    actions = torch.full((256, 3), -0.8) + 0.01 * torch.randn(256, 3)
    proposals = augmenter.explore_actions(actions, num=24)

    assert proposals.shape == (24, 3)
    assert proposals.abs().max() <= 1.0 + 1e-4
    # Every proposal moved off the cluster it was seeded from, and the set as
    # a whole is more spread out than the history it augments.
    assert (proposals - actions.mean(dim=0)).norm(dim=1).min() > 0.05
    assert proposals.std() > actions.std()


class _Sample:
    """Minimal stand-in for a ReplayBufferSample."""

    def __init__(self, obs, next_obs, actions):
        self.obs = obs
        self.next_obs = next_obs
        self.actions = actions
        self.rewards = torch.rand(obs.shape[0], 1)
        self.dones = torch.zeros(obs.shape[0], 1)


def test_generate_is_gated_on_dynamics_signal(cf):
    """When actions carry no information about the transition, augment nothing.

    A dynamics model fitted on action-independent transitions predicts noise;
    writing those predictions into the replay buffer would only corrupt the
    value targets, so the augmenter has to abstain.
    """
    env = FakeVectorEnv()
    augmenter = cf.CounterfactualAugmenter(env, torch.device("cpu"), k_actions=4)
    obs = torch.randn(512, 6)
    actions = torch.rand(512, 3) * 2 - 1
    next_obs = obs + torch.randn(512, 6) * 0.5

    batch = augmenter.generate(_Sample(obs, next_obs, actions), num_states=16)
    assert batch.obs.shape[0] == 0
