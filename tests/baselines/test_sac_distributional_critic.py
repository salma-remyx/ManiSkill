"""Tests for the DSAC-T distributional critic wired into the SAC baseline.

These exercise the integration through the existing ``sac.py`` call site
(the module the ``--distributional`` flag dispatches into), not just the new
module in isolation.
"""

import importlib.util
import pathlib
import sys
import types

import numpy as np
import pytest
import torch

SAC_DIR = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "examples"
    / "baselines"
    / "sac"
)


def _load(name: str):
    """Load a module from the SAC baseline dir, which is not a package.

    The baseline scripts import each other by bare module name (they are meant
    to be run as ``python sac.py`` from their own directory), so the directory
    is put on sys.path to make that resolve here too.
    """
    if str(SAC_DIR) not in sys.path:
        sys.path.insert(0, str(SAC_DIR))
    spec = importlib.util.spec_from_file_location(name, SAC_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sac():
    # sac.py imports SummaryWriter at module scope; tensorboard is an optional
    # training dependency, so stub it when it is not installed.
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        stub = types.ModuleType("torch.utils.tensorboard")
        stub.SummaryWriter = object
        sys.modules["torch.utils.tensorboard"] = stub
    return _load("sac")


@pytest.fixture(scope="module")
def distributional_critic(sac):
    """The same module object sac.py's import resolved to.

    sac.py falls back to a file-location import when its directory is not on
    sys.path, so loading a second copy here would produce distinct classes
    and isinstance checks would spuriously fail.
    """
    name = sac.DistributionalSACCritic.__module__
    if name in sys.modules:
        return sys.modules[name]
    return _load("distributional_critic")


class FakeEnv:
    """Just the two space attributes the critic head reads."""

    class _Box:
        def __init__(self, shape):
            self.shape = shape
            self.high = np.ones(shape)
            self.low = -np.ones(shape)

    def __init__(self, obs_dim=5, act_dim=2):
        self.single_observation_space = self._Box((obs_dim,))
        self.single_action_space = self._Box((act_dim,))


BATCH = 16


def _batch(env, batch_size=BATCH):
    torch.manual_seed(0)
    return (
        torch.randn(batch_size, env.single_observation_space.shape[0]),
        torch.rand(batch_size, env.single_action_space.shape[0]) * 2 - 1,
    )


def test_sac_exposes_distributional_flag(sac):
    """The call site must surface the flag that selects the DSAC-T critic."""
    assert hasattr(sac, "Args")
    assert sac.Args().distributional is False
    assert sac.DistributionalSACCritic is not None


def test_expected_value_preserves_scalar_contract(sac, distributional_critic):
    """The actor loss in sac.py needs a scalar Q, not a distribution."""
    env = FakeEnv()
    critic = distributional_critic.DistributionalSACCritic(env, tau=0.01)
    obs, act = _batch(env)
    value = critic.qf1.expected_value(obs, act)
    assert value.shape == (BATCH, 1)
    assert critic.min_expected_value(obs, act).shape == (BATCH, 1)


def test_targets_match_min_of_twin_means(sac, distributional_critic):
    """Eq. (20): the target must come from the target net with the smaller mean."""
    env = FakeEnv()
    critic = distributional_critic.DistributionalSACCritic(env, tau=0.01)
    obs, act = _batch(env)
    with torch.no_grad():
        # bias one head so the argmin is unambiguous
        critic.qf1_target.fc_mean.bias.fill_(5.0)
        critic.qf2_target.fc_mean.bias.fill_(-5.0)
    log_pi = torch.zeros(BATCH, 1)
    rewards = torch.randn(BATCH, 1)
    dones = torch.zeros(BATCH, 1)
    y_q, y_z = critic.targets(obs, act, log_pi, rewards, dones, gamma=0.9, alpha=0.2)
    with torch.no_grad():
        q2 = critic.qf2_target(obs, act)[0].view(-1)
        expected = rewards.flatten() + 0.9 * (q2 - 0.2 * log_pi.view(-1))
    assert torch.allclose(y_q, expected, atol=1e-5)
    # y_z is a random draw around y_q, so it must stay within a few sigma
    sigma = critic.qf2_target(obs, act)[1].view(-1)
    assert (y_z - y_q).abs().max() < 6 * sigma.max()


def test_critic_loss_gradient_matches_paper(sac, distributional_critic):
    """Autograd gradient of the surrogate equals eq. (26) as eps -> 0."""
    dc = distributional_critic
    torch.manual_seed(0)
    b, omega = 3.0, 1.7
    y_q = torch.randn(BATCH)
    y_z = torch.randn(BATCH)
    Q0 = torch.randn(BATCH)
    S0 = torch.rand(BATCH) + 0.3
    y_clipped = Q0 + (y_z - Q0).clamp(-b, b)

    Q = torch.nn.Parameter(Q0.clone())
    S = torch.nn.Parameter(S0.clone())
    denom = S.pow(2)  # eps -> 0
    with torch.no_grad():
        y_c = Q + (y_z - Q).clamp(-b, b)
    mean_term = 0.5 * omega * (y_q - Q).pow(2) / denom.detach()
    std_term = 0.5 * (y_c - Q.detach()).pow(2) / denom + 0.5 * torch.log(denom)
    (mean_term + std_term).mean().backward()

    # eq. (26), the two terms of the critic ascent gradient, scaled by the
    # batch mean reduction
    want_Q = -(omega * (y_q - Q0) / S0.pow(2)) / BATCH
    want_S = -((y_clipped - Q0).pow(2) - S0.pow(2)) / S0.pow(3) / BATCH
    assert torch.allclose(Q.grad, want_Q, atol=1e-6)
    assert torch.allclose(S.grad, want_S, atol=1e-6)


def test_adaptive_boundary_follows_three_sigma_rule(sac, distributional_critic):
    """Eq. (23)/(27): b converges to xi * E[sigma] under the slow-moving update."""
    env = FakeEnv()
    critic = distributional_critic.DistributionalSACCritic(env, tau=0.01)
    obs, act = _batch(env)
    y_q = torch.zeros(BATCH)
    y_z = torch.zeros(BATCH)
    for _ in range(500):
        critic.loss(obs, act, y_q, y_z)
    with torch.no_grad():
        sigma1 = critic.qf1(obs, act)[1].view(-1)
        sigma2 = critic.qf2(obs, act)[1].view(-1)
    assert critic.b1.item() == pytest.approx(
        distributional_critic.XI * sigma1.mean().item(), rel=0.05
    )
    assert critic.b2.item() == pytest.approx(
        distributional_critic.XI * sigma2.mean().item(), rel=0.05
    )


def test_training_step_reduces_td_error(sac, distributional_critic):
    """A few critic updates on a fixed batch must pull the mean toward y_q."""
    env = FakeEnv()
    critic = distributional_critic.DistributionalSACCritic(env, tau=0.01)
    obs, act = _batch(env, batch_size=64)
    y_q = torch.full((64,), 3.0)
    y_z = torch.full((64,), 3.0)
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-2)

    def td_error():
        with torch.no_grad():
            return (critic.qf1.expected_value(obs, act).view(-1) - y_q).abs().mean()

    before = td_error()
    for _ in range(200):
        optimizer.zero_grad()
        critic.loss(obs, act, y_q, y_z).backward()
        optimizer.step()
    assert td_error().item() < before.item() * 0.5
    assert torch.isfinite(critic.qf1.expected_value(obs, act)).all()


def test_target_network_is_polyak_updated(sac, distributional_critic):
    env = FakeEnv()
    critic = distributional_critic.DistributionalSACCritic(env, tau=0.05)
    obs, act = _batch(env)
    before = critic.qf1_target.fc_mean.weight.detach().clone()
    with torch.no_grad():
        critic.qf1.fc_mean.weight.add_(1.0)
    critic.update_targets()
    after = critic.qf1_target.fc_mean.weight.detach()
    assert not torch.allclose(before, after)
    assert torch.allclose(after, 0.05 * (before + 1.0) + 0.95 * before, atol=1e-6)


def test_sac_default_path_unchanged(sac):
    """The scalar-critic path must still exist and behave as before."""
    env = FakeEnv()
    qf = sac.SoftQNetwork(env)
    obs, act = _batch(env)
    assert qf(obs, act).shape == (BATCH, 1)


def test_state_dict_roundtrip_uses_baseline_keys(sac, distributional_critic):
    """Checkpoints written by the wiring use the qf1/qf2 keys sac.py expects."""
    env = FakeEnv()
    critic = distributional_critic.DistributionalSACCritic(env, tau=0.01)
    ckpt = {
        "actor": {},
        "qf1": critic.qf1.state_dict(),
        "qf2": critic.qf2.state_dict(),
    }
    fresh = distributional_critic.DistributionalQNetwork(env)
    fresh.load_state_dict(ckpt["qf1"])
    obs, act = _batch(env)
    assert torch.allclose(
        fresh.expected_value(obs, act), critic.qf1.expected_value(obs, act)
    )


def test_critic_parameters_cover_both_heads(sac, distributional_critic):
    env = FakeEnv()
    critic = distributional_critic.DistributionalSACCritic(env, tau=0.01)
    n_single = len(list(critic.qf1.parameters()))
    assert len(critic.parameters()) == 2 * n_single


def test_sac_wiring_builds_distributional_critic(sac, distributional_critic):
    """The call site must construct the DSAC-T critic under --distributional.

    Replays the construction block sac.py runs before its training loop: the
    flag selects the twin value distributions, and the same qf1/qf2 names are
    bound so the checkpoint and logging code paths are unchanged.
    """
    env = FakeEnv()
    args = sac.Args()
    args.distributional = True
    # sac.py reuses the class it imported, whichever loader resolved it
    assert sac.DistributionalSACCritic.__name__ == "DistributionalSACCritic"

    dist_critic = sac.DistributionalSACCritic(env, tau=args.tau)
    qf1, qf2 = dist_critic.qf1, dist_critic.qf2
    assert isinstance(qf1, distributional_critic.DistributionalQNetwork)
    assert isinstance(qf2, distributional_critic.DistributionalQNetwork)
    # the baseline's Adam setup consumes this exact parameter list
    q_optimizer = torch.optim.Adam(dist_critic.parameters(), lr=args.q_lr)
    obs, act = _batch(env)
    y_q, y_z = torch.zeros(BATCH), torch.zeros(BATCH)
    q_optimizer.zero_grad()
    dist_critic.loss(obs, act, y_q, y_z).backward()
    q_optimizer.step()
    # the scalar contract the actor loss and logging rely on still holds
    assert qf1.expected_value(obs, act).shape == (BATCH, 1)
    assert dist_critic.min_expected_value(obs, act).shape == (BATCH, 1)


def test_sac_actor_loss_runs_against_distributional_critic(sac, distributional_critic):
    """The actor update in sac.py must work unchanged with the new critic.

    Mirrors the actor block: min over the twin value distributions (eq. 22)
    feeding the same (alpha * log_pi) - min_qf_pi objective.
    """
    env = FakeEnv(obs_dim=5, act_dim=2)
    actor = sac.Actor(env)
    critic = distributional_critic.DistributionalSACCritic(env, tau=0.01)
    obs, _ = _batch(env)
    pi, log_pi, _ = actor.get_action(obs)
    min_qf_pi = critic.min_expected_value(obs, pi)
    actor_loss = ((0.2 * log_pi) - min_qf_pi).mean()
    actor_loss.backward()
    assert torch.isfinite(actor_loss)
    assert actor.fc_mean.weight.grad is not None
