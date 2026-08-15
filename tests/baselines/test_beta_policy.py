"""Tests for the bounded-support Beta SAC policy head (arXiv:2409.04971).

The head is wired into the SAC baseline's ``Actor`` via the ``--policy beta``
flag; these tests exercise the actor at that call site, the differentiable
Beta CDF it rests on, and the implicit reparameterization gradient formula
that is the paper's core contribution.
"""

import importlib.util
import pathlib
import sys
import types

import gymnasium as gym
import numpy as np
import pytest
import torch

SAC_DIR = pathlib.Path(__file__).parents[2] / "examples" / "baselines" / "sac"
sys.path.insert(0, str(SAC_DIR))


_MODULES = {}


def _import(name):
    if name in _MODULES:
        return _MODULES[name]
    # sac.py imports torch.utils.tensorboard, which needs the tensorboard
    # package at import time; stub it when it is not installed so the actor
    # can be tested without the full training-stack dependencies.
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        stub = types.ModuleType("tensorboard")
        stub.__path__ = []  # mark as a package so submodule imports resolve
        stub.__version__ = "2.0"
        compat = types.ModuleType("tensorboard.compat")
        tf_stub = types.ModuleType("tensorboard.compat.tf")
        summary = types.ModuleType("tensorboard.summary")
        writer = types.ModuleType("tensorboard.summary.writer")
        record_writer = types.ModuleType("tensorboard.summary.writer.record_writer")

        class _RecordWriter:  # pragma: no cover - unused by these tests
            pass

        record_writer.RecordWriter = _RecordWriter
        writer.record_writer = record_writer
        summary.writer = writer
        stub.summary = summary
        stub.compat = compat
        compat.tf = tf_stub
        sys.modules["tensorboard"] = stub
        sys.modules["tensorboard.compat"] = compat
        sys.modules["tensorboard.compat.tf"] = tf_stub
        sys.modules["tensorboard.summary"] = summary
        sys.modules["tensorboard.summary.writer"] = writer
        sys.modules["tensorboard.summary.writer.record_writer"] = record_writer

    # also stub torch.utils.tensorboard itself if its import chain would fail
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    except ImportError:
        tb_utils = types.ModuleType("torch.utils.tensorboard")

        class _SummaryWriter:  # pragma: no cover - unused by these tests
            def __init__(self, *args, **kwargs):
                pass

            def add_scalar(self, *args, **kwargs):
                pass

            def close(self):
                pass

        tb_utils.SummaryWriter = _SummaryWriter
        sys.modules["torch.utils.tensorboard"] = tb_utils

    spec = importlib.util.spec_from_file_location(name, SAC_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULES[name] = module
    return module


@pytest.fixture
def sac():
    return _import("sac")


@pytest.fixture
def beta_policy():
    return _import("beta_policy")


class FakeEnv:
    """Minimal stand-in for the vectorized env, exposing the spaces the actor reads."""

    def __init__(self, obs_dim=8, act_dim=3):
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, (obs_dim,), np.float32
        )
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (act_dim,), np.float32)


def test_actor_beta_head_is_wired_into_call_site(sac, beta_policy):
    """The SAC Actor must route sampling/eval through the Beta head when policy='beta'."""
    env = FakeEnv()
    actor = sac.Actor(env, policy="beta").double()
    obs = torch.randn(5, 8, dtype=torch.float64)

    action, log_prob, mean = actor.get_action(obs)
    assert action.shape == (5, 3)
    assert log_prob.shape == (5, 1)
    assert mean.shape == (5, 3)
    # bounded support: no tanh squash is needed, the sample already respects the bounds
    assert (action >= -1.0).all() and (action <= 1.0).all()
    assert torch.isfinite(log_prob).all()

    eval_action = actor.get_eval_action(obs)
    assert eval_action.shape == (5, 3)
    assert (eval_action >= -1.0).all() and (eval_action <= 1.0).all()

    # the head is the module under test, not a re-implementation
    assert type(actor.beta_head).__name__ == "BetaPolicyHead"
    assert actor.beta_head.fc_alpha.in_features == 256


def test_actor_beta_head_gradients_flow_to_parameters(sac):
    """Implicit reparameterization gradients must reach the actor's linear layers,
    which is what lets SAC train the Beta policy without an explicit rsample path."""
    env = FakeEnv()
    actor = sac.Actor(env, policy="beta").double()
    obs = torch.randn(16, 8, dtype=torch.float64)

    action, log_prob, _ = actor.get_action(obs)
    loss = (action.sum() + log_prob.sum()) / 16
    loss.backward()

    grads = [p.grad for p in actor.beta_head.parameters()]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
    backbone_grads = [
        p.grad for p in actor.backbone.parameters() if p.grad is not None
    ]
    assert backbone_grads, "no gradient reached the shared backbone"


def test_actor_gaussian_head_unchanged(sac):
    """The default policy must behave exactly as before the wiring."""
    env = FakeEnv()
    actor = sac.Actor(env).double()
    obs = torch.randn(4, 8, dtype=torch.float64)
    action, log_prob, mean = actor.get_action(obs)
    assert action.shape == (4, 3)
    assert (action.abs() <= 1.0).all()
    assert torch.isfinite(log_prob).all()
    # the Gaussian head still carries its fc_mean/fc_logstd layers untouched
    assert hasattr(actor, "fc_mean") and hasattr(actor, "fc_logstd")
    assert not hasattr(actor, "beta_head")


def test_beta_cdf_matches_scipy(beta_policy):
    scipy_special = pytest.importorskip("scipy.special")
    from beta_policy import beta_cdf

    torch.manual_seed(0)
    cases = {
        "a>=1": (
            1 + torch.rand(200) * 20,
            0.1 + torch.rand(200) * 20,
            torch.rand(200) * 0.98 + 0.01,
        ),
        "a<1": (
            0.1 + torch.rand(200) * 0.9,
            0.1 + torch.rand(200) * 20,
            torch.rand(200) * 0.9 + 0.05,
        ),
        "large concentrations": (
            torch.full((100,), 50.0),
            30 + torch.rand(100) * 40,
            torch.rand(100) * 0.9 + 0.05,
        ),
    }
    for name, (a, b, x) in cases.items():
        ours = beta_cdf(a, b, x).double()
        ref = scipy_special.betainc(
            a.double().numpy(), b.double().numpy(), x.double().numpy()
        )
        assert np.abs(ours.numpy() - ref).max() < 1e-4, name


def test_implicit_reparam_grads_match_torch_rsample(beta_policy):
    """The CDF-based implicit gradient dx/dphi = -(dF/dphi)/(dF/dx) must agree
    with the gradient torch's own Beta.rsample produces at the same sample."""
    from beta_policy import implicit_reparam_grads

    torch.manual_seed(3)
    a = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([5.0], requires_grad=True)
    sample = torch.distributions.Beta(a, b).rsample()

    d_torch = torch.autograd.grad(sample.sum(), a)[0]
    d_implicit, _ = implicit_reparam_grads(
        torch.tensor([2.0]), torch.tensor([5.0]), sample.detach()
    )
    assert torch.isclose(d_implicit, d_torch, rtol=1e-4)


def test_implicit_reparam_grads_match_finite_difference(beta_policy):
    """Cross-check dx/da and dx/db against a numeric inverse-CDF finite difference."""
    scipy_special = pytest.importorskip("scipy.special")
    scipy_optimize = pytest.importorskip("scipy.optimize")
    from beta_policy import implicit_reparam_grads

    a, b, x = 2.0, 5.0, 0.3
    da, db = implicit_reparam_grads(
        torch.tensor([a]), torch.tensor([b]), torch.tensor([x])
    )

    u = scipy_special.betainc(a, b, x)

    def icdf(av, bv):
        return scipy_optimize.brentq(
            lambda t: scipy_special.betainc(av, bv, t) - u, 1e-12, 1 - 1e-12
        )

    eps = 1e-5
    numeric_da = (icdf(a + eps, b) - icdf(a - eps, b)) / (2 * eps)
    numeric_db = (icdf(a, b + eps) - icdf(a, b - eps)) / (2 * eps)
    assert torch.isclose(da[0], torch.tensor(numeric_da), rtol=1e-3)
    assert torch.isclose(db[0], torch.tensor(numeric_db), rtol=1e-3)
