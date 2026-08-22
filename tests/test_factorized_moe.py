"""Tests for the factorized (multilinear) MoE option of the SAC MoE baseline.

These go through the baseline's own public surface: ``sac_moe.build_moe`` (the
factory the training script now routes every value/Q head through) with
``sac_moe.SoftQNetwork`` / ``sac_moe.VNetwork`` as the expert modules, exactly
as ``sac_moe.py``'s ``__main__`` block does.
"""

import pathlib
import sys

import pytest
import torch

SAC_MOE_DIR = (
    pathlib.Path(__file__, "..", "..", "examples", "baselines", "experimental", "sac_moe")
    .resolve()
)
sys.path.insert(0, str(SAC_MOE_DIR))

# `sac_moe.py` imports SummaryWriter for its (unused-in-tests) logging, and
# tensorboard is not part of the minimal test environment. Stub it out rather
# than making the baseline import heavier than the code under test needs.
try:
    import torch.utils.tensorboard  # noqa: F401
except ImportError:
    import types

    _tb = types.ModuleType("torch.utils.tensorboard")

    class _SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

        def close(self):
            pass

    _tb.SummaryWriter = _SummaryWriter
    sys.modules["torch.utils.tensorboard"] = _tb

from sac_moe import Args, MoE, SoftQNetwork, VNetwork, build_moe  # noqa: E402
from factorized_mu_moe import (  # noqa: E402
    FactorizedMoE,
    factor_param_count,
    materialize_experts,
)


class _Box:
    def __init__(self, shape):
        self.shape = shape


class _FakeEnv:
    """Just the two attributes the SAC MoE networks read off the env."""

    single_observation_space = _Box((21,))
    single_action_space = _Box((7,))


OBS_DIM = 21
ACT_DIM = 7


def _rand_obs(batch=32):
    torch.manual_seed(0)
    return torch.randn(batch, OBS_DIM)


def test_default_args_still_build_the_dense_moe():
    """The opt-in flag must not change the baseline's default architecture."""
    net = build_moe(SoftQNetwork, _FakeEnv(), Args())
    assert isinstance(net, MoE)
    assert len(net.experts) == Args().num_experts


@pytest.mark.parametrize("variant", ["cp", "tr"])
def test_factorized_moe_matches_materialized_experts(variant):
    """The fast forward pass equals the naive computation on the full tensor.

    Evaluating every (implicit) expert on the materialized weight tensor and
    mixing with the same coefficients is the paper's Eq. 1, so this is the
    check that the factorized fast path (Eq. 2 / Eq. 3) computes the same layer.
    """
    args = Args(moe_mode="mumoe", moe_variant=variant, num_experts=8, moe_rank=16)
    net = build_moe(SoftQNetwork, _FakeEnv(), args).eval()
    assert isinstance(net, FactorizedMoE)

    obs, act = _rand_obs(), torch.randn(32, ACT_DIM)
    with torch.no_grad():
        fast = net(obs, act)
        # Recompute the layer the slow way: shared backbone, then Eq. 1 over
        # every expert's materialized weight matrix.
        x = torch.cat([obs, act], dim=-1)
        h = net.backbone(x)
        coefs = net.expert_coefficients(x)
        W = materialize_experts(net)  # [N, hidden, out]
        naive = torch.einsum("bn,nio,bi->bo", coefs, W, h)

    assert fast.shape == naive.shape == (32, 1)
    assert torch.allclose(fast, naive, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("variant", ["cp", "tr"])
def test_factorized_moe_backward_through_routing(variant):
    """Routing is differentiable: gradients reach both the factors and the gate.

    This is the property sparse top-K MoEs give up, and the reason the paper
    needs no load-balancing loss.
    """
    args = Args(moe_mode="mumoe", moe_variant=variant, num_experts=16, moe_rank=8)
    net = build_moe(SoftQNetwork, _FakeEnv(), args)
    out = net(_rand_obs(), torch.randn(32, ACT_DIM)).sum()
    out.backward()

    for name in ("U1", "U2", "U3"):
        param = getattr(net, name)
        assert param.grad is not None, f"no gradient reached {name}"
        assert param.grad.abs().sum() > 0
    assert net.gating.proj.weight.grad.abs().sum() > 0


def test_factorized_moe_value_head_obs_only():
    """The V-network call site passes observations only (no action concat)."""
    args = Args(moe_mode="mumoe", num_experts=32, moe_rank=16)
    net = build_moe(VNetwork, _FakeEnv(), args)
    assert net.backbone[0].in_features == OBS_DIM
    assert net(_rand_obs()).shape == (32, 1)


def test_factorized_moe_supports_far_more_experts_than_dense():
    """The point of the factorization: N grows without evaluating each expert.

    A dense MoE holds one full weight matrix per expert; the factorized layer
    holds only the shared factors. At 4096 experts the dense stack would need
    N*hidden*out weights for the head alone, an order of magnitude more.
    """
    num_experts = 4096
    args = Args(moe_mode="mumoe", num_experts=num_experts, moe_rank=64)
    net = build_moe(SoftQNetwork, _FakeEnv(), args)
    factor_params = factor_param_count(net)
    dense_equiv = num_experts * net.hidden_dim * net.out_dim

    assert factor_params < dense_equiv / 3
    # and it still runs, in both variants, without materializing anything
    for variant in ("cp", "tr"):
        variant_args = Args(
            moe_mode="mumoe",
            moe_variant=variant,
            num_experts=num_experts,
            moe_rank=64,
        )
        net_v = build_moe(SoftQNetwork, _FakeEnv(), variant_args)
        assert net_v(_rand_obs(), torch.randn(32, ACT_DIM)).shape == (32, 1)


def test_build_moe_rejects_unknown_mode():
    with pytest.raises(ValueError):
        build_moe(SoftQNetwork, _FakeEnv(), Args(moe_mode="bogus"))
