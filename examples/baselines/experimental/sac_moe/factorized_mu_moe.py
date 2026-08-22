"""Factorized (multilinear) mixture-of-experts layers for the SAC MoE baseline.

The dense MoE in ``sac_moe.py`` computes *every* expert for every input and
mixes them with softmax weights (``torch.stack([expert(x) for expert in
self.experts])``), so its cost grows linearly with the number of experts: one
full MLP forward pass per expert, plus a gating network, on every call.  This
module adds the multilinear MoE ("muMoE") layer of

    Oldfield et al., "Multilinear Mixture of Experts: Scalable Expert
    Specialization through Factorization", NeurIPS 2024 (arXiv:2402.12550)

reimplemented from the paper -- the reference implementation carries no
license, so nothing is vendored here.  A muMoE parameterizes the whole stack of
expert weight matrices as a single order-3 tensor ``W`` in ``R^{N x I x O}``
and computes the mixture *implicitly* in factorized form, never materializing
``W``.  Expert counts can therefore scale far beyond what a dense MoE can
afford, while routing stays fully differentiable (no top-K selection).

Forward pass (paper Eq. 1)::

    a = phi(G^T z)              # expert coefficients, phi = entmax1.5
    y = W x_1 a x_2 z           # mode-n tensor products

Two factorizations of ``W`` are provided, each with the fast einsum forward
pass derived in the paper's Appendix B:

* ``variant="cp"`` (Eq. 2): CP-rank-R sum of outer products; forward cost
  ``R(N + I + O)`` instead of the dense ``NIO``.
* ``variant="tr"`` (Eq. 3): Tensor-Ring format, whose cost grows with the
  expert count only through the small ``R1*N*R2`` term -- the paper's
  recommended choice once N reaches a few hundred.

:class:`FactorizedMoE` is a drop-in replacement for the ``MoE`` wrapper in
``sac_moe.py``: it keeps the expert MLPs' input/hidden layers as a *shared*
backbone and replaces only their final linear layer with the muMoE layer, and
swaps the softmax gating network for an entmax1.5 router over the same input
features.  :func:`materialize_experts` exposes the implicit expert weight
matrices for inspection, which is the property the paper exploits for expert
interpretability and editing.
"""

from typing import Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def entmax1_5(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparse 1.5-entmax over ``dim`` (Peters et al. 2019; Correia et al. 2019).

    A convex generalization of softmax whose output sums to one but is exactly
    sparse: many expert coefficients are zero, which the paper finds yields
    more monosemantic experts than softmax (its Appendix H.1).  Solved by
    bisection on the shift ``tau``, which keeps the result differentiable.
    """
    n = logits.shape[dim]
    transposed = logits.transpose(dim, -1)
    flat = transposed.reshape(-1, n)

    # p(tau) = clamp(x - tau)^2 must sum to 1; tau lies in [max-1, max].
    mu = flat.max(dim=-1, keepdim=True).values
    lo = mu - 1.0
    hi = mu.expand_as(flat).clone()
    for _ in range(50):
        tau = (lo + hi) / 2.0
        total = torch.clamp(flat - tau, min=0.0).pow(2).sum(dim=-1, keepdim=True)
        too_big = total > 1.0
        lo = torch.where(too_big, tau, lo)
        hi = torch.where(too_big, hi, tau)

    tau = (lo + hi) / 2.0
    p = torch.clamp(flat - tau, min=0.0).pow(2)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return p.reshape(transposed.shape).transpose(dim, -1)


class MuMoEGating(nn.Module):
    """Differentiable router computing the expert coefficients ``a = phi(G^T z)``.

    Replaces the dense baseline's 4-layer softmax ``Gating`` network with a
    single linear projection, batch normalization over the logits, and the
    sparse entmax1.5 activation -- the configuration used throughout the paper
    (Sections 3.1 and 4).  Normalizing the logits before the activation is
    what keeps the sparsity pattern stable early in training (App. H.3).
    """

    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, num_experts)
        self.norm = nn.BatchNorm1d(num_experts)
        self.num_experts = num_experts

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.proj(z)
        if self.training and z.size(0) > 1:
            logits = self.norm(logits)
        return entmax1_5(logits, dim=-1)


# ---------------------------------------------------------------------------
# Factorized layer
# ---------------------------------------------------------------------------
class FactorizedMoE(nn.Module):
    """muMoE over the final linear layer of a shared-backbone stack of experts.

    ``expert_module(env)`` is constructed once and its final ``nn.Linear`` is
    replaced by the factorized N-expert layer, so the module slots into the
    same call sites as the dense ``MoE`` (same constructor signature, same
    ``forward(x, a=None)`` contract) and reads its dimensions from the expert
    network it is given.
    """

    def __init__(
        self,
        num_experts: int,
        expert_module,
        env,
        variant: str = "cp",
        rank: int = 64,
        tr_ranks: Sequence[int] = (4, 4),
    ):
        super().__init__()
        if variant not in ("cp", "tr"):
            raise ValueError(f"unknown muMoE variant {variant!r} (expected 'cp' or 'tr')")

        expert = expert_module(env)
        net = expert.net
        self.backbone = net[:-1]
        head = net[-1]
        if not isinstance(head, nn.Linear):
            raise TypeError("expert network must end with nn.Linear to host the muMoE layer")
        in_dim, hidden_dim, out_dim = net[0].in_features, head.in_features, head.out_features

        self.variant = variant
        self.num_experts = num_experts
        self.rank = rank
        self.in_dim, self.hidden_dim, self.out_dim = in_dim, hidden_dim, out_dim

        # U(-k, k) with k = 1/fan_in, matching nn.Linear's default init.
        def _uniform(*shape, fan_in):
            k = 1.0 / fan_in
            return torch.empty(*shape).uniform_(-k, k)

        if variant == "cp":
            self.U1 = nn.Parameter(_uniform(rank, num_experts, fan_in=num_experts))
            self.U2 = nn.Parameter(_uniform(rank, hidden_dim, fan_in=hidden_dim))
            self.U3 = nn.Parameter(_uniform(rank, out_dim, fan_in=hidden_dim))
            with torch.no_grad():
                # Expert-mode factors replicate the weight matrix along the
                # expert mode (plus noise) so all experts start identical
                # rather than as pure noise (paper App. F.2).
                self.U1.normal_(1.0, 1.0)
        else:
            r1, r2 = tr_ranks
            r3 = rank
            self.tr_ranks = (r1, r2, r3)
            self.U1 = nn.Parameter(_uniform(r1, num_experts, r2, fan_in=num_experts))
            self.U2 = nn.Parameter(_uniform(r2, hidden_dim, r3, fan_in=hidden_dim))
            self.U3 = nn.Parameter(_uniform(r3, out_dim, r1, fan_in=hidden_dim))
            with torch.no_grad():
                # Same idea in TR format: a (noisy) diagonal along the ring
                # replicates the weight matrix for every expert at init.
                diag = torch.randn(r1, r2)
                self.U1.copy_(torch.eye(r1, r2).unsqueeze(1) * diag.unsqueeze(1))

        self.gating = MuMoEGating(in_dim, num_experts)

    def expert_coefficients(self, z: torch.Tensor) -> torch.Tensor:
        """The routing weights ``a`` for each input, shape ``[B, num_experts]``."""
        return self.gating(z)

    def forward(self, x: torch.Tensor, a: torch.Tensor = None) -> torch.Tensor:
        if a is not None:
            x = torch.cat([x, a], dim=-1)
        h = self.backbone(x)
        coef = self.gating(x)
        if self.variant == "cp":
            # Eq. 2: y = sum_r (U2 z)_r (U1 a)_r u^(3)_r
            z_r = torch.einsum("bi,ri->br", h, self.U2)
            a_r = torch.einsum("bn,rn->br", coef, self.U1)
            return (z_r * a_r) @ self.U3
        # Eq. 3: contract the first two TR cores with a and z, multiply, then
        # contract with the final core.
        r1, r2, r3 = self.tr_ranks
        f1 = torch.einsum("bn,rnk->brk", coef, self.U1)  # [B, R1, R2]
        f2 = torch.einsum("bi,rik->brk", h, self.U2)  # [B, R2, R3]
        prod = torch.einsum("brk,bkj->brj", f1, f2)  # [B, R1, R3]
        return torch.einsum("brj,jor->bo", prod, self.U3)  # [B, O]

    def extra_repr(self) -> str:
        if self.variant == "cp":
            factors = f"rank={self.rank}"
        else:
            factors = "ranks={}".format(self.tr_ranks)
        return f"num_experts={self.num_experts}, variant={self.variant!r}, {factors}"


def materialize_experts(layer: FactorizedMoE) -> torch.Tensor:
    """Materialize the implicit expert weight tensor ``W`` in ``R^{N x I x O}``.

    Only for inspection/analysis -- calling this at training time defeats the
    point of the factorization.  Exposed because per-expert weight matrices are
    what the paper's interpretability and model-editing results operate on.
    """
    if layer.variant == "cp":
        # W[n, i, o] = sum_r U1[r, n] * U2[r, i] * U3[r, o]
        return torch.einsum("rn,ri,ro->nio", layer.U1, layer.U2, layer.U3)
    r1, r2, r3 = layer.tr_ranks
    # W[n, i, o] = tr( U1[:, n, :] @ U2[i] @ U3[o].T ), with U3 indexed [r3, o, r1]
    return torch.einsum("rnk,kis,sok->nio", layer.U1, layer.U2, layer.U3)


def factor_param_count(layer: FactorizedMoE) -> int:
    """Parameters in the factorized expert tensor (excludes the shared backbone).

    The dense MoE needs ``N * I * O`` for the same experts; this is what the
    factorization saves.
    """
    return layer.U1.numel() + layer.U2.numel() + layer.U3.numel()
