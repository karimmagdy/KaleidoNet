"""Hard-Concrete distribution for L0-regularized pruning.

Implements Louizos et al. 2018 (ICLR), "Learning Sparse Neural
Networks through L_0 Regularization". The Hard-Concrete distribution
is a continuous relaxation of Bernoulli with a tractable expected-
L0 penalty:

    z = clip(sigmoid((log u - log(1-u) + log_alpha) / beta) * (zeta - gamma) + gamma, 0, 1)

with u ~ Uniform(0,1) at training time and the deterministic
expectation z = clip(sigmoid(log_alpha) * (zeta - gamma) + gamma, 0, 1)
at inference time. The expected L0 norm is analytic:

    E[||z||_0] = sum_i sigmoid(log_alpha_i - beta * log(-gamma / zeta))

Hyperparameters (matching the Louizos 2018 paper):
    beta = 2/3       (concrete temperature; lower -> more discrete)
    gamma = -0.1     (left-stretch endpoint; below 0)
    zeta = 1.1       (right-stretch endpoint; above 1)

The hard-clip squashes the stretched concrete output to [0, 1] so
that exact zeros and exact ones occur with non-zero probability,
giving a true L0 penalty rather than a soft surrogate.

Drop-in replacement for the Gumbel-sigmoid masks in
kaleidonet/core/elastic.py: same forward/backward signature, same
straight-through interpretation in the inference path.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# Canonical Louizos 2018 hyperparameters
BETA = 2.0 / 3.0
GAMMA = -0.1
ZETA = 1.1
EPS = 1e-6


def _l0_log_correction() -> float:
    """The constant ``-beta * log(-gamma / zeta)`` used in the
    expected-L0 penalty. Pre-computed for efficiency.
    """
    return -BETA * math.log(-GAMMA / ZETA)


L0_LOG_CORRECTION = _l0_log_correction()


def hard_concrete_sample(log_alpha: torch.Tensor, training: bool = True) -> torch.Tensor:
    """Sample from the Hard-Concrete distribution.

    During training, draws u ~ U(0,1) and applies the stretched concrete
    transformation. At inference, returns the deterministic expectation
    (no noise). In both modes, the output is hard-clipped to [0, 1].

    Args:
        log_alpha: Per-element location logits (any shape).
        training: If True, sample stochastically; otherwise return mean.

    Returns:
        z: Tensor of the same shape as log_alpha, with values in [0, 1].
    """
    if training:
        u = torch.rand_like(log_alpha).clamp_(min=EPS, max=1.0 - EPS)
        s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + log_alpha) / BETA)
    else:
        # Deterministic expectation: u replaced by 0.5 -> log u - log(1-u) = 0
        s = torch.sigmoid(log_alpha / BETA)
    s_stretched = s * (ZETA - GAMMA) + GAMMA
    return s_stretched.clamp(min=0.0, max=1.0)


def expected_l0(log_alpha: torch.Tensor) -> torch.Tensor:
    """Analytic expected L0 norm of the Hard-Concrete distribution.

    Returns a scalar (sum over elements of ``log_alpha``). Useful as
    the differentiable surrogate for the L0 penalty term in the loss.

    The closed form (Eq. 12 of Louizos et al. 2018):
        E[||z||_0] = sum_i sigmoid(log_alpha_i - beta * log(-gamma / zeta))
    """
    return torch.sigmoid(log_alpha - L0_LOG_CORRECTION).sum()


def expected_active_fraction(log_alpha: torch.Tensor) -> torch.Tensor:
    """Mean expected-active rate (E[||z||_0] / N) for use as a
    differentiable FLOPs-fraction proxy. Returns a scalar in [0, 1].
    """
    return torch.sigmoid(log_alpha - L0_LOG_CORRECTION).mean()


# ---------------------------------------------------------------------------
# Module wrapper: HardConcreteMask
# ---------------------------------------------------------------------------

class HardConcreteMask(nn.Module):
    """Per-feature Hard-Concrete mask drop-in replacement for the
    Gumbel-sigmoid mask in kaleidonet.core.elastic.ElasticLinear.

    Parameter is ``log_alpha`` (one logit per output feature). Forward
    returns the [0, 1]-valued mask via ``hard_concrete_sample``. The
    expected-L0 norm is exposed via ``expected_l0()`` and the
    expected-active fraction via ``expected_active_fraction()`` for
    use in the loss.

    The hyperparameters BETA, GAMMA, ZETA follow Louizos 2018.
    """

    def __init__(self, num_features: int, log_alpha_init: float = 0.0):
        super().__init__()
        self.num_features = num_features
        # Per-feature location logit; init at 0 -> ~50% expected active rate
        self.log_alpha = nn.Parameter(torch.full((num_features,), log_alpha_init))

    def forward(self) -> torch.Tensor:
        return hard_concrete_sample(self.log_alpha, training=self.training)

    def expected_l0(self) -> torch.Tensor:
        return expected_l0(self.log_alpha)

    def expected_active_fraction(self) -> torch.Tensor:
        return expected_active_fraction(self.log_alpha)

    def hard_active_count(self) -> int:
        """Number of features that are deterministically active at the
        current parameters (used for evaluation / logging). A feature
        is "deterministically active" if the deterministic expectation
        of its Hard-Concrete mask is > 0.5.
        """
        with torch.no_grad():
            z = hard_concrete_sample(self.log_alpha, training=False)
            return int((z > 0.5).sum().item())
