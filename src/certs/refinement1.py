"""Refinement 1: boundedness-aware certificate B_N^CH'.

The piecewise closed form C_[0,1](alpha_bar, F, eta) is the sharp two-moment
upper bound on mu([0, 1/2 + eta]) over P([0,1]) with mean=alpha_bar, var=F.

The Case A/B split is governed by the sign of p_0, the lower atom of the
(unconstrained) optimal two-point distribution on R matching the
mean-variance constraints:

  * Case A (p_0 >= 0): F <= (1-alpha_bar)(m-eta). The unconstrained
    Cantelli optimizer already lives in [0,1].
  * Case B (p_0 = 0): F > (1-alpha_bar)(m-eta). The optimizer is pushed
    against the lower boundary 0.

Do NOT describe this split as "Hausdorff admissibility"; that was an
incorrect terminological framing in the Appendix A draft v0. Derivation
reference: Appendix A.5 (v12).
"""
from __future__ import annotations
import math
from ._optimize import minimize_eta
from .theorem1 import _validate_moments


def C_01(alpha_bar: float, F: float, eta: float) -> float:
    """Piecewise Refinement 1 closed form C_[0,1](alpha_bar, F, eta).

    Case A: F <= (1-alpha_bar)(m-eta)  ->  F / (F + (m-eta)^2)
            (optimal two-point distribution has lower atom p_0 >= 0)
    Case B: F >  (1-alpha_bar)(m-eta)  ->  (1-alpha_bar)
                                            + (alpha_bar(1-alpha_bar) - F)/(1/2 - eta)
            (optimal two-point distribution has lower atom p_0 = 0)
    """
    m = alpha_bar - 0.5
    one_minus = 1.0 - alpha_bar
    boundary = one_minus * (m - eta)
    if F <= boundary:  # Case A: p_0 >= 0
        return F / (F + (m - eta) ** 2)
    # Case B: p_0 = 0
    return one_minus + (alpha_bar * one_minus - F) / (0.5 - eta)


def ch_prime_objective(eta: float, alpha_bar: float, F: float, N: int) -> float:
    return C_01(alpha_bar, F, eta) + math.exp(-2.0 * N * eta ** 2)


def B_N_CH_prime(alpha_bar: float, F: float, N: int) -> float:
    """Compute B_N^CH' via 1-D minimization over eta in (0, m), using the
    same scipy + dense-grid-fallback pattern as B_N_CH. Clipped to [0,1].
    """
    _validate_moments(alpha_bar, F)
    if N < 1:
        raise ValueError(f"N must be a positive integer, got {N}")
    m = alpha_bar - 0.5
    if m <= 0:
        return 1.0
    res = minimize_eta(lambda e: ch_prime_objective(e, alpha_bar, F, N), m)
    return float(min(1.0, max(0.0, res["value"])))
