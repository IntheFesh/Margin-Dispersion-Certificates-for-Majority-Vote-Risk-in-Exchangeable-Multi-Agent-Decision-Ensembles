"""Theorem 1: Cantelli-Hoeffding closed-form certificate B_N^CH.

Pure functions: arrays/floats in, floats out. No I/O, no side effects.
"""
from __future__ import annotations
import math
from ._optimize import minimize_eta

_F_SLACK = 1e-12


def _validate_moments(alpha_bar: float, F: float) -> None:
    if not (0.0 <= alpha_bar <= 1.0):
        raise ValueError(f"alpha_bar must be in [0,1], got {alpha_bar}")
    upper = alpha_bar * (1.0 - alpha_bar) + _F_SLACK
    if not (0.0 <= F <= upper):
        raise ValueError(
            f"F must be in [0, alpha_bar*(1-alpha_bar)]={upper - _F_SLACK:.6g}, got {F}"
        )


def ch_objective(eta: float, alpha_bar: float, F: float, N: int) -> float:
    """The Cantelli-Hoeffding integrand minimized over eta."""
    m = alpha_bar - 0.5
    return F / (F + (m - eta) ** 2) + math.exp(-2.0 * N * eta ** 2)


def B_N_CH(alpha_bar: float, F: float, N: int) -> float:
    """Compute B_N^CH via 1-D minimization over eta in (0, m).

    Returns 1.0 if m <= 0 (no certificate available). Uses
    scipy.optimize.minimize_scalar with a 2000-point dense-grid fallback
    (see src/certs/_optimize.py). Return value clipped to [0, 1].
    """
    _validate_moments(alpha_bar, F)
    if N < 1:
        raise ValueError(f"N must be a positive integer, got {N}")
    m = alpha_bar - 0.5
    if m <= 0:
        return 1.0
    res = minimize_eta(lambda e: ch_objective(e, alpha_bar, F, N), m)
    return float(min(1.0, max(0.0, res["value"])))
