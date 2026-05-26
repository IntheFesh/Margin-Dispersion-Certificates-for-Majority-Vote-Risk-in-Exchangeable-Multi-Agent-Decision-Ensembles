"""Lemma 1 (V14 §4): marginal-only ceiling v_N^mean(alpha_bar).

The mean-only sup of R_N(mu) is achieved by a 2-atom measure on
{a_mean(N), 1}; the interior atom a_mean(N) solves

    g_N(a) + (1 - a) g_N'(a) = 0

in (0, 1/2). The supremum value is

    v_N^mean(alpha_bar) = (1 - alpha_bar) * |g_N'(a_mean(N))|.

Pure functions: no I/O, no side effects.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq

from .theorem3 import g_N


def g_N_prime(a: float, N: int) -> float:
    """g_N'(a) = -N * C(N-1, r) * a^r * (1-a)^(N-1-r), where r = floor(N/2).

    Derivation: differentiating g_N(a) = P(Bin(N,a) <= r) gives a telescoping
    sum that collapses to a single term, the classical formula above.
    """
    r = math.floor(N / 2)
    if a <= 0.0 or a >= 1.0:
        return 0.0
    log_binom = math.lgamma(N) - math.lgamma(r + 1) - math.lgamma(N - r)
    return -N * math.exp(log_binom + r * math.log(a) + (N - 1 - r) * math.log(1.0 - a))


def _tangent_residual(a: float, N: int) -> float:
    """The defining equation: g_N(a) + (1 - a) * g_N'(a)."""
    return float(g_N(a, N)) + (1.0 - a) * g_N_prime(a, N)


def a_mean(N: int, tol: float = 1e-12) -> float:
    """Unique root of g_N(a) + (1 - a) g_N'(a) = 0 in (0, 1/2).

    Brentq with bracketing endpoints; the residual is positive near 0 (since
    g_N(0) = 1) and negative near 1/2 for odd N (verified by the V14 §4.2 table).
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    lo, hi = 1e-9, 0.5 - 1e-9
    f_lo = _tangent_residual(lo, N)
    f_hi = _tangent_residual(hi, N)
    if f_lo * f_hi > 0:
        raise RuntimeError(
            f"a_mean: residual same sign at endpoints for N={N}: f({lo})={f_lo:.4e}, f({hi})={f_hi:.4e}"
        )
    return brentq(_tangent_residual, lo, hi, args=(N,), xtol=tol)


def v_N_mean(alpha_bar: float, N: int) -> float:
    """Mean-only sup: v_N^mean(alpha_bar) = (1 - alpha_bar) * |g_N'(a_mean(N))|.

    Valid for alpha_bar in (1/2, 1); raises ValueError otherwise.
    """
    if not (0.5 < alpha_bar < 1.0):
        raise ValueError(f"v_N_mean requires alpha_bar in (1/2, 1), got {alpha_bar}")
    a_star = a_mean(N)
    return (1.0 - alpha_bar) * abs(g_N_prime(a_star, N))
