"""Theorem 3: sharp two-moment envelope B_N^star via discretized moment LP.

    B_N^star = sup_{nu in P([0,1])} { int g_N dnu :
                  int a dnu = alpha_bar, int a^2 dnu = alpha_bar^2 + F }

where g_N(a) = P(Bin(N,a) <= floor(N/2)). The primal is a moment LP on a
uniform grid; the dual is the quadratic-majorant problem. Strong duality
holds for the discretized LP, so |primal - dual| is a numerical-bug
detector (checked in verify.py, not part of the proof).
"""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import binom
from scipy.optimize import linprog

_F_SLACK = 1e-12


def g_N(a: np.ndarray | float, N: int) -> np.ndarray | float:
    """g_N(a) = P(Bin(N,a) <= floor(N/2)) = strict success-majority failure."""
    return binom.cdf(math.floor(N / 2), N, a)


def _validate_moments(alpha_bar: float, F: float) -> None:
    if not (0.0 <= alpha_bar <= 1.0):
        raise ValueError(f"alpha_bar must be in [0,1], got {alpha_bar}")
    upper = alpha_bar * (1.0 - alpha_bar) + _F_SLACK
    if not (0.0 <= F <= upper):
        raise ValueError(
            f"F must be in [0, alpha_bar*(1-alpha_bar)]={upper - _F_SLACK:.6g}, got {F}"
        )


def B_N_star(
    alpha_bar: float,
    F: float,
    N: int,
    n_grid: int = 2001,
    return_dual: bool = False,
) -> float | tuple[float, float]:
    """Sharp two-moment envelope via discretized moment LP.

    Primal: max int g_N dnu s.t. moment constraints, nu in M+([0,1]).
    Discretized on a uniform grid of n_grid points in [0,1] via
    scipy.optimize.linprog(method='highs').

    If return_dual=True, also computes the dual LP (quadratic majorant)
    and returns (primal, dual). The caller in verify.py must check
    |primal - dual| < 1e-8 and abort the pipeline if violated.
    """
    _validate_moments(alpha_bar, F)
    if N < 1:
        raise ValueError(f"N must be a positive integer, got {N}")
    if n_grid < 3:
        raise ValueError("n_grid must be >= 3")

    # Degenerate F = 0: the only distribution with mean alpha_bar and zero
    # variance is the point mass delta_{alpha_bar}, so the envelope is exactly
    # g_N(alpha_bar). Handling it here avoids LP infeasibility when alpha_bar
    # is not on the uniform grid (the common case for empirical alpha_bar_hat).
    if F <= 1e-12:
        val = float(min(1.0, max(0.0, float(g_N(alpha_bar, N)))))
        return (val, val) if return_dual else val

    # Insert alpha_bar as an explicit grid node so the mean constraint is
    # exactly representable and the LP stays feasible for any 0 < F <=
    # alpha_bar(1-alpha_bar), even when alpha_bar is off the uniform grid.
    a = np.union1d(np.linspace(0.0, 1.0, n_grid), np.array([float(alpha_bar)]))
    g = np.asarray(g_N(a, N), dtype=float)
    m2 = alpha_bar ** 2 + F

    # Primal: max g^T nu s.t. [1; a; a^2] nu = [1; alpha_bar; m2], nu >= 0.
    A_eq = np.vstack([np.ones_like(a), a, a ** 2])
    b_eq = np.array([1.0, alpha_bar, m2])
    res_p = linprog(c=-g, A_eq=A_eq, b_eq=b_eq, bounds=(0.0, None), method="highs")
    if not res_p.success:
        raise RuntimeError(
            f"B_N_star primal LP failed (alpha_bar={alpha_bar}, F={F}, N={N}): {res_p.message}"
        )
    primal = float(-res_p.fun)
    primal = float(min(1.0, max(0.0, primal)))

    if not return_dual:
        return primal

    # Dual: min b^T y s.t. y0 + a y1 + a^2 y2 >= g(a) for all grid a, y free.
    # linprog form: min f^T y s.t. A_ub y <= b_ub.
    f = b_eq
    A_ub = -A_eq.T  # shape (n_grid, 3): -[1, a_i, a_i^2]
    b_ub = -g
    res_d = linprog(c=f, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * 3, method="highs")
    if not res_d.success:
        raise RuntimeError(
            f"B_N_star dual LP failed (alpha_bar={alpha_bar}, F={F}, N={N}): {res_d.message}"
        )
    dual = float(res_d.fun)
    return primal, dual
