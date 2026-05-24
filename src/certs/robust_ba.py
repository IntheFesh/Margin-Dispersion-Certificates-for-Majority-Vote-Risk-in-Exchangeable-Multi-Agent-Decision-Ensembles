"""Robust boundedness-aware bound C_rob and the secondary certificate term.

    C_rob(L_alpha, U_F, eta) = sup_{(a,f) in A_delta} C_[0,1](a, f, eta),
    A_delta = { (a,f): L_alpha <= a <= 1, 0 <= f <= min(U_F, a(1-a)) }.

The supremum is computed by a 51x51 grid plus multi-start
scipy.optimize.minimize refinement (minimizing -C_01).
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import minimize
from .refinement1 import C_01

_GRID = 51


def C_rob(L_alpha: float, U_F: float, eta: float) -> float:
    """Supremum of C_01 over the confidence box A_delta. Requires
    L_alpha > 1/2 (the certificate-issued event)."""
    if L_alpha <= 0.5:
        raise ValueError("C_rob requires L_alpha > 1/2 (issued event)")
    if not (0.0 < eta < (L_alpha - 0.5) + 1e-15):
        raise ValueError("eta must satisfy 0 < eta < m_L = L_alpha - 1/2")
    U_F = min(0.25, max(0.0, U_F))

    a_grid = np.linspace(L_alpha, 1.0, _GRID)
    best = -math.inf
    best_a, best_t = L_alpha, 0.0
    for a in a_grid:
        f_cap = min(U_F, a * (1.0 - a))
        if f_cap < 0:
            continue
        f_grid = np.linspace(0.0, f_cap, _GRID)
        for f in f_grid:
            v = C_01(float(a), float(f), eta)
            if v > best:
                best = v
                best_a = float(a)
                best_t = (float(f) / f_cap) if f_cap > 0 else 0.0

    # Multi-start scipy refinement in (a, t) with f = t * min(U_F, a(1-a)).
    def neg_obj(x: np.ndarray) -> float:
        a = min(1.0, max(L_alpha, x[0]))
        t = min(1.0, max(0.0, x[1]))
        f_cap = min(U_F, a * (1.0 - a))
        f = t * f_cap
        return -C_01(a, f, eta)

    starts = [
        np.array([best_a, best_t]),
        np.array([L_alpha, 1.0]),
        np.array([(L_alpha + 1.0) / 2.0, 0.5]),
    ]
    for x0 in starts:
        try:
            res = minimize(
                neg_obj,
                x0,
                method="L-BFGS-B",
                bounds=[(L_alpha, 1.0), (0.0, 1.0)],
            )
            if res.success:
                cand = -float(res.fun)
                if cand > best:
                    best = cand
        except (ValueError, RuntimeError):
            # Grid result already provides a valid lower estimate of the sup.
            continue
    return float(min(1.0, max(0.0, best)))


def robust_ba_objective(eta: float, L_alpha: float, U_F: float, N: int) -> float:
    return C_rob(L_alpha, U_F, eta) + math.exp(-2.0 * N * eta ** 2)
