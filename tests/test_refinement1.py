import numpy as np
import pytest
from scipy.optimize import linprog

from src.certs.refinement1 import C_01, B_N_CH_prime


def _lp_tail_mass(alpha_bar, F, eta, n_grid=2001):
    """sup_{nu in P([0,1])} nu([0, 1/2+eta]) s.t. mean=alpha_bar, var=F."""
    a = np.linspace(0.0, 1.0, n_grid)
    # Closed interval [0, 1/2+eta]; +1e-9 makes boundary inclusion robust to
    # the knife-edge case where 1/2+eta lands exactly on a grid point.
    c = (a <= 0.5 + eta + 1e-9).astype(float)
    A_eq = np.vstack([np.ones_like(a), a, a ** 2])
    b_eq = np.array([1.0, alpha_bar, alpha_bar ** 2 + F])
    res = linprog(c=-c, A_eq=A_eq, b_eq=b_eq, bounds=(0.0, None), method="highs")
    assert res.success, res.message
    return float(-res.fun)


def test_C01_vs_lp_case_A_and_B():
    # 5 Case A (F <= boundary) + 5 Case B (F > boundary) points.
    points = []
    for alpha_bar in (0.60, 0.65, 0.70, 0.75, 0.80):
        m = alpha_bar - 0.5
        eta = m / 2.0
        boundary = (1 - alpha_bar) * (m - eta)
        points.append((alpha_bar, 0.4 * boundary, eta))  # Case A
    for alpha_bar in (0.60, 0.65, 0.70, 0.75, 0.80):
        m = alpha_bar - 0.5
        eta = m / 2.0
        boundary = (1 - alpha_bar) * (m - eta)
        F_B = min(2.0 * boundary, 0.95 * alpha_bar * (1 - alpha_bar))
        assert F_B > boundary
        points.append((alpha_bar, F_B, eta))
    for alpha_bar, F, eta in points:
        lp = _lp_tail_mass(alpha_bar, F, eta)
        cf = C_01(alpha_bar, F, eta)
        assert abs(cf - lp) < 1e-4, (alpha_bar, F, eta, cf, lp)


def test_C01_continuity_at_boundary():
    for alpha_bar in (0.58, 0.63, 0.70, 0.77, 0.83):
        m = alpha_bar - 0.5
        eta = m / 3.0
        Fstar = (1 - alpha_bar) * (m - eta)
        case_a = Fstar / (Fstar + (m - eta) ** 2)
        case_b = (1 - alpha_bar) + (alpha_bar * (1 - alpha_bar) - Fstar) / (0.5 - eta)
        target = (1 - alpha_bar) / (0.5 - eta)
        assert abs(case_a - target) < 1e-10
        assert abs(case_b - target) < 1e-10
        assert abs(C_01(alpha_bar, Fstar, eta) - target) < 1e-10


def test_B_N_CH_prime_range_and_vacuity():
    assert 0.0 <= B_N_CH_prime(0.7, 0.05, 15) <= 1.0
    assert B_N_CH_prime(0.5, 0.0, 15) == 1.0
