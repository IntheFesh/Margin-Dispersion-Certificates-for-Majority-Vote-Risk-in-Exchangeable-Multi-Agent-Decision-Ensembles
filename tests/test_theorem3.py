import pytest
from src.certs.theorem3 import B_N_star, g_N
from scipy.stats import binom


POINTS = [(0.6, 0.04, 3), (0.7, 0.05, 7), (0.55, 0.02, 15), (0.65, 0.06, 31), (0.75, 0.08, 63)]


def test_strong_duality():
    for alpha_bar, F, N in POINTS:
        primal, dual = B_N_star(alpha_bar, F, N, return_dual=True)
        assert abs(primal - dual) < 1e-8, (alpha_bar, F, N, primal, dual)


def test_g_N_matches_binomial_cdf():
    import math
    for N in (3, 7, 15):
        for a in (0.3, 0.5, 0.7):
            assert g_N(a, N) == pytest.approx(binom.cdf(math.floor(N / 2), N, a), abs=1e-12)


def test_B_N_star_in_range():
    for alpha_bar, F, N in POINTS:
        v = B_N_star(alpha_bar, F, N)
        assert 0.0 <= v <= 1.0


def test_B_N_star_validation():
    with pytest.raises(ValueError):
        B_N_star(0.6, 0.5, 3)  # F exceeds 0.24


def test_B_N_star_off_grid_alpha_and_zero_F():
    # Empirical alpha_bar is generally off the uniform grid; F_hat can be ~0.
    # B_N_star must not raise and at F=0 equals the point-mass value g_N(a).
    import math
    a = 0.821703125  # deliberately off the 1/2000 grid
    v = B_N_star(a, 0.0, 3)
    assert v == pytest.approx(binom.cdf(math.floor(3 / 2), 3, a), abs=1e-9)
    # Tiny positive F off-grid must also be feasible (no LP infeasibility).
    v2 = B_N_star(a, 1e-4, 7)
    assert 0.0 <= v2 <= 1.0
