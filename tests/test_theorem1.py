import math
import numpy as np
import pytest

from src.certs import _optimize
from src.certs._optimize import minimize_eta
from src.certs.theorem1 import B_N_CH, ch_objective


def _dense_min(alpha_bar, F, N):
    m = alpha_bar - 0.5
    grid = np.linspace(1e-7, m - 1e-7, 2000)
    vals = [ch_objective(float(e), alpha_bar, F, N) for e in grid]
    return min(1.0, float(min(vals)))


def test_dense_vs_scipy_consistency():
    for alpha_bar, F, N in [(0.7, 0.05, 15), (0.6, 0.02, 7), (0.8, 0.1, 31)]:
        scipy_val = B_N_CH(alpha_bar, F, N)
        dense_val = _dense_min(alpha_bar, F, N)
        assert abs(scipy_val - dense_val) < 1e-4


def test_hoeffding_limit_F_zero():
    # At F = 0 the Cantelli term vanishes and B_N^CH -> exp(-2 N m^2).
    for alpha_bar, N in [(0.7, 15), (0.65, 31), (0.8, 7)]:
        m = alpha_bar - 0.5
        assert B_N_CH(alpha_bar, 0.0, N) == pytest.approx(math.exp(-2 * N * m * m), abs=1e-4)


def test_monotone_in_F():
    vals = [B_N_CH(0.7, F, 15) for F in (0.0, 0.01, 0.05, 0.1, 0.2)]
    assert all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1))


def test_monotone_in_N():
    vals = [B_N_CH(0.7, 0.05, N) for N in (3, 7, 15, 31, 63)]
    assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1))


def test_vacuity_when_no_margin():
    assert B_N_CH(0.5, 0.0, 15) == 1.0
    assert B_N_CH(0.45, 0.0, 15) == 1.0


def test_range_and_validation():
    assert 0.0 <= B_N_CH(0.7, 0.05, 15) <= 1.0
    with pytest.raises(ValueError):
        B_N_CH(0.7, 0.5, 15)  # F exceeds alpha_bar(1-alpha_bar)=0.21
    with pytest.raises(ValueError):
        B_N_CH(0.7, 0.05, 0)  # N must be >= 1


def test_dense_grid_fallback_exercised(monkeypatch):
    # Force scipy to fail and confirm the dense-grid fallback is used.
    def boom(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(_optimize, "minimize_scalar", boom)
    res = minimize_eta(lambda e: ch_objective(e, 0.7, 0.05, 15), 0.2)
    assert res["method"] == "dense-grid-fallback"
    assert 0.0 <= res["value"] <= 1.0
    assert 0.0 < res["eta_star"] < 0.2


def test_scipy_path_default():
    res = minimize_eta(lambda e: ch_objective(e, 0.7, 0.05, 15), 0.2)
    assert res["method"] == "scipy-bounded"
