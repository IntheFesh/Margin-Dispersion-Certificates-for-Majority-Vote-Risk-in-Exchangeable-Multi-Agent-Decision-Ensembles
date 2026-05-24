import numpy as np
import pytest

from src.certs.empirical import empirical_certificate, bonferroni_cell_budget
from src.certs.refusal import classify_refusal


def _make_X(M, K, alpha, seed=0):
    rng = np.random.default_rng(seed)
    a = np.full(M, alpha)
    return (rng.random((M, K)) < a[:, None]).astype(int)


def test_issued_when_margin_positive():
    X = _make_X(500, 256, 0.9, seed=1)
    dc = bonferroni_cell_budget(0.05, 6)
    out = empirical_certificate(X, N=63, delta_cell=dc, use_BA=True)
    assert out["m_L"] > 0
    assert out["R_N_cert"] is not None
    assert 0.0 <= out["R_N_cert"] <= 1.0
    assert out["R_N_cert_eta_star"] is not None
    assert out["R_N_cert_dominating_term"] in ("cantelli", "hoeffding")
    assert out["R_N_BA_cert"] is not None
    # The boundedness-aware certificate exploits support [0,1], so it is at
    # least as tight as (<=) the plain Cantelli-based certificate.
    assert out["R_N_BA_cert"] <= out["R_N_cert"] + 1e-9


def test_refusal_when_no_margin():
    X = _make_X(200, 64, 0.5, seed=2)  # alpha ~ 0.5 -> m_L <= 0
    dc = bonferroni_cell_budget(0.05, 6)
    out = empirical_certificate(X, N=15, delta_cell=dc)
    assert out["R_N_cert"] is None
    cls = classify_refusal(out, epsilon=0.3)
    assert cls["mode"] in ("no_direction", "failure_margin", "success_margin")


def test_failure_side_certificate_for_weak_model():
    X = _make_X(500, 256, 0.1, seed=3)  # weak model -> failure-side margin
    dc = bonferroni_cell_budget(0.05, 6)
    out = empirical_certificate(X, N=63, delta_cell=dc)
    assert out["m_beta_L"] > 0
    assert out["Q_N_cert"] is not None


def test_U_F_clipped_range():
    X = _make_X(100, 32, 0.8, seed=4)
    dc = bonferroni_cell_budget(0.05, 12)
    out = empirical_certificate(X, N=31, delta_cell=dc)
    assert 0.0 <= out["U_F"] <= 0.25


def test_estimation_only_input_shape():
    with pytest.raises(ValueError):
        empirical_certificate(np.zeros(10), N=15, delta_cell=0.01)
