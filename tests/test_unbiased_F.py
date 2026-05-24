import numpy as np
import pytest

from src.certs.moments import unbiased_F_hat


def test_unbiased_F_hat_beta_mixture():
    rng = np.random.default_rng(7)
    M, K = 2000, 32
    beta_var = 12 * 8 / ((20 ** 2) * 21)  # Var of Beta(12,8) = 0.0114286
    F_hats = []
    biased = []
    for t in range(100):
        alpha = rng.beta(12, 8, size=M)
        X = (rng.random((M, K)) < alpha[:, None]).astype(int)
        _, F_hat = unbiased_F_hat(X)
        F_hats.append(F_hat)
        Z = X.mean(axis=1)
        biased.append(float((Z ** 2).mean() - Z.mean() ** 2))
    mean_unbiased = float(np.mean(F_hats))
    mean_biased = float(np.mean(biased))
    # Unbiased estimator: within +/- 1e-3 of the population variance.
    assert abs(mean_unbiased - beta_var) < 1e-3
    # Biased plug-in trips the sentinel (bias ~ E[a(1-a)]/K >> 1e-3).
    assert abs(mean_biased - beta_var) > 1e-3


def test_unbiased_F_hat_validation():
    with pytest.raises(ValueError):
        unbiased_F_hat(np.zeros((1, 4), dtype=int))  # M < 2
    with pytest.raises(ValueError):
        unbiased_F_hat(np.zeros((4, 1), dtype=int))  # K < 2
    with pytest.raises(ValueError):
        unbiased_F_hat(np.full((4, 4), 2, dtype=int))  # non-binary
