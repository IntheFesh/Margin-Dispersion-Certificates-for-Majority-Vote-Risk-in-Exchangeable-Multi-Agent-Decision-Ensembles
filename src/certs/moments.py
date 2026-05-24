"""Unbiased two-stage Bernoulli-mixture dispersion estimator F_hat.

Given M instances each with K_full i.i.d. samples given alpha_m, the
per-instance success rate is Z_m. The unbiased second-moment estimator
uses the per-instance U-statistic

    U_m^(2) = Z_m (K Z_m - 1) / (K - 1),    E[U_m^(2) | alpha_m] = alpha_m^2,

equivalently  Z_m^2 - Z_m(1-Z_m)/(K-1). Then

    F_hat = (1/M) sum_m U_m^(2) - alpha_bar_hat^2.

The biased plug-in  (1/M) sum_m Z_m^2 - alpha_bar_hat^2  is FORBIDDEN in the
certificate path (it carries the per-instance binomial variance
contribution E[alpha(1-alpha)]/K).
"""
from __future__ import annotations
import numpy as np


def unbiased_F_hat(successes: np.ndarray) -> tuple[float, float]:
    """Return (alpha_bar_hat, F_hat) with the per-instance binomial-variance
    correction. Raises ValueError if K_full < 2 or M < 2 or non-binary."""
    X = np.asarray(successes)
    if X.ndim != 2:
        raise ValueError("successes must be a 2-D (M, K) array")
    M, K = X.shape
    if M < 2:
        raise ValueError("M must be >= 2 for F_hat")
    if K < 2:
        raise ValueError("K_full must be >= 2 for the unbiased correction")
    uniq = np.unique(X)
    if not np.all(np.isin(uniq, (0, 1))):
        raise ValueError("successes must be binary {0,1}")
    Z = X.mean(axis=1)
    alpha_hat = float(Z.mean())
    U2 = Z * (K * Z - 1.0) / (K - 1.0)
    F_hat = float(U2.mean() - alpha_hat ** 2)
    return alpha_hat, F_hat


def second_moment_hat(successes: np.ndarray) -> float:
    """Unbiased estimator of E[alpha^2] = mean_m U_m^(2)."""
    X = np.asarray(successes, dtype=float)
    M, K = X.shape
    if K < 2:
        raise ValueError("K_full must be >= 2")
    Z = X.mean(axis=1)
    return float((Z * (K * Z - 1.0) / (K - 1.0)).mean())
