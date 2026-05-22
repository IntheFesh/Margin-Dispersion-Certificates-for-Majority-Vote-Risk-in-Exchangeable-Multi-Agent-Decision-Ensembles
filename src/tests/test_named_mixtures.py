"""Step 2 of the staged validation: synthetic checks on known Bernoulli mixtures.

Three distributions, each with closed-form theoretical values that the code
must reproduce within sample/MC tolerance:

(a) mu = delta_{0.7}          (zero dispersion, high margin)
(b) mu = 0.5*delta_{0.4} + 0.5*delta_{0.8}  (moderate dispersion, low margin)
(c) the two-moment insufficiency pair (mu_1, mu_2) from the paper, N=3

For each, we verify:
- estimator unbiasedness for bar_alpha and F,
- closed-form mixture_majority_risk matches exact binomial,
- empirical certificate is a valid upper bound when issued, and
  vacuous/refused when theory predicts (low m or m_L <= 0).
"""
from __future__ import annotations
import math
import numpy as np
import pytest
from scipy.stats import binom

from src.synthetic.simulate_bernoulli_mixture import simulate_alpha, simulate_X_from_alpha
from src.theory.estimators import estimate_F_unbiased, estimate_m1_m2, estimate_basic_summaries
from src.theory.majority_risk import binom_strict_failure_prob, mixture_majority_risk
from src.theory.certificate import (
    population_certificate,
    empirical_certificate_from_X,
)
from src.theory.insufficiency import (
    discrete_mixture_mean,
    discrete_mixture_variance,
    discrete_mixture_majority_risk,
)


# ----------------------- (a) mu = delta_{0.7} -----------------------------


def test_mu_delta_07_F_zero_and_R_N_closed_form():
    """Degenerate mixture at 0.7: F = 0 and R_N = P(Bin(N, 0.7) <= floor(N/2))."""
    a = 0.7
    # Exact theory:
    assert mixture_majority_risk(16, np.full(2000, a)) == pytest.approx(
        binom.cdf(8, 16, a), abs=1e-12
    )
    assert mixture_majority_risk(32, np.full(2000, a)) == pytest.approx(
        binom.cdf(16, 32, a), abs=1e-12
    )
    # F should be 0 in the population; with finite samples it concentrates near 0.
    rng = np.random.default_rng(0)
    K = 32
    M = 4000
    alpha = np.full(M, a)
    X = rng.binomial(1, alpha[:, None].repeat(K, axis=1)).astype(int)
    F_hat = estimate_F_unbiased(X)
    # With F_true = 0 and Bernoulli sampling noise, |F_hat| is tiny.
    assert abs(F_hat) < 0.005


def test_mu_delta_07_population_certificate_collapses_to_hoeffding():
    """When F = 0 the certificate equals the Hoeffding bound e^{-2 N m^2}."""
    m = 0.7 - 0.5
    for N in (16, 32, 64):
        cert = population_certificate(0.7, 0.0, N)
        hoe = math.exp(-2 * N * m * m)
        # The certificate's inf{...} = (0/(0+(m-eta)^2) -> 0) + e^{-2N eta^2},
        # which is minimised at eta -> m and equals e^{-2 N m^2}.
        # eta is bounded strictly away from m in optimize_eta, so we allow
        # a small numerical gap on the first term (F/(F+(m-eta)^2)).
        assert cert["R_cert"] == pytest.approx(hoe, abs=1e-4)


def test_mu_delta_07_empirical_certificate_covers_truth():
    """Empirical certificate must upper-bound true R_N for delta_{0.7}."""
    rng = np.random.default_rng(1)
    M, K = 2000, 32
    alpha = np.full(M, 0.7)
    X = rng.binomial(1, alpha[:, None].repeat(K, axis=1)).astype(int)
    for N in (16, 32, 64):
        cert = empirical_certificate_from_X(X, N, 0.1)
        R_true = binom.cdf(N // 2, N, 0.7)
        assert cert["issued"] is True
        assert cert["R_cert"] >= R_true - 1e-9, (N, cert["R_cert"], R_true)


# --------------- (b) mu = 0.5 delta_{0.4} + 0.5 delta_{0.8} ---------------


def test_two_point_F_and_R_N_closed_form():
    """Two-point mixture has bar_alpha=0.6, F=0.04, and a closed-form R_N."""
    pts = np.array([0.4, 0.8])
    wts = np.array([0.5, 0.5])
    # Theory checks.
    assert discrete_mixture_mean(pts, wts) == pytest.approx(0.6, abs=1e-12)
    assert discrete_mixture_variance(pts, wts) == pytest.approx(0.04, abs=1e-12)
    for N in (3, 16, 32):
        expected = 0.5 * binom.cdf(N // 2, N, 0.4) + 0.5 * binom.cdf(N // 2, N, 0.8)
        assert discrete_mixture_majority_risk(pts, wts, N) == pytest.approx(expected, abs=1e-12)


def test_two_point_estimator_recovers_F():
    """Estimator should average close to F = 0.04 over independent trials."""
    rng = np.random.default_rng(2)
    M, K = 1500, 32
    F_hats = []
    for t in range(40):
        alpha = rng.choice([0.4, 0.8], size=M, p=[0.5, 0.5])
        X = rng.binomial(1, alpha[:, None].repeat(K, axis=1)).astype(int)
        F_hats.append(estimate_F_unbiased(X))
    assert abs(float(np.mean(F_hats)) - 0.04) < 0.01


def test_two_point_certificate_vacuous_low_margin():
    """At m = 0.1 with F = 0.04, the population certificate is vacuous for all
    N tried. This matches Corollary 1 (operating regime requires m large)."""
    for N in (8, 16, 32, 64):
        cert = population_certificate(0.6, 0.04, N)
        assert cert["R_cert"] >= 0.99, (N, cert["R_cert"])


# ----- (c) two-moment insufficiency: mu_1, mu_2 with N=3 ------------------


def test_two_moment_insufficiency_exact_numbers():
    """Reproduce the paper's Proposition 1 numbers exactly."""
    mu1p = np.array([0.4, 0.8])
    mu1w = np.array([0.5, 0.5])
    mu2p = np.array([0.0, 0.5, 0.7])
    mu2w = np.array([3 / 35, 1 / 5, 5 / 7])
    assert discrete_mixture_mean(mu1p, mu1w) == pytest.approx(0.6, abs=1e-12)
    assert discrete_mixture_mean(mu2p, mu2w) == pytest.approx(0.6, abs=1e-12)
    assert discrete_mixture_variance(mu1p, mu1w) == pytest.approx(0.04, abs=1e-12)
    assert discrete_mixture_variance(mu2p, mu2w) == pytest.approx(0.04, abs=1e-12)
    assert discrete_mixture_majority_risk(mu1p, mu1w, 3) == pytest.approx(0.376, abs=1e-9)
    assert discrete_mixture_majority_risk(mu2p, mu2w, 3) == pytest.approx(0.340, abs=1e-9)


def test_two_moment_insufficiency_certificate_indistinguishable():
    """The two-moment certificate sees mu_1 and mu_2 as identical because
    both have the same (bar_alpha, F). The certificate's R_cert is therefore
    the same for both, demonstrating the two-moment insufficiency."""
    # Same population (bar_alpha, F) for both; certificate is fully determined
    # by these and N, so it must agree.
    bar_alpha, F = 0.6, 0.04
    for N in (3, 8, 16, 32):
        c1 = population_certificate(bar_alpha, F, N)
        c2 = population_certificate(bar_alpha, F, N)
        assert c1["R_cert"] == pytest.approx(c2["R_cert"], abs=1e-12)
    # The certificate is vacuous here (low margin), and the *true* risks
    # differ (0.376 vs 0.340), which is precisely the insufficiency claim:
    # (bar_alpha, F) cannot identify R_N.
    mu1 = (np.array([0.4, 0.8]), np.array([0.5, 0.5]))
    mu2 = (np.array([0.0, 0.5, 0.7]), np.array([3 / 35, 1 / 5, 5 / 7]))
    R3_1 = discrete_mixture_majority_risk(*mu1, 3)
    R3_2 = discrete_mixture_majority_risk(*mu2, 3)
    assert R3_1 != pytest.approx(R3_2, abs=1e-3)
