import math
import numpy as np
from scipy.stats import binom


def _R_N_discrete(points, weights, N):
    k = math.floor(N / 2)
    return float(sum(w * binom.cdf(k, N, a) for a, w in zip(points, weights)))


MU1_P = [0.4, 0.8]
MU1_W = [0.5, 0.5]
MU2_P = [0.0, 0.5, 0.7]
MU2_W = [3 / 35, 1 / 5, 5 / 7]


def test_proposition1_means_and_variances():
    for P, W in [(MU1_P, MU1_W), (MU2_P, MU2_W)]:
        p = np.array(P)
        w = np.array(W)
        mean = float((w * p).sum())
        var = float((w * (p - mean) ** 2).sum())
        assert mean == __import__("pytest").approx(0.6, abs=1e-12)
        assert var == __import__("pytest").approx(0.04, abs=1e-12)


def test_proposition1_R3_values():
    assert _R_N_discrete(MU1_P, MU1_W, 3) == __import__("pytest").approx(0.376, abs=1e-9)
    assert _R_N_discrete(MU2_P, MU2_W, 3) == __import__("pytest").approx(0.340, abs=1e-9)
