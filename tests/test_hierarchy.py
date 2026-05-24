import math
import numpy as np
import pandas as pd
from scipy.stats import binom

from src.certs.theorem1 import B_N_CH
from src.certs.refinement1 import B_N_CH_prime
from src.certs.theorem3 import B_N_star
from src.certs.verify import check_hierarchy, max_hierarchy_gap


def _snap(a, n_grid=2001):
    step = 1.0 / (n_grid - 1)
    return round(a / step) * step


def test_hierarchy_on_random_points():
    rng = np.random.default_rng(12)
    rows = []
    for _ in range(20):
        alpha_bar = _snap(float(rng.uniform(0.55, 0.85)))
        F = float(rng.uniform(0.0, 0.5 * alpha_bar * (1 - alpha_bar)))
        N = int(rng.choice([3, 7, 15, 31, 63]))
        b_ch = B_N_CH(alpha_bar, F, N)
        b_chp = B_N_CH_prime(alpha_bar, F, N)
        b_star = B_N_star(alpha_bar, F, N)
        rows.append({"alpha_bar": alpha_bar, "F": F, "N": N, "B_CH": b_ch, "B_CH_prime": b_chp, "B_star": b_star})
    df = pd.DataFrame(rows)
    check_hierarchy(df, tol=1e-6)  # raises on violation
    assert max_hierarchy_gap(df) <= 1e-6


def test_B_star_dominates_proposition1_mixtures():
    # Both mixtures have mean 0.6, var 0.04; B_3^star must dominate their R_3.
    b_star = B_N_star(0.6, 0.04, 3)
    k = math.floor(3 / 2)
    R3_mu1 = 0.5 * binom.cdf(k, 3, 0.4) + 0.5 * binom.cdf(k, 3, 0.8)
    R3_mu2 = (3 / 35) * binom.cdf(k, 3, 0.0) + (1 / 5) * binom.cdf(k, 3, 0.5) + (5 / 7) * binom.cdf(k, 3, 0.7)
    assert b_star >= R3_mu1 - 1e-6
    assert b_star >= R3_mu2 - 1e-6
