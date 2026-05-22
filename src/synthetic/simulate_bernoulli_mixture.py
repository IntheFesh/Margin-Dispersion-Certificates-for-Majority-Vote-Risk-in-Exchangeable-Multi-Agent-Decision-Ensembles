from __future__ import annotations
import numpy as np


def simulate_alpha(M: int, distribution: dict, seed: int) -> np.ndarray:
    if M <= 0:
        raise ValueError("M must be positive")
    rng = np.random.default_rng(seed)
    kind = distribution.get("name")
    if kind == "beta":
        a, b = float(distribution["a"]), float(distribution["b"])
        alpha = rng.beta(a, b, size=M)
    elif kind == "two_point":
        p = float(distribution["p"])
        lo, hi = float(distribution["a_low"]), float(distribution["a_high"])
        alpha = np.where(rng.random(M) < p, lo, hi)
    elif kind == "three_point":
        w = np.asarray(distribution["weights"], dtype=float)
        pts = np.asarray(distribution["points"], dtype=float)
        if len(w) != 3 or len(pts) != 3:
            raise ValueError("three_point needs 3 weights and 3 points")
        if not np.isclose(w.sum(), 1.0):
            raise ValueError("weights must sum to 1")
        alpha = rng.choice(pts, size=M, p=w)
    elif kind == "uniform_low_high":
        lo, hi = float(distribution["low"]), float(distribution["high"])
        alpha = rng.uniform(lo, hi, size=M)
    else:
        raise ValueError(f"Unsupported distribution: {kind}")
    if np.any((alpha < 0) | (alpha > 1)):
        raise ValueError("alpha values must be in [0,1]")
    return alpha


def simulate_X_from_alpha(alpha: np.ndarray, K: int, seed: int) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=float)
    if alpha.ndim != 1:
        raise ValueError("alpha must be 1D")
    if K < 1:
        raise ValueError("K must be >= 1")
    if np.any((alpha < 0) | (alpha > 1)):
        raise ValueError("alpha values must be in [0,1]")
    rng = np.random.default_rng(seed)
    probs = np.repeat(alpha[:, None], K, axis=1)
    return rng.binomial(1, probs).astype(int)
