from __future__ import annotations
import numpy as np


def split_estimation_reference(X: np.ndarray, K_est: int, seed: int) -> dict:
    """Split a binary correctness matrix [M, K_ref] into disjoint estimation
    and reference column subsets and assert disjointness.

    Returns a dict with the two submatrices and the integer column indices
    used. Raises ValueError on shape or disjointness issues.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D [M, K]")
    M, K = X.shape
    if not isinstance(K_est, (int, np.integer)) or K_est <= 0:
        raise ValueError("K_est must be a positive integer")
    if K_est >= K:
        raise ValueError(f"K_est ({K_est}) must be strictly less than K ({K})")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(K)
    est_idx = np.sort(perm[:K_est])
    ref_idx = np.sort(perm[K_est:])
    overlap = np.intersect1d(est_idx, ref_idx)
    if overlap.size != 0:
        raise ValueError("estimation and reference index sets must be disjoint")
    X_est = X[:, est_idx]
    X_ref = X[:, ref_idx]
    return {
        "X_est": X_est,
        "X_ref": X_ref,
        "est_idx": est_idx,
        "ref_idx": ref_idx,
        "K_est": int(K_est),
        "K_ref": int(K - K_est),
    }


def assert_disjoint_instance_sets(set_a: set, set_b: set, name_a: str = "A", name_b: str = "B") -> None:
    """Validate that two instance_id sets are disjoint or raise ValueError."""
    inter = set_a & set_b
    if inter:
        raise ValueError(
            f"data leakage: {name_a} and {name_b} share {len(inter)} instance ids; sample: {list(inter)[:5]}"
        )
