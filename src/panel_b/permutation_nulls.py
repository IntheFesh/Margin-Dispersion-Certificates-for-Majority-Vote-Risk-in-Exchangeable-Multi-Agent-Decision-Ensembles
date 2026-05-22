from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr


def _upper_tri_corr(A: np.ndarray, B: np.ndarray, method: str) -> float:
    iu = np.triu_indices(A.shape[0], 1)
    x, y = A[iu], B[iu]
    if method == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    if method == "spearman":
        return float(spearmanr(x, y).correlation)
    raise ValueError(f"unsupported correlation method: {method}")


def permutation_null(
    observed: np.ndarray,
    comparator: np.ndarray,
    n_perm: int,
    seed: int,
    method: str = "spearman",
) -> np.ndarray:
    """Row/column permutation null on the comparator matrix.

    Both matrices must be square and the same size. Permutation is applied
    jointly to rows and columns so that symmetry is preserved.
    """
    observed = np.asarray(observed, dtype=float)
    comparator = np.asarray(comparator, dtype=float)
    if observed.shape != comparator.shape:
        raise ValueError("observed/comparator shape mismatch")
    if observed.ndim != 2 or observed.shape[0] != observed.shape[1]:
        raise ValueError("expected square matrices")
    rng = np.random.default_rng(seed)
    m = observed.shape[0]
    vals = np.empty(n_perm, dtype=float)
    for k in range(n_perm):
        p = rng.permutation(m)
        comp_p = comparator[p][:, p]
        vals[k] = _upper_tri_corr(observed, comp_p, method)
    return vals


def family_aware_permutation_null(
    observed: np.ndarray,
    comparator: np.ndarray,
    families: list[str],
    n_perm: int,
    seed: int,
    method: str = "spearman",
) -> np.ndarray:
    """Permutation null where labels are permuted only within family blocks."""
    observed = np.asarray(observed, dtype=float)
    comparator = np.asarray(comparator, dtype=float)
    if observed.shape != comparator.shape:
        raise ValueError("observed/comparator shape mismatch")
    fam = np.asarray(families)
    m = observed.shape[0]
    if len(fam) != m:
        raise ValueError("families length must match matrix size")
    rng = np.random.default_rng(seed)
    vals = np.empty(n_perm, dtype=float)
    idx = np.arange(m)
    for k in range(n_perm):
        p = idx.copy()
        for f in np.unique(fam):
            loc = np.where(fam == f)[0]
            p[loc] = rng.permutation(loc)
        comp_p = comparator[p][:, p]
        vals[k] = _upper_tri_corr(observed, comp_p, method)
    return vals


def quantile_at_observed(observed_stat: float, null_dist: np.ndarray) -> float:
    """Return the (right-tail) fraction of null values >= observed_stat."""
    null_dist = np.asarray(null_dist, dtype=float)
    if null_dist.size == 0:
        raise ValueError("null distribution is empty")
    return float((null_dist >= observed_stat).mean())
