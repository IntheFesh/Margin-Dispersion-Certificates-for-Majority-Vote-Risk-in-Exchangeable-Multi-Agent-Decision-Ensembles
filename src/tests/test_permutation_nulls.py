import numpy as np
import pytest
from src.panel_b.permutation_nulls import (
    permutation_null,
    family_aware_permutation_null,
    quantile_at_observed,
)


def test_permutation_null_shape():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    A = (A + A.T) / 2
    B = rng.normal(size=(6, 6))
    B = (B + B.T) / 2
    null = permutation_null(A, B, n_perm=50, seed=1, method="spearman")
    assert null.shape == (50,)


def test_family_aware_permutation_null_shape():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 8))
    A = (A + A.T) / 2
    B = rng.normal(size=(8, 8))
    B = (B + B.T) / 2
    fams = ["x"] * 4 + ["y"] * 4
    null = family_aware_permutation_null(A, B, fams, n_perm=30, seed=2)
    assert null.shape == (30,)


def test_quantile_at_observed():
    null = np.linspace(0, 1, 11)
    p = quantile_at_observed(0.5, null)
    assert 0.4 < p < 0.7


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        permutation_null(np.zeros((3, 3)), np.zeros((4, 4)), n_perm=2, seed=0)
