import numpy as np
import pytest
from src.panel_a.split_reference_estimation import (
    split_estimation_reference,
    assert_disjoint_instance_sets,
)


def test_split_disjoint_and_shapes():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(50, 16))
    out = split_estimation_reference(X, K_est=4, seed=1)
    assert out["X_est"].shape == (50, 4)
    assert out["X_ref"].shape == (50, 12)
    assert set(out["est_idx"]).isdisjoint(set(out["ref_idx"]))
    assert len(set(out["est_idx"])) == 4
    assert len(set(out["ref_idx"])) == 12


def test_split_invalid_K_est_raises():
    X = np.zeros((10, 4), int)
    with pytest.raises(ValueError):
        split_estimation_reference(X, K_est=4, seed=0)
    with pytest.raises(ValueError):
        split_estimation_reference(X, K_est=0, seed=0)


def test_split_invalid_X_shape_raises():
    with pytest.raises(ValueError):
        split_estimation_reference(np.zeros(10, int), K_est=2, seed=0)


def test_disjoint_instance_sets_raises():
    with pytest.raises(ValueError):
        assert_disjoint_instance_sets({"a", "b"}, {"b", "c"})


def test_disjoint_instance_sets_ok():
    assert_disjoint_instance_sets({"a", "b"}, {"c", "d"}) is None
