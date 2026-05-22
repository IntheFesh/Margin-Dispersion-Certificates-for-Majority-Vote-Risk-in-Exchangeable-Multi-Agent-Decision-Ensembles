import numpy as np
import pytest
from src.panel_b.cka import linear_cka, compute_pairwise_cka, assert_probe_correctness_disjoint, cka_layer_sweep


def test_cka_self_is_one():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 8))
    assert abs(linear_cka(X, X) - 1.0) < 1e-9


def test_cka_orthogonal_low():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    Y = rng.normal(size=(40, 4))
    v = linear_cka(X, Y)
    assert 0.0 <= v <= 1.0
    assert v < 0.5  # random matrices not strongly aligned


def test_cka_dimension_mismatch_raises():
    X = np.zeros((10, 3))
    Y = np.zeros((11, 3))
    with pytest.raises(ValueError):
        linear_cka(X, Y)


def test_compute_pairwise_cka():
    rng = np.random.default_rng(0)
    reps = {"a": rng.normal(size=(20, 5)), "b": rng.normal(size=(20, 5)), "c": rng.normal(size=(20, 5))}
    df = compute_pairwise_cka(reps)
    assert set(df.columns) == {"model_i", "model_j", "cka"}
    assert len(df) == 3


def test_probe_correctness_disjoint_raises():
    with pytest.raises(ValueError):
        assert_probe_correctness_disjoint(["a", "b"], ["b", "c"])


def test_cka_layer_sweep_inconsistent_raises():
    reps = {
        "a": {"last": np.zeros((10, 3)), "penultimate": np.zeros((10, 3))},
        "b": {"last": np.zeros((10, 3))},
    }
    with pytest.raises(ValueError):
        cka_layer_sweep(reps)
