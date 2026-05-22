import numpy as np
import pytest
from src.theory.estimators import compute_z, compute_u2_per_instance, estimate_F_unbiased, cross_instance_u_statistic

def test_compute_z_values():
    X=np.array([[1,0,1],[0,0,1]])
    assert np.allclose(compute_z(X), [2/3,1/3])

def test_u2_small_hand():
    X=np.array([[1,1,0]])
    assert np.allclose(compute_u2_per_instance(X), [1/3])

def test_F_hand_calc():
    X=np.array([[1,1],[1,0]])
    assert abs(estimate_F_unbiased(X)-0.0)<1e-12

def test_invalid_X_raises():
    with pytest.raises(ValueError): compute_z(np.array([[0,2]]))

def test_K_lt_2_raises_for_F():
    with pytest.raises(ValueError): estimate_F_unbiased(np.array([[1],[0]]))

def test_M_lt_2_cross_u_raises():
    with pytest.raises(ValueError): cross_instance_u_statistic(np.array([0.5]))
