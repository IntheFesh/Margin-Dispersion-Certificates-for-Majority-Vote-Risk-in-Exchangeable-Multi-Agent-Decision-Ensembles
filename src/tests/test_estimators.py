import numpy as np
from src.theory.estimators import compute_z,compute_u2_per_instance,estimate_F_unbiased

def test_estimators_shapes():
 X=np.array([[1,0,1],[1,1,0]])
 assert compute_z(X).shape==(2,)
 assert np.allclose(compute_u2_per_instance(X),[1/3,1/3])
 assert isinstance(estimate_F_unbiased(X),float)
