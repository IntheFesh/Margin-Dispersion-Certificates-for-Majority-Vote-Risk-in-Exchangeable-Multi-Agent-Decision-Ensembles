import numpy as np
from src.theory.majority_risk import mixture_majority_risk

def test_mixture_risk_range():
 r=mixture_majority_risk(3,np.array([0.4,0.8]))
 assert 0<r<1
