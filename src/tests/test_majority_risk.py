import numpy as np
from src.theory.majority_risk import binom_strict_failure_prob, mixture_majority_risk

def test_binom_n3_manual():
    a=0.7
    manual=(1-a)**3 + 3*a*(1-a)**2
    assert abs(binom_strict_failure_prob(3,a)-manual)<1e-12

def test_mixture_weighted():
    al=np.array([0.4,0.8])
    r=mixture_majority_risk(3,al)
    exp=(binom_strict_failure_prob(3,0.4)+binom_strict_failure_prob(3,0.8))/2
    assert abs(r-exp)<1e-12

def test_even_tie_is_failure():
    assert abs(binom_strict_failure_prob(2,0.5)-0.75)<1e-12
