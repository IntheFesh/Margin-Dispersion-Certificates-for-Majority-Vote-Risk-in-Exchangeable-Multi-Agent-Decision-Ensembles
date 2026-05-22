import numpy as np
from src.synthetic.simulate_bernoulli_mixture import simulate_alpha, simulate_X_from_alpha

def test_simulate_alpha_and_X():
    a = simulate_alpha(100, {"name":"two_point","p":0.5,"a_low":0.2,"a_high":0.8}, seed=1)
    assert a.shape == (100,)
    X = simulate_X_from_alpha(a, 7, seed=2)
    assert X.shape == (100,7)
    assert np.all(np.isin(X, [0,1]))
