import numpy as np
import warnings
from src.panel_b.pairwise_stats import raw_covariance, normalized_correlation

def test_raw_covariance_correct():
    x=np.array([0,1,1,0]); y=np.array([0,1,0,1])
    assert abs(raw_covariance(x,y)-0.0)<1e-12

def test_normalized_rho_correct():
    x=np.array([0,1,0,1]); y=np.array([0,1,0,1])
    assert abs(normalized_correlation(x,y)-1.0)<1e-12

def test_zero_variance_nan_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        r=normalized_correlation(np.array([1,1,1]),np.array([0,1,0]))
        assert np.isnan(r)
        assert any('zero variance' in str(ww.message) for ww in w)
