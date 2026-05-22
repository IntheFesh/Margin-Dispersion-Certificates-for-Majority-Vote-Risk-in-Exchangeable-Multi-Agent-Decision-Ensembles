from src.theory.certificate import *

def test_confidence_bounds_monotonicity():
    b1=compute_confidence_bounds(0.7,0.55,100,0.1)
    b2=compute_confidence_bounds(0.7,0.55,400,0.1)
    assert b2['eps_delta'] < b1['eps_delta']

def test_refuse_and_issue():
    r=empirical_certificate_from_summaries(0.51,0.26,10,16,0.1)
    assert r['issued'] is False
    i=empirical_certificate_from_summaries(0.9,0.81,500,16,0.1)
    assert i['issued'] is True

def test_optimize_eta_and_range():
    o=optimize_eta(0.2,0.01,64)
    assert 0<o['eta_star']<0.2 and 0<=o['objective_value']<=1

def test_population_largeN_to_cantelli():
    p=population_certificate(0.7,0.02,5000)
    c=asymptotic_cantelli_bound(0.7,0.02)
    assert p['R_cert'] <= 1.0 and c <= 1.0
