import numpy as np
from .majority_risk import binom_strict_failure_prob

def discrete_mixture_mean(points,weights):
    p=np.asarray(points,float); w=np.asarray(weights,float); return float(np.sum(w*p))

def discrete_mixture_variance(points,weights):
    p=np.asarray(points,float); w=np.asarray(weights,float); m=discrete_mixture_mean(p,w); return float(np.sum(w*(p-m)**2))

def discrete_mixture_majority_risk(points,weights,N):
    p=np.asarray(points,float); w=np.asarray(weights,float); return float(np.sum(w*np.array([binom_strict_failure_prob(N,x) for x in p])))

def reproduce_two_moment_insufficiency():
    mu1p=np.array([0.4,0.8]); mu1w=np.array([0.5,0.5]); mu2p=np.array([0.0,0.5,0.7]); mu2w=np.array([3/35,1/5,5/7])
    return {"mu1_mean":discrete_mixture_mean(mu1p,mu1w),"mu2_mean":discrete_mixture_mean(mu2p,mu2w),"mu1_var":discrete_mixture_variance(mu1p,mu1w),"mu2_var":discrete_mixture_variance(mu2p,mu2w),"R3_mu1":discrete_mixture_majority_risk(mu1p,mu1w,3),"R3_mu2":discrete_mixture_majority_risk(mu2p,mu2w,3)}
