from __future__ import annotations
import math,itertools
import numpy as np
from scipy.stats import binom

def binom_strict_failure_prob(N:int, alpha:float)->float:
    return float(binom.cdf(math.floor(N/2),N,alpha))

def mixture_majority_risk(N:int, alpha_values:np.ndarray)->float:
    a=np.asarray(alpha_values,dtype=float)
    return float(np.mean([binom_strict_failure_prob(N,x) for x in a]))

def monte_carlo_reference_risk(N:int, alpha_ref:np.ndarray)->float:
    return mixture_majority_risk(N,alpha_ref)

def observed_majority_failure_from_samples(X:np.ndarray,N:int,num_subsets:int|None,seed:int)->dict:
    X=np.asarray(X)
    if X.ndim!=2: raise ValueError('X must be 2D')
    M,K=X.shape
    if N<1 or N>K: raise ValueError('N out of range')
    rng=np.random.default_rng(seed); total=math.comb(K,N)
    if total<=5000 and (num_subsets is None or num_subsets>=total):
        subs=list(itertools.combinations(range(K),N))
    else:
        if num_subsets is None: raise ValueError('num_subsets required for large combinatorial space')
        target=min(num_subsets,total); seen=set(); subs=[]
        while len(subs)<target:
            idx=tuple(sorted(rng.choice(K,size=N,replace=False).tolist()))
            if idx not in seen: seen.add(idx); subs.append(idx)
    vals=[]
    for s in subs:
        votes=X[:,list(s)].sum(axis=1)
        vals.append(float(np.mean(votes<=math.floor(N/2))))
    arr=np.asarray(vals,float)
    se=float(arr.std(ddof=1)/np.sqrt(len(arr))) if len(arr)>1 else 0.0
    return {"mean_failure":float(arr.mean()),"stderr":se,"num_subsets_used":len(subs),"total_subsets":total}
