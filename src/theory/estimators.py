from __future__ import annotations
import numpy as np

def _validate_binary_matrix(X: np.ndarray) -> np.ndarray:
    X=np.asarray(X)
    if X.ndim!=2: raise ValueError("X must be 2D")
    vals=np.unique(X)
    if not np.all(np.isin(vals,[0,1])): raise ValueError("X must be binary with values in {0,1}")
    return X.astype(float)

def compute_z(X: np.ndarray) -> np.ndarray:
    X=_validate_binary_matrix(X)
    return X.mean(axis=1)

def compute_u2_per_instance(X: np.ndarray) -> np.ndarray:
    X=_validate_binary_matrix(X)
    M,K=X.shape
    if K<2: raise ValueError("K must be >=2")
    s=X.sum(axis=1)
    return (s*(s-1))/(K*(K-1))

def cross_instance_u_statistic(z: np.ndarray) -> float:
    z=np.asarray(z,dtype=float)
    if z.ndim!=1: raise ValueError("z must be 1D")
    M=len(z)
    if M<2: raise ValueError("M must be >=2")
    return float((z.sum()**2-(z**2).sum())/(M*(M-1)))

def estimate_m1_m2(X: np.ndarray) -> dict:
    z=compute_z(X); u2=compute_u2_per_instance(X)
    return {"m1":float(z.mean()),"m2":float(u2.mean()),"Z":z,"U2":u2}

def estimate_F_unbiased(X: np.ndarray) -> float:
    X=_validate_binary_matrix(X)
    M,K=X.shape
    if K<2: raise ValueError("K must be >=2")
    if M<2: raise ValueError("M must be >=2")
    s=estimate_m1_m2(X)
    return float(s["m2"]-cross_instance_u_statistic(s["Z"]))

def estimate_basic_summaries(X: np.ndarray) -> dict:
    X=_validate_binary_matrix(X)
    M,K=X.shape
    m=estimate_m1_m2(X); F=estimate_F_unbiased(X)
    return {"M":M,"K":K,"bar_alpha_hat":m["m1"],"margin_hat":m["m1"]-0.5,"m2_hat":m["m2"],"F_hat_unbiased":F,"F_hat_clipped":min(0.25,max(0.0,F))}
