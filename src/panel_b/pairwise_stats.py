from __future__ import annotations
import warnings
import numpy as np


def raw_covariance(x: np.ndarray, y: np.ndarray) -> float:
    x=np.asarray(x,float); y=np.asarray(y,float)
    if x.shape!=y.shape: raise ValueError("shape mismatch")
    return float(np.mean((x-x.mean())*(y-y.mean())))


def normalized_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x=np.asarray(x,float); y=np.asarray(y,float)
    vx,vy=np.var(x),np.var(y)
    if vx==0 or vy==0:
        warnings.warn("zero variance model; rho undefined", RuntimeWarning)
        return float("nan")
    return float(raw_covariance(x,y)/np.sqrt(vx*vy))
