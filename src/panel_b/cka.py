from __future__ import annotations
import numpy as np
import pandas as pd

def center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    Xc,Yc=center(X),center(Y)
    num=np.linalg.norm(Yc.T@Xc,ord='fro')**2
    den=np.linalg.norm(Xc.T@Xc,ord='fro')*np.linalg.norm(Yc.T@Yc,ord='fro')
    if den==0: raise ValueError('Degenerate features for CKA')
    return float(num/den)

def compute_pairwise_cka(repr_map: dict[str,np.ndarray]) -> pd.DataFrame:
    keys=sorted(repr_map.keys()); rows=[]
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            rows.append({'model_i':a,'model_j':b,'cka':linear_cka(repr_map[a],repr_map[b])})
    return pd.DataFrame(rows)
