from __future__ import annotations
import numpy as np

def permutation_null(observed: np.ndarray, comparator: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    rng=np.random.default_rng(seed)
    m=observed.shape[0]
    vals=[]
    iu=np.triu_indices(m,1)
    for _ in range(n_perm):
        p=rng.permutation(m)
        comp_p=comparator[p][:,p]
        x=observed[iu]; y=comp_p[iu]
        vals.append(float(np.corrcoef(x,y)[0,1]))
    return np.asarray(vals,float)

def family_aware_permutation_null(observed: np.ndarray, comparator: np.ndarray, families: list[str], n_perm: int, seed: int) -> np.ndarray:
    rng=np.random.default_rng(seed); fam=np.asarray(families); m=len(fam)
    vals=[]; iu=np.triu_indices(m,1)
    idx=np.arange(m)
    for _ in range(n_perm):
        p=idx.copy()
        for f in np.unique(fam):
            loc=np.where(fam==f)[0]
            p[loc]=rng.permutation(loc)
        comp_p=comparator[p][:,p]
        vals.append(float(np.corrcoef(observed[iu],comp_p[iu])[0,1]))
    return np.asarray(vals,float)
