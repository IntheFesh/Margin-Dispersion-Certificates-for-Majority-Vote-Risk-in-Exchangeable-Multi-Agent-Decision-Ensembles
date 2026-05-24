"""Instance-level estimation/oracle pool split.

The split is INSTANCE-LEVEL: each instance belongs to exactly one pool.
Sample columns are NEVER split across pools (that would mix estimation and
reference signals on the same instance). Pool index disjointness is
asserted at run time; any overlap raises ValueError.
"""
from __future__ import annotations
import numpy as np


def split_pools_by_instance(
    instance_ids: list[str],
    oracle_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split instance IDs into (estimation_ids, oracle_ids).

    The estimation pool feeds certificate construction; the oracle pool
    feeds the R_N^MC reference only. Disjoint at the instance level.
    """
    if not (0.0 < oracle_fraction < 1.0):
        raise ValueError(f"oracle_fraction must be in (0,1), got {oracle_fraction}")
    if len(instance_ids) < 2:
        raise ValueError("need at least 2 instances to split into two pools")
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError("instance_ids must be unique")

    rng = np.random.default_rng(seed)
    ids = list(instance_ids)
    perm = rng.permutation(len(ids))
    n_oracle = int(round(len(ids) * oracle_fraction))
    n_oracle = max(1, min(len(ids) - 1, n_oracle))
    oracle_idx = perm[:n_oracle]
    estimation_idx = perm[n_oracle:]

    oracle_ids = [ids[i] for i in oracle_idx]
    estimation_ids = [ids[i] for i in estimation_idx]

    overlap = set(estimation_ids) & set(oracle_ids)
    if overlap:
        raise ValueError(f"pool leakage: {len(overlap)} shared instance ids")
    return estimation_ids, oracle_ids
