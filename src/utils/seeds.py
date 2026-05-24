"""Deterministic seed derivation.

Every random operation derives its seed from the global seed plus a
structured key via a stable hash. No os-time seeding, no unseeded RNGs.
"""
from __future__ import annotations
import hashlib
import numpy as np

_MASK64 = (1 << 64) - 1


def derive_seed(global_seed: int, *key_parts: object) -> int:
    """Derive a stable 64-bit seed from the global seed and a structured key.

    Example: derive_seed(global_seed, cell_id, instance_idx, sample_idx,
    'generation'). Deterministic across runs and machines.
    """
    h = hashlib.sha256()
    h.update(str(int(global_seed)).encode())
    for part in key_parts:
        h.update(b"|")
        h.update(str(part).encode())
    return int.from_bytes(h.digest()[:8], "big") & _MASK64


def rng_for(global_seed: int, *key_parts: object) -> np.random.Generator:
    """Return a numpy Generator seeded deterministically from the key."""
    return np.random.default_rng(derive_seed(global_seed, *key_parts))
