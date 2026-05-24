import pytest
from src.protocols.pools import split_pools_by_instance


def test_split_500_half():
    ids = [f"inst_{i}" for i in range(500)]
    est, orc = split_pools_by_instance(ids, oracle_fraction=0.5, seed=0)
    assert len(est) == 250
    assert len(orc) == 250
    assert set(est).isdisjoint(set(orc))
    assert set(est) | set(orc) == set(ids)


def test_split_deterministic():
    ids = [f"x{i}" for i in range(100)]
    a = split_pools_by_instance(ids, 0.5, seed=42)
    b = split_pools_by_instance(ids, 0.5, seed=42)
    assert a == b


def test_invalid_oracle_fraction():
    ids = [f"x{i}" for i in range(10)]
    with pytest.raises(ValueError):
        split_pools_by_instance(ids, 0.0, seed=0)
    with pytest.raises(ValueError):
        split_pools_by_instance(ids, 1.0, seed=0)


def test_duplicate_ids_raise():
    with pytest.raises(ValueError):
        split_pools_by_instance(["a", "a", "b"], 0.5, seed=0)
