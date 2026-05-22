from __future__ import annotations
import numpy as np
import pandas as pd


def center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two feature matrices [n, d_i].
    Raises ValueError on degenerate inputs (zero feature variance), since
    silent fallback is disallowed."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("CKA inputs must be 2D")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"CKA expects same number of probe rows; got {X.shape[0]} vs {Y.shape[0]}")
    Xc, Yc = center(X), center(Y)
    num = float(np.linalg.norm(Yc.T @ Xc, ord="fro") ** 2)
    den = float(np.linalg.norm(Xc.T @ Xc, ord="fro") * np.linalg.norm(Yc.T @ Yc, ord="fro"))
    if den == 0:
        raise ValueError("Degenerate features for CKA (zero feature variance)")
    return float(num / den)


def compute_pairwise_cka(repr_map: dict[str, np.ndarray]) -> pd.DataFrame:
    """Compute pairwise linear CKA across a dict of {model_id: features}."""
    keys = sorted(repr_map.keys())
    rows = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            rows.append({"model_i": a, "model_j": b, "cka": linear_cka(repr_map[a], repr_map[b])})
    return pd.DataFrame(rows)


def assert_probe_correctness_disjoint(probe_ids, correctness_ids) -> None:
    """Probe set must be disjoint from the correctness-evaluation set.
    Raises ValueError on any overlap (no silent fallback)."""
    probe = set(map(str, probe_ids))
    corr = set(map(str, correctness_ids))
    inter = probe & corr
    if inter:
        raise ValueError(
            f"Panel B leakage: CKA probe set overlaps correctness eval ({len(inter)} shared ids); "
            f"sample: {list(inter)[:5]}"
        )


def cka_layer_sweep(repr_by_model_by_layer: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    """Compute pairwise CKA for multiple layer specs.

    Input: {model_id: {layer_name: features}}.
    All models must expose the same set of layer keys; an inconsistency
    raises ValueError.
    """
    if not repr_by_model_by_layer:
        return pd.DataFrame(columns=["layer", "model_i", "model_j", "cka"])
    layer_sets = [set(v.keys()) for v in repr_by_model_by_layer.values()]
    common = set.intersection(*layer_sets)
    if not common:
        raise ValueError("layer-sweep: no common layer names across all models")
    if any(s != common for s in layer_sets):
        raise ValueError("layer-sweep: model entries expose inconsistent layer names")
    frames = []
    for layer in sorted(common):
        per_model = {m: repr_by_model_by_layer[m][layer] for m in repr_by_model_by_layer}
        d = compute_pairwise_cka(per_model)
        d.insert(0, "layer", layer)
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["layer", "model_i", "model_j", "cka"])
