from __future__ import annotations
import warnings
import numpy as np
import pandas as pd


def raw_covariance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("shape mismatch")
    return float(np.mean((x - x.mean()) * (y - y.mean())))


def normalized_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    vx, vy = float(np.var(x)), float(np.var(y))
    if vx == 0 or vy == 0:
        warnings.warn("zero variance model; rho undefined", RuntimeWarning)
        return float("nan")
    return float(raw_covariance(x, y) / np.sqrt(vx * vy))


def build_pairwise_dataframe(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Construct per-benchmark pairwise correctness stats.

    eval_df must include: instance_id, benchmark, model_id, family, correct.
    Output columns: benchmark, model_i, model_j, family_i, family_j,
    same_family, accuracy_i, accuracy_j, abs_accuracy_diff, n_shared,
    C_ij (raw covariance), rho_ij (normalized correctness correlation).
    """
    required = {"instance_id", "benchmark", "model_id", "family", "correct"}
    missing = required - set(eval_df.columns)
    if missing:
        raise ValueError(f"eval_df missing required columns: {sorted(missing)}")
    fam = eval_df.drop_duplicates("model_id").set_index("model_id")["family"].to_dict()
    rows = []
    for bench, g in eval_df.groupby("benchmark"):
        piv = g.pivot_table(index="instance_id", columns="model_id", values="correct", aggfunc="first")
        models = list(piv.columns)
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                pair = piv[[a, b]].dropna()
                if pair.empty:
                    continue
                xa = pair[a].to_numpy(dtype=float)
                xb = pair[b].to_numpy(dtype=float)
                acc_a = float(xa.mean())
                acc_b = float(xb.mean())
                rows.append(
                    {
                        "benchmark": bench,
                        "model_i": a,
                        "model_j": b,
                        "family_i": fam.get(a),
                        "family_j": fam.get(b),
                        "same_family": fam.get(a) == fam.get(b),
                        "accuracy_i": acc_a,
                        "accuracy_j": acc_b,
                        "abs_accuracy_diff": abs(acc_a - acc_b),
                        "n_shared": int(len(pair)),
                        "C_ij": raw_covariance(xa, xb),
                        "rho_ij": normalized_correlation(xa, xb),
                    }
                )
    return pd.DataFrame(rows)
