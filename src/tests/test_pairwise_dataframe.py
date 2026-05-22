import pandas as pd
import pytest
from src.panel_b.pairwise_stats import build_pairwise_dataframe


def _toy_predictions():
    rows = []
    for i in range(10):
        for m, fam, scale, p_correct in [("m1", "f1", "S", 0.9), ("m2", "f1", "S", 0.4), ("m3", "f2", "S", 0.5)]:
            rows.append({"instance_id": f"x{i}", "benchmark": "bench", "model_id": m, "family": fam, "scale": scale, "correct": int((i * (1 if m == "m1" else 7 if m == "m2" else 3)) % 5 < int(p_correct * 5))})
    return pd.DataFrame(rows)


def test_build_pairwise_dataframe_columns():
    df = _toy_predictions()
    out = build_pairwise_dataframe(df)
    expected = {"benchmark", "model_i", "model_j", "family_i", "family_j", "same_family", "accuracy_i", "accuracy_j", "abs_accuracy_diff", "n_shared", "C_ij", "rho_ij"}
    assert expected.issubset(out.columns)
    assert len(out) == 3  # C(3,2)


def test_build_pairwise_dataframe_missing_columns_raises():
    df = pd.DataFrame({"instance_id": [], "benchmark": []})
    with pytest.raises(ValueError):
        build_pairwise_dataframe(df)
