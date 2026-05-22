import pandas as pd
import pytest
from src.data_adapters.leaderboard import load_per_instance_predictions


def test_load_per_instance_predictions(tmp_path):
    df = pd.DataFrame({
        "instance_id": ["i1", "i2", "i3"],
        "model_id": ["m1", "m1", "m1"],
        "family": ["f", "f", "f"],
        "correct": [1, 0, 1],
    })
    (tmp_path / "m1.csv").write_text(df.to_csv(index=False))
    loaded = load_per_instance_predictions(str(tmp_path))
    assert set(loaded.columns) >= {"instance_id", "model_id", "family", "correct"}
    assert len(loaded) == 3


def test_load_per_instance_predictions_missing_col(tmp_path):
    df = pd.DataFrame({"instance_id": ["i1"], "model_id": ["m1"], "correct": [1]})  # missing family
    (tmp_path / "m1.csv").write_text(df.to_csv(index=False))
    with pytest.raises(ValueError):
        load_per_instance_predictions(str(tmp_path))


def test_load_per_instance_predictions_no_dir():
    with pytest.raises(FileNotFoundError):
        load_per_instance_predictions("/nonexistent_dir_for_test_xx")
