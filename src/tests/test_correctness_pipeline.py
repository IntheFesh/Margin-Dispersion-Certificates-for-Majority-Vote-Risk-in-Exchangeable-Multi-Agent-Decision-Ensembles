import pandas as pd
import pytest
from src.inference.correctness import evaluate_predictions, REQUIRED_COLUMNS


def _toy_row(**kw):
    base = {
        "instance_id": "i0",
        "benchmark": "ARC-Challenge",
        "model_id": "m",
        "protocol": "A1",
        "agent_id": "a0",
        "sample_id": 0,
        "prompt_id": "p0",
        "seed": 0,
        "temperature": 0.0,
        "raw_output": "The answer is (B).",
        "gold_answer": "B",
    }
    base.update(kw)
    return base


def test_evaluate_predictions_correct_and_invalid():
    df = pd.DataFrame([
        _toy_row(raw_output="The answer is (B).", gold_answer="B"),
        _toy_row(instance_id="i1", raw_output="<no choice>", gold_answer="A"),
        _toy_row(instance_id="i2", benchmark="GSM8K", raw_output="result #### 42", gold_answer="42"),
        _toy_row(instance_id="i3", benchmark="GSM8K", raw_output="nope", gold_answer="3"),
    ])
    out = evaluate_predictions(df)
    assert out.loc[0, "correct"] == 1 and out.loc[0, "invalid_parse"] == False
    assert out.loc[1, "correct"] == 0 and out.loc[1, "invalid_parse"] == True
    assert out.loc[2, "correct"] == 1
    assert out.loc[3, "invalid_parse"] == True
    assert out.loc[3, "correct"] == 0
    # raw_output is preserved
    assert "raw_output" in out.columns


def test_missing_column_raises():
    df = pd.DataFrame([{c: 0 for c in REQUIRED_COLUMNS if c != "raw_output"}])
    with pytest.raises(ValueError):
        evaluate_predictions(df)
