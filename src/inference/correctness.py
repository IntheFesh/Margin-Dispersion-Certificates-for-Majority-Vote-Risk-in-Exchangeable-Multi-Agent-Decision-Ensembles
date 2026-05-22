from __future__ import annotations
import pandas as pd
from .answer_extractors import parse_multiple_choice_answer, parse_gsm8k_numeric_answer

REQUIRED_COLUMNS = ["instance_id","benchmark","model_id","protocol","agent_id","sample_id","prompt_id","seed","temperature","raw_output","gold_answer"]

def evaluate_predictions(df: pd.DataFrame, valid_labels: list[str] | None = None) -> pd.DataFrame:
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    out = df.copy()
    parsed, invalid, correct = [], [], []
    for _, r in out.iterrows():
        if r["benchmark"] == "GSM8K":
            pa, inv = parse_gsm8k_numeric_answer(str(r["raw_output"]))
        else:
            labels = valid_labels or ["A", "B", "C", "D", "E", "F"]
            pa, inv = parse_multiple_choice_answer(str(r["raw_output"]), labels)
        parsed.append(pa)
        invalid.append(bool(inv))
        correct.append(0 if inv else int(str(pa) == str(r["gold_answer"])))
    out["parsed_answer"] = parsed
    out["correct"] = correct
    out["invalid_parse"] = invalid
    return out[[*REQUIRED_COLUMNS, "parsed_answer", "gold_answer", "correct", "invalid_parse"]]
