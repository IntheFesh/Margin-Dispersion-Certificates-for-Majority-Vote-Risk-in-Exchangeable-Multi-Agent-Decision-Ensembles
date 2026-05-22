from __future__ import annotations
import json, re
import pandas as pd
from datasets import load_dataset

_NUM_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")

def _extract_final_number(text: str) -> str:
    part = text.split("####")[-1] if "####" in text else text
    matches = _NUM_RE.findall(part)
    if not matches:
        raise ValueError("No numeric answer found in GSM8K reference")
    return matches[-1].replace(",", "")

def load_dataset_split(split: str, max_instances: int | None, seed: int) -> pd.DataFrame:
    ds = load_dataset("gsm8k", "main", split=split)
    df = ds.to_pandas()
    rows = []
    for i, r in df.iterrows():
        rows.append({"instance_id": f"gsm8k_{split}_{i}", "benchmark": "GSM8K", "prompt": r["question"], "choices": json.dumps([]), "gold_answer": _extract_final_number(str(r["answer"])), "metadata_json": json.dumps({"split": split})})
    out = pd.DataFrame(rows)
    if max_instances is not None:
        out = out.sample(n=min(max_instances, len(out)), random_state=seed).reset_index(drop=True)
    return out
