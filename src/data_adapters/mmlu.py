from __future__ import annotations
import json
import pandas as pd
from datasets import load_dataset

LABELS=["A","B","C","D"]

def load_dataset_split(split: str, max_instances: int | None, seed: int) -> pd.DataFrame:
    ds = load_dataset("cais/mmlu", "all", split=split)
    rows=[]
    for i,r in enumerate(ds):
        choices=[f"{LABELS[j]}. {c}" for j,c in enumerate(r["choices"])]
        rows.append({"instance_id":f"mmlu_{split}_{i}","benchmark":"MMLU","prompt":r["question"],"choices":json.dumps(choices),"gold_answer":LABELS[int(r["answer"])],"metadata_json":json.dumps({"split":split,"subject":r.get("subject")})})
    out=pd.DataFrame(rows)
    if max_instances is not None: out=out.sample(n=min(max_instances,len(out)),random_state=seed).reset_index(drop=True)
    return out
