from __future__ import annotations
import json
import pandas as pd
from datasets import load_dataset

def load_dataset_split(split: str, max_instances: int | None, seed: int) -> pd.DataFrame:
    ds = load_dataset("ai2_arc", "ARC-Challenge", split=split)
    rows=[]
    for i,r in enumerate(ds):
        labels=r["choices"]["label"]; texts=r["choices"]["text"]
        choices=[f"{l}. {t}" for l,t in zip(labels,texts)]
        rows.append({"instance_id":f"arc_challenge_{split}_{i}","benchmark":"ARC-Challenge","prompt":r["question"],"choices":json.dumps(choices),"gold_answer":str(r["answerKey"]).strip().upper(),"metadata_json":json.dumps({"split":split})})
    out=pd.DataFrame(rows)
    if max_instances is not None: out=out.sample(n=min(max_instances,len(out)),random_state=seed).reset_index(drop=True)
    return out
