from __future__ import annotations
import pandas as pd

def run_eval_from_predictions(predictions_csv: str, output_csv: str) -> None:
    df=pd.read_csv(predictions_csv)
    req=['instance_id','benchmark','model_id','family','scale','raw_output','parsed_answer','gold_answer','correct','invalid_parse']
    miss=[c for c in req if c not in df.columns]
    if miss: raise ValueError(f'Missing required columns: {miss}')
    df[req].to_csv(output_csv,index=False)
