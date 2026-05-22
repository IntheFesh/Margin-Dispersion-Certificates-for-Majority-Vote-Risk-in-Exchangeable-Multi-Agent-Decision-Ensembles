from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_leaderboard_predictions(input_dir: str) -> pd.DataFrame:
    p=Path(input_dir)
    files=list(p.glob('*.csv'))
    if not files: raise FileNotFoundError('No prediction CSV files found')
    dfs=[]
    for f in files:
        d=pd.read_csv(f)
        if {'instance_id','model_id','family','correct'} <= set(d.columns):
            dfs.append(d[['instance_id','model_id','family','correct']].copy())
    if not dfs: raise ValueError('No per-instance prediction files with required columns')
    return pd.concat(dfs,ignore_index=True)
