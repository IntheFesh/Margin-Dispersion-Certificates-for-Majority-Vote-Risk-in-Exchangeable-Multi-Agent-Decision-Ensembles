import json
from pathlib import Path
import pandas as pd

def save_json(obj,path):
 Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True))

def save_csv(df: pd.DataFrame, path):
 Path(path).parent.mkdir(parents=True,exist_ok=True); df.to_csv(path,index=False)
