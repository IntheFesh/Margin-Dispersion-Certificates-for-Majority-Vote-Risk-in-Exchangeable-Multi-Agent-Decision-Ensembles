import yaml
from pathlib import Path

def load_config(path):
 p=Path(path)
 if not p.exists(): raise FileNotFoundError(path)
 cfg=yaml.safe_load(p.read_text())
 if not isinstance(cfg,dict): raise ValueError("Config must be mapping")
 return cfg
