from __future__ import annotations
from pathlib import Path
import shutil
import yaml


def load_yaml_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    cfg = yaml.safe_load(p.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("YAML config must be a mapping")
    return cfg


def resolve_runtime_config(config_path: str, output_dir: str | None = None, seed: int | None = None) -> dict:
    cfg = load_yaml_config(config_path)
    if output_dir is not None:
        cfg["output_dir"] = output_dir
    if seed is not None:
        cfg["seed"] = int(seed)
    if "output_dir" not in cfg:
        raise ValueError("Config must specify output_dir or pass --output_dir")
    if "seed" not in cfg:
        raise ValueError("Config must specify seed or pass --seed")
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, out / "config_used.yaml")
    return cfg
