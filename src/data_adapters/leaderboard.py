from __future__ import annotations
from pathlib import Path
import json
import pandas as pd


REQUIRED_COLUMNS = {"instance_id", "model_id", "family", "correct"}


def load_per_instance_predictions(input_dir: str) -> pd.DataFrame:
    """Load per-instance prediction CSVs from a directory.

    Each file must contain at minimum: instance_id, model_id, family, correct.
    Models that only provide aggregate scores must be skipped at the
    file-curation stage; this loader does not synthesise missing rows.

    Raises FileNotFoundError if the directory does not exist and ValueError
    if no file has the required schema.
    """
    p = Path(input_dir)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"leaderboard input directory not found: {p}")
    files = sorted([f for f in p.glob("*.csv")])
    if not files:
        raise FileNotFoundError(f"No CSV files found under {p}")
    parts = []
    for f in files:
        df = pd.read_csv(f)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"{f.name} missing required columns: {sorted(missing)}")
        for col in REQUIRED_COLUMNS:
            if df[col].isna().any():
                raise ValueError(f"{f.name} contains NaN in required column '{col}'")
        if not set(df["correct"].unique()).issubset({0, 1}):
            raise ValueError(f"{f.name}: correct must be in {{0,1}}")
        if df.duplicated(subset=["instance_id", "model_id"]).any():
            raise ValueError(f"{f.name}: duplicate (instance_id, model_id) rows")
        parts.append(df[list(REQUIRED_COLUMNS)])
    if not parts:
        raise ValueError("No usable per-instance prediction files were found")
    return pd.concat(parts, ignore_index=True)
