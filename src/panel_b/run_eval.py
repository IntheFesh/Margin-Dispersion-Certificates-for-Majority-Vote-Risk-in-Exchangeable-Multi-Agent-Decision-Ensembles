from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from src.config.load_config import resolve_runtime_config
from src.io.report import write_markdown_report
from src.panel_b.pairwise_stats import build_pairwise_dataframe


REQUIRED_EVAL_COLUMNS = [
    "instance_id",
    "benchmark",
    "model_id",
    "family",
    "scale",
    "raw_output",
    "parsed_answer",
    "gold_answer",
    "correct",
    "invalid_parse",
]


def validate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_EVAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required Panel B prediction columns: {missing}")
    if df.duplicated(subset=["instance_id", "model_id"]).any():
        raise ValueError("Duplicate (instance_id, model_id) pairs in predictions")
    if not set(df["correct"].dropna().unique()).issubset({0, 1}):
        raise ValueError("correct must be in {0,1}")
    return df[REQUIRED_EVAL_COLUMNS].copy()


def run_eval_from_predictions(predictions_csv: str, out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(predictions_csv)
    val = validate_predictions(df)
    val.to_csv(out / "panel_b_evaluation.csv", index=False)
    pair = build_pairwise_dataframe(val)
    pair.to_csv(out / "panel_b_pairwise_stats.csv", index=False)
    invalid_rate = float(val["invalid_parse"].mean()) if len(val) else 0.0
    return {
        "n_rows": int(len(val)),
        "n_models": int(val["model_id"].nunique()),
        "invalid_parse_rate": invalid_rate,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--validate_only", action="store_true")
    a = ap.parse_args()
    cfg = resolve_runtime_config(a.config, a.output_dir, a.seed)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if a.validate_only or cfg.get("validate_only", False):
        write_markdown_report(
            str(out / "panel_b_eval_report.md"),
            "Panel B Evaluation Report",
            "validate_only",
            a.config,
            cfg,
            int(cfg["seed"]),
            0,
            0,
            0.0,
            "validate_only run; no inference executed",
            [],
            [
                "Panel B is external grounding only; observational and non-causal.",
                "Full evaluation requires real Hugging Face model resources and probe/eval splits.",
            ],
            ["panel_b_evaluation.csv", "panel_b_pairwise_stats.csv"],
        )
        return
    in_csv = cfg.get("predictions_csv")
    if not in_csv or not Path(in_csv).exists():
        raise FileNotFoundError(
            "Panel B predictions_csv must point to real per-(instance, model) outputs"
        )
    summary = run_eval_from_predictions(in_csv, str(out))
    main_md = (
        f"| metric | value |\n|---|---|\n"
        f"| rows | {summary['n_rows']} |\n"
        f"| models | {summary['n_models']} |\n"
        f"| invalid_parse_rate | {summary['invalid_parse_rate']:.4f} |\n"
    )
    write_markdown_report(
        str(out / "panel_b_report.md"),
        "Panel B Evaluation Report",
        "full",
        a.config,
        cfg,
        int(cfg["seed"]),
        summary["n_rows"],
        summary["n_models"],
        summary["invalid_parse_rate"],
        main_md,
        ["results/figures/cka_alignment.png"],
        ["Panel B is external grounding only; observational and non-causal."],
        [],
    )


if __name__ == "__main__":
    main()
