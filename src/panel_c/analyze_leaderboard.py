from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.config.load_config import resolve_runtime_config
from src.io.report import write_markdown_report
from src.panel_c.load_leaderboard_predictions import load_per_instance_predictions
from src.panel_b.pairwise_stats import normalized_correlation


def run(input_dir: str, output_dir: str, config_path: str = "", config: dict | None = None, seed: int = 0) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = load_per_instance_predictions(input_dir)
    piv = d.pivot_table(index="instance_id", columns="model_id", values="correct", aggfunc="first")
    fam = d.drop_duplicates("model_id").set_index("model_id")["family"]
    rows = []
    models = list(piv.columns)
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            sub = piv[[a, b]].dropna()
            if sub.empty:
                continue
            xa, xb = sub[a].to_numpy(float), sub[b].to_numpy(float)
            rows.append({
                "model_i": a,
                "model_j": b,
                "n_shared": int(len(sub)),
                "accuracy_i": float(xa.mean()),
                "accuracy_j": float(xb.mean()),
                "rho_ij": normalized_correlation(xa, xb),
                "same_family": bool(fam[a] == fam[b]),
            })
    p = pd.DataFrame(rows)
    p.to_csv(out / "panel_c_pairwise_correctness.csv", index=False)
    if len(p):
        sg = p.groupby("same_family")["rho_ij"].mean()
        within = float(sg.get(True, np.nan))
        cross = float(sg.get(False, np.nan))
        gap = within - cross if not (np.isnan(within) or np.isnan(cross)) else float("nan")
    else:
        within = cross = gap = float("nan")
    pd.DataFrame([
        {"within_family_mean_rho": within, "cross_family_mean_rho": cross, "family_gap": gap}
    ]).to_csv(out / "panel_c_family_summary.csv", index=False)
    summary = {
        "n_models": int(len(models)),
        "n_pairs": int(len(p)),
        "within_family_mean_rho": within,
        "cross_family_mean_rho": cross,
        "family_gap": gap,
    }
    (out / "panel_c_summary.json").write_text(json.dumps(summary, indent=2))
    main_md = (
        f"| metric | value |\n|---|---|\n"
        f"| models | {summary['n_models']} |\n"
        f"| pairs | {summary['n_pairs']} |\n"
        f"| within-family mean rho | {within:.4f} |\n"
        f"| cross-family mean rho | {cross:.4f} |\n"
        f"| family gap | {gap:.4f} |\n"
    )
    write_markdown_report(
        str(out / "panel_c_report.md"),
        "Panel C Supplementary Report",
        "supplementary",
        config_path,
        config or {},
        seed,
        int(d["instance_id"].nunique()),
        len(models),
        0.0,
        main_md,
        [],
        [
            "Panel C is appendix-only.",
            "No CKA is computed and no representation-alignment claim is made.",
        ],
        [],
    )
    return summary


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
            str(out / "panel_c_report.md"),
            "Panel C Supplementary Report",
            "validate_only",
            a.config,
            cfg,
            int(cfg["seed"]),
            0,
            0,
            0.0,
            "validate_only run; no leaderboard predictions consumed",
            [],
            [
                "Panel C is appendix-only.",
                "No CKA is computed and no representation-alignment claim is made.",
                "Full Panel C run requires real per-instance leaderboard prediction files.",
            ],
            ["panel_c_pairwise_correctness.csv", "panel_c_family_summary.csv"],
        )
        return
    input_dir = cfg.get("leaderboard_input_dir")
    if not input_dir:
        raise ValueError("leaderboard_input_dir required")
    run(input_dir, str(out), config_path=a.config, config=cfg, seed=int(cfg["seed"]))


if __name__ == "__main__":
    main()
