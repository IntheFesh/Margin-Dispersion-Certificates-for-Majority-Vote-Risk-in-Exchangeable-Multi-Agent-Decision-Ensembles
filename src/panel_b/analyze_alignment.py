from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

from src.config.load_config import resolve_runtime_config
from src.io.report import write_markdown_report
from src.panel_b.permutation_nulls import (
    permutation_null,
    family_aware_permutation_null,
    quantile_at_observed,
)


def _matrix_from_pairs(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, list[str]]:
    models = sorted(set(df["model_i"]).union(df["model_j"]))
    idx = {m: i for i, m in enumerate(models)}
    M = np.full((len(models), len(models)), np.nan, dtype=float)
    for _, r in df.iterrows():
        i, j = idx[r["model_i"]], idx[r["model_j"]]
        M[i, j] = M[j, i] = r[value_col]
    np.fill_diagonal(M, 1.0 if value_col == "cka" else 0.0)
    return M, models


def _partial_spearman(y: np.ndarray, x: np.ndarray, z: np.ndarray) -> float:
    """Spearman correlation of y, x after partialling out z (rank-based)."""
    ry, rx, rz = rankdata(y), rankdata(x), rankdata(z)
    # residualise via least squares on ranks
    A = np.column_stack([rz, np.ones_like(rz)])
    by, _ = np.linalg.lstsq(A, ry, rcond=None)[0], None
    bx, _ = np.linalg.lstsq(A, rx, rcond=None)[0], None
    rho_y = ry - A @ by
    rho_x = rx - A @ bx
    return float(np.corrcoef(rho_y, rho_x)[0, 1])


def analyze(
    pairwise_csv: str,
    cka_csv: str,
    out_dir: str,
    n_perm: int = 10000,
    seed: int = 42,
    config_path: str = "",
    config: dict | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv(pairwise_csv)
    c = pd.read_csv(cka_csv)
    required_p = {"benchmark", "model_i", "model_j", "rho_ij", "C_ij", "family_i", "family_j", "same_family", "abs_accuracy_diff"}
    missing = required_p - set(p.columns)
    if missing:
        raise ValueError(f"pairwise CSV missing columns: {sorted(missing)}")
    if not {"model_i", "model_j", "cka"}.issubset(c.columns):
        raise ValueError("cka CSV missing model_i/model_j/cka columns")
    if "benchmark" in c.columns:
        m = p.merge(c, on=["benchmark", "model_i", "model_j"], how="inner")
    else:
        m = p.merge(c, on=["model_i", "model_j"], how="inner")

    aggregate_rho = float(spearmanr(m["rho_ij"], m["cka"]).correlation)
    raw_vs_cka = float(spearmanr(m["C_ij"], m["cka"]).correlation)
    partial_rho = _partial_spearman(m["rho_ij"].to_numpy(float), m["cka"].to_numpy(float), m["abs_accuracy_diff"].to_numpy(float))

    by_bench = (
        m.groupby("benchmark")
        .apply(lambda d: float(spearmanr(d["rho_ij"], d["cka"]).correlation))
        .to_dict()
    )

    families = sorted(set(m["family_i"]).union(m["family_j"]))
    loo = {}
    for f in families:
        sub = m[(m["family_i"] != f) & (m["family_j"] != f)]
        if len(sub) >= 3 and sub["rho_ij"].std() > 0 and sub["cka"].std() > 0:
            loo[f] = float(spearmanr(sub["rho_ij"], sub["cka"]).correlation)
        else:
            loo[f] = None

    within = float(m.loc[m["same_family"], "rho_ij"].mean()) if (m["same_family"]).any() else float("nan")
    cross = float(m.loc[~m["same_family"], "rho_ij"].mean()) if (~m["same_family"]).any() else float("nan")

    rho_mat, models = _matrix_from_pairs(m.drop_duplicates(["model_i", "model_j"]), "rho_ij")
    cka_mat, models2 = _matrix_from_pairs(m.drop_duplicates(["model_i", "model_j"]), "cka")
    if models != models2:
        raise ValueError("rho_ij and cka pairwise indices disagree")
    fam_map = {**{r["model_i"]: r["family_i"] for _, r in m.iterrows()},
               **{r["model_j"]: r["family_j"] for _, r in m.iterrows()}}
    fams = [fam_map[mm] for mm in models]
    null_rc = permutation_null(rho_mat, cka_mat, n_perm=n_perm, seed=seed, method="spearman")
    null_fa = family_aware_permutation_null(rho_mat, cka_mat, fams, n_perm=n_perm, seed=seed + 1, method="spearman")
    q99_rc = float(np.quantile(null_rc, 0.99))
    q99_fa = float(np.quantile(null_fa, 0.99))
    pvalue_rc = quantile_at_observed(aggregate_rho, null_rc)
    pvalue_fa = quantile_at_observed(aggregate_rho, null_fa)

    pd.DataFrame({"row_col_null": null_rc, "family_aware_null": null_fa}).to_csv(out / "panel_b_permutation_nulls.csv", index=False)

    summary = {
        "aggregate_spearman_rho_vs_cka": aggregate_rho,
        "spearman_raw_cov_vs_cka": raw_vs_cka,
        "partial_spearman_rho_vs_cka_given_abs_acc_diff": partial_rho,
        "benchmarkwise_spearman": by_bench,
        "leave_one_family_out_spearman": loo,
        "within_family_mean_rho": within,
        "cross_family_mean_rho": cross,
        "row_col_null_q99": q99_rc,
        "family_aware_null_q99": q99_fa,
        "pvalue_row_col": pvalue_rc,
        "pvalue_family_aware": pvalue_fa,
        "confirmatory_pass_family_aware": bool(aggregate_rho > q99_fa),
        "n_pairs": int(len(m)),
        "n_perm": int(n_perm),
        "note": "Panel B is external grounding only; observational and non-causal. Does not validate Theorem 2.",
    }
    (out / "panel_b_alignment_summary.json").write_text(json.dumps(summary, indent=2))

    main_md = (
        f"| metric | value |\n|---|---|\n"
        f"| aggregate Spearman (rho vs CKA) | {aggregate_rho:.4f} |\n"
        f"| raw cov vs CKA Spearman | {raw_vs_cka:.4f} |\n"
        f"| partial Spearman (controlling abs acc diff) | {partial_rho:.4f} |\n"
        f"| family-aware null q99 | {q99_fa:.4f} |\n"
        f"| row/col null q99 | {q99_rc:.4f} |\n"
        f"| confirmatory pass (vs family-aware q99) | {summary['confirmatory_pass_family_aware']} |\n"
        f"| within-family mean rho | {within:.4f} |\n"
        f"| cross-family mean rho | {cross:.4f} |\n"
        f"| n_pairs | {len(m)} |\n"
    )
    warnings_list = []
    if not summary["confirmatory_pass_family_aware"]:
        warnings_list.append("aggregate Spearman did not exceed the 99th percentile of the family-aware null")
    write_markdown_report(
        str(out / "panel_b_report.md"),
        "Panel B Alignment Report",
        "full",
        config_path,
        config or {},
        seed,
        num_instances=int(len(m)),
        num_models_or_samples=len(models),
        invalid_parse_rate=0.0,
        main_table_md=main_md,
        figure_links=["results/figures/cka_alignment.png", "results/figures/cka_alignment.pdf"],
        warnings=["Panel B is external grounding only; observational and non-causal."] + warnings_list,
        missing_outputs=[],
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
            str(out / "panel_b_report.md"),
            "Panel B Alignment Report",
            "validate_only",
            a.config,
            cfg,
            int(cfg["seed"]),
            0,
            0,
            0.0,
            "validate_only run; no inputs required",
            [],
            [
                "Panel B is external grounding only; observational and non-causal.",
                "Full alignment analysis requires real pairwise stats and CKA CSVs.",
            ],
            ["panel_b_pairwise_stats.csv", "panel_b_cka.csv"],
        )
        return
    pw = cfg.get("pairwise_stats_csv")
    ck = cfg.get("cka_csv")
    if not pw or not ck or not Path(pw).exists() or not Path(ck).exists():
        raise FileNotFoundError(
            "Need pairwise_stats_csv and cka_csv pointing to real Panel B artefacts"
        )
    analyze(pw, ck, str(out), n_perm=int(cfg.get("n_perm", 10000)), seed=int(cfg["seed"]), config_path=a.config, config=cfg)


if __name__ == "__main__":
    main()
