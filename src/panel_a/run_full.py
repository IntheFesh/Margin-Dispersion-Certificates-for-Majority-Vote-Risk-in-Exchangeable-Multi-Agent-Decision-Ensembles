from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config.load_config import resolve_runtime_config
from src.io.report import write_markdown_report
from src.theory.estimators import estimate_basic_summaries
from src.theory.certificate import (
    empirical_certificate_from_X,
    margin_only_hoeffding_baseline,
    asymptotic_cantelli_bound,
)
from src.theory.majority_risk import monte_carlo_reference_risk
from src.panel_a.split_reference_estimation import split_estimation_reference

REQUIRED_INPUT_COLUMNS = [
    "instance_id",
    "benchmark",
    "protocol",
    "regime",
    "sample_id",
    "correct",
]


def _validate_inputs(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"full correctness CSV missing columns: {missing}")
    if not set(df["correct"].dropna().unique()).issubset({0, 1}):
        raise ValueError("correct column must be in {0,1}")
    if df[REQUIRED_INPUT_COLUMNS].isna().any().any():
        raise ValueError("required columns contain NaN values")
    if df.duplicated(subset=["benchmark", "protocol", "regime", "instance_id", "sample_id"]).any():
        raise ValueError("duplicate (benchmark, protocol, regime, instance_id, sample_id) rows")


def _matrix(df: pd.DataFrame, K: int) -> np.ndarray:
    piv = df.pivot_table(index="instance_id", columns="sample_id", values="correct", aggfunc="first").sort_index(axis=1)
    if piv.isna().any().any():
        raise ValueError("pivoted correctness matrix contains NaN; check inputs")
    if piv.shape[1] < K:
        raise ValueError(f"insufficient K_ref columns: have {piv.shape[1]}, need {K}")
    return piv.iloc[:, :K].to_numpy(dtype=int)


def run(cfg: dict, config_path: str = "") -> dict:
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cfg["full_correctness_csv"])
    _validate_inputs(df)
    seed = int(cfg["seed"])
    K_ref = int(cfg["K_ref"])
    delta_global = float(cfg["global_delta"])
    B_boot = int(cfg["B_boot"])
    # §3.3 / §7: Bonferroni adjustment. The paper defines C as the number of
    # (protocol, benchmark) cells; we extend this conservatively to include
    # regime (each stress-test regime has a separate (L_alpha, U_F)
    # construction). Within a cell at fixed K_est, all N values share the
    # statistical budget since the binomial tail is analytic.
    cells = sorted({
        (b, p, r) for b, p, r in df[["benchmark", "protocol", "regime"]].itertuples(index=False, name=None)
    })
    C = max(1, len(cells))
    delta = delta_global / C
    rows = []
    boot = []
    for (bench, proto, reg), g in df.groupby(["benchmark", "protocol", "regime"]):
        X = _matrix(g, K_ref)
        for K_est in cfg["K_est_values"]:
            split = split_estimation_reference(X, int(K_est), seed)
            X_est, X_ref = split["X_est"], split["X_ref"]
            a_ref = X_ref.mean(axis=1)
            base = estimate_basic_summaries(X_est)
            for N in cfg["N_values"]:
                cert = empirical_certificate_from_X(X_est, int(N), delta)
                R_MC = float(monte_carlo_reference_risk(int(N), a_ref))
                hoe = float(margin_only_hoeffding_baseline(base["bar_alpha_hat"], int(N)))
                can = float(asymptotic_cantelli_bound(base["bar_alpha_hat"], base["F_hat_clipped"]))
                rows.append({
                    "benchmark": bench,
                    "protocol": proto,
                    "regime": reg,
                    "K_est": int(K_est),
                    "N": int(N),
                    "M": int(X_est.shape[0]),
                    "K_ref": int(K_ref),
                    "delta_global": delta_global,
                    "delta_cell": delta,
                    "m_hat": float(base["margin_hat"]),
                    "F_hat_unbiased": float(base["F_hat_unbiased"]),
                    "F_hat_clipped": float(base["F_hat_clipped"]),
                    "L_alpha": float(cert["L_alpha"]),
                    "U_F": float(cert["U_F"]),
                    "m_L": float(cert["m_L"]),
                    "issued": bool(cert["issued"]),
                    "eta_star": cert["eta_star"],
                    "R_cert": float(cert["R_cert"]),
                    "R_MC": R_MC,
                    "R_hoeffding": hoe,
                    "R_cantelli": can,
                    "slack": float(cert["R_cert"] - R_MC),
                    "undercoverage": int(cert["R_cert"] < R_MC),
                    "coverage_cert": int(cert["R_cert"] >= R_MC),
                    "coverage_hoeffding": int(hoe >= R_MC),
                    "coverage_cantelli": int(can >= R_MC),
                    "nonvacuous_lt_1": int(cert["R_cert"] < 1.0),
                    "nonvacuous_lt_0_7": int(cert["R_cert"] < 0.7),
                    "nonvacuous_lt_0_3": int(cert["R_cert"] < 0.3),
                })
                # §4.5 Analysis 1: bootstrap-resample queries within the cell;
                # recompute BOTH R_cert and R_MC on the resampled instances and
                # record per-replicate coverage = 1{R_cert_b >= R_MC_b}.
                rng = np.random.default_rng(seed + int(N) * 17 + int(K_est) * 31)
                M_cell = X_est.shape[0]
                for b in range(B_boot):
                    idx = rng.integers(0, M_cell, size=M_cell)
                    xb_est = X_est[idx, :]
                    xb_ref = X_ref[idx, :]
                    a_ref_b = xb_ref.mean(axis=1)
                    cb = empirical_certificate_from_X(xb_est, int(N), delta)
                    R_MC_b = float(monte_carlo_reference_risk(int(N), a_ref_b))
                    boot.append({
                        "benchmark": bench,
                        "protocol": proto,
                        "regime": reg,
                        "K_est": int(K_est),
                        "N": int(N),
                        "boot_id": int(b),
                        "R_cert_boot": float(cb["R_cert"]),
                        "R_MC_boot": R_MC_b,
                        "coverage_indicator_boot": int(cb["R_cert"] >= R_MC_b),
                    })
    m = pd.DataFrame(rows)
    mb = pd.DataFrame(boot)
    m.to_csv(out / "panel_a_cell_metrics.csv", index=False)
    mb.to_csv(out / "panel_a_bootstrap_metrics.csv", index=False)
    m[
        [
            "benchmark","protocol","regime","K_est","N","R_MC","R_cert","R_hoeffding","R_cantelli",
            "coverage_cert","coverage_hoeffding","coverage_cantelli","slack",
        ]
    ].to_csv(out / "panel_a_baseline_comparison.csv", index=False)
    nv = (
        m.groupby(["benchmark", "protocol", "regime", "K_est", "N"])
        .agg(
            nonvacuous=("R_cert", lambda s: float((s < 1).mean())),
            lt07=("R_cert", lambda s: float((s < 0.7).mean())),
            lt03=("R_cert", lambda s: float((s < 0.3).mean())),
            mean_R_cert=("R_cert", "mean"),
            mean_R_MC=("R_MC", "mean"),
        )
        .reset_index()
    )
    nv.to_csv(out / "panel_a_nonvacuity_summary.csv", index=False)

    q_m = m["m_hat"].quantile([1 / 3, 2 / 3])
    q_f = m["F_hat_unbiased"].quantile([1 / 3, 2 / 3])

    def bucket(r):
        hm = r["m_hat"] >= q_m.iloc[1]
        lf = r["F_hat_unbiased"] <= q_f.iloc[0]
        ln = r["N"] >= 32
        lm = r["m_hat"] <= q_m.iloc[0]
        hf = r["F_hat_unbiased"] >= q_f.iloc[1]
        sn = r["N"] < 32
        if hm and lf and ln:
            return "high-m/low-F/large-N"
        if lm and hf and sn:
            return "low-m/high-F/small-N"
        return "other"

    m["stratum"] = m.apply(bucket, axis=1)
    strat = m.groupby("stratum")["nonvacuous_lt_1"].mean().to_dict()

    baselines = []
    for name, col in [("certificate", "R_cert"), ("hoeffding", "R_hoeffding"), ("cantelli", "R_cantelli")]:
        cov = float((m[col] >= m["R_MC"]).mean())
        sub = m.loc[m[col] >= m["R_MC"]]
        slack = float((sub[col] - sub["R_MC"]).mean()) if len(sub) else float("nan")
        nonv = float((m[col] < 1).mean())
        sp = float(spearmanr(m[col], m["R_MC"]).correlation) if len(m) > 1 else float("nan")
        baselines.append({
            "method": name,
            "undercoverage_rate": 1 - cov,
            "mean_slack_cond_coverage": slack,
            "nonvacuity_rate": nonv,
            "spearman_with_R_MC": sp,
        })
    pd.DataFrame(baselines).to_csv(out / "panel_a_baseline_summary.csv", index=False)

    # Bootstrap per-cell coverage rate against R_MC (§4.5 Analysis 1).
    boot_cov = (
        mb.groupby(["benchmark", "protocol", "regime", "K_est", "N"])["coverage_indicator_boot"]
        .mean()
        .reset_index(name="bootstrap_coverage_rate")
    )
    boot_cov.to_csv(out / "panel_a_bootstrap_coverage.csv", index=False)

    summary = {
        "num_cells": int(len(m)),
        "C_protocol_benchmark_regime_cells": int(C),
        "delta_global": delta_global,
        "delta_cell": delta,
        "fraction_nonvacuous": float((m["R_cert"] < 1.0).mean()) if len(m) else 0.0,
        "coverage_rate_cert_vs_R_MC": float(m["coverage_cert"].mean()) if len(m) else 0.0,
        "bootstrap_coverage_mean": float(boot_cov["bootstrap_coverage_rate"].mean()) if len(boot_cov) else float("nan"),
        "bootstrap_coverage_min": float(boot_cov["bootstrap_coverage_rate"].min()) if len(boot_cov) else float("nan"),
        "stratified_nonvacuity": {k: float(v) for k, v in strat.items()},
        "baselines": baselines,
        "bootstrap_replicates_per_cell": B_boot,
    }
    (out / "panel_a_summary.json").write_text(json.dumps(summary, indent=2))

    main_md_rows = [
        "| metric | value |",
        "|---|---|",
        f"| cells (rows) | {summary['num_cells']} |",
        f"| (protocol,benchmark,regime) cells C | {C} |",
        f"| delta_global | {delta_global:.4f} |",
        f"| delta_cell (Bonferroni) | {delta:.6f} |",
        f"| non-vacuous fraction (R_cert<1) | {summary['fraction_nonvacuous']:.4f} |",
        f"| pointwise coverage vs R_MC | {summary['coverage_rate_cert_vs_R_MC']:.4f} |",
        f"| bootstrap coverage mean | {summary['bootstrap_coverage_mean']:.4f} |",
        f"| bootstrap coverage min | {summary['bootstrap_coverage_min']:.4f} |",
        "",
        "| method | undercov | mean slack | non-vac rate | Spearman vs R_MC |",
        "|---|---|---|---|---|",
    ]
    for b in baselines:
        main_md_rows.append(
            f"| {b['method']} | {b['undercoverage_rate']:.4f} | {b['mean_slack_cond_coverage']:.4f} | {b['nonvacuity_rate']:.4f} | {b['spearman_with_R_MC']:.4f} |"
        )
    main_md_rows += ["", "| stratum | non-vacuity |", "|---|---|"]
    for k, v in strat.items():
        main_md_rows.append(f"| {k} | {v:.4f} |")
    main_md = "\n".join(main_md_rows)

    warnings_list = []
    if summary["fraction_nonvacuous"] < 0.2:
        warnings_list.append("Less than 20% of cells have R_cert<1; near-universal vacuity.")
    write_markdown_report(
        str(out / "panel_a_report.md"),
        "Panel A Full Report",
        "full",
        config_path,
        cfg,
        seed,
        int(df["instance_id"].nunique()),
        int(df["sample_id"].nunique()),
        invalid_parse_rate=float(df.get("invalid_parse", pd.Series([0] * len(df))).mean()) if "invalid_parse" in df.columns else 0.0,
        main_table_md=main_md,
        figure_links=[
            "results/figures/design_space.png",
            "results/figures/certificate_vs_reference.png",
            "results/figures/baseline_comparison.png",
            "results/figures/nonvacuity.png",
        ],
        warnings=["Empirical coverage check is not a formal proof of the bound."] + warnings_list,
        missing_outputs=[],
    )
    return summary


def main() -> None:
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
            str(out / "panel_a_report.md"),
            "Panel A Full Report",
            "validate_only",
            a.config,
            cfg,
            int(cfg["seed"]),
            0,
            0,
            0.0,
            "validate_only run; no inference executed",
            [],
            ["Real Panel A full inference not run; requires real per-(instance, sample) correctness outputs."],
            [
                "panel_a_cell_metrics.csv",
                "panel_a_bootstrap_metrics.csv",
                "panel_a_baseline_comparison.csv",
                "panel_a_nonvacuity_summary.csv",
                "panel_a_baseline_summary.csv",
            ],
        )
        return
    run(cfg, config_path=a.config)


if __name__ == "__main__":
    main()
