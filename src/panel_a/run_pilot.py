from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.config.load_config import resolve_runtime_config
from src.io.report import write_markdown_report
from src.theory.certificate import empirical_certificate_from_X
from src.theory.majority_risk import monte_carlo_reference_risk
from src.theory.estimators import estimate_basic_summaries
from src.panel_a.split_reference_estimation import split_estimation_reference

REQUIRED_INPUT_COLUMNS = [
    "instance_id",
    "benchmark",
    "protocol",
    "sample_id",
    "correct",
]


def _validate_inputs(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"pilot correctness CSV missing columns: {missing}")
    if not set(df["correct"].dropna().unique()).issubset({0, 1}):
        raise ValueError("correct column must be in {0,1}")
    if df[REQUIRED_INPUT_COLUMNS].isna().any().any():
        raise ValueError("required columns contain NaN values")
    if df.duplicated(subset=["benchmark", "protocol", "instance_id", "sample_id"]).any():
        raise ValueError("duplicate (benchmark, protocol, instance_id, sample_id) rows")


def _matrix_from_long(df: pd.DataFrame, K: int) -> np.ndarray:
    piv = df.pivot_table(index="instance_id", columns="sample_id", values="correct", aggfunc="first")
    if piv.isna().any().any():
        raise ValueError(
            "pivoted correctness matrix contains NaN; check that every (instance_id, sample_id) cell is populated"
        )
    if piv.shape[1] < K:
        raise ValueError(
            f"not enough sample_id columns to build matrix: have {piv.shape[1]}, need {K}"
        )
    return piv.sort_index(axis=1).iloc[:, :K].to_numpy(dtype=int)


def run(cfg: dict, config_path: str = "") -> dict:
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    in_path = cfg.get("pilot_correctness_csv")
    if not in_path or not Path(in_path).exists():
        raise FileNotFoundError(
            "pilot_correctness_csv must point to real per-(instance, sample) evaluation outputs"
        )
    df = pd.read_csv(in_path)
    _validate_inputs(df)
    seed = int(cfg["seed"])
    K_ref = int(cfg["K_ref"])
    K_est = int(cfg["K_est"])
    delta = float(cfg["global_delta"])
    rows = []
    invalid_rate_acc = 0.0
    n_total = 0
    for (bench, proto), g in df.groupby(["benchmark", "protocol"]):
        X = _matrix_from_long(g, K_ref)
        split = split_estimation_reference(X, K_est, seed)
        X_est, X_ref = split["X_est"], split["X_ref"]
        alpha_ref = X_ref.mean(axis=1)
        base = estimate_basic_summaries(X_est)
        n_total += int((g["correct"]).notna().sum())
        invalid_rate_acc += float(g.get("invalid_parse", pd.Series([0] * len(g))).mean()) if "invalid_parse" in g.columns else 0.0
        for N in cfg["N_values"]:
            cert = empirical_certificate_from_X(X_est, int(N), delta)
            R_MC = monte_carlo_reference_risk(int(N), alpha_ref)
            slack = float(cert["R_cert"] - R_MC)
            rows.append({
                "benchmark": bench,
                "protocol": proto,
                "N": int(N),
                "M": int(X_est.shape[0]),
                "K_ref": int(K_ref),
                "K_est": int(K_est),
                "m_hat": float(base["margin_hat"]),
                "F_hat_unbiased": float(base["F_hat_unbiased"]),
                "F_hat_clipped": float(base["F_hat_clipped"]),
                "L_alpha": float(cert["L_alpha"]),
                "U_F": float(cert["U_F"]),
                "m_L": float(cert["m_L"]),
                "issued": bool(cert["issued"]),
                "eta_star": cert["eta_star"],
                "R_cert": float(cert["R_cert"]),
                "R_MC": float(R_MC),
                "coverage_indicator": int(cert["R_cert"] >= R_MC),
                "slack": slack,
                "undercoverage": int(cert["R_cert"] < R_MC),
                "nonvacuous_lt_1": int(cert["R_cert"] < 1.0),
                "nonvacuous_lt_0_7": int(cert["R_cert"] < 0.7),
                "nonvacuous_lt_0_3": int(cert["R_cert"] < 0.3),
            })
    met = pd.DataFrame(rows)
    met.to_csv(out / "pilot_metrics.csv", index=False)
    frac_nonv = float((met["R_cert"] < 1.0).mean()) if len(met) else 0.0
    coverage_rate = float(met["coverage_indicator"].mean()) if len(met) else 0.0
    invalid_rate = float(df["invalid_parse"].mean()) if "invalid_parse" in df.columns else 0.0
    summary = {
        "num_cells": int(len(met)),
        "fraction_nonvacuous": frac_nonv,
        "warning_near_universal_vacuity": frac_nonv < 0.2,
        "coverage_rate_vs_R_MC": coverage_rate,
        "invalid_parse_rate": invalid_rate,
    }
    (out / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    warnings_list = []
    if summary["warning_near_universal_vacuity"]:
        warnings_list.append(
            "Pilot indicates near-universal vacuity; adjust protocols before full Panel A. (Pilot does not auto-adjust configs.)"
        )
    main_md = (
        f"| metric | value |\n|---|---|\n"
        f"| cells | {summary['num_cells']} |\n"
        f"| non-vacuous fraction (R_cert<1) | {frac_nonv:.4f} |\n"
        f"| coverage vs R_MC | {coverage_rate:.4f} |\n"
        f"| invalid parse rate | {invalid_rate:.4f} |\n"
    )
    write_markdown_report(
        str(out / "pilot_report.md"),
        "Panel A Pilot Report",
        "pilot",
        config_path,
        cfg,
        seed,
        int(df["instance_id"].nunique()),
        int(df["sample_id"].nunique()),
        invalid_rate,
        main_md,
        ["results/figures/design_space.png", "results/figures/certificate_vs_reference.png"],
        warnings_list + ["Panel A pilot is a gate; no automatic config changes."],
        [],
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--validate_only", action="store_true")
    args = ap.parse_args()
    cfg = resolve_runtime_config(args.config, args.output_dir, args.seed)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if args.validate_only or cfg.get("validate_only", False):
        write_markdown_report(
            str(out / "pilot_report.md"),
            "Panel A Pilot Report",
            "validate_only",
            args.config,
            cfg,
            int(cfg["seed"]),
            0,
            0,
            0.0,
            "validate_only run; no inference executed",
            [],
            [
                "Real pilot inference not run; pilot requires per-(instance, sample) correctness outputs.",
            ],
            ["pilot_metrics.csv", "pilot_summary.json"],
        )
        return
    run(cfg, config_path=args.config)


if __name__ == "__main__":
    main()
