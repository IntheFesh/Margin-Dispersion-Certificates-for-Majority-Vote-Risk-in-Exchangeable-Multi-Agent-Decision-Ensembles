from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from src.config.load_config import resolve_runtime_config
from src.synthetic.simulate_bernoulli_mixture import simulate_alpha, simulate_X_from_alpha
from src.theory.estimators import estimate_F_unbiased, estimate_basic_summaries
from src.theory.certificate import empirical_certificate_from_X
from src.theory.majority_risk import mixture_majority_risk
from src.theory.insufficiency import reproduce_two_moment_insufficiency


def run(cfg: dict) -> dict:
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    syn = cfg["synthetic"]

    rows = []
    for t in range(int(syn["estimator_trials"])):
        alpha = simulate_alpha(int(syn["M"]), syn["alpha_distribution"], seed + t)
        X = simulate_X_from_alpha(alpha, int(syn["K"]), seed + 10_000 + t)
        rows.append({"trial": t, "F_hat": estimate_F_unbiased(X), "F_true": float(np.var(alpha, ddof=0))})
    est_df = pd.DataFrame(rows)
    est_df["error"] = est_df["F_hat"] - est_df["F_true"]
    est_df.to_csv(out / "estimator_check.csv", index=False)

    cert_rows = []
    for t in range(int(syn["certificate_trials"])):
        alpha = simulate_alpha(int(syn["M"]), syn["alpha_distribution"], seed + 200_000 + t)
        X = simulate_X_from_alpha(alpha, int(syn["K"]), seed + 400_000 + t)
        cert = empirical_certificate_from_X(X, int(syn["N"]), float(cfg["global_delta"]))
        r_true = mixture_majority_risk(int(syn["N"]), alpha)
        cert_rows.append({"trial": t, "R_cert": cert["R_cert"], "R_true": r_true, "covered": int(cert["R_cert"] >= r_true), "issued": int(cert["issued"]), "eta_star": cert["eta_star"]})
    cert_df = pd.DataFrame(cert_rows)
    cert_df.to_csv(out / "certificate_check.csv", index=False)

    high_alpha = simulate_alpha(int(syn["M"]), syn["nonvacuity_high"], seed + 700_000)
    high_X = simulate_X_from_alpha(high_alpha, int(syn["K"]), seed + 710_000)
    low_alpha = simulate_alpha(int(syn["M"]), syn["nonvacuity_low"], seed + 720_000)
    low_X = simulate_X_from_alpha(low_alpha, int(syn["K"]), seed + 730_000)
    high_cert = empirical_certificate_from_X(high_X, int(syn["N"]), float(cfg["global_delta"]))
    low_cert = empirical_certificate_from_X(low_X, int(syn["N"]), float(cfg["global_delta"]))

    insuff = reproduce_two_moment_insufficiency()
    summary = {
        "estimator_mean_error": float(est_df["error"].mean()),
        "coverage_rate": float(cert_df["covered"].mean()),
        "nominal_target": float(1 - cfg["global_delta"]),
        "high_setting_R_cert": float(high_cert["R_cert"]),
        "low_setting_R_cert": float(low_cert["R_cert"]),
        "insufficiency": insuff,
    }
    (out / "synthetic_summary.json").write_text(json.dumps(summary, indent=2))

    report = ["# Synthetic Checks", "", f"Coverage rate: {summary['coverage_rate']:.4f} (target ~ {summary['nominal_target']:.4f})", f"Estimator mean error: {summary['estimator_mean_error']:.6f}", f"High-margin low-dispersion R_cert: {summary['high_setting_R_cert']:.4f}", f"Low-margin/high-dispersion R_cert: {summary['low_setting_R_cert']:.4f}"]
    if summary["high_setting_R_cert"] >= 1.0:
        report.append("WARNING: high-margin setting was vacuous.")
    if summary["low_setting_R_cert"] < 0.95:
        report.append("WARNING: low/high-dispersion setting not near vacuous.")
    (out / "synthetic_report.md").write_text("\n".join(report) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--validate_only", action="store_true")
    args = ap.parse_args()
    cfg = resolve_runtime_config(args.config, args.output_dir, args.seed)
    if args.validate_only or cfg.get("validate_only", False):
        return
    run(cfg)


if __name__ == "__main__":
    main()
