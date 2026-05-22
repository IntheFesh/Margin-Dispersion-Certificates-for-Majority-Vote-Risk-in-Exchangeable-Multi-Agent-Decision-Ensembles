from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.config.load_config import resolve_runtime_config
from src.io.report import write_markdown_report
from src.synthetic.simulate_bernoulli_mixture import simulate_alpha, simulate_X_from_alpha
from src.theory.estimators import estimate_F_unbiased, estimate_basic_summaries
from src.theory.certificate import empirical_certificate_from_X, population_certificate
from src.theory.majority_risk import mixture_majority_risk
from src.theory.insufficiency import reproduce_two_moment_insufficiency


def _N_values(syn: dict) -> list[int]:
    if "N_values" in syn:
        return [int(x) for x in syn["N_values"]]
    return [int(syn["N"])]


def run(cfg: dict, config_path: str = "") -> dict:
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    syn = cfg["synthetic"]
    delta = float(cfg["global_delta"])

    # Estimator unbiasedness check (averaged over trials).
    est_rows = []
    for t in range(int(syn["estimator_trials"])):
        alpha = simulate_alpha(int(syn["M"]), syn["alpha_distribution"], seed + t)
        X = simulate_X_from_alpha(alpha, int(syn["K"]), seed + 10_000 + t)
        est_rows.append({
            "trial": t,
            "F_hat": estimate_F_unbiased(X),
            "F_true": float(np.var(alpha, ddof=0)),
            "mean_alpha": float(alpha.mean()),
        })
    est_df = pd.DataFrame(est_rows)
    est_df["error"] = est_df["F_hat"] - est_df["F_true"]
    est_df.to_csv(out / "estimator_check.csv", index=False)

    # Empirical certificate coverage check across multiple N.
    cert_rows = []
    N_values = _N_values(syn)
    primary_N = int(syn.get("N", N_values[0]))
    for t in range(int(syn["certificate_trials"])):
        alpha = simulate_alpha(int(syn["M"]), syn["alpha_distribution"], seed + 200_000 + t)
        X = simulate_X_from_alpha(alpha, int(syn["K"]), seed + 400_000 + t)
        for N in N_values:
            cert = empirical_certificate_from_X(X, int(N), delta)
            r_true = mixture_majority_risk(int(N), alpha)
            cert_rows.append({
                "trial": t,
                "N": int(N),
                "R_cert": float(cert["R_cert"]),
                "R_true": float(r_true),
                "issued": bool(cert["issued"]),
                "eta_star": cert["eta_star"],
                "covered": int(cert["R_cert"] >= r_true),
            })
    cert_df = pd.DataFrame(cert_rows)
    cert_df.to_csv(out / "certificate_check.csv", index=False)
    coverage_by_N = cert_df.groupby("N")["covered"].mean().to_dict()

    # Non-vacuity demonstration (population certificate and empirical certificate).
    high_alpha = simulate_alpha(int(syn["M"]), syn["nonvacuity_high"], seed + 700_000)
    high_X = simulate_X_from_alpha(high_alpha, int(syn["K"]), seed + 710_000)
    low_alpha = simulate_alpha(int(syn["M"]), syn["nonvacuity_low"], seed + 720_000)
    low_X = simulate_X_from_alpha(low_alpha, int(syn["K"]), seed + 730_000)
    nv_rows = []
    for label, alpha, X in [("high", high_alpha, high_X), ("low", low_alpha, low_X)]:
        for N in N_values:
            emp = empirical_certificate_from_X(X, int(N), delta)
            pop = population_certificate(float(alpha.mean()), float(np.var(alpha, ddof=0)), int(N))
            base = estimate_basic_summaries(X)
            r_true = mixture_majority_risk(int(N), alpha)
            nv_rows.append({
                "setting": label,
                "N": int(N),
                "M": int(X.shape[0]),
                "K": int(X.shape[1]),
                "bar_alpha": float(alpha.mean()),
                "F_true": float(np.var(alpha, ddof=0)),
                "F_hat_unbiased": float(base["F_hat_unbiased"]),
                "empirical_R_cert": float(emp["R_cert"]),
                "population_R_cert": float(pop["R_cert"]),
                "R_true": float(r_true),
                "issued_empirical": bool(emp["issued"]),
                "eta_star_population": pop.get("eta_star"),
            })
    nv_df = pd.DataFrame(nv_rows)
    nv_df.to_csv(out / "nonvacuity_check.csv", index=False)

    insuff = reproduce_two_moment_insufficiency()
    pd.DataFrame([insuff]).to_csv(out / "insufficiency_check.csv", index=False)

    # Aggregate summary.
    summary = {
        "estimator_mean_error": float(est_df["error"].mean()),
        "estimator_std_error": float(est_df["error"].std(ddof=1)),
        "coverage_rate_by_N": {int(k): float(v) for k, v in coverage_by_N.items()},
        "primary_N": primary_N,
        "primary_coverage_rate": float(coverage_by_N.get(primary_N, float("nan"))),
        "nominal_target": 1 - delta,
        "high_setting_min_empirical_R_cert": float(nv_df.loc[nv_df["setting"] == "high", "empirical_R_cert"].min()),
        "high_setting_min_population_R_cert": float(nv_df.loc[nv_df["setting"] == "high", "population_R_cert"].min()),
        "low_setting_min_empirical_R_cert": float(nv_df.loc[nv_df["setting"] == "low", "empirical_R_cert"].min()),
        "insufficiency": insuff,
    }
    (out / "synthetic_summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown report including config provenance and warnings.
    main_md_lines = [
        "| metric | value |",
        "|---|---|",
        f"| estimator mean error | {summary['estimator_mean_error']:.6f} |",
        f"| estimator std error | {summary['estimator_std_error']:.6f} |",
        f"| primary N | {primary_N} |",
        f"| primary coverage rate | {summary['primary_coverage_rate']:.4f} |",
        f"| nominal coverage target | {summary['nominal_target']:.4f} |",
        f"| high-setting min empirical R_cert | {summary['high_setting_min_empirical_R_cert']:.4f} |",
        f"| high-setting min population R_cert | {summary['high_setting_min_population_R_cert']:.4f} |",
        f"| low-setting min empirical R_cert | {summary['low_setting_min_empirical_R_cert']:.4f} |",
        "",
        "| insufficiency | mu1 | mu2 |",
        "|---|---|---|",
        f"| mean | {insuff['mu1_mean']:.6f} | {insuff['mu2_mean']:.6f} |",
        f"| variance | {insuff['mu1_var']:.6f} | {insuff['mu2_var']:.6f} |",
        f"| R3 | {insuff['R3_mu1']:.6f} | {insuff['R3_mu2']:.6f} |",
    ]
    warnings_list: list[str] = []
    if summary["high_setting_min_empirical_R_cert"] >= 1.0 and summary["high_setting_min_population_R_cert"] >= 1.0:
        warnings_list.append("Neither population nor empirical R_cert dropped below 1 in the high-margin setting; check non-vacuity config.")
    if summary["low_setting_min_empirical_R_cert"] < 0.95:
        warnings_list.append("Low/high-dispersion setting was unexpectedly non-vacuous; check non-vacuity config.")
    if abs(summary["primary_coverage_rate"] - (1 - delta)) > 0.15:
        warnings_list.append(f"Primary coverage rate {summary['primary_coverage_rate']:.4f} deviates from nominal {(1-delta):.4f} by more than 0.15; Monte Carlo fluctuation.")
    write_markdown_report(
        str(out / "synthetic_report.md"),
        "Synthetic Checks Report",
        "synthetic",
        config_path,
        cfg,
        seed,
        int(syn["M"]),
        int(syn["K"]),
        0.0,
        "\n".join(main_md_lines),
        [],
        warnings_list,
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
    if args.validate_only or cfg.get("validate_only", False):
        return
    run(cfg, config_path=args.config)


if __name__ == "__main__":
    main()
