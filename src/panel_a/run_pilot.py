from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from src.config.load_config import resolve_runtime_config
from src.theory.certificate import empirical_certificate_from_X
from src.theory.majority_risk import monte_carlo_reference_risk
from src.theory.estimators import estimate_basic_summaries


def _matrix_from_long(df: pd.DataFrame, K: int) -> np.ndarray:
    piv = df.pivot_table(index="instance_id", columns="sample_id", values="correct", aggfunc="first")
    if piv.shape[1] < K:
        raise ValueError("Not enough sample_id columns to build matrix")
    return piv.sort_index(axis=1).iloc[:, :K].to_numpy(dtype=int)


def run(cfg: dict) -> None:
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    in_path = cfg.get("pilot_correctness_csv")
    if not in_path or not Path(in_path).exists():
        raise FileNotFoundError("pilot_correctness_csv must point to real evaluation outputs")
    df = pd.read_csv(in_path)
    req=["instance_id","benchmark","protocol","sample_id","correct"]
    for c in req:
        if c not in df.columns: raise ValueError(f"Missing column {c}")
    rows=[]
    for (bench, proto), g in df.groupby(["benchmark", "protocol"]):
        X = _matrix_from_long(g, int(cfg["K_ref"]))
        M = X.shape[0]; K_est = int(cfg["K_est"])
        if K_est >= X.shape[1]: raise ValueError("K_est must be less than K_ref")
        X_est, X_ref = X[:, :K_est], X[:, K_est:]
        alpha_ref = X_ref.mean(axis=1)
        base = estimate_basic_summaries(X_est)
        for N in cfg["N_values"]:
            cert = empirical_certificate_from_X(X_est, int(N), float(cfg["global_delta"]))
            R_MC = monte_carlo_reference_risk(int(N), alpha_ref)
            rows.append({"benchmark":bench,"protocol":proto,"N":N,"R_cert":cert["R_cert"],"R_MC":R_MC,"coverage":int(cert["R_cert"]>=R_MC),"nonvacuous":int(cert["R_cert"]<1.0),"nonvacuous_lt_07":int(cert["R_cert"]<0.7),"nonvacuous_lt_03":int(cert["R_cert"]<0.3),"slack":cert["R_cert"]-R_MC,"undercoverage":int(cert["R_cert"]<R_MC),"m_hat":base["margin_hat"],"F_hat":base["F_hat_unbiased"],"m_L":cert["m_L"],"U_F":cert["U_F"],"eta_star":cert["eta_star"]})
    met = pd.DataFrame(rows)
    met.to_csv(out / "pilot_metrics.csv", index=False)
    frac_nonv = float((met["R_cert"] < 1.0).mean()) if len(met) else 0.0
    summary = {"num_cells": int(len(met)), "fraction_nonvacuous": frac_nonv, "warning_near_universal_vacuity": frac_nonv < 0.2}
    (out / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    report = ["# Panel A Pilot Report", f"Cells: {len(met)}", f"Fraction R_cert < 1: {frac_nonv:.3f}"]
    if frac_nonv < 0.2:
        report.append("Pilot indicates near-universal vacuity; adjust protocols before full Panel A.")
    (out / "pilot_report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--validate_only", action="store_true")
    args=ap.parse_args()
    cfg=resolve_runtime_config(args.config, args.output_dir, args.seed)
    if args.validate_only or cfg.get("validate_only", False):
        return
    run(cfg)

if __name__ == "__main__":
    main()
