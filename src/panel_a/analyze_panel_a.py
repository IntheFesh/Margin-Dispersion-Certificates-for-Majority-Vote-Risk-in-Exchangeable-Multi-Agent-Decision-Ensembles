from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _stratify(d: pd.DataFrame) -> dict:
    q_m = d["m_hat"].quantile([1 / 3, 2 / 3])
    q_f = d["F_hat_unbiased"].quantile([1 / 3, 2 / 3]) if "F_hat_unbiased" in d.columns else d["F_hat"].quantile([1 / 3, 2 / 3])
    f_col = "F_hat_unbiased" if "F_hat_unbiased" in d.columns else "F_hat"
    res = {}
    for name, mask in [
        ("high-m/low-F/large-N", (d["m_hat"] >= q_m.iloc[1]) & (d[f_col] <= q_f.iloc[0]) & (d["N"] >= 32)),
        ("low-m/high-F/small-N", (d["m_hat"] <= q_m.iloc[0]) & (d[f_col] >= q_f.iloc[1]) & (d["N"] < 32)),
    ]:
        s = d[mask]
        res[name] = float((s["R_cert"] < 1).mean()) if len(s) else float("nan")
    return res


def _baselines(d: pd.DataFrame) -> list[dict]:
    out = []
    cols = [("certificate", "R_cert")]
    if "R_hoeffding" in d.columns:
        cols.append(("hoeffding", "R_hoeffding"))
    if "R_cantelli" in d.columns:
        cols.append(("cantelli", "R_cantelli"))
    for name, col in cols:
        cov = float((d[col] >= d["R_MC"]).mean())
        sub = d.loc[d[col] >= d["R_MC"]]
        slack = float((sub[col] - sub["R_MC"]).mean()) if len(sub) else float("nan")
        nonv = float((d[col] < 1).mean())
        sp = float(spearmanr(d[col], d["R_MC"]).correlation) if len(d) > 1 else float("nan")
        out.append({
            "method": name,
            "undercoverage_rate": 1 - cov,
            "mean_slack_cond_coverage": slack,
            "nonvacuity_rate": nonv,
            "spearman_with_R_MC": sp,
        })
    return out


def analyze_panel_a(metrics_csv: str, out_dir: str | None = None) -> dict:
    d = pd.read_csv(metrics_csv)
    if "R_cert" not in d.columns or "R_MC" not in d.columns:
        raise ValueError("metrics CSV must include R_cert and R_MC columns")
    if "m_hat" not in d.columns:
        raise ValueError("metrics CSV must include m_hat column")
    res = {
        "stratified_nonvacuity": _stratify(d),
        "baseline_comparison": _baselines(d),
        "nonvacuity_rates": {
            "lt_1": float((d["R_cert"] < 1.0).mean()),
            "lt_0_7": float((d["R_cert"] < 0.7).mean()),
            "lt_0_3": float((d["R_cert"] < 0.3).mean()),
        },
    }
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        Path(out_dir, "panel_a_analysis_summary.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--output_dir")
    a = ap.parse_args()
    res = analyze_panel_a(a.metrics_csv, a.output_dir)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
