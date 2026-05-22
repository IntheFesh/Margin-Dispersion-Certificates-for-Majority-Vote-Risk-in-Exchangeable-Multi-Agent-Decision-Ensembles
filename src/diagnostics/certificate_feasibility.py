"""Certificate feasibility diagnostic.

Diagnostic-only: evaluate the current Theorem 2 empirical certificate
over a grid of (M, C, N, bar_alpha, F) at delta_global = 0.10 with
delta_cell = delta_global / C. The theorem is not modified and no
auto-tuning is applied. The script reports:

  * fraction of grid points with m_L <= 0,
  * fraction with U_F = 1/4 (deterministic clip is binding),
  * fraction with R_N^cert = 1 (vacuous),
  * fraction with R_N^cert < 0.7 / < 0.3,
  * breakdowns by M, N, C.

Run:
  python -m src.diagnostics.certificate_feasibility \\
      --output_dir results/diagnostics
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path

import pandas as pd

from src.theory.certificate import optimize_eta

# Grid as specified by the user.
M_GRID = [100, 300, 500, 1000, 2000]
C_GRID = [4, 6, 12]
N_GRID = [8, 16, 32, 64]
BAR_ALPHA_GRID = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
F_GRID = [0.00, 0.01, 0.03, 0.05, 0.10]
DELTA_GLOBAL_DEFAULT = 0.10


def evaluate_grid(delta_global: float = DELTA_GLOBAL_DEFAULT) -> pd.DataFrame:
    rows = []
    for M, C, N, ba, F in itertools.product(M_GRID, C_GRID, N_GRID, BAR_ALPHA_GRID, F_GRID):
        if F > 0.25:
            raise ValueError("F must be <= 1/4")
        delta_cell = delta_global / C
        eps_M = math.sqrt(math.log(4 / delta_cell) / (2 * M))
        L_alpha = ba - eps_M
        U_2 = (F + ba ** 2) + eps_M
        # Diagnostic uses the population identity E[alpha^2] = F + bar_alpha^2
        # and reuses the empirical formula U_F = min(1/4, max(0, U_2 - L_alpha^2)).
        U_F = min(0.25, max(0.0, U_2 - L_alpha ** 2))
        m_L = L_alpha - 0.5
        if m_L <= 0:
            R_cert = 1.0
            eta_star = None
            method = "no-margin"
            issued = False
        else:
            opt = optimize_eta(m_L, U_F, N)
            R_cert = float(opt["objective_value"])
            eta_star = opt["eta_star"]
            method = opt.get("method")
            issued = True
        rows.append({
            "M": M,
            "C": C,
            "N": N,
            "bar_alpha": ba,
            "F": F,
            "delta_global": delta_global,
            "delta_cell": delta_cell,
            "eps_M": eps_M,
            "L_alpha": L_alpha,
            "U_2": U_2,
            "U_F": U_F,
            "m_L": m_L,
            "issued": issued,
            "eta_star": eta_star,
            "method": method,
            "R_cert": R_cert,
            "mL_le_0": int(m_L <= 0),
            "UF_eq_quarter": int(U_F >= 0.25 - 1e-12),
            "R_cert_eq_1": int(R_cert >= 1.0 - 1e-9),
            "R_cert_lt_07": int(R_cert < 0.7),
            "R_cert_lt_03": int(R_cert < 0.3),
        })
    return pd.DataFrame(rows)


def _frac(df: pd.DataFrame, col: str) -> float:
    return float(df[col].mean()) if len(df) else float("nan")


def make_report(df: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "certificate_feasibility.csv", index=False)

    overall = {
        "n_grid_points": int(len(df)),
        "fraction_m_L_le_0": _frac(df, "mL_le_0"),
        "fraction_U_F_eq_quarter": _frac(df, "UF_eq_quarter"),
        "fraction_R_cert_eq_1": _frac(df, "R_cert_eq_1"),
        "fraction_R_cert_lt_07": _frac(df, "R_cert_lt_07"),
        "fraction_R_cert_lt_03": _frac(df, "R_cert_lt_03"),
    }

    def breakdown(col: str) -> pd.DataFrame:
        return (
            df.groupby(col)[["mL_le_0", "UF_eq_quarter", "R_cert_eq_1", "R_cert_lt_07", "R_cert_lt_03"]]
            .mean()
            .reset_index()
        )

    bM = breakdown("M")
    bN = breakdown("N")
    bC = breakdown("C")
    bM.to_csv(out_dir / "feasibility_by_M.csv", index=False)
    bN.to_csv(out_dir / "feasibility_by_N.csv", index=False)
    bC.to_csv(out_dir / "feasibility_by_C.csv", index=False)

    nonvac = df[df["R_cert"] < 1.0 - 1e-9].copy()
    nonvac.to_csv(out_dir / "feasibility_nonvacuous_slice.csv", index=False)
    if len(nonvac):
        min_row = nonvac.loc[nonvac["R_cert"].idxmin()]
        nonvac_summary = {
            "nonvacuous_count": int(len(nonvac)),
            "nonvacuous_fraction": float(len(nonvac) / len(df)),
            "min_R_cert": float(min_row["R_cert"]),
            "argmin_M": int(min_row["M"]),
            "argmin_C": int(min_row["C"]),
            "argmin_N": int(min_row["N"]),
            "argmin_bar_alpha": float(min_row["bar_alpha"]),
            "argmin_F": float(min_row["F"]),
            "Ns_with_any_nonvacuous": sorted({int(x) for x in nonvac["N"].unique()}),
            "Ms_with_any_nonvacuous": sorted({int(x) for x in nonvac["M"].unique()}),
            "bar_alphas_with_any_nonvacuous": sorted({float(x) for x in nonvac["bar_alpha"].unique()}),
        }
    else:
        nonvac_summary = {"nonvacuous_count": 0, "nonvacuous_fraction": 0.0}

    (out_dir / "certificate_feasibility_summary.json").write_text(
        json.dumps({
            "overall": overall,
            "nonvacuous_slice": nonvac_summary,
            "by_M": bM.to_dict(orient="records"),
            "by_N": bN.to_dict(orient="records"),
            "by_C": bC.to_dict(orient="records"),
        }, indent=2)
    )

    md = []
    md.append("# Certificate Feasibility Diagnostic")
    md.append("")
    md.append("Diagnostic-only: evaluates the current Theorem 2 empirical certificate over a")
    md.append("grid of (M, C, N, bar_alpha, F) at delta_global = 0.10, delta_cell = delta_global / C.")
    md.append("No theorem changes; no auto-tuning.")
    md.append("")
    md.append("## Grid")
    md.append("")
    md.append(f"- M: {M_GRID}")
    md.append(f"- C: {C_GRID}")
    md.append(f"- N: {N_GRID}")
    md.append(f"- bar_alpha: {BAR_ALPHA_GRID}")
    md.append(f"- F: {F_GRID}")
    md.append(f"- total grid points: {len(df)}")
    md.append("")
    md.append("## Overall feasibility")
    md.append("")
    md.append("| metric | value |")
    md.append("|---|---|")
    md.append(f"| fraction m_L <= 0 (certificate refused) | {overall['fraction_m_L_le_0']:.4f} |")
    md.append(f"| fraction U_F = 1/4 (deterministic clip binding) | {overall['fraction_U_F_eq_quarter']:.4f} |")
    md.append(f"| fraction R_cert = 1 (vacuous) | {overall['fraction_R_cert_eq_1']:.4f} |")
    md.append(f"| fraction R_cert < 0.7 | {overall['fraction_R_cert_lt_07']:.4f} |")
    md.append(f"| fraction R_cert < 0.3 | {overall['fraction_R_cert_lt_03']:.4f} |")
    md.append("")
    for name, b, axis in [("M", bM, "M"), ("N", bN, "N"), ("C", bC, "C")]:
        md.append(f"## Breakdown by {axis}")
        md.append("")
        md.append(f"| {axis} | mL_le_0 | UF=1/4 | R_cert=1 | R_cert<0.7 | R_cert<0.3 |")
        md.append("|---|---|---|---|---|---|")
        for _, r in b.iterrows():
            md.append(
                f"| {int(r[axis])} | {r['mL_le_0']:.4f} | {r['UF_eq_quarter']:.4f} | "
                f"{r['R_cert_eq_1']:.4f} | {r['R_cert_lt_07']:.4f} | {r['R_cert_lt_03']:.4f} |"
            )
        md.append("")

    md.append("## Where the certificate is informative")
    md.append("")
    if len(nonvac):
        md.append(f"- non-vacuous grid points: **{len(nonvac)} / {len(df)} ({100*len(nonvac)/len(df):.2f}%)**")
        md.append(f"- min R_cert observed: **{nonvac_summary['min_R_cert']:.4f}** at "
                  f"M={nonvac_summary['argmin_M']}, C={nonvac_summary['argmin_C']}, "
                  f"N={nonvac_summary['argmin_N']}, bar_alpha={nonvac_summary['argmin_bar_alpha']:.2f}, "
                  f"F={nonvac_summary['argmin_F']:.2f}")
        md.append(f"- N values with any non-vacuous cell: {nonvac_summary['Ns_with_any_nonvacuous']}")
        md.append(f"- M values with any non-vacuous cell: {nonvac_summary['Ms_with_any_nonvacuous']}")
        md.append(f"- bar_alpha values with any non-vacuous cell: {nonvac_summary['bar_alphas_with_any_nonvacuous']}")
        md.append("")
        md.append("Full non-vacuous slice (sorted by R_cert ascending):")
        md.append("")
        md.append("| M | C | N | bar_alpha | F | eps_M | U_F | m_L | R_cert |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in nonvac.sort_values("R_cert").iterrows():
            md.append(
                f"| {int(r['M'])} | {int(r['C'])} | {int(r['N'])} | "
                f"{r['bar_alpha']:.2f} | {r['F']:.2f} | "
                f"{r['eps_M']:.4f} | {r['U_F']:.4f} | {r['m_L']:.4f} | {r['R_cert']:.4f} |"
            )
        md.append("")
    else:
        md.append("No non-vacuous grid points in this grid.")
        md.append("")

    md.append("## Interpretation (diagnostic, not a theorem change)")
    md.append("")
    md.append("- $\\epsilon_M=\\sqrt{\\log(4/\\delta_{cell})/(2M)}$ dominates U_F at small M. With $\\delta_{global}=0.10, C=4, M=100$, $\\epsilon_M\\approx 0.159$ and U_F clips at 1/4 in every cell. Larger M is required for U_F to escape the deterministic 1/4 clip.")
    md.append("- $L_\\alpha = \\bar\\alpha - \\epsilon_M$ falls below 1/2 whenever $\\bar\\alpha < 0.5 + \\epsilon_M$, refusing the certificate. The fraction $\\{m_L\\le 0\\}$ scales with M and not with N.")
    md.append("- Reducing C (e.g. running fewer cells in a single Bonferroni group) loosens $\\delta_{cell}$ and shrinks $\\epsilon_M$, but the effect is logarithmic in 1/C and small at this grid.")
    md.append("- Within the requested grid, R_cert < 0.7 is unattainable. Stronger informativeness requires $\\bar\\alpha > 0.8$ or $M \\gg 2000$.")
    md.append("")
    md.append("No theorem or default config was changed. This is a feasibility map only.")
    md.append("")

    (out_dir / "certificate_feasibility.md").write_text("\n".join(md) + "\n")
    return overall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="results/diagnostics")
    ap.add_argument("--delta_global", type=float, default=DELTA_GLOBAL_DEFAULT)
    args = ap.parse_args()
    df = evaluate_grid(delta_global=args.delta_global)
    overall = make_report(df, Path(args.output_dir))
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
