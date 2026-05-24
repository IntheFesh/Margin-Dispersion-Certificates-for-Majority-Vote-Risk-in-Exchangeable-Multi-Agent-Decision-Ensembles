"""Phase 1.2 no-GPU design-space pre-check (go / no-go gate).

Sweeps the ``(alpha_bar, F_relative, N, M)`` grid declared in
``configs/design_grid.yaml`` and, for every grid point, computes:

  * the closed-form certificates B_N^CH (Theorem 1), B_N^CH' (Refinement 1),
    and the sharp two-moment envelope B_N^star with its dual (Theorem 3);
  * an empirical-certificate informativeness proxy: the four-event Bonferroni
    per-cell budget delta_cell and the corresponding Hoeffding radius at M,
    plus a "would the cell issue?" margin check (m_L = L_alpha - 1/2 > 0).

It then runs the numerical bug detectors :func:`src.certs.verify.check_hierarchy`
(B_N^star <= B_N^CH' <= B_N^CH) and :func:`src.certs.verify.check_primal_dual_gap`
(|primal - dual| < 1e-8); either raises and aborts on violation. The full
informativeness table is written to ``outputs/precheck/informativeness.csv`` and
a go/no-go summary to ``outputs/precheck/go_no_go.json``, and the design-space
figure is rendered into ``outputs/figures/``.

No GPU, no network, no datasets: this script is the runnable pre-registration
gate. It is NOT executed in Phase 0.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
import yaml

from src.certs.empirical import bonferroni_cell_budget, hoeffding_radius
from src.certs.refinement1 import B_N_CH_prime
from src.certs.theorem1 import B_N_CH
from src.certs.theorem3 import B_N_star
from src.certs.verify import (
    check_hierarchy,
    check_primal_dual_gap,
    max_hierarchy_gap,
    max_primal_dual_gap,
)
from src.figures.design_space import render_design_space
from src.utils.logging import JsonlLogger
from src.utils.provenance import capture, copy_config

# Informativeness proxy: a closed-form certificate below this is "informative".
_INFORMATIVE_THRESHOLD: float = 0.5


def _load_grid(config_path: Path) -> dict:
    """Load and minimally validate the design grid config (odd N enforced)."""
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    required = {"alpha_bar_grid", "F_grid_relative", "N_grid", "M_grid", "delta_global"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"design grid config missing keys: {sorted(missing)}")
    for N in cfg["N_grid"]:
        if int(N) % 2 == 0:
            raise ValueError(
                f"N_grid must be ODD under strict success-majority (ties=failures), got {N}"
            )
    return cfg


def _sweep(cfg: dict) -> pd.DataFrame:
    """Evaluate every certificate over the (alpha_bar, F_rel, N, M) grid.

    ``F`` is the ABSOLUTE dispersion ``F_rel * alpha_bar * (1 - alpha_bar)`` so
    every point is moment-feasible by construction. The empirical proxy
    columns (delta_cell, eps, would-issue) depend on M; the closed-form
    certificate columns do not.
    """
    alpha_grid = [float(a) for a in cfg["alpha_bar_grid"]]
    f_rel_grid = [float(f) for f in cfg["F_grid_relative"]]
    n_grid = [int(n) for n in cfg["N_grid"]]
    m_grid = [int(m) for m in cfg["M_grid"]]
    delta_global = float(cfg["delta_global"])

    # C = number of (alpha_bar, F_rel, N) certificate cells (Bonferroni budget).
    C = len(alpha_grid) * len(f_rel_grid) * len(n_grid)
    delta_cell = bonferroni_cell_budget(delta_global, C)

    rows: list[dict] = []
    for alpha_bar, f_rel, N in itertools.product(alpha_grid, f_rel_grid, n_grid):
        F = f_rel * alpha_bar * (1.0 - alpha_bar)
        b_ch = B_N_CH(alpha_bar, F, N)
        b_chp = B_N_CH_prime(alpha_bar, F, N)
        primal, dual = B_N_star(alpha_bar, F, N, return_dual=True)
        for M in m_grid:
            eps = hoeffding_radius(delta_cell, M)
            # Population-margin proxy: with no sampling noise the lower CI is
            # alpha_bar - eps; the success direction is certifiable iff its
            # margin exceeds 0.
            m_lower = (alpha_bar - eps) - 0.5
            rows.append(
                {
                    "alpha_bar": alpha_bar,
                    "F_relative": f_rel,
                    "F": F,
                    "N": N,
                    "M": M,
                    "delta_global": delta_global,
                    "C_cells": C,
                    "delta_cell": delta_cell,
                    "eps_delta": eps,
                    "B_CH": b_ch,
                    "B_CH_prime": b_chp,
                    "B_star": primal,
                    "primal": primal,
                    "dual": dual,
                    "m_lower_proxy": m_lower,
                    "would_issue": bool(m_lower > 0.0),
                    "informative_B_star": bool(primal < _INFORMATIVE_THRESHOLD),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Design-space no-GPU pre-check (go/no-go).")
    parser.add_argument("--config", default="configs/design_grid.yaml", help="design grid YAML")
    parser.add_argument("--output_dir", default="outputs/precheck", help="output directory")
    parser.add_argument("--figures_dir", default="outputs/figures", help="figure output directory")
    parser.add_argument("--seed", type=int, default=0, help="global seed (provenance only here)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(Path("outputs/logs") / "01_design_space_precheck.jsonl")

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"design grid config not found: {config_path}")
    copy_config(str(config_path), str(output_dir))
    provenance = capture(args.seed, [str(config_path)])

    cfg = _load_grid(config_path)

    with logger.timed("design_space_precheck.sweep", config=str(config_path)) as ctx:
        df = _sweep(cfg)
        ctx["n_points"] = int(len(df))

    # Numerical bug detectors (raise + abort on any violation).
    check_hierarchy(df, tol=1e-6)
    check_primal_dual_gap(df, tol=1e-8)
    hier_gap = max_hierarchy_gap(df)
    pd_gap = max_primal_dual_gap(df)

    csv_path = output_dir / "informativeness.csv"
    df.to_csv(csv_path, index=False)

    # Go/no-go: there must exist at least one grid point that is both
    # informative (B_star below threshold) AND issuable (positive margin) at
    # the largest M, otherwise the design is uninformative everywhere.
    M_max = int(max(cfg["M_grid"]))
    at_mmax = df[df["M"] == M_max]
    n_informative_issuable = int((at_mmax["informative_B_star"] & at_mmax["would_issue"]).sum())
    go = n_informative_issuable > 0

    summary = {
        "decision": "GO" if go else "NO-GO",
        "go": bool(go),
        "n_grid_points": int(len(df)),
        "n_cells_C": int(df["C_cells"].iloc[0]),
        "delta_global": float(cfg["delta_global"]),
        "delta_cell": float(df["delta_cell"].iloc[0]),
        "M_max": M_max,
        "n_informative_issuable_at_Mmax": n_informative_issuable,
        "frac_informative_B_star_overall": float(df["informative_B_star"].mean()),
        "hierarchy_gap_max": hier_gap,
        "primal_dual_gap_max": pd_gap,
        "informativeness_csv": str(csv_path),
        "provenance": provenance,
    }
    summary_path = output_dir / "go_no_go.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    # Design-space figure (uses ODD reference sizes from the grid, default 15/63).
    n_choices = [int(n) for n in cfg["N_grid"] if int(n) % 2 == 1]
    n_refs = (15, 63) if {15, 63}.issubset(set(n_choices)) else tuple(n_choices[:2])
    figures_dir = Path(args.figures_dir)
    render_design_space(
        out_pdf=figures_dir / "design_space.pdf",
        out_png=figures_dir / "design_space.png",
        N_refs=n_refs,
    )

    logger.event(
        "design_space_precheck.done",
        decision=summary["decision"],
        n_points=summary["n_grid_points"],
        hierarchy_gap_max=hier_gap,
        primal_dual_gap_max=pd_gap,
        csv=str(csv_path),
        summary_json=str(summary_path),
    )

    print(f"Design-space pre-check: {summary['decision']}")
    print(f"  grid points: {summary['n_grid_points']}  (C={summary['n_cells_C']} cells)")
    print(f"  delta_cell:  {summary['delta_cell']:.3e}")
    print(f"  hierarchy gap max:   {hier_gap:.3e}  (expected < 1e-6)")
    print(f"  primal-dual gap max: {pd_gap:.3e}  (expected < 1e-8)")
    print(f"  informative+issuable at M={M_max}: {n_informative_issuable}")
    print(f"  wrote {csv_path}")
    print(f"  wrote {summary_path}")


if __name__ == "__main__":
    main()
