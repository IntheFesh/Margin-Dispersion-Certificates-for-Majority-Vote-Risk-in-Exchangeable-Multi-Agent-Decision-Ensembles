"""Phase 2 figure rendering: design space, Analysis 7 stacked bar, budget curves.

Renders the three paper figures from the analysis outputs into
``outputs/figures/`` (each as BOTH a 300-DPI PNG and a PDF):

  * ``design_space``  -- the B_N^CH' certificate over the (mean-margin,
    dispersion) plane for two ODD reference sizes, optionally overlaying the
    empirical operating points read from the Panel A per-cell certificates;
  * ``stacked_bar``   -- the Analysis 7 PRINCIPAL figure: the four-component
    conservativeness decomposition, from
    ``outputs/analyses/analysis_7_conservativeness.csv``;
  * ``budget_curves`` -- R_N^cert vs N per cell, from
    ``outputs/analyses/analysis_5_budget_curves.csv`` (falling back to the
    Panel A summary's per-cell certificate rows when that CSV carries the same
    columns).

A required input CSV that is absent raises ``FileNotFoundError`` -- this script
NEVER fabricates data or draws an empty figure. No GPU/network; it does NOT run
in Phase 0 (it needs the analysis CSVs from ``05_run_analyses.py``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.figures.budget_curves import render_budget_curves
from src.figures.design_space import render_design_space
from src.figures.stacked_bar import render_analysis7_stacked_bar
from src.utils.logging import JsonlLogger
from src.utils.provenance import capture, copy_config


def _require_csv(path: Path) -> pd.DataFrame:
    """Read a required analysis CSV, raising ``FileNotFoundError`` if absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"required analysis CSV not found: {path}; run 05_run_analyses.py first"
        )
    return pd.read_csv(path)


def _overlay_points(panel_a_dir: Path) -> pd.DataFrame | None:
    """Derive design-space overlay operating points from the Panel A summary.

    Maps each issued cell to ``(m = alpha_bar_hat - 1/2, F = F_hat)``. Returns
    ``None`` (no overlay) if the Panel A summary is absent -- the design-space
    background is still meaningful without empirical points.
    """
    summary_path = panel_a_dir / "panel_a_summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    rows = summary.get("cell_certificates", [])
    pts = [
        {"m": float(r["alpha_bar_hat"]) - 0.5, "F": float(r["F_hat"])}
        for r in rows
        if r.get("alpha_bar_hat") is not None and r.get("F_hat") is not None
    ]
    return pd.DataFrame(pts) if pts else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Panel A figures from analysis outputs.")
    parser.add_argument("--config", default="configs/design_grid.yaml")
    parser.add_argument("--analyses_dir", default="outputs/analyses")
    parser.add_argument("--panel_a_dir", default="outputs/panel_a")
    parser.add_argument("--output_dir", default="outputs/figures")
    parser.add_argument("--seed", type=int, default=0, help="global seed (provenance only)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(Path("outputs/logs") / "06_render_figures.jsonl")

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    copy_config(str(config_path), str(output_dir))
    provenance = capture(args.seed, [str(config_path)])

    analyses_dir = Path(args.analyses_dir)
    panel_a_dir = Path(args.panel_a_dir)

    # --- Design space (background + optional empirical overlay points). ------
    overlay = _overlay_points(panel_a_dir)
    render_design_space(
        out_pdf=output_dir / "design_space.pdf",
        out_png=output_dir / "design_space.png",
        N_refs=(15, 63),
        overlay_points=overlay,
    )
    logger.event(
        "figure.done",
        figure="design_space",
        overlay_points=(0 if overlay is None else int(len(overlay))),
    )

    # --- Analysis 7 principal: conservativeness stacked bar. -----------------
    decomp_df = _require_csv(analyses_dir / "analysis_7_conservativeness.csv")
    render_analysis7_stacked_bar(
        decomp_df,
        out_pdf=output_dir / "analysis7_stacked_bar.pdf",
        out_png=output_dir / "analysis7_stacked_bar.png",
    )
    logger.event("figure.done", figure="analysis7_stacked_bar", n_rows=int(len(decomp_df)))

    # --- Budget curves: R_N^cert vs N per cell. ------------------------------
    budget_csv = analyses_dir / "analysis_5_budget_curves.csv"
    if budget_csv.exists():
        curves_df = pd.read_csv(budget_csv)
    else:
        # Fallback to the Panel A per-cell certificate rows (same columns).
        summary_path = panel_a_dir / "panel_a_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"neither {budget_csv} nor {summary_path} is available for budget curves"
            )
        with summary_path.open("r", encoding="utf-8") as fh:
            curves_df = pd.DataFrame(json.load(fh)["cell_certificates"])
    render_budget_curves(
        curves_df,
        out_pdf=output_dir / "budget_curves.pdf",
        out_png=output_dir / "budget_curves.png",
    )
    logger.event("figure.done", figure="budget_curves", n_rows=int(len(curves_df)))

    logger.event("figures.complete", output_dir=str(output_dir))
    print(f"Rendered design_space, analysis7_stacked_bar, budget_curves (PNG+PDF) to {output_dir}")
    _ = provenance


if __name__ == "__main__":
    main()
