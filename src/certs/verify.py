"""Hierarchy and primal-dual verification (numerical bug detectors).

The hierarchy R_N(mu) <= B_N^star <= B_N^CH' <= B_N^CH is the theorem; this
module checks the computable upper-bound chain on every cell and aborts the
pipeline on any violation. The R_N(mu) <= B_N^star side is exercised on
known mixtures (Proposition 1) in tests/test_hierarchy.py.
"""
from __future__ import annotations
import pandas as pd


def check_hierarchy(cells: pd.DataFrame, tol: float = 1e-6) -> None:
    """Verify B_N_star <= B_N_CH_prime <= B_N_CH (within tol) per row.

    Required columns: alpha_bar, F, N, B_CH, B_CH_prime, B_star.
    Raises RuntimeError with diagnostics on any violation.
    """
    required = {"alpha_bar", "F", "N", "B_CH", "B_CH_prime", "B_star"}
    missing = required - set(cells.columns)
    if missing:
        raise ValueError(f"check_hierarchy missing columns: {sorted(missing)}")
    for idx, r in cells.iterrows():
        if r["B_star"] > r["B_CH_prime"] + tol:
            raise RuntimeError(
                f"Hierarchy violation at row {idx}: B_star={r['B_star']:.9f} > "
                f"B_CH_prime={r['B_CH_prime']:.9f} (gap {r['B_star'] - r['B_CH_prime']:.2e}); "
                f"alpha_bar={r['alpha_bar']}, F={r['F']}, N={r['N']}"
            )
        if r["B_CH_prime"] > r["B_CH"] + tol:
            raise RuntimeError(
                f"Hierarchy violation at row {idx}: B_CH_prime={r['B_CH_prime']:.9f} > "
                f"B_CH={r['B_CH']:.9f} (gap {r['B_CH_prime'] - r['B_CH']:.2e}); "
                f"alpha_bar={r['alpha_bar']}, F={r['F']}, N={r['N']}"
            )


def max_hierarchy_gap(cells: pd.DataFrame) -> float:
    """Return the largest signed violation (positive = violation)."""
    g1 = (cells["B_star"] - cells["B_CH_prime"]).max()
    g2 = (cells["B_CH_prime"] - cells["B_CH"]).max()
    return float(max(g1, g2))


def check_primal_dual_gap(cells: pd.DataFrame, tol: float = 1e-8) -> None:
    """Verify |primal - dual| < tol per B_N_star computation.

    Required columns: alpha_bar, F, N, primal, dual.
    Raises RuntimeError on violation."""
    required = {"alpha_bar", "F", "N", "primal", "dual"}
    missing = required - set(cells.columns)
    if missing:
        raise ValueError(f"check_primal_dual_gap missing columns: {sorted(missing)}")
    gaps = (cells["primal"] - cells["dual"]).abs()
    bad = gaps[gaps >= tol]
    if len(bad):
        i = bad.idxmax()
        r = cells.loc[i]
        raise RuntimeError(
            f"Primal-dual gap {gaps[i]:.2e} >= tol {tol} at row {i}: "
            f"primal={r['primal']:.9f}, dual={r['dual']:.9f}; "
            f"alpha_bar={r['alpha_bar']}, F={r['F']}, N={r['N']}"
        )


def max_primal_dual_gap(cells: pd.DataFrame) -> float:
    return float((cells["primal"] - cells["dual"]).abs().max())
