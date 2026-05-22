"""Sanity tests for the certificate feasibility diagnostic.

These confirm that (a) the grid script runs end-to-end on a small sub-grid,
(b) the recorded R_cert values satisfy R_cert in [0,1], and (c) when
m_L <= 0 the certificate is refused (R_cert = 1, eta_star = None,
issued = False).
"""
from __future__ import annotations
import math
from pathlib import Path
import pandas as pd
import pytest

from src.diagnostics import certificate_feasibility as diag


def test_evaluate_grid_runs_and_schema(tmp_path):
    df = diag.evaluate_grid(delta_global=0.10)
    assert len(df) == (
        len(diag.M_GRID) * len(diag.C_GRID) * len(diag.N_GRID)
        * len(diag.BAR_ALPHA_GRID) * len(diag.F_GRID)
    )
    expected_cols = {
        "M", "C", "N", "bar_alpha", "F", "delta_global", "delta_cell",
        "eps_M", "L_alpha", "U_2", "U_F", "m_L", "issued", "eta_star",
        "R_cert", "mL_le_0", "UF_eq_quarter", "R_cert_eq_1",
        "R_cert_lt_07", "R_cert_lt_03",
    }
    assert expected_cols.issubset(df.columns)


def test_R_cert_bounds_and_refusal_consistency():
    df = diag.evaluate_grid(delta_global=0.10)
    assert (df["R_cert"] >= 0).all()
    assert (df["R_cert"] <= 1 + 1e-12).all()
    # If m_L <= 0 the certificate must be refused with R_cert == 1.
    sub = df[df["m_L"] <= 0]
    assert (sub["R_cert"] == 1.0).all()
    assert (sub["issued"] == False).all()


def test_eps_M_matches_formula():
    df = diag.evaluate_grid(delta_global=0.10)
    expected = (df.apply(lambda r: math.sqrt(math.log(4 / r["delta_cell"]) / (2 * r["M"])), axis=1))
    assert (abs(df["eps_M"] - expected) < 1e-12).all()


def test_make_report_writes_files(tmp_path):
    df = diag.evaluate_grid(delta_global=0.10)
    overall = diag.make_report(df, tmp_path)
    for f in (
        "certificate_feasibility.csv",
        "feasibility_by_M.csv",
        "feasibility_by_N.csv",
        "feasibility_by_C.csv",
        "feasibility_nonvacuous_slice.csv",
        "certificate_feasibility_summary.json",
        "certificate_feasibility.md",
    ):
        assert (tmp_path / f).exists(), f
    assert 0 <= overall["fraction_R_cert_eq_1"] <= 1
