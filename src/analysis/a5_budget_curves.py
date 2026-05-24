"""Analysis 5: certificate budget curves (R_N_cert vs N per cell).

For the budget-curve figure we emit, per cell, the certificate-vs-N curve:
``R_N_cert`` as a function of the ensemble size ``N``, together with the
optimizer ``eta*`` and the dominating term (Cantelli vs Hoeffding) at each
``N``. As ``N`` grows the Hoeffding ``exp(-2 N eta^2)`` term shrinks and the
curve transitions from budget-dominated (small ``N``) to dispersion-dominated
(large ``N``); the per-``N`` dominating-term column makes that crossover
visible.

Cells are identified by whichever of ``cell_id``, ``protocol``, ``benchmark``,
``model`` columns are present (a stable composite key); every cell's rows are
sorted by ``N`` to form the curve. Rows where the certificate refused
(``R_N_cert`` is None/NaN) are retained with ``R_N_cert = NaN`` so a gap in the
curve is explicit rather than silently dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["budget_curves"]

# Candidate columns that, in combination, identify one cell across N.
_CELL_KEY_CANDIDATES: tuple[str, ...] = (
    "cell_id",
    "protocol",
    "benchmark",
    "model",
    "K_est",
    "K_full",
)


def _cell_key_columns(cells_df: pd.DataFrame) -> list[str]:
    keys = [c for c in _CELL_KEY_CANDIDATES if c in cells_df.columns]
    if not keys:
        # No identifying columns: treat the whole frame as one cell.
        return []
    return keys


def _first_present(cells_df: pd.DataFrame, *names: str) -> str | None:
    for nm in names:
        if nm in cells_df.columns:
            return nm
    return None


def budget_curves(cells_df: pd.DataFrame) -> pd.DataFrame:
    """Produce the per-cell certificate-vs-N budget curve.

    Parameters
    ----------
    cells_df:
        Per-``(cell, N)`` rows. Requires ``N`` and ``R_N_cert``; uses
        ``R_N_cert_eta_star`` and ``R_N_cert_dominating_term`` when present, and
        any of ``cell_id``/``protocol``/``benchmark``/``model``/``K_est`` as the
        cell key.

    Returns
    -------
    pd.DataFrame
        One row per ``(cell, N)`` sorted within each cell by ``N``, with columns
        for the cell key, ``N``, ``R_N_cert`` (NaN where refused),
        ``R_N_cert_eta_star``, ``R_N_cert_dominating_term`` and a per-cell
        ``cell_index``.
    """
    if not isinstance(cells_df, pd.DataFrame):
        raise TypeError("cells_df must be a pandas DataFrame")
    if "N" not in cells_df.columns:
        raise ValueError("cells_df must contain 'N'")
    if "R_N_cert" not in cells_df.columns:
        raise ValueError("cells_df must contain 'R_N_cert'")

    key_cols = _cell_key_columns(cells_df)
    eta_col = _first_present(cells_df, "R_N_cert_eta_star")
    dom_col = _first_present(cells_df, "R_N_cert_dominating_term")
    method_col = _first_present(cells_df, "R_N_cert_method")

    work = cells_df.copy()
    work["N"] = pd.to_numeric(work["N"], errors="coerce")
    work["R_N_cert"] = pd.to_numeric(work["R_N_cert"], errors="coerce")

    out_rows: list[dict] = []
    if key_cols:
        grouped = work.groupby(key_cols, dropna=False, sort=True)
        cell_iter = enumerate(grouped)
    else:
        cell_iter = enumerate([((), work)])

    for cell_index, (key, sub) in cell_iter:
        sub_sorted = sub.sort_values("N")
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_dict = dict(zip(key_cols, key_tuple)) if key_cols else {}
        for _, r in sub_sorted.iterrows():
            row: dict = dict(key_dict)
            row["cell_index"] = int(cell_index)
            row["N"] = (None if pd.isna(r["N"]) else int(r["N"]))
            r_cert = r["R_N_cert"]
            row["R_N_cert"] = (None if pd.isna(r_cert) else float(r_cert))
            row["refused"] = bool(pd.isna(r_cert))
            row["R_N_cert_eta_star"] = (
                None if eta_col is None or pd.isna(r[eta_col]) else float(r[eta_col])
            )
            row["R_N_cert_dominating_term"] = (
                None if dom_col is None else r[dom_col]
            )
            row["R_N_cert_method"] = (
                None if method_col is None else r[method_col]
            )
            out_rows.append(row)

    result = pd.DataFrame(out_rows)
    # Stable ordering for the figure: by cell then N.
    sort_cols = (["cell_index"] if "cell_index" in result.columns else []) + (
        ["N"] if "N" in result.columns else []
    )
    if sort_cols:
        result = result.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    return result
