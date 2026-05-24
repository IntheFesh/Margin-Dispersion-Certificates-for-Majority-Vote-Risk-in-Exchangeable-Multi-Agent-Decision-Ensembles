"""Analysis 6: refusal-mode decomposition.

Applies :func:`src.certs.refusal.classify_refusal` to every ``(cell, N)`` row
and tabulates how often each of the four refusal/issuance modes occurs:

  * ``no_direction``       -- m_L <= 0 AND m_beta_L <= 0 (no certifiable side);
  * ``success_margin``     -- success side has a margin but is not certified;
  * ``failure_margin``     -- failure side has a margin but is not certified;
  * ``bidirectional_cert`` -- at least one side is certified at epsilon.

Within the two margin-refusal modes we further split the sub-mode into
``dispersion_dominated`` (the Cantelli term dominates at eta*) vs
``budget_dominated`` (the Hoeffding term dominates), as recorded by the
per-side ``*_dominating_term`` columns.

The per-row dict consumed by ``classify_refusal`` is reconstructed from the
DataFrame columns produced by :func:`src.certs.empirical.empirical_certificate`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.certs.refusal import classify_refusal

__all__ = ["refusal_decomposition"]

# Columns classify_refusal reads off a cell dict.
_REQUIRED = ("m_L", "m_beta_L")
_OPTIONAL = (
    "R_N_cert",
    "Q_N_cert",
    "R_N_cert_dominating_term",
    "Q_N_cert_dominating_term",
)

_MODES = ("no_direction", "success_margin", "failure_margin", "bidirectional_cert")
_SUB_MODES = ("dispersion_dominated", "budget_dominated")


def _none_if_nan(v: object) -> object:
    """Map NaN/NA back to None so classify_refusal sees a refused cert."""
    try:
        if v is None or (np.isscalar(v) and pd.isna(v)):
            return None
    except (TypeError, ValueError):
        return v
    return v


def _row_to_cell(row: pd.Series) -> dict:
    cell: dict = {}
    for col in _REQUIRED:
        cell[col] = float(row[col])
    for col in _OPTIONAL:
        cell[col] = _none_if_nan(row[col]) if col in row.index else None
    return cell


def refusal_decomposition(
    cells_df: pd.DataFrame, epsilon: float = 0.3
) -> pd.DataFrame:
    """Classify and tabulate refusal modes over ``(cell, N)`` rows.

    Parameters
    ----------
    cells_df:
        Per-``(cell, N)`` rows. Must contain ``m_L`` and ``m_beta_L``; uses
        ``R_N_cert``, ``Q_N_cert`` and their ``*_dominating_term`` columns when
        present. ``N`` and ``benchmark`` are used for grouped tabulation.
    epsilon:
        Certification target passed to ``classify_refusal``.

    Returns
    -------
    pd.DataFrame
        Long-format counts table with columns ``group``, ``group_value``,
        ``mode``, ``sub_mode`` (``"all"`` for non-margin modes), ``count`` and
        ``fraction`` (within the group).
    """
    if not isinstance(cells_df, pd.DataFrame):
        raise TypeError("cells_df must be a pandas DataFrame")
    if not (0.0 < epsilon < 1.0):
        raise ValueError(f"epsilon must be in (0,1), got {epsilon}")
    missing = [c for c in _REQUIRED if c not in cells_df.columns]
    if missing:
        raise ValueError(f"cells_df missing required columns: {missing}")

    # Per-row classification.
    modes: list[str] = []
    sub_modes: list[object] = []
    for _, row in cells_df.iterrows():
        cell = _row_to_cell(row)
        res = classify_refusal(cell, epsilon)
        modes.append(res["mode"])
        sub_modes.append(res["sub_mode"])

    work = cells_df.copy()
    work["_mode"] = modes
    # Non-margin modes have sub_mode None; label them "all" for tabulation.
    work["_sub_mode"] = [sm if sm is not None else "all" for sm in sub_modes]

    def _tabulate(df: pd.DataFrame, group_label: str, group_value: object) -> list[dict]:
        n = int(len(df))
        rows: list[dict] = []
        counts = df.groupby(["_mode", "_sub_mode"], dropna=False).size()
        for (mode, sub_mode), count in counts.items():
            rows.append(
                {
                    "group": group_label,
                    "group_value": group_value,
                    "mode": mode,
                    "sub_mode": sub_mode,
                    "count": int(count),
                    "fraction": (int(count) / n) if n else float("nan"),
                }
            )
        return rows

    out_rows: list[dict] = []
    out_rows.extend(_tabulate(work, "marginal", "all"))
    if "N" in work.columns:
        for n_val, sub in work.groupby("N"):
            out_rows.extend(_tabulate(sub, "N", int(n_val)))
    if "benchmark" in work.columns:
        for bench, sub in work.groupby("benchmark"):
            out_rows.extend(_tabulate(sub, "benchmark", bench))

    result = pd.DataFrame(out_rows)
    # Order modes/sub-modes consistently for the figure/table.
    mode_order = {mode: i for i, mode in enumerate(_MODES)}
    sub_order = {"all": 0, **{sm: i + 1 for i, sm in enumerate(_SUB_MODES)}}
    if not result.empty:
        result["_mode_ord"] = result["mode"].map(lambda x: mode_order.get(x, 99))
        result["_sub_ord"] = result["sub_mode"].map(lambda x: sub_order.get(x, 99))
        result = (
            result.sort_values(["group", "group_value", "_mode_ord", "_sub_ord"])
            .drop(columns=["_mode_ord", "_sub_ord"])
            .reset_index(drop=True)
        )
    return result
