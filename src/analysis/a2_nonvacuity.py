"""Analysis 2: non-vacuity rates of the issued certificate.

A certificate row is NON-VACUOUS at threshold ``t`` iff it is issued and
``R_N_cert < t``. A refused row (``R_N_cert`` is ``None``/``NaN``) is treated as
VACUOUS at every threshold (it is, by definition, not ``< t``): refusal is the
maximally conservative outcome, so it must never be counted as informative.

We report the fraction of ``(cell, N)`` rows that are non-vacuous at each
threshold, marginally and grouped by ``N`` and by ``benchmark``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["nonvacuity_rates"]

# Default thresholds: 1.0 (any non-trivial bound), 0.7, 0.3.
_DEFAULT_THRESHOLDS: tuple[float, ...] = (1.0, 0.7, 0.3)


def _cert_values(cells_df: pd.DataFrame) -> np.ndarray:
    """Extract the issued certificate column as float with NaN for refusals."""
    if "R_N_cert" not in cells_df.columns:
        raise ValueError("cells_df must contain an 'R_N_cert' column")
    # None -> NaN; refusals therefore fail every strict `< t` comparison.
    return pd.to_numeric(cells_df["R_N_cert"], errors="coerce").to_numpy(dtype=float)


def _nonvacuous_mask(values: np.ndarray, threshold: float) -> np.ndarray:
    """1 where issued AND strictly below threshold; NaN (refused) -> 0."""
    # np.less with NaN yields False, exactly the desired vacuous treatment.
    return (values < threshold) & np.isfinite(values)


def _rate_frame(
    df: pd.DataFrame,
    values: np.ndarray,
    thresholds: Sequence[float],
    group_label: str,
    group_value: object,
) -> list[dict]:
    rows: list[dict] = []
    n = int(len(df))
    n_issued = int(np.isfinite(values).sum())
    for t in thresholds:
        mask = _nonvacuous_mask(values, float(t))
        rows.append(
            {
                "group": group_label,
                "group_value": group_value,
                "threshold": float(t),
                "n_rows": n,
                "n_issued": n_issued,
                "n_nonvacuous": int(mask.sum()),
                "nonvacuity_rate": (float(mask.sum()) / n) if n > 0 else float("nan"),
            }
        )
    return rows


def nonvacuity_rates(
    cells_df: pd.DataFrame,
    thresholds: Sequence[float] = _DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Fraction of ``(cell, N)`` rows with ``R_N_cert < threshold``.

    Refused rows (``R_N_cert`` None/NaN) count as vacuous.

    Parameters
    ----------
    cells_df:
        Per-``(cell, N)`` rows. Must contain ``R_N_cert``; ``N`` and
        ``benchmark`` are used for grouping when present.
    thresholds:
        Strict upper thresholds at which to measure non-vacuity.

    Returns
    -------
    pd.DataFrame
        Long-format table with columns ``group`` (``"marginal"``, ``"N"``,
        ``"benchmark"``), ``group_value``, ``threshold``, ``n_rows``,
        ``n_issued``, ``n_nonvacuous`` and ``nonvacuity_rate``.
    """
    if not isinstance(cells_df, pd.DataFrame):
        raise TypeError("cells_df must be a pandas DataFrame")
    thresholds = tuple(float(t) for t in thresholds)
    if not thresholds:
        raise ValueError("thresholds must be non-empty")

    values = _cert_values(cells_df)

    out_rows: list[dict] = []
    # Marginal.
    out_rows.extend(_rate_frame(cells_df, values, thresholds, "marginal", "all"))

    # Grouped by N (kept separate so an odd-N grid reads cleanly).
    if "N" in cells_df.columns:
        for n_val, idx in cells_df.groupby("N").groups.items():
            pos = cells_df.index.get_indexer(list(idx))
            out_rows.extend(
                _rate_frame(
                    cells_df.loc[idx], values[pos], thresholds, "N", int(n_val)
                )
            )

    # Grouped by benchmark.
    if "benchmark" in cells_df.columns:
        for bench, idx in cells_df.groupby("benchmark").groups.items():
            pos = cells_df.index.get_indexer(list(idx))
            out_rows.extend(
                _rate_frame(
                    cells_df.loc[idx], values[pos], thresholds, "benchmark", bench
                )
            )

    return pd.DataFrame(out_rows)
