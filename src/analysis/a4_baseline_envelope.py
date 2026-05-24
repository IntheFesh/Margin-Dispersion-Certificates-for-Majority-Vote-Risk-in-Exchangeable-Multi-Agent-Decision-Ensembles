"""Analysis 4: limit-case BASELINES (NOT certificates).

This module computes two LIMIT-CASE reference quantities used purely to
demonstrate that the unified Cantelli-Hoeffding bound ``B_N^CH`` dominates
EITHER limit:

  * ``margin_only_hoeffding`` = exp(-2 N m^2)   -- the dispersion-free (F -> 0)
    Hoeffding limit: pretend the per-instance success rates are a point mass at
    the mean, so only the margin ``m`` matters.
  * ``asymptotic_cantelli``   = F / (F + m^2)    -- the budget-free (N -> infinity)
    Cantelli limit: pretend the sample average concentrates perfectly, so only
    the dispersion ``F`` and margin ``m`` matter.

In both, ``m = alpha_bar_hat - 0.5`` and BOTH quantities are set to ``1.0``
when ``m <= 0`` (no favorable margin -> the trivial vacuous value).

CRITICAL FRAMING (do not violate)
---------------------------------
These two quantities are LIMIT-CASE BASELINES, NOT certificates. They are:
  * NOT members of the certificate hierarchy ``R_N <= B_N^star <= B_N^CH' <=
    B_N^CH``;
  * NOT issued with the four-event Hoeffding confidence bounds (no delta_cell,
    no union bound, no estimation-pool guarantee). They are computed directly
    from point estimates ``(alpha_bar_hat, F_hat)`` for ILLUSTRATION ONLY;
  * NOT to be reported, plotted, or referred to as certificates anywhere.
They exist solely so the figures can show that the unified ``B_N^CH`` (a real
certificate) is no worse than -- and typically strictly tighter than -- each of
these one-sided limits. The comparison columns below quantify that gap; they
never promote a baseline to certificate status.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["baseline_envelope"]


def _margin(cells_df: pd.DataFrame) -> np.ndarray:
    """m = alpha_bar_hat - 0.5 (accepts alpha_bar as a fallback name)."""
    for col in ("alpha_bar_hat", "alpha_bar"):
        if col in cells_df.columns:
            return pd.to_numeric(cells_df[col], errors="coerce").to_numpy(float) - 0.5
    raise ValueError("cells_df must contain 'alpha_bar_hat' or 'alpha_bar'")


def _dispersion(cells_df: pd.DataFrame) -> np.ndarray:
    """F_hat (accepts F as a fallback name)."""
    for col in ("F_hat", "F"):
        if col in cells_df.columns:
            return pd.to_numeric(cells_df[col], errors="coerce").to_numpy(float)
    raise ValueError("cells_df must contain 'F_hat' or 'F'")


def baseline_envelope(cells_df: pd.DataFrame) -> pd.DataFrame:
    """Append the two limit-case baseline columns and comparison diagnostics.

    Adds (these are NOT certificates):
      * ``margin_only_hoeffding`` = exp(-2 N m^2), =1.0 if m <= 0;
      * ``asymptotic_cantelli``   = F / (F + m^2),  =1.0 if m <= 0.

    Comparison columns versus the issued certificate ``R_N_cert`` (and the MC
    risk ``R_N_MC`` when present) are added to quantify how much the unified
    ``B_N^CH`` certificate improves on each limit; they never relabel a baseline
    as a certificate.

    Parameters
    ----------
    cells_df:
        Per-``(cell, N)`` rows; requires a margin source (``alpha_bar_hat`` or
        ``alpha_bar``), a dispersion source (``F_hat`` or ``F``) and ``N``.

    Returns
    -------
    pd.DataFrame
        Copy of ``cells_df`` with the baseline and comparison columns added.
    """
    if not isinstance(cells_df, pd.DataFrame):
        raise TypeError("cells_df must be a pandas DataFrame")
    if "N" not in cells_df.columns:
        raise ValueError("cells_df must contain 'N'")

    out = cells_df.copy()
    m = _margin(cells_df)
    F = _dispersion(cells_df)
    N = pd.to_numeric(cells_df["N"], errors="coerce").to_numpy(float)

    favorable = m > 0.0

    # margin_only_hoeffding = exp(-2 N m^2); vacuous 1.0 where m <= 0.
    hoeff = np.where(favorable, np.exp(-2.0 * N * np.square(m)), 1.0)
    # asymptotic_cantelli = F / (F + m^2); vacuous 1.0 where m <= 0.
    denom = F + np.square(m)
    with np.errstate(divide="ignore", invalid="ignore"):
        cant = np.where(favorable & (denom > 0), F / denom, 1.0)
    cant = np.where(favorable, np.clip(cant, 0.0, 1.0), 1.0)

    out["margin_only_hoeffding"] = hoeff
    out["asymptotic_cantelli"] = cant
    # The dominating baseline at this row is the larger (looser) of the two
    # limits; the unified B_N^CH should not exceed it (illustration, not proof).
    out["baseline_envelope_max"] = np.maximum(hoeff, cant)

    # --- Comparison diagnostics (baselines stay baselines). ------------------
    if "R_N_cert" in out.columns:
        R_cert = pd.to_numeric(out["R_N_cert"], errors="coerce").to_numpy(float)
        # Slack of each limit relative to the issued certificate (baseline -
        # cert); positive means the certificate is tighter than the limit.
        out["slack_hoeffding_vs_cert"] = hoeff - R_cert
        out["slack_cantelli_vs_cert"] = cant - R_cert
        out["cert_tighter_than_both_limits"] = (R_cert <= hoeff) & (R_cert <= cant)

    if "R_N_MC" in out.columns:
        R_mc = pd.to_numeric(out["R_N_MC"], errors="coerce").to_numpy(float)
        # A limit "undercovers" if it falls BELOW the realized MC risk -- i.e.
        # it would be an INVALID upper bound. Reported only to show why the
        # limits are not safe to use as certificates.
        out["hoeffding_undercovers_MC"] = hoeff < R_mc
        out["cantelli_undercovers_MC"] = cant < R_mc
        out["hoeffding_slack_vs_MC"] = hoeff - R_mc
        out["cantelli_slack_vs_MC"] = cant - R_mc

    return out
