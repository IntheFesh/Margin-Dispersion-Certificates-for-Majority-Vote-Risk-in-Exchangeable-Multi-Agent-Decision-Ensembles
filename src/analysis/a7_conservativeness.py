"""Analysis 7 (PRINCIPAL): conservativeness decomposition of R_N_cert - R_N_MC.

The end-to-end conservativeness of the issued certificate relative to the
Monte-Carlo majority-vote risk is decomposed into four additive, telescoping
gaps along the certificate chain
``R_N^MC -> B_N^star -> B_N^CH' -> B_N^CH -> R_N^cert``:

  * ``A = B_N_star    - R_N_MC``     -- sharp two-moment envelope vs realized MC
    risk: the price of summarizing the population by its first two moments
    (mean, dispersion) instead of the full per-instance success-rate law;
  * ``B = B_N_CH'     - B_N_star``   -- closed-form boundedness-aware relaxation
    (Refinement 1) vs the sharp moment LP;
  * ``C = B_N_CH      - B_N_CH'``    -- dropping the [0,1] boundedness term
    (Theorem 1) vs the boundedness-aware form;
  * ``D = R_N_cert    - B_N_CH``     -- the confidence/estimation gap: plugging
    the one-sided Hoeffding upper-confidence moments into the closed form
    instead of the (unknown) true ``(alpha_bar, F)``.

By construction A + B + C + D telescopes to ``R_N_cert - R_N_MC``; the
``sum_check_ok`` column asserts this to 1e-9 per row (a numerical-bug detector,
not part of the proof).

Proxy/exact distinction (framework A.7 / A.12): terms ``A`` and ``D`` carry a
proxy-vs-exact subtlety. ``B_N_star`` here is computed on the *plug-in* moments
stored in the row (``alpha_bar``/``F``), so ``A`` mixes the moment-summarization
gap with any plug-in error -- the exact ``A`` would use the population moments.
``D`` is the confidence inflation that makes the chain a valid certificate
rather than a point estimate; it is the only term that depends on
``delta_cell``/``M`` and is exactly the difference between the certificate's
upper-confidence moments and the plug-in moments fed to ``B_N_CH`` here. ``B``
and ``C`` are exact analytic relaxations and carry no such proxy ambiguity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.certs.refinement1 import B_N_CH_prime
from src.certs.theorem1 import B_N_CH
from src.certs.theorem3 import B_N_star

__all__ = ["conservativeness_decomposition"]

_SUM_TOL = 1e-9


def _moment_columns(cells_df: pd.DataFrame) -> tuple[str, str]:
    """Resolve the (alpha_bar, F) source columns, preferring exact names."""
    alpha_col = next(
        (c for c in ("alpha_bar", "alpha_bar_hat") if c in cells_df.columns), None
    )
    f_col = next((c for c in ("F", "F_hat") if c in cells_df.columns), None)
    if alpha_col is None:
        raise ValueError("cells_df must contain 'alpha_bar' or 'alpha_bar_hat'")
    if f_col is None:
        raise ValueError("cells_df must contain 'F' or 'F_hat'")
    return alpha_col, f_col


def _clip_F(alpha_bar: float, F: float) -> float:
    """Clip F into the feasible [0, alpha_bar(1-alpha_bar)] for the certs.

    The certs functions validate ``0 <= F <= alpha_bar(1-alpha_bar)``. Plug-in
    moments can land marginally outside that box (sampling noise / numerical
    error); we clip into the feasible box rather than crash, which is the
    conservative choice for an upper-bound computation.
    """
    upper = alpha_bar * (1.0 - alpha_bar)
    return float(min(max(F, 0.0), upper))


def conservativeness_decomposition(cells_df: pd.DataFrame) -> pd.DataFrame:
    """Decompose ``R_N_cert - R_N_MC`` into the four telescoping gaps A,B,C,D.

    For each ``(cell, N)`` row, recompute ``B_N_CH``, ``B_N_CH'`` and
    ``B_N_star`` from the row's ``(alpha_bar, F, N)`` and form

        A = B_N_star    - R_N_MC
        B = B_N_CH'     - B_N_star
        C = B_N_CH      - B_N_CH'
        D = R_N_cert    - B_N_CH

    Parameters
    ----------
    cells_df:
        Per-``(cell, N)`` rows. Requires ``N``, ``R_N_cert``, ``R_N_MC`` and a
        moment source (``alpha_bar``/``alpha_bar_hat`` and ``F``/``F_hat``).
        Rows where ``R_N_cert`` is refused (None/NaN) or ``m = alpha_bar - 0.5
        <= 0`` are still emitted, with the un-computable gaps set to NaN and
        ``sum_check_ok = False`` (no silent drop).

    Returns
    -------
    pd.DataFrame
        Copy of ``cells_df`` augmented with ``B_N_CH``, ``B_N_CH_prime``,
        ``B_N_star``, ``gap_A``, ``gap_B``, ``gap_C``, ``gap_D``,
        ``total_conservativeness``, ``sum_check_residual`` and ``sum_check_ok``.
    """
    if not isinstance(cells_df, pd.DataFrame):
        raise TypeError("cells_df must be a pandas DataFrame")
    for col in ("N", "R_N_cert", "R_N_MC"):
        if col not in cells_df.columns:
            raise ValueError(f"cells_df must contain '{col}'")
    alpha_col, f_col = _moment_columns(cells_df)

    out = cells_df.copy()
    b_ch_list: list[float] = []
    b_chp_list: list[float] = []
    b_star_list: list[float] = []
    gap_a: list[float] = []
    gap_b: list[float] = []
    gap_c: list[float] = []
    gap_d: list[float] = []
    total: list[float] = []
    residual: list[float] = []
    ok: list[bool] = []

    nan = float("nan")
    for _, row in cells_df.iterrows():
        alpha_bar = float(row[alpha_col])
        F = _clip_F(alpha_bar, float(row[f_col]))
        N = int(row["N"])
        R_cert = pd.to_numeric(pd.Series([row["R_N_cert"]]), errors="coerce").iloc[0]
        R_mc = float(row["R_N_MC"])
        m = alpha_bar - 0.5

        # The bound chain is only defined for a favorable margin; below it the
        # certs return the vacuous 1.0 and R_N_cert is refused. Emit NaN gaps.
        if m <= 0 or pd.isna(R_cert):
            b_ch_list.append(nan)
            b_chp_list.append(nan)
            b_star_list.append(nan)
            gap_a.append(nan)
            gap_b.append(nan)
            gap_c.append(nan)
            gap_d.append(nan)
            total.append(nan if pd.isna(R_cert) else float(R_cert) - R_mc)
            residual.append(nan)
            ok.append(False)
            continue

        R_cert = float(R_cert)
        b_ch = B_N_CH(alpha_bar, F, N)
        b_chp = B_N_CH_prime(alpha_bar, F, N)
        b_star = B_N_star(alpha_bar, F, N)

        A = b_star - R_mc
        B = b_chp - b_star
        C = b_ch - b_chp
        D = R_cert - b_ch
        tot = R_cert - R_mc
        res = (A + B + C + D) - tot

        b_ch_list.append(b_ch)
        b_chp_list.append(b_chp)
        b_star_list.append(b_star)
        gap_a.append(A)
        gap_b.append(B)
        gap_c.append(C)
        gap_d.append(D)
        total.append(tot)
        residual.append(res)
        is_ok = abs(res) < _SUM_TOL
        ok.append(is_ok)
        # Telescoping is algebraically exact; a breach signals a numerical bug.
        assert is_ok, (
            f"conservativeness sum check failed: A+B+C+D-({R_cert}-{R_mc})="
            f"{res:.3e} exceeds {_SUM_TOL} (alpha_bar={alpha_bar}, F={F}, N={N})"
        )

    out["B_N_CH"] = b_ch_list
    out["B_N_CH_prime"] = b_chp_list
    out["B_N_star"] = b_star_list
    out["gap_A"] = gap_a
    out["gap_B"] = gap_b
    out["gap_C"] = gap_c
    out["gap_D"] = gap_d
    out["total_conservativeness"] = total
    out["sum_check_residual"] = residual
    out["sum_check_ok"] = ok
    return out
