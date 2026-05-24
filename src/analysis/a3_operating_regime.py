"""Analysis 3: operating-regime characterization of the certificate.

Where in ``(margin, dispersion, ensemble-size)`` space does the certificate
become informative? We report

  (a) a non-vacuity indicator summary (how often a usable bound is issued);
  (b) a stratified regime table crossing
        * high/low margin   m_hat = alpha_bar_hat - 0.5  (split at m_hat > 0),
        * high/low dispersion F_hat (terciles -> low/mid/high),
        * small/large N      (N >= 31 = "large"; the design grid is ODD so 31
          is the first grid point at/above the small-N boundary);
  (c) an optional continuous logit-style summary regressing
        logit(clip(R_N_cert)) ~ m_hat + F_hat + log(N).

Predicted directions (framework A.3 / A.6):
  * higher margin m  -> MORE non-vacuity, LOWER R_N_cert (negative coef);
  * higher dispersion F -> LESS non-vacuity, HIGHER R_N_cert (positive coef);
  * larger N         -> MORE non-vacuity, LOWER R_N_cert (negative coef on
    log N).

Dependency note: ``statsmodels`` is imported LAZILY and only as an optional
nicety. If it is unavailable we fall back to an ordinary numpy least-squares
fit on the logit response and report the OLS coefficients (no Logit MLE, no
silent omission of the regression -- the method used is recorded).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["operating_regime"]

_LARGE_N = 31  # odd-grid small/large boundary: N >= 31 is "large".
_LOGIT_CLIP = 1e-6


def _margin(cells_df: pd.DataFrame) -> np.ndarray:
    """m_hat = alpha_bar_hat - 0.5 (accepts alpha_bar as a fallback name)."""
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


def _cert(cells_df: pd.DataFrame) -> np.ndarray:
    if "R_N_cert" not in cells_df.columns:
        raise ValueError("cells_df must contain 'R_N_cert'")
    return pd.to_numeric(cells_df["R_N_cert"], errors="coerce").to_numpy(float)


def _tercile_labels(x: np.ndarray) -> np.ndarray:
    """Label finite values as low/mid/high by terciles; NaN -> 'unknown'.

    Degenerate (constant or near-constant) inputs collapse to a single 'mid'
    label rather than raising, so a homogeneous grid still tabulates.
    """
    labels = np.full(x.shape, "unknown", dtype=object)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return labels
    vals = x[finite]
    q1, q2 = np.quantile(vals, [1.0 / 3.0, 2.0 / 3.0])
    if not (q1 < q2):  # degenerate split
        labels[finite] = "mid"
        return labels
    lab = np.where(vals <= q1, "low", np.where(vals <= q2, "mid", "high"))
    labels[finite] = lab
    return labels


def _logit_fit(m: np.ndarray, F: np.ndarray, N: np.ndarray, R: np.ndarray) -> dict:
    """Regress logit(clip(R)) on [1, m, F, log N] over issued rows.

    Prefers a statsmodels OLS (lazy import) for richer diagnostics; falls back
    to numpy ``lstsq`` if statsmodels is absent. Returns coefficients keyed by
    name plus the method and sample size, or a structured 'skipped' record when
    too few issued rows are available.
    """
    issued = np.isfinite(R) & np.isfinite(m) & np.isfinite(F) & np.isfinite(N) & (N > 0)
    n = int(issued.sum())
    if n < 4:  # need > #params for a meaningful fit
        return {
            "method": "skipped",
            "reason": f"only {n} issued rows; need >= 4 for a 3-predictor fit",
            "n": n,
        }
    Rc = np.clip(R[issued], _LOGIT_CLIP, 1.0 - _LOGIT_CLIP)
    y = np.log(Rc / (1.0 - Rc))
    X = np.column_stack(
        [np.ones(n), m[issued], F[issued], np.log(N[issued].astype(float))]
    )
    names = ["intercept", "m_hat", "F_hat", "log_N"]

    try:
        import statsmodels.api as sm  # lazy, optional

        model = sm.OLS(y, X).fit()
        coefs = {nm: float(c) for nm, c in zip(names, model.params)}
        return {
            "method": "statsmodels-OLS-on-logit",
            "n": n,
            "coefficients": coefs,
            "rsquared": float(model.rsquared),
        }
    except Exception:  # noqa: BLE001 - statsmodels missing or failed -> numpy
        beta, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        coefs = {nm: float(c) for nm, c in zip(names, beta)}
        ss_res = float(np.sum((y - X @ beta) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return {
            "method": "numpy-lstsq-on-logit",
            "n": n,
            "coefficients": coefs,
            "rsquared": r2,
            "rank": int(rank),
        }


def operating_regime(cells_df: pd.DataFrame) -> dict:
    """Characterize the certificate's operating regime.

    Parameters
    ----------
    cells_df:
        Per-``(cell, N)`` rows. Requires a margin source (``alpha_bar_hat`` or
        ``alpha_bar``), a dispersion source (``F_hat`` or ``F``), ``N`` and
        ``R_N_cert``.

    Returns
    -------
    dict
        ``{"nonvacuity_summary": {...}, "stratified_regime": <DataFrame>,
        "continuous_logit": {...}, "predicted_directions": {...}}``.
    """
    if not isinstance(cells_df, pd.DataFrame):
        raise TypeError("cells_df must be a pandas DataFrame")
    if "N" not in cells_df.columns:
        raise ValueError("cells_df must contain 'N'")

    m = _margin(cells_df)
    F = _dispersion(cells_df)
    R = _cert(cells_df)
    N = pd.to_numeric(cells_df["N"], errors="coerce").to_numpy(float)

    issued = np.isfinite(R)
    n_rows = int(len(cells_df))
    n_issued = int(issued.sum())

    # (a) Non-vacuity / issuance indicator summary.
    nonvacuity_summary = {
        "n_rows": n_rows,
        "n_issued": n_issued,
        "issuance_rate": (n_issued / n_rows) if n_rows else float("nan"),
        "mean_R_N_cert_issued": (
            float(np.nanmean(R[issued])) if n_issued else float("nan")
        ),
        "median_R_N_cert_issued": (
            float(np.nanmedian(R[issued])) if n_issued else float("nan")
        ),
        "frac_issued_below_0.3": (
            float(np.mean(R[issued] < 0.3)) if n_issued else float("nan")
        ),
    }

    # (b) Stratified regime table.
    margin_stratum = np.where(
        np.isfinite(m), np.where(m > 0.0, "high_m", "low_m"), "unknown"
    )
    disp_stratum = _tercile_labels(F)
    size_stratum = np.where(
        np.isfinite(N), np.where(N >= _LARGE_N, "large_N", "small_N"), "unknown"
    )

    strat = pd.DataFrame(
        {
            "margin_stratum": margin_stratum,
            "dispersion_stratum": disp_stratum,
            "size_stratum": size_stratum,
            "_issued": issued.astype(float),
            "_R": R,
        }
    )

    def _agg(g: pd.DataFrame) -> pd.Series:
        issued_mask = g["_issued"] > 0.5
        n = int(len(g))
        ni = int(issued_mask.sum())
        rvals = g.loc[issued_mask, "_R"].to_numpy(float)
        # Non-vacuity at 0.3 over ALL rows in the group (refusals count as
        # vacuous), so issuance and tightness compose into a single fraction.
        n_nonvac = int(np.sum(rvals < 0.3))
        return pd.Series(
            {
                "n_rows": n,
                "n_issued": ni,
                "issuance_rate": (ni / n) if n else float("nan"),
                "mean_R_N_cert_issued": (float(np.mean(rvals)) if ni else float("nan")),
                "nonvacuity_rate_0.3": (n_nonvac / n) if n else float("nan"),
            }
        )

    stratified_regime = (
        strat.groupby(
            ["margin_stratum", "dispersion_stratum", "size_stratum"],
            dropna=False,
        )
        .apply(_agg, include_groups=False)
        .reset_index()
    )

    # (c) Continuous logit-style summary.
    continuous_logit = _logit_fit(m, F, N, R)

    predicted_directions = {
        "m_hat": "negative (higher margin -> lower R_N_cert, more non-vacuity)",
        "F_hat": "positive (higher dispersion -> higher R_N_cert, less non-vacuity)",
        "log_N": "negative (larger N -> lower R_N_cert, more non-vacuity)",
    }

    return {
        "nonvacuity_summary": nonvacuity_summary,
        "stratified_regime": stratified_regime,
        "continuous_logit": continuous_logit,
        "predicted_directions": predicted_directions,
    }
