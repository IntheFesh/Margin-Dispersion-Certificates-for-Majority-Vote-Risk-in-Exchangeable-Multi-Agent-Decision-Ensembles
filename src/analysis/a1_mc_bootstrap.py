"""Analysis 1: instance-level bootstrap of certificate coverage vs R_N^MC.

We resample INSTANCES (rows) with replacement -- never samples (columns) -- to
quantify the joint variability of (a) the issued certificate ``R_N^cert`` built
on the estimation pool and (b) the Monte-Carlo majority-vote risk reference
``R_N^MC`` built on the disjoint oracle pool. Both quantities are recomputed
inside every replicate from independently resampled rows; neither side is held
fixed, so the coverage estimate reflects the full sampling distribution of the
estimation/oracle split.

For one replicate ``b``:

  (1) Draw ``M_est`` estimation rows WITH REPLACEMENT and recompute
      ``R_N^cert`` on the resampled ``(rows x K_full)`` submatrix via
      :func:`src.certs.empirical.empirical_certificate`.
  (2) INDEPENDENTLY draw ``M_oracle`` oracle rows WITH REPLACEMENT and recompute
      ``R_N^MC = mean_m P(Bin(N, alpha_ref_m) <= floor(N/2))`` using the
      per-instance oracle success rate ``alpha_ref_m`` = resampled row mean.
  (3) Record ``coverage_indicator = 1{R_N^cert >= R_N^MC}`` (the certificate is
      a valid UPPER bound on the realized risk iff it dominates the MC risk).

Refusal handling: when ``R_N^cert`` is ``None`` (the cell refuses on the
estimation resample, e.g. ``m_L <= 0``), the replicate contributes
``coverage_indicator = None`` and is EXCLUDED from the coverage-rate
denominator. This keeps the coverage rate conditional on issuance and avoids
silently scoring a refusal as either a hit or a miss.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom

from src.certs.empirical import empirical_certificate
from src.utils.seeds import rng_for

__all__ = ["analysis_1_bootstrap"]


def _validate_matrix(name: str, X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D (M, K) array, got ndim={arr.ndim}")
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError(f"{name} must have M>=2 rows and K>=2 columns, got {arr.shape}")
    return arr


def _r_n_mc(oracle_rows: np.ndarray, N: int) -> float:
    """R_N^MC on a resampled oracle submatrix.

    Per-instance oracle success rate ``alpha_ref_m`` = row mean; the
    majority-vote failure risk for that instance is
    ``P(Bin(N, alpha_ref_m) <= floor(N/2))``; R_N^MC is their mean over rows.
    """
    alpha_ref = oracle_rows.mean(axis=1)
    k = math.floor(N / 2)
    per_instance = binom.cdf(k, N, alpha_ref)
    return float(np.mean(per_instance))


def analysis_1_bootstrap(
    estimation_X: np.ndarray,
    oracle_X: np.ndarray,
    N: int,
    delta_cell: float,
    B_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Bootstrap certificate coverage of the MC majority-vote risk.

    Parameters
    ----------
    estimation_X:
        Estimation-pool success matrix ``(M_est, K_full)`` of binary {0,1}
        entries. Feeds ``R_N^cert`` only.
    oracle_X:
        Oracle-pool success matrix ``(M_oracle, K_ref)`` of binary {0,1}
        entries. Feeds ``R_N^MC`` only; never certificate construction.
    N:
        Ensemble size (odd in the design grid; strict success-majority uses
        ``floor(N/2)``).
    delta_cell:
        Per-cell confidence budget forwarded to ``empirical_certificate``.
    B_boot:
        Number of bootstrap replicates.
    seed:
        Global seed; replicate ``b`` uses ``rng_for(seed, "a1", b)``.

    Returns
    -------
    dict
        ``{"per_replicate": list[dict], "coverage_rate": float,
        "R_cert_ci": (lo, hi)}`` where the CI is the 2.5/97.5 percentile of the
        issued (non-None) ``R_N^cert`` values, and ``coverage_rate`` excludes
        refused replicates from its denominator.
    """
    est = _validate_matrix("estimation_X", estimation_X)
    orc = _validate_matrix("oracle_X", oracle_X)
    if N < 1:
        raise ValueError(f"N must be a positive integer, got {N}")
    if not (0.0 < delta_cell < 1.0):
        raise ValueError(f"delta_cell must be in (0,1), got {delta_cell}")
    if B_boot < 1:
        raise ValueError(f"B_boot must be >= 1, got {B_boot}")

    M_est = est.shape[0]
    M_oracle = orc.shape[0]

    per_replicate: list[dict] = []
    covered = 0
    scored = 0  # replicates with an issued (non-refused) certificate
    issued_certs: list[float] = []

    for b in range(B_boot):
        rng = rng_for(seed, "a1", b)
        # (1) Estimation resample (rows, with replacement) -> R_N^cert.
        est_idx = rng.integers(0, M_est, size=M_est)
        est_boot = est[est_idx, :]
        cert = empirical_certificate(est_boot, N, delta_cell, use_BA=False)
        R_cert = cert["R_N_cert"]

        # (2) INDEPENDENT oracle resample (rows, with replacement) -> R_N^MC.
        orc_idx = rng.integers(0, M_oracle, size=M_oracle)
        orc_boot = orc[orc_idx, :]
        R_mc = _r_n_mc(orc_boot, N)

        # (3) Coverage indicator (None when the certificate refuses).
        if R_cert is None:
            coverage_indicator = None
            reason = cert.get("reason")
        else:
            coverage_indicator = int(R_cert >= R_mc)
            reason = None
            issued_certs.append(float(R_cert))
            scored += 1
            covered += coverage_indicator

        per_replicate.append(
            {
                "b": b,
                "R_N_cert": (None if R_cert is None else float(R_cert)),
                "R_N_MC": float(R_mc),
                "coverage_indicator": coverage_indicator,
                "alpha_bar_hat": float(cert["alpha_bar_hat"]),
                "F_hat": float(cert["F_hat"]),
                "m_L": float(cert["m_L"]),
                "refused": R_cert is None,
                "reason": reason,
            }
        )

    coverage_rate = (covered / scored) if scored > 0 else float("nan")
    if issued_certs:
        lo = float(np.percentile(issued_certs, 2.5))
        hi = float(np.percentile(issued_certs, 97.5))
    else:
        lo, hi = float("nan"), float("nan")

    return {
        "per_replicate": per_replicate,
        "coverage_rate": float(coverage_rate),
        "R_cert_ci": (lo, hi),
        "n_replicates": int(B_boot),
        "n_scored": int(scored),
        "n_refused": int(B_boot - scored),
        "N": int(N),
        "delta_cell": float(delta_cell),
    }
