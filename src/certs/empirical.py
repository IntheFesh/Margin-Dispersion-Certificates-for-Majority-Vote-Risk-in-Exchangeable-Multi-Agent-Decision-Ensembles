"""Empirical certificates: R_N^cert, Q_N^cert, R_{N,BA}^cert (Theorem 2).

All quantities are built from the ESTIMATION POOL only. The oracle/reference
pool never feeds certificate construction.

Confidence bounds use a four-event one-sided Hoeffding union bound per cell
{L_alpha, U_alpha, U_2, reserve}, each with delta_cell/4, so the radius is

    eps_delta = sqrt(log(4/delta_cell) / (2 M)).
"""
from __future__ import annotations
import math
import numpy as np
from typing import Callable

from ._optimize import minimize_eta
from .moments import unbiased_F_hat, second_moment_hat
from .robust_ba import robust_ba_objective


def bonferroni_cell_budget(delta_global: float, C: int) -> float:
    """delta_cell = delta_global / C, where C is the number of
    (protocol, benchmark, K_est) cells. Raises on invalid inputs."""
    if not (0.0 < delta_global < 1.0):
        raise ValueError("delta_global must be in (0,1)")
    if C < 1:
        raise ValueError("C (number of cells) must be >= 1")
    return delta_global / C


def hoeffding_radius(delta_cell: float, M: int) -> float:
    """Four-event union-bound one-sided Hoeffding radius."""
    if not (0.0 < delta_cell < 1.0):
        raise ValueError("delta_cell must be in (0,1)")
    if M < 1:
        raise ValueError("M must be >= 1")
    return math.sqrt(math.log(4.0 / delta_cell) / (2.0 * M))


def optimize_eta(objective: Callable[[float], float], margin: float) -> dict:
    """Minimize ``objective`` over eta in (0, margin). Records the optimizer
    used as 'method' (scipy-bounded or dense-grid-fallback)."""
    return minimize_eta(objective, margin)


def _cantelli_hoeffding_inf(U_F: float, margin: float, N: int) -> dict:
    """inf over eta in (0, margin) of U_F/(U_F+(margin-eta)^2) + exp(-2 N eta^2),
    returning the value, eta_star, optimizer method, and the dominating term."""
    def obj(eta: float) -> float:
        return U_F / (U_F + (margin - eta) ** 2) + math.exp(-2.0 * N * eta ** 2)

    res = optimize_eta(obj, margin)
    eta = res["eta_star"]
    if eta is None:
        return {"value": 1.0, "eta_star": None, "method": res["method"], "dominating_term": None}
    cantelli = U_F / (U_F + (margin - eta) ** 2)
    hoeffding = math.exp(-2.0 * N * eta ** 2)
    dominating = "cantelli" if cantelli >= hoeffding else "hoeffding"
    return {
        "value": float(min(1.0, max(0.0, res["value"]))),
        "eta_star": float(eta),
        "method": res["method"],
        "dominating_term": dominating,
    }


def empirical_certificate(
    successes: np.ndarray,
    N: int,
    delta_cell: float,
    use_BA: bool = False,
) -> dict:
    """Compute the empirical certificates on the estimation pool.

    Returns a dict with L_alpha, U_alpha, U_2, U_F, m_L, m_beta_L,
    R_N_cert, Q_N_cert, R_N_BA_cert (None if not requested or refused),
    eta_star / dominating term per issued certificate, and a `reason`
    string for the refusal mode.

    The input matrix MUST come from the estimation pool. It must never
    include oracle/reference-pool rows.
    """
    X = np.asarray(successes)
    if X.ndim != 2:
        raise ValueError("successes must be (M, K_full)")
    M, K = X.shape
    if N < 1:
        raise ValueError("N must be a positive integer")

    alpha_hat, F_hat = unbiased_F_hat(X)
    m2_hat = second_moment_hat(X)
    eps = hoeffding_radius(delta_cell, M)

    # Clip all confidence bounds to their natural ranges:
    # alpha is a probability, so L_alpha, U_alpha in [0,1];
    # E[alpha^2] in [0,1] for alpha in [0,1], so U_2 in [0,1].
    L_alpha = max(0.0, alpha_hat - eps)
    U_alpha = min(1.0, alpha_hat + eps)
    U_2 = min(1.0, max(0.0, m2_hat + eps))
    # Admissibility envelope: F = E[alpha^2] - (E[alpha])^2 = Var(alpha) <= alpha_bar(1 - alpha_bar).
    # Worst-case over alpha_bar in [L_alpha, U_alpha]:
    if L_alpha <= 0.5 <= U_alpha:
        adm_max = 0.25
    else:
        adm_max = max(L_alpha * (1.0 - L_alpha), U_alpha * (1.0 - U_alpha))
    # U_F is the tightest of: Hausdorff ceiling 1/4, admissibility envelope, Hoeffding raw bound.
    U_F = min(0.25, adm_max, max(0.0, U_2 - L_alpha ** 2))
    m_L = L_alpha - 0.5
    m_beta_L = (1.0 - U_alpha) - 0.5

    out: dict = {
        "M": int(M),
        "K_full": int(K),
        "N": int(N),
        "delta_cell": float(delta_cell),
        "alpha_bar_hat": float(alpha_hat),
        "F_hat": float(F_hat),
        "E_alpha2_hat": float(m2_hat),
        "eps_delta": float(eps),
        "L_alpha": float(L_alpha),
        "U_alpha": float(U_alpha),
        "U_2": float(U_2),
        "U_F": float(U_F),
        "m_L": float(m_L),
        "m_beta_L": float(m_beta_L),
        "R_N_cert": None,
        "R_N_cert_eta_star": None,
        "R_N_cert_dominating_term": None,
        "R_N_cert_method": None,
        "Q_N_cert": None,
        "Q_N_cert_eta_star": None,
        "Q_N_cert_dominating_term": None,
        "Q_N_cert_method": None,
        "R_N_BA_cert": None,
        "R_N_BA_cert_eta_star": None,
        "R_N_BA_cert_method": None,
    }

    # Success-side certificate.
    if m_L > 0:
        r = _cantelli_hoeffding_inf(U_F, m_L, N)
        out["R_N_cert"] = r["value"]
        out["R_N_cert_eta_star"] = r["eta_star"]
        out["R_N_cert_dominating_term"] = r["dominating_term"]
        out["R_N_cert_method"] = r["method"]

    # Failure-side certificate (failure-indicator symmetry; same U_F).
    if m_beta_L > 0:
        q = _cantelli_hoeffding_inf(U_F, m_beta_L, N)
        out["Q_N_cert"] = q["value"]
        out["Q_N_cert_eta_star"] = q["eta_star"]
        out["Q_N_cert_dominating_term"] = q["dominating_term"]
        out["Q_N_cert_method"] = q["method"]

    # Robust boundedness-aware certificate (secondary).
    if use_BA and m_L > 0:
        res = optimize_eta(lambda e: robust_ba_objective(e, L_alpha, U_F, N), m_L)
        out["R_N_BA_cert"] = float(min(1.0, max(0.0, res["value"])))
        out["R_N_BA_cert_eta_star"] = res["eta_star"]
        out["R_N_BA_cert_method"] = res["method"]

    if m_L <= 0 and m_beta_L <= 0:
        out["reason"] = "no-direction refusal: m_L <= 0 and m_beta_L <= 0"
    elif m_L <= 0:
        out["reason"] = "success-side not certifiable: m_L <= 0"
    elif m_beta_L <= 0:
        out["reason"] = "failure-side not certifiable: m_beta_L <= 0"
    else:
        out["reason"] = "both directions admit a margin"
    return out
