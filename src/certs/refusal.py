"""Refusal taxonomy (4 modes) for each cell x N.

Modes:
  * no_direction       : m_L <= 0 AND m_beta_L <= 0
  * success_margin     : m_L > 0 but R_N_cert > epsilon
  * failure_margin     : m_beta_L > 0 but Q_N_cert > epsilon
  * bidirectional_cert : min(R_N_cert, Q_N_cert) <= epsilon

Within a margin refusal, sub-classify as dispersion-dominated (Cantelli
term dominates at eta*) vs budget-dominated (Hoeffding term dominates).
"""
from __future__ import annotations


def classify_refusal(cell: dict, epsilon: float) -> dict:
    """Classify a single empirical_certificate() result dict.

    Returns {mode, sub_mode}. ``sub_mode`` is None outside margin refusals.
    """
    m_L = cell["m_L"]
    m_beta_L = cell["m_beta_L"]
    R = cell.get("R_N_cert")
    Q = cell.get("Q_N_cert")

    if m_L <= 0 and m_beta_L <= 0:
        return {"mode": "no_direction", "sub_mode": None}

    certified = False
    issued_vals = [v for v in (R, Q) if v is not None]
    if issued_vals and min(issued_vals) <= epsilon:
        certified = True
    if certified:
        return {"mode": "bidirectional_cert", "sub_mode": None}

    # Not certified, but at least one direction has a margin -> margin refusal.
    if m_L > 0 and (R is None or R > epsilon):
        sub = _sub_mode(cell, side="success")
        return {"mode": "success_margin", "sub_mode": sub}
    if m_beta_L > 0 and (Q is None or Q > epsilon):
        sub = _sub_mode(cell, side="failure")
        return {"mode": "failure_margin", "sub_mode": sub}

    return {"mode": "no_direction", "sub_mode": None}


def _sub_mode(cell: dict, side: str) -> str | None:
    term = cell.get("R_N_cert_dominating_term") if side == "success" else cell.get("Q_N_cert_dominating_term")
    if term == "cantelli":
        return "dispersion_dominated"
    if term == "hoeffding":
        return "budget_dominated"
    return None
