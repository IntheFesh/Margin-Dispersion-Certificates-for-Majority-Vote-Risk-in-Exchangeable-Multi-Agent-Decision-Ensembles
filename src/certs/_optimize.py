"""Shared one-dimensional eta optimizer for the certificate bounds.

The certificate objectives all have the form

    J(eta) = cantelli_term(eta) + exp(-2 N eta^2),  eta in (0, margin).

The primary optimizer is scipy.optimize.minimize_scalar(method='bounded');
the ONLY documented silent fallback in the codebase is a 2000-point dense
grid on np.linspace(1e-7, margin - 1e-7, 2000), used only when scipy raises
ValueError/RuntimeError or returns success=False. The chosen optimizer is
recorded as 'method' in the returned dict.
"""
from __future__ import annotations
from typing import Callable
import numpy as np
from scipy.optimize import minimize_scalar

GRID_POINTS = 2000
_LO = 1e-7


def minimize_eta(objective: Callable[[float], float], margin: float) -> dict:
    """Minimize ``objective`` over eta in (0, margin).

    Returns a dict with keys ``eta_star``, ``value`` (clipped to [0, 1]),
    and ``method`` in {'scipy-bounded', 'dense-grid-fallback'}.
    """
    if margin <= 0:
        # No favorable margin: there is no interior eta. Caller should not
        # reach here for an issued certificate; we surface a vacuous value.
        return {"eta_star": None, "value": 1.0, "method": "no-margin"}
    lo = _LO
    hi = margin - _LO
    if hi <= lo:
        # Degenerate tiny margin: evaluate at the midpoint deterministically.
        eta = margin / 2.0
        return {"eta_star": float(eta), "value": float(min(1.0, objective(eta))), "method": "degenerate-midpoint"}

    method = "scipy-bounded"
    try:
        res = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
        if not getattr(res, "success", False):
            raise RuntimeError("minimize_scalar did not converge")
        eta = float(res.x)
        val = float(res.fun)
    except (ValueError, RuntimeError):
        grid = np.linspace(lo, hi, GRID_POINTS)
        vals = np.array([objective(float(g)) for g in grid])
        idx = int(np.argmin(vals))
        eta = float(grid[idx])
        val = float(vals[idx])
        method = "dense-grid-fallback"
    return {"eta_star": eta, "value": float(min(1.0, max(0.0, val))), "method": method}
