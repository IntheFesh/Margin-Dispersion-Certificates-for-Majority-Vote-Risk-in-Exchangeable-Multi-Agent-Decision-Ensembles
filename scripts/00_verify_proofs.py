"""Phase 0 numerical proof verification.

Verifies, with no GPU/network:
  * hierarchy  B_N^star <= B_N^CH' <= B_N^CH on a grid (check_hierarchy);
  * primal-dual gap of the Theorem 3 LP (< 1e-8);
  * closed-form vs LP agreement for Refinement 1 C_01 (< 1e-4);
  * unbiasedness of F_hat on a Beta-mixture simulation (|bias| < 1e-3).

Writes a structured event to outputs/logs/phase_0.jsonl and prints the
gaps. Raises (aborts) on any hierarchy or primal-dual violation.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from src.certs.theorem1 import B_N_CH
from src.certs.refinement1 import B_N_CH_prime, C_01
from src.certs.theorem3 import B_N_star
from src.certs.moments import unbiased_F_hat
from src.certs.verify import (
    check_hierarchy,
    check_primal_dual_gap,
    max_hierarchy_gap,
    max_primal_dual_gap,
)
from src.utils.logging import JsonlLogger

ALPHA_BAR = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
F_REL = [0.0, 0.1, 0.4, 0.8]
N_GRID = [3, 7, 15, 31, 63]


def _hierarchy_and_dual() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, dual_rows = [], []
    for alpha_bar, f_rel, N in itertools.product(ALPHA_BAR, F_REL, N_GRID):
        F = f_rel * alpha_bar * (1.0 - alpha_bar)
        b_ch = B_N_CH(alpha_bar, F, N)
        b_chp = B_N_CH_prime(alpha_bar, F, N)
        primal, dual = B_N_star(alpha_bar, F, N, return_dual=True)
        rows.append({"alpha_bar": alpha_bar, "F": F, "N": N, "B_CH": b_ch, "B_CH_prime": b_chp, "B_star": primal})
        dual_rows.append({"alpha_bar": alpha_bar, "F": F, "N": N, "primal": primal, "dual": dual})
    return pd.DataFrame(rows), pd.DataFrame(dual_rows)


def _lp_tail_mass(alpha_bar: float, F: float, eta: float, n_grid: int = 2001) -> float:
    a = np.linspace(0.0, 1.0, n_grid)
    c = (a <= 0.5 + eta + 1e-9).astype(float)
    A_eq = np.vstack([np.ones_like(a), a, a ** 2])
    b_eq = np.array([1.0, alpha_bar, alpha_bar ** 2 + F])
    res = linprog(c=-c, A_eq=A_eq, b_eq=b_eq, bounds=(0.0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"C_01 verification LP failed: {res.message}")
    return float(-res.fun)


def _lp_formula_gap() -> float:
    worst = 0.0
    for alpha_bar in (0.60, 0.65, 0.70, 0.75, 0.80):
        m = alpha_bar - 0.5
        eta = m / 2.0
        boundary = (1 - alpha_bar) * (m - eta)
        for F in (0.4 * boundary, min(2.0 * boundary, 0.95 * alpha_bar * (1 - alpha_bar))):
            gap = abs(C_01(alpha_bar, F, eta) - _lp_tail_mass(alpha_bar, F, eta))
            worst = max(worst, gap)
    return worst


def _F_hat_bias(trials: int = 50, M: int = 2000, K: int = 32) -> float:
    rng = np.random.default_rng(20260524)
    beta_var = 12 * 8 / ((20 ** 2) * 21)
    errs = []
    for _ in range(trials):
        alpha = rng.beta(12, 8, size=M)
        X = (rng.random((M, K)) < alpha[:, None]).astype(int)
        _, F_hat = unbiased_F_hat(X)
        errs.append(F_hat - beta_var)
    return float(np.mean(errs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="outputs/logs")
    args = ap.parse_args()
    logger = JsonlLogger(Path(args.output_dir) / "phase_0.jsonl")

    hier_df, dual_df = _hierarchy_and_dual()
    check_hierarchy(hier_df, tol=1e-6)  # raises on violation
    check_primal_dual_gap(dual_df, tol=1e-8)  # raises on violation
    hier_gap = max_hierarchy_gap(hier_df)
    pd_gap = max_primal_dual_gap(dual_df)
    lp_gap = _lp_formula_gap()
    fhat_bias = _F_hat_bias()

    summary = {
        "hierarchy_gap_max": hier_gap,
        "primal_dual_gap_max": pd_gap,
        "lp_formula_gap_max": lp_gap,
        "F_hat_bias_mean": fhat_bias,
        "n_grid_points": int(len(hier_df)),
    }
    logger.event("verify_proofs", **summary)

    print("Proof verification (Phase 0):")
    print(f"  hierarchy gap (B*<=B_CH'<=B_CH): {hier_gap:.3e}  (expected < 1e-6)")
    print(f"  primal-dual gap (Theorem 3):     {pd_gap:.3e}  (expected < 1e-8)")
    print(f"  LP-formula gap (Refinement 1):   {lp_gap:.3e}  (expected < 1e-4)")
    print(f"  F_hat bias (unbiased estimator): {fhat_bias:+.3e}  (expected |.| < 1e-3)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
