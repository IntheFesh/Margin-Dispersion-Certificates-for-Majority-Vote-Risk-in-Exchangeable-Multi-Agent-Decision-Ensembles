"""Figure 1: margin-dispersion design space."""
from __future__ import annotations
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.theory.certificate import population_certificate


def _r_grid(m_grid: np.ndarray, F_grid: np.ndarray, N: int) -> np.ndarray:
    R = np.full((len(F_grid), len(m_grid)), 1.0)
    for i, F in enumerate(F_grid):
        for j, m in enumerate(m_grid):
            if m <= 0:
                R[i, j] = 1.0
            else:
                R[i, j] = population_certificate(0.5 + m, max(F, 1e-12), N)["R_cert"]
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="CSV with columns m_hat, F_hat (or m, F)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--N_small", type=int, default=16)
    ap.add_argument("--N_large", type=int, default=64)
    a = ap.parse_args()

    in_path = Path(a.input_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path}")
    df = pd.read_csv(in_path)
    if "m_hat" in df.columns:
        m_vals = df["m_hat"].to_numpy(float)
    elif "m" in df.columns:
        m_vals = df["m"].to_numpy(float)
    elif "bar_alpha" in df.columns:
        m_vals = df["bar_alpha"].to_numpy(float) - 0.5
    else:
        raise ValueError(f"Need margin column (m_hat / m / bar_alpha); got {list(df.columns)}")
    for cand in ("F_hat_unbiased", "F_hat", "F_true", "F"):
        if cand in df.columns:
            f_vals = df[cand].to_numpy(float)
            break
    else:
        raise ValueError(f"Need dispersion column; got {list(df.columns)}")

    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    m_grid = np.linspace(0.005, 0.5, 60)
    F_grid = np.linspace(0.0, 0.25, 60)
    R_small = _r_grid(m_grid, F_grid, a.N_small)
    R_large = _r_grid(m_grid, F_grid, a.N_large)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, R, N in [(axes[0], R_small, a.N_small), (axes[1], R_large, a.N_large)]:
        cs = ax.contourf(m_grid, F_grid, R, levels=[0, 0.3, 0.7, 1.0001], colors=["#1b7837", "#fdb863", "#b2182b"])
        ax.scatter(m_vals, f_vals, s=14, c="black", edgecolors="white", linewidths=0.5, label="cells")
        ax.set_xlabel("margin m = bar_alpha - 1/2")
        ax.set_title(f"R_cert for N={N}\n(certified low-risk / intermediate / uninformative)")
        ax.legend(loc="upper right")
    axes[0].set_ylabel("dispersion F")
    fig.colorbar(cs, ax=axes, shrink=0.8, label="R_cert region (low/inter/uninformative)")
    fig.savefig(out_dir / "design_space.png", dpi=160, bbox_inches="tight")
    fig.savefig(out_dir / "design_space.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
