"""Figure 5/6: CKA alignment and raw covariance vs normalized rho."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="merged Panel B pairwise CSV with rho_ij, C_ij, cka")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--null_csv", default=None, help="optional permutation null CSV")
    a = ap.parse_args()
    p = Path(a.input_csv)
    if not p.exists():
        raise FileNotFoundError(f"Missing input file: {p}")
    df = pd.read_csv(p)
    needed = {"rho_ij", "cka"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Need columns {needed}; got {list(df.columns)}")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    has_cov = "C_ij" in df.columns
    sp_rho = float(spearmanr(df["rho_ij"], df["cka"]).correlation) if len(df) > 1 else float("nan")
    sp_cov = float(spearmanr(df["C_ij"], df["cka"]).correlation) if has_cov and len(df) > 1 else float("nan")

    null_q99 = None
    if a.null_csv and Path(a.null_csv).exists():
        nd = pd.read_csv(a.null_csv)
        if "family_aware_null" in nd.columns:
            null_q99 = float(np.quantile(nd["family_aware_null"], 0.99))

    fig, axes = plt.subplots(1, 2 if has_cov else 1, figsize=(9 if has_cov else 5, 4), squeeze=False)
    ax = axes[0][0]
    ax.scatter(df["cka"], df["rho_ij"], s=14, c="black", alpha=0.6)
    ax.set_xlabel("CKA")
    ax.set_ylabel("normalised rho_ij")
    title = f"rho vs CKA (Spearman={sp_rho:.3f})"
    if null_q99 is not None:
        title += f"; family-aware null q99={null_q99:.3f}"
    ax.set_title(title)
    if has_cov:
        ax = axes[0][1]
        ax.scatter(df["cka"], df["C_ij"], s=14, c="black", alpha=0.6)
        ax.set_xlabel("CKA")
        ax.set_ylabel("raw covariance C_ij")
        ax.set_title(f"C_ij vs CKA (Spearman={sp_cov:.3f})")
    fig.tight_layout()
    fig.savefig(out / "cka_alignment.png", dpi=160, bbox_inches="tight")
    fig.savefig(out / "cka_alignment.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
