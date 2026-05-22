"""Figure 3: baseline comparison (margin-only, asymptotic Cantelli, certificate)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="baseline_comparison CSV from Panel A full")
    ap.add_argument("--output_dir", required=True)
    a = ap.parse_args()
    p = Path(a.input_csv)
    if not p.exists():
        raise FileNotFoundError(f"Missing input file: {p}")
    df = pd.read_csv(p)
    if not {"R_MC", "R_cert"}.issubset(df.columns):
        raise ValueError("baseline CSV must contain R_MC and R_cert")
    cols = []
    for c in ("R_cert", "R_hoeffding", "R_cantelli"):
        if c in df.columns:
            cols.append(c)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax_slack, ax_undercov = axes
    width = 0.25
    Ns = sorted(df["N"].unique())
    x = np.arange(len(Ns))
    for k, c in enumerate(cols):
        sub = df.groupby("N").apply(lambda d: ((d[c] - d["R_MC"]).where(d[c] >= d["R_MC"]).mean()))
        ax_slack.bar(x + (k - 1) * width, [sub.get(n, np.nan) for n in Ns], width=width, label=c)
    ax_slack.set_xticks(x)
    ax_slack.set_xticklabels(Ns)
    ax_slack.set_xlabel("N")
    ax_slack.set_ylabel("mean slack | coverage")
    ax_slack.legend()

    for k, c in enumerate(cols):
        sub = df.groupby("N").apply(lambda d: float((d[c] < d["R_MC"]).mean()))
        ax_undercov.bar(x + (k - 1) * width, [sub.get(n, np.nan) for n in Ns], width=width, label=c)
    ax_undercov.set_xticks(x)
    ax_undercov.set_xticklabels(Ns)
    ax_undercov.set_xlabel("N")
    ax_undercov.set_ylabel("undercoverage rate")
    ax_undercov.legend()
    fig.tight_layout()
    fig.savefig(out / "baseline_comparison.png", dpi=160, bbox_inches="tight")
    fig.savefig(out / "baseline_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
