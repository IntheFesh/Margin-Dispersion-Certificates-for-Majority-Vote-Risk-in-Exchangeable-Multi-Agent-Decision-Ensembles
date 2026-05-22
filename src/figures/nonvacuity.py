"""Figure 4: non-vacuity operating regime."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="cell metrics or non-vacuity summary CSV with R_cert, N, benchmark/protocol")
    ap.add_argument("--output_dir", required=True)
    a = ap.parse_args()
    p = Path(a.input_csv)
    if not p.exists():
        raise FileNotFoundError(f"Missing input file: {p}")
    df = pd.read_csv(p)
    if "R_cert" not in df.columns or "N" not in df.columns:
        raise ValueError("input must contain R_cert and N")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df["nonvacuous"] = (df["R_cert"] < 1.0).astype(int)
    df["lt07"] = (df["R_cert"] < 0.7).astype(int)
    df["lt03"] = (df["R_cert"] < 0.3).astype(int)
    g = df.groupby("N")[["nonvacuous", "lt07", "lt03"]].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    Ns = list(g.index)
    width = 0.25
    x = np.arange(len(Ns))
    ax.bar(x - width, g["nonvacuous"].values, width=width, label="R_cert<1")
    ax.bar(x, g["lt07"].values, width=width, label="R_cert<0.7")
    ax.bar(x + width, g["lt03"].values, width=width, label="R_cert<0.3")
    ax.set_xticks(x)
    ax.set_xticklabels(Ns)
    ax.set_xlabel("N")
    ax.set_ylabel("non-vacuity rate")
    ax.set_title("certified low-risk / intermediate / uninformative cells by N")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "nonvacuity.png", dpi=160, bbox_inches="tight")
    fig.savefig(out / "nonvacuity.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
