"""Figure 2: certificate vs Monte Carlo reference, faceted by protocol/benchmark."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="metrics CSV with R_cert, R_MC, N (and optionally benchmark, protocol)")
    ap.add_argument("--output_dir", required=True)
    a = ap.parse_args()
    in_path = Path(a.input_csv)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path}")
    df = pd.read_csv(in_path)
    required = {"R_cert", "N"}
    if not required.issubset(df.columns):
        raise ValueError(f"need columns R_cert and N; got {list(df.columns)}")
    has_R_MC = "R_MC" in df.columns or "R_true" in df.columns
    mc_col = "R_MC" if "R_MC" in df.columns else ("R_true" if "R_true" in df.columns else None)
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    facets = []
    if "protocol" in df.columns and "benchmark" in df.columns:
        facets = sorted(set(zip(df["benchmark"], df["protocol"])))
    elif "benchmark" in df.columns:
        facets = [(b, "") for b in sorted(df["benchmark"].unique())]
    else:
        facets = [("all", "")]
    fig, axes = plt.subplots(1, max(1, len(facets)), figsize=(4 * max(1, len(facets)), 4), squeeze=False)
    for ax, (b, p) in zip(axes[0], facets):
        sub = df
        if b != "all":
            sub = sub[sub["benchmark"] == b]
        if p:
            sub = sub[sub["protocol"] == p]
        agg = sub.groupby("N").agg(R_cert=("R_cert", "mean"), R_MC=(mc_col, "mean") if mc_col else ("R_cert", "mean"))
        ax.plot(agg.index, agg["R_cert"], marker="o", label="R_cert")
        if mc_col:
            ax.plot(agg.index, agg["R_MC"], marker="s", label="R_MC")
        ax.set_xlabel("N")
        ax.set_ylabel("risk")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{b}{(' / ' + p) if p else ''}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "certificate_vs_reference.png", dpi=160, bbox_inches="tight")
    fig.savefig(out_dir / "certificate_vs_reference.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
