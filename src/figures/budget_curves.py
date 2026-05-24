"""Budget curves: issued certificate ``R_N^cert`` versus ensemble size ``N``.

One line per cell (a ``(protocol, benchmark, model)`` / ``cell_id`` group),
tracing how the issued certificate moves as the ensemble size ``N`` grows over
the odd design grid ``(3, 7, 15, 31, 63, 127)``. This visualizes the
sample-budget trade-off: larger ``N`` tightens the Hoeffding/concentration term
but the certificate is floored by the irreducible dispersion (Cantelli) term.

Refused ``(cell, N)`` points (``R_N_cert`` is ``None``/``NaN``) leave a gap in
that cell's line (they are never imputed as 0 or 1). Pure matplotlib; saves
BOTH a 300-DPI PNG and a PDF. Missing required columns raise ``ValueError``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Column aliases accepted for the cell grouping key and the certificate value.
_CELL_ALIASES: tuple[str, ...] = ("cell_id", "cell")
_CELL_COMPONENT_KEYS: tuple[str, ...] = ("protocol", "benchmark", "model")
_CERT_ALIASES: tuple[str, ...] = ("R_N_cert", "R_N_cert_value", "cert")


def _resolve_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    for name in aliases:
        if name in df.columns:
            return name
    return None


def _cell_keys(df: pd.DataFrame) -> pd.Series:
    """Return a per-row cell label.

    Prefers an explicit ``cell_id``/``cell`` column; otherwise composes it from
    ``protocol``/``benchmark``/``model`` (whichever are present). Raises
    ``ValueError`` if no grouping information is available at all.
    """
    explicit = _resolve_column(df, _CELL_ALIASES)
    if explicit is not None:
        return df[explicit].astype(str)
    present = [k for k in _CELL_COMPONENT_KEYS if k in df.columns]
    if not present:
        raise ValueError(
            "budget_curves needs a cell grouping column; expected one of "
            f"{list(_CELL_ALIASES)} or some of {list(_CELL_COMPONENT_KEYS)}, "
            f"available columns: {sorted(df.columns)}"
        )
    return df[present].astype(str).agg(" / ".join, axis=1)


def render_budget_curves(
    curves_df: pd.DataFrame,
    out_pdf: Path | None = None,
    out_png: Path | None = None,
) -> None:
    """Render ``R_N^cert`` vs ``N`` budget curves (one line per cell).

    Parameters
    ----------
    curves_df:
        Long-format table with one row per ``(cell, N)``. Must contain an
        ``N`` column and a certificate column (``R_N_cert``); the cell grouping
        is taken from ``cell_id``/``cell`` or composed from
        ``protocol``/``benchmark``/``model``.
    out_pdf, out_png:
        Output paths. At least one MUST be provided, else ``ValueError``.

    Raises
    ------
    ValueError
        If neither output path is given, if ``curves_df`` is empty, or if a
        required column (``N`` or the certificate) is missing.
    """
    if out_pdf is None and out_png is None:
        raise ValueError("render_budget_curves requires at least one of out_pdf, out_png")
    if not isinstance(curves_df, pd.DataFrame):
        raise TypeError("curves_df must be a pandas DataFrame")
    if len(curves_df) == 0:
        raise ValueError("curves_df is empty; cannot render budget curves")
    if "N" not in curves_df.columns:
        raise ValueError(
            f"budget_curves requires an 'N' column; available columns: {sorted(curves_df.columns)}"
        )
    cert_col = _resolve_column(curves_df, _CERT_ALIASES)
    if cert_col is None:
        raise ValueError(
            f"budget_curves requires a certificate column; expected one of "
            f"{list(_CERT_ALIASES)}, available columns: {sorted(curves_df.columns)}"
        )

    df = curves_df.copy()
    df["__cell__"] = _cell_keys(df)
    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    df[cert_col] = pd.to_numeric(df[cert_col], errors="coerce")

    import matplotlib.pyplot as plt

    from src.figures.style import COLORBLIND_COLORS, apply_style

    apply_style()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    cells = sorted(df["__cell__"].dropna().unique().tolist())
    for i, cell in enumerate(cells):
        sub = df[df["__cell__"] == cell].sort_values("N")
        color = COLORBLIND_COLORS[i % len(COLORBLIND_COLORS)]
        ax.plot(
            sub["N"].to_numpy(dtype=float),
            sub[cert_col].to_numpy(dtype=float),  # NaN (refused) breaks the line
            marker="o",
            color=color,
            label=str(cell),
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"ensemble size $N$ (odd grid)")
    ax.set_ylabel(r"issued certificate $R_N^{\mathrm{cert}}$")
    ax.set_title(r"Budget curves: $R_N^{\mathrm{cert}}$ vs ensemble size $N$, per cell")
    ax.set_ylim(0.0, 1.0)
    # Show every odd N present as an explicit tick (avoids log-scale clutter).
    present_N = sorted(int(n) for n in df["N"].dropna().unique())
    if present_N:
        ax.set_xticks(present_N)
        ax.set_xticklabels([str(n) for n in present_N])
    ax.legend(loc="best", ncol=1, fontsize=8)
    fig.tight_layout()

    if out_pdf is not None:
        out_pdf = Path(out_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, format="pdf")
    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, format="png", dpi=300)
    plt.close(fig)
