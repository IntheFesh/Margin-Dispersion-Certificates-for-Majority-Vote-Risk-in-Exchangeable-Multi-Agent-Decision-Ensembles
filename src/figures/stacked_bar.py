"""Analysis 7 principal figure: four-component conservativeness decomposition.

The total slack between the Monte-Carlo majority-vote risk ``R_N^MC`` and the
issued certificate ``R_N^cert`` is attributed, bottom-up, to four cumulative
layers (Analysis 7 conservativeness decomposition):

  * ``A`` -- information loss (margin/dispersion summary vs the realized risk);
  * ``B`` -- bounded-support relaxation (the [0,1] feasibility tightening);
  * ``C`` -- unrestricted relaxation (the two-moment Chebyshev/Cantelli step);
  * ``D`` -- concentration / Hoeffding deviation (finite-N tail allowance).

Each ``(cell, N)`` configuration is one stacked bar; bars are sorted by
``R_N_cert`` descending. The four non-negative components stack from
``R_N^MC`` up to ``R_N^cert`` (their sum equals ``R_N_cert - R_N_MC``), and a
short black horizontal marker is drawn at ``R_N_cert`` (the top of the stack)
so the realized certificate level is legible against the decomposition.

Pure matplotlib; saves BOTH a 300-DPI PNG and a PDF. Missing required columns
raise ``ValueError`` (this module never fabricates a decomposition).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Canonical bottom-up layer order and the column aliases accepted for each.
# We accept either the terse single-letter names (A, B, C, D) or the
# descriptive ``component_*`` names emitted by src.analysis.a7_conservativeness.
_LAYER_ORDER: tuple[str, ...] = ("A", "B", "C", "D")
_LAYER_ALIASES: dict[str, tuple[str, ...]] = {
    "A": ("A", "component_A"),
    "B": ("B", "component_B"),
    "C": ("C", "component_C"),
    "D": ("D", "component_D"),
}
_LAYER_DESCRIPTIONS: dict[str, str] = {
    "A": "A: information loss",
    "B": "B: bounded-support relaxation",
    "C": "C: unrestricted relaxation",
    "D": "D: concentration",
}
# Aliases accepted for the certificate level annotated on top of each bar.
_CERT_ALIASES: tuple[str, ...] = ("R_N_cert", "R_N_cert_value", "cert")


def _resolve_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str:
    """Return the first column in ``df`` matching one of ``aliases``.

    Raises ``ValueError`` (a no-fake-data guard) if none of the aliases are
    present, listing what was searched for.
    """
    for name in aliases:
        if name in df.columns:
            return name
    raise ValueError(
        f"stacked-bar decomposition is missing a required column; expected one "
        f"of {list(aliases)}, available columns: {sorted(df.columns)}"
    )


def _config_label(row: pd.Series) -> str:
    """Build a compact x-tick label for one (cell, N) configuration."""
    parts: list[str] = []
    for key in ("protocol", "benchmark", "model", "cell_id", "cell"):
        if key in row.index and pd.notna(row[key]):
            parts.append(str(row[key]))
            break
    if "N" in row.index and pd.notna(row["N"]):
        parts.append(f"N={int(row['N'])}")
    return "\n".join(parts) if parts else ""


def render_analysis7_stacked_bar(
    decomp_df: pd.DataFrame,
    out_pdf: Path | None = None,
    out_png: Path | None = None,
) -> None:
    """Render the Analysis 7 four-component conservativeness stacked bar.

    Parameters
    ----------
    decomp_df:
        One row per ``(cell, N)`` configuration. Must contain the four
        component columns (``A``/``B``/``C``/``D`` or ``component_A``..``D``)
        and a certificate-level column (``R_N_cert``). Optional descriptor
        columns (``protocol``, ``benchmark``, ``model``, ``cell_id``, ``N``)
        are used for the x-tick labels.
    out_pdf, out_png:
        Output paths. At least one MUST be provided, else ``ValueError``.
        Whichever are given are written (PDF and/or 300-DPI PNG).

    Raises
    ------
    ValueError
        If neither output path is given, if ``decomp_df`` is empty, or if a
        required component / certificate column is missing. (No empty or
        fabricated figure is ever drawn.)
    """
    if out_pdf is None and out_png is None:
        raise ValueError(
            "render_analysis7_stacked_bar requires at least one of out_pdf, out_png"
        )
    if not isinstance(decomp_df, pd.DataFrame):
        raise TypeError("decomp_df must be a pandas DataFrame")
    if len(decomp_df) == 0:
        raise ValueError("decomp_df is empty; cannot render the conservativeness decomposition")

    # Resolve component columns (raises with context if any are missing).
    layer_cols = {layer: _resolve_column(decomp_df, _LAYER_ALIASES[layer]) for layer in _LAYER_ORDER}
    cert_col = _resolve_column(decomp_df, _CERT_ALIASES)

    # Sort by issued certificate, descending; refusals (NaN cert) sort last.
    df = decomp_df.copy()
    df[cert_col] = pd.to_numeric(df[cert_col], errors="coerce")
    df = df.sort_values(by=cert_col, ascending=False, na_position="last").reset_index(drop=True)

    import matplotlib.pyplot as plt

    from src.figures.style import COLORBLIND_COLORS, apply_style

    apply_style()

    n_bars = len(df)
    x = range(n_bars)
    # Optional MC baseline: bars stack from R_N^MC if present, else from 0.
    if "R_N_MC" in df.columns:
        baseline = pd.to_numeric(df["R_N_MC"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        baseline = [0.0] * n_bars

    fig_w = max(6.5, 0.5 * n_bars + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 4.8))

    layer_colors = {
        "A": COLORBLIND_COLORS[0],
        "B": COLORBLIND_COLORS[2],
        "C": COLORBLIND_COLORS[4],
        "D": COLORBLIND_COLORS[3],
    }

    bottoms = list(baseline)
    for layer in _LAYER_ORDER:
        heights = pd.to_numeric(df[layer_cols[layer]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            width=0.8,
            color=layer_colors[layer],
            edgecolor="white",
            linewidth=0.4,
            label=_LAYER_DESCRIPTIONS[layer],
        )
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    # Black horizontal markers at R_N_cert (top of each stack).
    cert_vals = df[cert_col].to_numpy(dtype=float)
    half = 0.4
    for xi, c in zip(x, cert_vals):
        if c == c:  # not NaN
            ax.hlines(c, xi - half, xi + half, colors="black", linewidth=1.6, zorder=6)
    # Single legend proxy for the certificate marker.
    ax.hlines([], [], [], colors="black", linewidth=1.6, label=r"$R_N^{\mathrm{cert}}$")

    ax.set_xticks(list(x))
    ax.set_xticklabels([_config_label(r) for _, r in df.iterrows()], rotation=90, fontsize=7)
    ax.set_ylabel(r"risk / cumulative slack from $R_N^{\mathrm{MC}}$ to $R_N^{\mathrm{cert}}$")
    ax.set_xlabel("configuration (cell, N), sorted by certificate")
    ax.set_title("Analysis 7: conservativeness decomposition of the certificate slack")
    ax.set_ylim(0.0, max(1.0, float(min(1.0, max(bottoms))) + 0.02))
    ax.legend(loc="upper right", ncol=1)
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
