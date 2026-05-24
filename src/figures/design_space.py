"""Design-space figure: B_N^CH' over the (mean-margin, dispersion) plane.

The background is the boundedness-aware certificate B_N^CH' (Refinement 1,
:func:`src.certs.refinement1.B_N_CH_prime`) evaluated on a grid of
``x = m`` (mean margin, in ``[0, 0.5]``) and ``y = F`` (dispersion, in
``[0, 0.25]``), for two ODD reference ensemble sizes (default 15 and 63).

Coordinate bridge: ``alpha_bar = m + 0.5``. The certificate is only defined
where ``F <= alpha_bar * (1 - alpha_bar)`` (the moment-feasible region); cells
outside it are masked (NaN) and drawn blank.

The certified-risk surface is partitioned into three regions, labelled via
:data:`src.figures.style.REGION_LABELS`:

  * ``certified low-risk``       : ``B_N^CH' < 0.2``
  * ``intermediate``             : ``0.2 <= B_N^CH' < 0.8``
  * ``uninformative-certificate``: ``B_N^CH' >= 0.8`` (near the vacuous bound 1)

Optionally overlays empirical operating points from ``overlay_points`` (a
DataFrame with columns ``m`` and ``F``). Saves BOTH a 300-DPI PNG and a PDF.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Region thresholds on the certified failure probability B_N^CH'.
_LOW_THRESHOLD: float = 0.2
_HIGH_THRESHOLD: float = 0.8


def _certificate_grid(
    m_axis: np.ndarray,
    F_axis: np.ndarray,
    N: int,
) -> np.ndarray:
    """Evaluate B_N^CH' over the (m, F) mesh for one reference size N.

    Returns a 2-D array indexed ``[F_index, m_index]`` (image orientation),
    with NaN at moment-infeasible cells ``F > alpha_bar * (1 - alpha_bar)``.
    """
    from src.certs.refinement1 import B_N_CH_prime  # local import: keeps module light

    grid = np.full((F_axis.size, m_axis.size), np.nan, dtype=float)
    for i, m in enumerate(m_axis):
        alpha_bar = float(m) + 0.5
        feasible_F = alpha_bar * (1.0 - alpha_bar)
        for j, F in enumerate(F_axis):
            if float(F) > feasible_F:
                continue  # outside the moment-feasible region -> leave NaN
            grid[j, i] = B_N_CH_prime(alpha_bar, float(F), N)
    return grid


def render_design_space(
    out_pdf: Path | None = None,
    out_png: Path | None = None,
    N_refs: tuple[int, ...] = (15, 63),
    overlay_points: pd.DataFrame | None = None,
    grid_resolution: int = 80,
) -> None:
    """Render the design-space certificate figure for two ODD reference sizes.

    Parameters
    ----------
    out_pdf, out_png:
        Output paths. At least one MUST be provided, else ``ValueError``.
        Whichever are given are written (PDF and/or 300-DPI PNG).
    N_refs:
        Two ODD reference ensemble sizes (default ``(15, 63)``). A non-odd N
        raises ``ValueError`` (strict success-majority forbids even N: ties).
    overlay_points:
        Optional DataFrame with columns ``m`` and ``F``; each row is scattered
        on every panel as an empirical operating point.
    grid_resolution:
        Number of samples along each axis of the background mesh.

    Raises
    ------
    ValueError
        If neither output path is given, if ``N_refs`` is empty / contains an
        even or non-positive N, or if ``overlay_points`` lacks ``m``/``F``.
    """
    if out_pdf is None and out_png is None:
        raise ValueError("render_design_space requires at least one of out_pdf, out_png")
    if len(N_refs) == 0:
        raise ValueError("N_refs must contain at least one reference size")
    for N in N_refs:
        if int(N) < 1:
            raise ValueError(f"reference size N must be positive, got {N}")
        if int(N) % 2 == 0:
            raise ValueError(f"reference size N must be ODD (strict success-majority), got {N}")
    if grid_resolution < 2:
        raise ValueError("grid_resolution must be >= 2")

    if overlay_points is not None:
        missing = {"m", "F"} - set(overlay_points.columns)
        if missing:
            raise ValueError(f"overlay_points missing columns: {sorted(missing)}")

    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D

    from src.figures.style import COLORBLIND_COLORS, REGION_LABELS, apply_style

    apply_style()

    m_axis = np.linspace(0.0, 0.5, grid_resolution)
    F_axis = np.linspace(0.0, 0.25, grid_resolution)
    extent = (0.0, 0.5, 0.0, 0.25)

    # Discrete 3-region colormap (light green / amber / grey), colorblind-safe.
    region_colors = ["#56B4E9", "#E69F00", "#999999"]
    cmap = ListedColormap(region_colors)
    cmap.set_bad(color="white")
    bounds = [0.0, _LOW_THRESHOLD, _HIGH_THRESHOLD, 1.0]
    norm = BoundaryNorm(bounds, cmap.N)

    n_panels = len(N_refs)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5.2 * n_panels, 4.6), squeeze=False, sharey=True
    )
    axes_flat = axes[0]

    contour_levels = [_LOW_THRESHOLD, _HIGH_THRESHOLD]
    mesh = None
    for ax, N in zip(axes_flat, N_refs):
        grid = _certificate_grid(m_axis, F_axis, int(N))
        masked = np.ma.masked_invalid(grid)

        mesh = ax.imshow(
            masked,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
        )
        # Region-boundary contour lines for B_N^CH' = 0.2 and 0.8.
        MM, FF = np.meshgrid(m_axis, F_axis)
        try:
            ax.contour(
                MM,
                FF,
                np.ma.filled(masked, np.nan),
                levels=contour_levels,
                colors="black",
                linewidths=0.8,
            )
        except Exception:  # noqa: BLE001 - contour can fail on degenerate masks
            pass

        if overlay_points is not None and len(overlay_points) > 0:
            ax.scatter(
                overlay_points["m"].to_numpy(dtype=float),
                overlay_points["F"].to_numpy(dtype=float),
                s=28,
                facecolors="none",
                edgecolors=COLORBLIND_COLORS[1],
                linewidths=1.3,
                zorder=5,
                label="empirical operating point",
            )

        ax.set_title(rf"$B_N^{{\mathrm{{CH}}'}}$, reference $N={int(N)}$")
        ax.set_xlabel(r"mean margin $m = \bar{\alpha} - 1/2$")
        ax.set_xlim(0.0, 0.5)
        ax.set_ylim(0.0, 0.25)

    axes_flat[0].set_ylabel(r"dispersion $F$")

    # Shared legend describing the three regions (canonical v12 labels).
    region_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=region_colors[0],
               markeredgecolor="black", markersize=11, label=REGION_LABELS["low"]),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=region_colors[1],
               markeredgecolor="black", markersize=11, label=REGION_LABELS["mid"]),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=region_colors[2],
               markeredgecolor="black", markersize=11, label=REGION_LABELS["high"]),
    ]
    if overlay_points is not None and len(overlay_points) > 0:
        region_handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor=COLORBLIND_COLORS[1], markersize=8,
                   label="empirical operating point")
        )
    fig.legend(
        handles=region_handles,
        loc="lower center",
        ncol=min(4, len(region_handles)),
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        r"Design space: certified strict success-majority risk $B_N^{\mathrm{CH}'}(\bar{\alpha}, F)$",
        y=1.02,
    )
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
