"""Paper-ready matplotlib styling (pure matplotlib; NO seaborn).

``apply_style()`` sets global rcParams for the v12 figures, ``REGION_LABELS``
holds the canonical design-space region names (NEVER "safe"/"unsafe"), and
``COLORBLIND_COLORS`` is a small Wong/Okabe-Ito colorblind-friendly palette.
"""
from __future__ import annotations

import matplotlib

# Force a non-interactive backend so figures render on headless hosts (no
# display, no GPU). Set before pyplot is imported anywhere downstream.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Canonical region labels for the design-space figure. These are the ONLY
# permitted region names in v12: do not substitute "safe"/"unsafe".
REGION_LABELS: dict[str, str] = {
    "low": "certified low-risk",
    "mid": "intermediate",
    "high": "uninformative-certificate",
}

# Okabe-Ito colorblind-safe palette (hex), ordered for categorical use.
COLORBLIND_COLORS: list[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]


def apply_style() -> None:
    """Apply the v12 paper figure style to matplotlib's global rcParams.

    Sets serif fonts, readable label/tick sizes, 300-DPI save resolution,
    tight bounding boxes, and a colorblind-friendly default property cycle.
    Pure matplotlib; importing this module never imports seaborn.
    """
    plt.rcParams.update(
        {
            "figure.figsize": (6.5, 4.5),
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": plt.cycler(color=COLORBLIND_COLORS),
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "pdf.fonttype": 42,  # embed TrueType (editable text in PDF)
            "ps.fonttype": 42,
        }
    )
