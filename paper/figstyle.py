"""Shared plotting style for the paper's figures.

Figures are typeset through LaTeX so that they inherit the manuscript's own
fonts: `neurips_2026.sty` sets `\\rmdefault` to `ptm` (Times) and leaves math in
Computer Modern, and the preamble below reproduces exactly that pairing. Text
inside a figure therefore matches body text instead of reading as a pasted-in
plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("pdf")

import matplotlib.pyplot as plt


PAPER_DIR = Path(__file__).resolve().parent
REPO_DIR = PAPER_DIR.parent
RESULTS_DIR = REPO_DIR / "results"
OUTPUT_DIR = PAPER_DIR / "figures"

# \the\linewidth for neurips_2026.sty with the dblblindworkshop option is
# 397.485pt, i.e. exactly 5.5in at 72.27pt/in.
TEXT_WIDTH_IN = 5.5
HALF_WIDTH_IN = 2.66

# Okabe-Ito. Colourblind-safe and separable after greyscale conversion.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
NEUTRAL = "#555555"
GRID = "#D9D9D9"
MUTED = "#9E9E9E"


def use_paper_style() -> None:
    """Apply the manuscript's typography to all subsequent figures."""
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": r"\renewcommand{\rmdefault}{ptm}",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.2,
            "legend.frameon": False,
            "legend.handlelength": 1.4,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.2,
            "legend.borderpad": 0.0,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,
        }
    )


def panel_label(axis, letter: str, title: str, pad: float = 6.0) -> None:
    """Write a bold panel letter and title flush with the axis's left edge."""
    axis.set_title(
        rf"\textbf{{{letter}}}\hspace{{0.5em}}{title}", loc="left", pad=pad
    )


def despine(axis, keep_left: bool = True, keep_bottom: bool = True) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(keep_left)
    axis.spines["bottom"].set_visible(keep_bottom)


def save(fig, name: str) -> None:
    """Write the figure as vector PDF plus a PNG for quick inspection."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.pdf")
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=400)
    plt.close(fig)
