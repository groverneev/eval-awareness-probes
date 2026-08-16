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
PURPLE = "#CC79A7"
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


TURN_WIDTH = 1.0
TURN_HEIGHT = 0.52
TURN_GAP = 0.16

USER_FILL = "#E8E8E8"
MODEL_FILL = "#F6F6F6"


def turn_box(
    axis,
    x,
    y,
    label,
    fill,
    edge=MUTED,
    text_color="black",
    width=TURN_WIDTH,
    height=TURN_HEIGHT,
    fontsize=6.6,
) -> None:
    """One conversation turn, drawn as a rounded box."""
    from matplotlib.patches import FancyBboxPatch

    axis.add_patch(
        FancyBboxPatch(
            (x, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0,rounding_size=0.07",
            facecolor=fill,
            edgecolor=edge,
            linewidth=0.7,
        )
    )
    axis.text(
        x + width / 2,
        y,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        linespacing=1.25,
    )


def read_marker(axis, x, y, label) -> None:
    """Mark a read point with a caret below the turn it reads from."""
    axis.plot(
        [x], [y], marker="^", markersize=4.0, color=NEUTRAL, clip_on=False, zorder=4
    )
    axis.text(
        x,
        y - 0.16,
        label,
        ha="center",
        va="top",
        fontsize=6.0,
        color=NEUTRAL,
        linespacing=1.25,
    )


def span_bracket(axis, x_start, x_end, y, label) -> None:
    axis.plot(
        [x_start, x_start, x_end, x_end],
        [y - 0.09, y, y, y - 0.09],
        color=NEUTRAL,
        linewidth=0.7,
        clip_on=False,
    )
    axis.text(
        (x_start + x_end) / 2,
        y + 0.06,
        label,
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=NEUTRAL,
    )


def save(fig, name: str) -> None:
    """Write the figure as vector PDF plus a PNG for quick inspection."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{name}.pdf")
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=400)
    plt.close(fig)
