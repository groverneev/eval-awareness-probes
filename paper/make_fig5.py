"""Figure 5: the rejected leaking construction against the repaired control.

The first written evaluation cells reach near-ceiling probe AUROC at the final
boundary, but they are already near ceiling before the benchmark question
appears. Their synthetic histories carry the label, so those cells cannot
support a temporal-localization claim at all. Only the shared-prefix
recombination sits at chance before the final turn.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import figstyle
from figstyle import (
    BLUE,
    GRID,
    MODEL_FILL,
    MUTED,
    NEUTRAL,
    ORANGE,
    TURN_GAP,
    TURN_HEIGHT,
    TURN_WIDTH,
    USER_FILL,
    read_marker,
    span_bracket,
    turn_box,
)

# figstyle selects a non-interactive backend on import.
import matplotlib.pyplot as plt


ANALYSIS_PATH = RESULTS = figstyle.RESULTS_DIR / "s3_Llama-3.1-8B-Instruct_analysis.json"

LEAK_FILL = "#F7D9C4"

CELLS = [
    ("eval_disguised", "Disguised (rejected)", ORANGE, "o", True),
    ("eval_obvious", "Obvious (rejected)", ORANGE, "^", True),
    ("eval_shared", "Shared prefix (retained)", BLUE, "s", False),
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def panel_constructions(axis) -> None:
    """Why the rejected cells leak: the history itself is class-specific."""
    rows = [
        (
            0.95,
            "Rejected",
            [LEAK_FILL] * 4,
            LEAK_FILL,
            ORANGE,
            "whole history written per label",
        ),
        (
            -0.95,
            "Repaired",
            [USER_FILL, MODEL_FILL, USER_FILL, MODEL_FILL],
            "white",
            ORANGE,
            "history byte-identical across labels",
        ),
    ]

    for baseline, name, fills, ending_fill, ending_edge, note in rows:
        x = 0.0
        for fill in fills:
            turn_box(axis, x, baseline, "", fill)
            x += TURN_WIDTH + TURN_GAP
        prefix_end = x - TURN_GAP
        turn_box(
            axis,
            x,
            baseline,
            "final\nturn",
            ending_fill,
            edge=ending_edge,
            text_color=ending_edge,
            fontsize=6.0,
        )
        span_bracket(axis, 0.0, prefix_end, baseline + TURN_HEIGHT / 2 + 0.10, note)
        axis.text(
            -0.22,
            baseline,
            name,
            ha="right",
            va="center",
            fontsize=6.6,
            color=NEUTRAL,
        )
        read_marker(
            axis,
            prefix_end - TURN_WIDTH / 2,
            baseline - TURN_HEIGHT / 2 - 0.05,
            "pre-final read",
        )

    axis.text(
        -1.45,
        -1.92,
        "Shaded turns differ by label.",
        fontsize=6.0,
        color=NEUTRAL,
        ha="left",
        va="center",
    )

    axis.set_xlim(-1.55, 6.05)
    axis.set_ylim(-2.15, 1.62)
    axis.set_aspect("equal")
    axis.axis("off")
    figstyle.panel_label(axis, "a", "Where the label lives", pad=2.0)


def panel_read_points(axis, results: dict) -> None:
    positions = {"post_instruction_token": 0, "last_user_token": 1}
    for index, (key, label, color, marker, leaks) in enumerate(CELLS):
        offset = -0.22 + 0.22 * index
        for read_point, position in positions.items():
            mean, error = results[f"{key}|{read_point}"]["probe"]
            x = position + offset
            axis.vlines(x, 0.5, mean, color=color, linewidth=1.3, alpha=0.5)
            axis.errorbar(
                x,
                mean,
                yerr=error,
                color=color,
                marker=marker,
                markerfacecolor=color if not leaks else "white",
                markeredgecolor=color,
                markeredgewidth=0.9,
                markersize=4.6,
                linewidth=0,
                elinewidth=1.0,
                capsize=2.0,
                label=label if position == 0 else None,
            )

    axis.axhline(0.5, color=NEUTRAL, linestyle=(0, (3, 2)), linewidth=0.7)
    axis.annotate(
        "already near ceiling before\nthe benchmark question appears",
        xy=(-0.19, 0.943),
        xytext=(0.66, 0.80),
        fontsize=6.0,
        color=NEUTRAL,
        ha="center",
        va="top",
        linespacing=1.3,
        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7, shrinkB=3),
    )
    axis.text(-0.44, 0.512, "chance", fontsize=6.2, color=NEUTRAL, va="bottom")

    axis.set_xticks([0, 1], ["Pre-final", "Final"])
    axis.set_xlim(-0.5, 1.5)
    axis.set_ylim(0.45, 1.04)
    axis.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axis.set_ylabel("Probe AUROC")
    axis.grid(axis="y", color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)
    figstyle.panel_label(axis, "b", "What the probe reads")
    figstyle.despine(axis)


def main() -> None:
    figstyle.use_paper_style()

    results = load_json(ANALYSIS_PATH)["results"]

    fig = plt.figure(figsize=(figstyle.TEXT_WIDTH_IN, 2.30))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.26, bottom=0.16)
    panel_constructions(fig.add_subplot(grid[0]))
    axis_reads = fig.add_subplot(grid[1])
    panel_read_points(axis_reads, results)

    handles, labels = axis_reads.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3)

    figstyle.save(fig, "failed_temporal_control")


if __name__ == "__main__":
    main()
