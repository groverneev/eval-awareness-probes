"""Figure 4: what the prespecified steering intervention does and does not move."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import figstyle
from figstyle import BLUE, GRID, GREEN, MUTED, NEUTRAL, ORANGE, RESULTS_DIR

# figstyle selects a non-interactive backend on import.
import matplotlib.pyplot as plt


ANALYSIS_PATH = RESULTS_DIR / "steer_Llama-3.1-8B-Instruct_analysis.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Nine swept coefficients do not fit as nine tick labels at this width, so only
# the round ones are labelled; every coefficient still gets a marker.
LABELLED_ALPHAS = {-4.0, -1.0, 0.0, 1.0, 4.0}


def alpha_label(alpha: float) -> str:
    if alpha not in LABELLED_ALPHAS:
        return ""
    return f"{alpha:g}".replace("-", "$-$")


def draw_trend(axis, x, values, degenerate, color, marker):
    """Plot the sweep, breaking the trend line across degenerate coefficients.

    Coefficients whose generations violate the prespecified output-degeneracy
    rule are excluded from behavioural trends, so no line is drawn through
    them and their markers are left hollow and grey.
    """
    x = np.asarray(x, dtype=float)
    values = np.asarray(values, dtype=float)
    degenerate = np.asarray(degenerate, dtype=bool)

    kept = ~degenerate
    axis.plot(x[kept], values[kept], color=color, linewidth=1.3, zorder=2)
    axis.plot(
        x[kept],
        values[kept],
        linestyle="none",
        marker=marker,
        markersize=4.2,
        color=color,
        zorder=3,
    )
    axis.plot(
        x[degenerate],
        values[degenerate],
        linestyle="none",
        marker=marker,
        markersize=4.2,
        markerfacecolor="white",
        markeredgecolor=MUTED,
        markeredgewidth=0.9,
        zorder=3,
    )


def shade_degenerate(axis, x, degenerate) -> None:
    for position, is_degenerate in zip(x, degenerate):
        if is_degenerate:
            axis.axvspan(
                position - 0.5, position + 0.5, color=MUTED, alpha=0.13, linewidth=0
            )


def panel_readout(axis, readout: dict, x, labels) -> None:
    layer = readout["selected_layer"]
    rows = [row for row in readout["grid"] if row["layer"] == layer]
    rows.sort(key=lambda row: row["alpha"])
    means = np.array([row["mean"] for row in rows])
    sds = np.array([row["sd"] for row in rows])

    reference = readout["reference"]
    for value, label, color in (
        (reference["eval_shared_eval_mean"], "eval", ORANGE),
        (reference["eval_shared_deploy_mean"], "deploy", BLUE),
    ):
        axis.axhline(value, color=color, linestyle=(0, (3, 2)), linewidth=0.8)
        axis.text(
            len(x) - 0.6,
            value + 0.18,
            label,
            color=color,
            fontsize=6.2,
            ha="right",
            va="bottom",
        )

    axis.fill_between(x, means - sds, means + sds, color=GREEN, alpha=0.25, linewidth=0)
    axis.plot(x, means, color=GREEN, marker="D", markersize=3.6, linewidth=1.3)

    axis.set_xticks(x, labels)
    axis.set_xlim(-0.5, len(x) - 0.5)
    axis.set_ylabel("Independent readout")
    axis.grid(axis="y", color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)
    figstyle.panel_label(axis, "a", "Readout moves")
    figstyle.despine(axis)


def panel_rate(
    axis, rows, key, x, labels, color, marker, letter, title, ylabel
) -> None:
    degenerate = [row["degenerate"] for row in rows]
    shade_degenerate(axis, x, degenerate)
    draw_trend(axis, x, [row[key] for row in rows], degenerate, color, marker)

    axis.set_xticks(x, labels)
    axis.set_xlim(-0.5, len(x) - 0.5)
    axis.set_ylim(-0.05, 1.05)
    axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0], [r"0", r".25", r".5", r".75", r"1"])
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)
    figstyle.panel_label(axis, letter, title)
    figstyle.despine(axis)


def main() -> None:
    figstyle.use_paper_style()

    analysis = load_json(ANALYSIS_PATH)
    self_report = sorted(analysis["self_report"]["rows"], key=lambda r: r["alpha"])
    safety = sorted(analysis["safety"]["rows"], key=lambda r: r["alpha"])

    alphas = [row["alpha"] for row in self_report]
    x = np.arange(len(alphas))
    labels = [alpha_label(alpha) for alpha in alphas]

    fig, axes = plt.subplots(
        1, 3, figsize=(figstyle.TEXT_WIDTH_IN, 2.0), gridspec_kw={"wspace": 0.42}
    )

    panel_readout(axes[0], analysis["readout"], x, labels)
    panel_rate(
        axes[1],
        self_report,
        "evaluation_rate",
        x,
        labels,
        ORANGE,
        "o",
        "b",
        "Self-reports move",
        "Self-report rate",
    )
    panel_rate(
        axes[2],
        safety,
        "refusal_rate",
        x,
        labels,
        BLUE,
        "s",
        "c",
        "Refusals do not",
        "Refusal rate",
    )

    # All three panels share this axis; the centre panel's label sits at the
    # figure's midpoint, so one label serves the row without repetition.
    axes[1].set_xlabel(r"Intervention coefficient $\alpha$", labelpad=4)

    figstyle.save(fig, "steering_effects")


if __name__ == "__main__":
    main()
