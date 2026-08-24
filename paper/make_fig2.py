"""Figure 2: leakage check, text-comparator hierarchy, cross-model margins."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.ticker import FuncFormatter

import figstyle
from figstyle import BLUE, GRID, NEUTRAL, ORANGE, RESULTS_DIR

# figstyle selects a non-interactive backend on import.
import matplotlib.pyplot as plt


CONTROLS = [
    ("eval_shared", "Retained-history", BLUE, "o"),
    ("coherent_pairs", "Coherence-matched", ORANGE, "s"),
]

COMPARATORS = [
    ("length", "Length"),
    ("format", "Format"),
    ("bow", "BoW"),
    ("word_tfidf", "Word TF-IDF"),
    ("char_tfidf", "Char TF-IDF"),
]

FULL_CONTEXT_PATH = RESULTS_DIR / "full_context_linear_comparison/analysis.json"
MODEL_REPOSITORIES = {
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Gemma-3-12B": "google/gemma-3-12b-it",
    "Ministral-3-8B": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def panel_temporal(axis, estimator: dict) -> None:
    """Read-point contrast, drawn as chance-anchored stems.

    The two read points are categories, not samples of a continuum, so nothing
    is drawn between them.
    """
    for control_index, (key, label, color, marker) in enumerate(CONTROLS):
        offset = -0.17 + 0.34 * control_index
        for position, read_point in enumerate(
            ["post_instruction_token", "last_user_token"]
        ):
            mean, error = estimator[f"s3_{key}|{read_point}"]["dom"]
            x = position + offset
            axis.vlines(x, 0.5, mean, color=color, linewidth=1.3, alpha=0.55)
            axis.errorbar(
                x,
                mean,
                yerr=error,
                color=color,
                marker=marker,
                markersize=4.4,
                linewidth=0,
                elinewidth=1.0,
                capsize=2.0,
                label=label if position == 0 else None,
            )

    axis.axhline(0.5, color=NEUTRAL, linestyle=(0, (3, 2)), linewidth=0.7)
    axis.text(
        -0.42,
        0.513,
        "chance",
        color=NEUTRAL,
        fontsize=6.2,
        ha="left",
        va="bottom",
    )
    axis.set_xticks([0, 1], ["Pre-final", "Final"])
    axis.set_xlim(-0.5, 1.5)
    axis.set_ylim(0.45, 1.04)
    axis.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axis.set_ylabel("Probe AUROC")
    axis.grid(axis="y", color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)
    figstyle.panel_label(axis, "a", "Leakage check")
    figstyle.despine(axis)


def panel_hierarchy(axis, within: dict, full_context: dict) -> None:
    """Every text comparator against the probe, on the Llama slice."""
    rows = COMPARATORS + [("e5_interaction", "E5 interaction"), ("probe", "Probe")]
    y = np.arange(len(rows))
    height = 0.36

    for control_index, (key, label, color, _marker) in enumerate(CONTROLS):
        aurocs = within[key]["aurocs"]
        offset = height / 2 - height * control_index
        values = [
            (
                full_context[key]["text_auroc"]
                if name == "e5_interaction"
                else full_context[key]["probe_auroc"]
                if name == "probe"
                else aurocs[name]
            )
            for name, _ in rows
        ]
        bars = axis.barh(
            y + offset,
            [value - 0.5 for value in values],
            height=height,
            left=0.5,
            color=color,
            edgecolor="none",
            label=label,
        )
        # E5 is the main text comparator; sparse baselines provide the hierarchy.
        for bar_index, bar in enumerate(bars):
            if rows[bar_index][0] in dict(COMPARATORS):
                bar.set_alpha(0.30)
            elif rows[bar_index][0] == "e5_interaction":
                bar.set_alpha(0.72)

    axis.axvline(0.5, color=NEUTRAL, linewidth=0.7)
    axis.set_yticks(y, [label for _, label in rows])
    axis.set_ylim(-0.6, len(rows) - 0.4)
    axis.set_xlim(0.44, 1.02)
    axis.set_xticks([0.5, 0.75, 1.0])
    axis.set_xlabel("AUROC")
    axis.grid(axis="x", color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)
    figstyle.panel_label(axis, "b", "Text baseline hierarchy")
    figstyle.despine(axis)
    axis.tick_params(axis="y", length=0)


def panel_cross_model(axis_outlier, axis_main, models: dict) -> None:
    """Paired probe-minus-E5-interaction margins, on a broken axis.

    Gemma's coherence-matched margin remains separated from the near-zero
    margins, so the break keeps those intervals legible.
    """
    names = list(models.keys())
    y = np.arange(len(names))

    for control_index, (key, label, color, marker) in enumerate(CONTROLS):
        offset = -0.16 + 0.32 * control_index
        for index, name in enumerate(names):
            bootstrap = models[name][key]["probe_minus_text"]["bootstrap"]
            difference = bootstrap["difference"]
            low, high = bootstrap["ci_low"], bootstrap["ci_high"]
            significant = low > 0 or high < 0
            for axis in (axis_outlier, axis_main):
                axis.errorbar(
                    difference,
                    y[index] + offset,
                    xerr=[[difference - low], [high - difference]],
                    color=color,
                    marker=marker,
                    markerfacecolor=color if significant else "white",
                    markeredgecolor=color,
                    markeredgewidth=0.9,
                    markersize=4.2,
                    linewidth=0,
                    elinewidth=1.0,
                    capsize=2.0,
                    label=label if index == 0 and axis is axis_main else None,
                )

    for axis in (axis_outlier, axis_main):
        axis.set_ylim(len(names) - 0.5, -0.5)
        axis.grid(axis="x", color=GRID, linewidth=0.5)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.tick_params(axis="y", length=0)

    # Labels are stripped of their leading zero: the split axis leaves each
    # half too narrow for two full-width tick labels.
    formatter = FuncFormatter(
        lambda value, _: f"{value:.2f}".replace("0.", ".").replace("-", "$-$")
    )
    axis_outlier.xaxis.set_major_formatter(formatter)
    axis_main.xaxis.set_major_formatter(formatter)

    axis_outlier.set_xlim(-0.125, -0.060)
    axis_outlier.set_xticks([-0.10])
    axis_outlier.set_yticks(y, names)
    axis_outlier.spines["right"].set_visible(False)

    axis_main.set_xlim(-0.050, 0.012)
    axis_main.set_xticks([-0.04, 0.0])
    axis_main.set_yticks(y, [])
    axis_main.spines["left"].set_visible(False)
    axis_main.axvline(0.0, color=NEUTRAL, linewidth=0.7)

    # Diagonal break marks on the two inner edges of the split axis.
    for axis, x in ((axis_outlier, 1.0), (axis_main, 0.0)):
        for shift in (-0.028, 0.028):
            axis.plot(
                [x - 0.075 + shift, x + 0.075 + shift],
                [-0.045, 0.045],
                transform=axis.transAxes,
                color=NEUTRAL,
                linewidth=0.7,
                solid_capstyle="butt",
                clip_on=False,
                zorder=5,
            )

    figstyle.panel_label(axis_outlier, "c", "Relative to E5 interaction")
    axis_main.set_xlabel(r"Probe $-$ E5 interaction AUROC", x=0.05)


def main() -> None:
    figstyle.use_paper_style()

    estimator = load_json(
        RESULTS_DIR / "estimator_check_Llama-3.1-8B-Instruct_points.json"
    )["results"]
    llama = load_json(
        RESULTS_DIR / "strong_baseline_audit_Llama-3.1-8B-Instruct_slice3.json"
    )["within"]
    full_context = load_json(FULL_CONTEXT_PATH)["models"]
    models = {
        label: full_context[repository]
        for label, repository in MODEL_REPOSITORIES.items()
    }

    fig = plt.figure(figsize=(figstyle.TEXT_WIDTH_IN, 2.05))
    outer = fig.add_gridspec(
        1, 3, width_ratios=[0.78, 1.02, 1.40], wspace=0.58, bottom=0.26
    )
    axis_temporal = fig.add_subplot(outer[0])
    axis_hierarchy = fig.add_subplot(outer[1])
    inner = outer[2].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.10)
    axis_outlier = fig.add_subplot(inner[0])
    axis_main = fig.add_subplot(inner[1])

    panel_temporal(axis_temporal, estimator)
    panel_hierarchy(axis_hierarchy, llama, models["Llama-3.1-8B"])
    panel_cross_model(axis_outlier, axis_main, models)

    handles, labels = axis_temporal.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
    )

    figstyle.save(fig, "anchor_results")


if __name__ == "__main__":
    main()
