"""Reference-model diagnostics and the central cross-model comparison.

This builder writes three separate figures so each result can sit beside the
text that explains it: the Llama leakage check, the Llama text-baseline
hierarchy, and the four-model linear/nonlinear comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import figstyle
from figstyle import BLUE, GRID, NEUTRAL, ORANGE, RESULTS_DIR

# figstyle selects a non-interactive backend on import.
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


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
DECODER_PATH = RESULTS_DIR / "decoder_comparison/combined_analysis.json"
NONLINEAR_TEXT_PATH = (
    RESULTS_DIR / "nonlinear_followup/e5_prefix_ending_interaction_analysis.json"
)
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
    """Reference-model read-point contrast, drawn as chance-anchored stems."""
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
    axis.set_title("Reference-model leakage check", loc="left", pad=6.0)
    axis.grid(axis="y", color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)
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
    axis.set_title("Reference-model text baseline hierarchy", loc="left", pad=6.0)
    axis.grid(axis="x", color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)
    figstyle.despine(axis)
    axis.tick_params(axis="y", length=0)


def setup_margin_axis(axis, names: list[str], letter: str, title: str) -> None:
    y = np.arange(len(names))
    axis.axvline(0.0, color=NEUTRAL, linewidth=0.7)
    axis.set_xlim(-0.13, 0.055)
    axis.set_xticks([-0.10, -0.05, 0.0, 0.05])
    axis.set_ylim(len(names) - 0.5, -0.5)
    axis.set_yticks(y, names)
    axis.set_xlabel(r"Activation $-$ matched-text AUROC")
    axis.grid(axis="x", color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.tick_params(axis="y", length=0)
    figstyle.panel_label(axis, letter, title)
    figstyle.despine(axis)


def panel_linear(axis, models: dict) -> None:
    """Primary linear probe minus fixed-logistic E5 interaction."""
    names = list(models.keys())
    y = np.arange(len(names))

    for control_index, (key, label, color, marker) in enumerate(CONTROLS):
        offset = -0.14 + 0.28 * control_index
        for index, name in enumerate(names):
            bootstrap = models[name][key]["probe_minus_text"]["bootstrap"]
            difference = bootstrap["difference"]
            low, high = bootstrap["ci_low"], bootstrap["ci_high"]
            significant = low > 0 or high < 0
            axis.errorbar(
                difference,
                y[index] + offset,
                xerr=[[difference - low], [high - difference]],
                color=color,
                marker=marker,
                markerfacecolor=color if significant else "white",
                markeredgecolor=color,
                markeredgewidth=0.9,
                markersize=4.5,
                linewidth=0,
                elinewidth=1.0,
                capsize=2.0,
                label=label if index == 0 else None,
            )

    setup_margin_axis(axis, names, "a", "Primary linear comparison")


def panel_nonlinear(axis, decoder: dict, nonlinear_text: dict) -> None:
    """Activation RBF minus E5-interaction RBF across five fold seeds."""
    names = list(MODEL_REPOSITORIES.keys())
    y = np.arange(len(names))
    seeds = [int(seed) for seed in decoder["seeds"]]
    seed_jitter = np.linspace(-0.045, 0.045, len(seeds))

    for control_index, (key, label, color, marker) in enumerate(CONTROLS):
        offset = -0.14 + 0.28 * control_index
        text_values = np.asarray(
            [nonlinear_text["seeds"][str(seed)][key]["auroc"] for seed in seeds]
        )
        for index, name in enumerate(names):
            repository = MODEL_REPOSITORIES[name]
            activation_values = np.asarray(
                decoder["models"][repository][key]["aurocs"]["rbf"]["values"]
            )
            if len(activation_values) != len(text_values):
                raise ValueError("Activation and E5 analyses use different seed counts")
            margins = activation_values - text_values
            center = y[index] + offset
            axis.scatter(
                margins,
                center + seed_jitter,
                s=8,
                color=color,
                alpha=0.28,
                linewidths=0,
                zorder=2,
            )
            axis.plot(
                np.mean(margins),
                center,
                color=color,
                marker=marker,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.9,
                markersize=4.5,
                linewidth=0,
                label=label if index == 0 else None,
                zorder=3,
            )

    setup_margin_axis(axis, names, "b", "Post-result RBF audit")


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
    decoder = load_json(DECODER_PATH)
    nonlinear_text = load_json(NONLINEAR_TEXT_PATH)

    leakage_fig, leakage_axis = plt.subplots(figsize=(2.75, 1.75))
    panel_temporal(leakage_axis, estimator)
    handles, labels = leakage_axis.get_legend_handles_labels()
    leakage_fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2
    )
    leakage_fig.subplots_adjust(left=0.19, right=0.98, top=0.82, bottom=0.28)
    figstyle.save(leakage_fig, "leakage_check")

    hierarchy_fig, hierarchy_axis = plt.subplots(figsize=(4.15, 1.80))
    panel_hierarchy(hierarchy_axis, llama, models["Llama-3.1-8B"])
    handles = [Patch(facecolor=BLUE, alpha=0.72), Patch(facecolor=ORANGE, alpha=0.72)]
    labels = ["Retained-history", "Coherence-matched"]
    hierarchy_fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2
    )
    hierarchy_fig.subplots_adjust(left=0.25, right=0.99, top=0.83, bottom=0.27)
    figstyle.save(hierarchy_fig, "text_baseline_hierarchy")

    cross_model_fig, axes = plt.subplots(1, 2, figsize=(figstyle.TEXT_WIDTH_IN, 2.05))
    panel_linear(axes[0], models)
    panel_nonlinear(axes[1], decoder, nonlinear_text)
    handles, labels = axes[0].get_legend_handles_labels()
    cross_model_fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2
    )
    cross_model_fig.subplots_adjust(
        left=0.18, right=0.99, wspace=0.55, top=0.84, bottom=0.27
    )
    figstyle.save(cross_model_fig, "cross_model_comparison")


if __name__ == "__main__":
    main()
