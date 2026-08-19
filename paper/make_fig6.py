"""Figure 6: within-family Qwen3 scale ladder.

Panel a reports the preregistered primary linear margins and paired-bootstrap
intervals. Panel b reports the post-result nonlinear sensitivity comparison as
mean plus SD over five grouped-fold seeds. The latter is descriptive rather
than a second scale-trend test.
"""

from __future__ import annotations

import json

import numpy as np

import figstyle
from figstyle import BLUE, GRID, NEUTRAL, ORANGE, RESULTS_DIR

# figstyle selects a non-interactive backend on import.
import matplotlib.pyplot as plt


SCALE_PATH = RESULTS_DIR / "qwen_scale/v1/scale_analysis.json"
NONLINEAR_PATH = RESULTS_DIR / "qwen_scale/nonlinear_followup_v1/combined_analysis.json"
CONTROLS = [
    ("eval_shared", "Retained-history", BLUE, "o"),
    ("coherent_pairs", "Coherence-matched", ORANGE, "s"),
]


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def size_label(parameter_billions: float) -> str:
    if parameter_billions < 1:
        return f"{parameter_billions:.2f}B"
    return f"{parameter_billions:.0f}B"


def setup_axis(axis, x, labels, title, letter):
    axis.axhline(0.0, color=NEUTRAL, linewidth=0.7)
    axis.set_xticks(x, labels)
    axis.set_xlabel("Exact parameter count (log scale)")
    axis.set_ylabel(r"Activation $-$ text AUROC")
    axis.grid(axis="y", color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)
    figstyle.panel_label(axis, letter, title)
    figstyle.despine(axis)


def panel_linear(axis, scale):
    models = scale["models"]
    x = np.log10([model["parameter_count"] for model in models])
    labels = [size_label(model["parameter_billions"]) for model in models]

    for control_key, label, color, marker in CONTROLS:
        endpoint = f"within_{control_key}"
        values = np.asarray(
            [model["endpoints"][endpoint]["probe_minus_strongest_text"] for model in models]
        )
        lows = np.asarray(
            [model["endpoints"][endpoint]["margin_ci_low"] for model in models]
        )
        highs = np.asarray(
            [model["endpoints"][endpoint]["margin_ci_high"] for model in models]
        )
        for index, value in enumerate(values):
            positive = lows[index] > 0
            axis.errorbar(
                x[index],
                value,
                yerr=[[value - lows[index]], [highs[index] - value]],
                color=color,
                marker=marker,
                markerfacecolor=color if positive else "white",
                markeredgecolor=color,
                markeredgewidth=0.9,
                markersize=4.4,
                linewidth=0,
                elinewidth=1.0,
                capsize=2.0,
                label=label if index == 0 else None,
            )
        slope = scale["paired_bootstrap_slopes"][endpoint]
        fitted = np.polyval(np.polyfit(x, values, 1), x)
        axis.plot(x, fitted, color=color, linewidth=0.9, alpha=0.75)
        annotation = (
            f"slope={slope['slope_auroc_per_log10_parameter']:+.3f} "
            f"[{slope['ci_low']:+.3f}, {slope['ci_high']:+.3f}]"
        )
        y = 0.067 if control_key == "eval_shared" else -0.027
        axis.text(x[0], y, annotation, color=color, fontsize=6.1, ha="left")

    setup_axis(axis, x, labels, "Primary linear margins", "a")
    axis.set_ylim(-0.036, 0.078)


def panel_nonlinear(axis, scale, nonlinear):
    models = scale["models"]
    x = np.log10([model["parameter_count"] for model in models])
    labels = [size_label(model["parameter_billions"]) for model in models]

    for control_key, label, color, marker in CONTROLS:
        means = []
        errors = []
        for model in models:
            aggregate = nonlinear["models"][model["repository"]]["aggregate"]["within"][
                control_key
            ]["rbf_minus_text"]
            means.append(aggregate["mean"])
            errors.append(aggregate["sd"])
        means = np.asarray(means)
        axis.errorbar(
            x,
            means,
            yerr=np.asarray(errors),
            color=color,
            marker=marker,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.8,
            markersize=4.2,
            linewidth=0.9,
            elinewidth=1.0,
            capsize=2.0,
            label=label,
        )

    setup_axis(axis, x, labels, "RBF sensitivity (five seeds)", "b")
    axis.set_ylim(-0.010, 0.052)


def main() -> None:
    figstyle.use_paper_style()
    scale = load_json(SCALE_PATH)
    nonlinear = load_json(NONLINEAR_PATH)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(figstyle.TEXT_WIDTH_IN, 2.05),
        gridspec_kw={"wspace": 0.34, "bottom": 0.27},
    )
    panel_linear(axes[0], scale)
    panel_nonlinear(axes[1], scale, nonlinear)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    figstyle.save(fig, "qwen_scale_ladder")


if __name__ == "__main__":
    main()
