"""Generate the paper's main temporal-localization and cross-model figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PAPER_DIR = Path(__file__).resolve().parent
REPO_DIR = PAPER_DIR.parent
RESULTS_DIR = REPO_DIR / "results"
OUTPUT_DIR = PAPER_DIR / "figures"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def paired_margin(record: dict) -> tuple[float, float, float]:
    bootstrap = record["probe_minus_strongest"]["bootstrap"]
    return bootstrap["difference"], bootstrap["ci_low"], bootstrap["ci_high"]


def main() -> None:
    estimator = load_json(
        RESULTS_DIR / "estimator_check_Llama-3.1-8B-Instruct_points.json"
    )["results"]
    llama = load_json(
        RESULTS_DIR / "strong_baseline_audit_Llama-3.1-8B-Instruct_slice3.json"
    )["within"]

    replication_paths = {
        "Qwen3-8B": RESULTS_DIR
        / "cross_model/v2/Qwen-Qwen3-8B-a82fd54762/full/analysis.json",
        "Gemma-3-12B": RESULTS_DIR
        / "cross_model/v2/google-gemma-3-12b-it-4abc260aaf/full/analysis.json",
        "Ministral-3-8B": RESULTS_DIR
        / "cross_model/v2/mistralai-Ministral-3-8B-Instruct-2512-BF16-18a1530627/full/analysis.json",
    }
    models = {"Llama-3.1-8B": llama}
    models.update({name: load_json(path)["within"] for name, path in replication_paths.items()})

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    blue = "#0072B2"
    orange = "#D55E00"
    neutral = "#555555"
    grid = "#D0D0D0"

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(5.5, 2.55),
        gridspec_kw={"width_ratios": [0.78, 1.22], "wspace": 0.55},
    )

    controls = [
        (
            "Shared prefix",
            "s3_eval_shared|post_instruction_token",
            "s3_eval_shared|last_user_token",
            blue,
            "o",
        ),
        (
            "Matched coherence",
            "s3_coherent_pairs|post_instruction_token",
            "s3_coherent_pairs|last_user_token",
            orange,
            "s",
        ),
    ]
    for label, pre_key, final_key, color, marker in controls:
        means = [estimator[pre_key]["dom"][0], estimator[final_key]["dom"][0]]
        errors = [estimator[pre_key]["dom"][1], estimator[final_key]["dom"][1]]
        ax_a.errorbar(
            [0, 1],
            means,
            yerr=errors,
            color=color,
            marker=marker,
            markersize=5.2,
            linewidth=1.6,
            capsize=2.5,
            label=label,
        )
    ax_a.axhline(0.5, color=neutral, linestyle="--", linewidth=0.9)
    ax_a.set_xticks([0, 1], ["Pre-final", "Final"])
    ax_a.set_ylim(0.45, 1.025)
    ax_a.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_a.set_ylabel("Probe AUROC")
    ax_a.set_title("a  Temporal localization", loc="left", fontweight="bold")
    ax_a.grid(axis="y", color=grid, linewidth=0.5)
    ax_a.legend(frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.67))

    model_names = list(models.keys())
    y = np.arange(len(model_names))
    offsets = {"eval_shared": -0.13, "coherent_pairs": 0.13}
    styles = {
        "eval_shared": ("Shared prefix", blue, "o"),
        "coherent_pairs": ("Matched coherence", orange, "s"),
    }
    for control_key in ["eval_shared", "coherent_pairs"]:
        label, color, marker = styles[control_key]
        for idx, model_name in enumerate(model_names):
            difference, low, high = paired_margin(models[model_name][control_key])
            significant = low > 0
            ax_b.errorbar(
                difference,
                y[idx] + offsets[control_key],
                xerr=[[difference - low], [high - difference]],
                color=color,
                marker=marker,
                markerfacecolor=color if significant else "white",
                markeredgecolor=color,
                markersize=5.0,
                linewidth=1.2,
                capsize=2.2,
                label=label if idx == 0 else None,
            )
    ax_b.axvline(0.0, color=neutral, linewidth=0.9)
    ax_b.set_yticks(y, model_names)
    ax_b.invert_yaxis()
    ax_b.set_xlim(-0.34, 0.09)
    ax_b.set_xticks([-0.3, -0.2, -0.1, 0.0])
    ax_b.set_xlabel(r"Probe $-$ strongest text AUROC")
    ax_b.set_title("b  Beyond-text advantage", loc="left", fontweight="bold")
    ax_b.grid(axis="x", color=grid, linewidth=0.5)
    ax_b.legend(frameon=False, loc="lower left")

    for axis in (ax_a, ax_b):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(length=3, width=0.7)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "anchor_results.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "anchor_results.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
