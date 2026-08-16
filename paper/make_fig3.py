"""Figure 3: cross-construction transfer, probe against unigram counts.

The primary grid is the math-free one: GSM8K-derived items are removed from
every cell first, because digit density alone nearly solves some contrasts.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import figstyle
from figstyle import GREEN, GRID, MUTED, NEUTRAL, PURPLE, RESULTS_DIR

# figstyle selects a non-interactive backend on import.
import matplotlib.pyplot as plt


TRANSFER_PATH = (
    RESULTS_DIR / "s2b_Llama-3.1-8B-Instruct_transfer_multiseed_mathfree.json"
)

DISPLAY_NAMES = {
    "devb_standard": "Standard",
    "devb_fmt_matched": "Format-ctrl.",
    "ours_a_wc": "A vs. WC",
    "ours_a_tm": "A vs. TM",
    "ours_b_wc": "B vs. WC",
    "ours_b_tm": "B vs. TM",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def strength(auroc: float) -> float:
    """Separation strength: distance from chance, direction discarded."""
    return abs(auroc - 0.5)


def margin_matrix(matrices: list[dict], cells: list[str]) -> np.ndarray:
    """Seed-averaged probe-minus-unigram margin for every ordered transfer."""
    grid = np.full((len(cells), len(cells)), np.nan)
    for row, source in enumerate(cells):
        for column, target in enumerate(cells):
            if source == target:
                continue
            per_seed = [
                strength(matrix[source]["to"][target]["probe"]["auroc"])
                - strength(matrix[source]["to"][target]["bow"]["auroc"])
                for matrix in matrices
            ]
            grid[row, column] = float(np.mean(per_seed))
    return grid


def subset_summary(matrices, cells, pairs, shares_cell) -> list[tuple]:
    """Probe and unigram strength, plus the paired margin, for each subset."""
    subsets = [
        ("Disjoint", lambda s, t: not shares_cell(s, t)),
        ("Shared cell", shares_cell),
        ("All", lambda s, t: True),
    ]
    rows = []
    for label, predicate in subsets:
        transfers = [
            (source, target)
            for source, target in itertools.permutations(cells, 2)
            if predicate(source, target)
        ]
        per_seed_probe, per_seed_bow = [], []
        for matrix in matrices:
            entries = [matrix[s]["to"][t] for s, t in transfers]
            per_seed_probe.append(
                float(np.mean([strength(e["probe"]["auroc"]) for e in entries]))
            )
            per_seed_bow.append(
                float(np.mean([strength(e["bow"]["auroc"]) for e in entries]))
            )
        margins = np.array(per_seed_probe) - np.array(per_seed_bow)
        rows.append(
            (
                label,
                len(transfers),
                float(np.mean(per_seed_probe)),
                float(np.mean(per_seed_bow)),
                float(margins.mean()),
                # Population SD over the five split seeds, matching the
                # spread reported in the manuscript.
                float(margins.std(ddof=0)),
            )
        )
    return rows


def panel_heatmap(fig, axis, grid, cells, shares_cell) -> None:
    colormap = LinearSegmentedColormap.from_list(
        "margin", [PURPLE, "#FFFFFF", GREEN], N=256
    )
    colormap.set_bad("#F2F2F2")
    limit = float(np.nanmax(np.abs(grid)))

    image = axis.imshow(
        np.ma.masked_invalid(grid),
        cmap=colormap,
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )

    # Transfers that reuse a dataset cell are not independent evidence.
    for row, source in enumerate(cells):
        for column, target in enumerate(cells):
            if source != target and shares_cell(source, target):
                axis.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        hatch="/////",
                        edgecolor=NEUTRAL,
                        linewidth=0.0,
                        alpha=0.55,
                    )
                )

    labels = [DISPLAY_NAMES[cell] for cell in cells]
    axis.set_xticks(range(len(cells)), labels, rotation=38, ha="right")
    axis.set_yticks(range(len(cells)), labels)
    axis.set_xlabel("Target construction")
    axis.set_ylabel("Source construction")
    axis.set_xticks(np.arange(len(cells) + 1) - 0.5, minor=True)
    axis.set_yticks(np.arange(len(cells) + 1) - 0.5, minor=True)
    axis.grid(which="minor", color="white", linewidth=1.1)
    axis.tick_params(which="minor", length=0)
    axis.tick_params(which="major", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)

    bar = fig.colorbar(image, ax=axis, fraction=0.055, pad=0.03)
    bar.set_label(r"Probe $-$ unigram strength", labelpad=2)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2, width=0.6)

    figstyle.panel_label(axis, "a", "Every transfer direction")


def panel_subsets(axis, rows) -> None:
    y = np.arange(len(rows))
    for index, (_, _, probe, bow, _, _) in enumerate(rows):
        axis.plot(
            [bow, probe], [y[index], y[index]], color=MUTED, linewidth=1.1, zorder=1
        )
        axis.plot(
            bow,
            y[index],
            marker="o",
            markersize=4.4,
            color="white",
            markeredgecolor=NEUTRAL,
            markeredgewidth=1.0,
            zorder=2,
            label="Unigram" if index == 0 else None,
        )
        axis.plot(
            probe,
            y[index],
            marker="o",
            markersize=4.4,
            color=GREEN,
            zorder=2,
            label="Probe" if index == 0 else None,
        )

    for index, (_, _, probe, bow, margin, sd) in enumerate(rows):
        axis.text(
            (bow + probe) / 2,
            y[index] - 0.22,
            rf"$+{margin:.3f} \pm {sd:.3f}$",
            fontsize=6.4,
            color=NEUTRAL,
            ha="center",
            va="bottom",
        )

    axis.set_yticks(y, [f"{label}\n({count})" for label, count, *_ in rows])
    axis.set_ylim(len(rows) - 0.35, -0.65)
    axis.set_xlim(0.08, 0.48)
    axis.set_xticks([0.2, 0.3, 0.4])
    axis.set_xlabel("Mean separation strength")
    axis.grid(axis="x", color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", bbox_to_anchor=(1.03, 1.03))
    figstyle.panel_label(axis, "b", "By transfer subset")
    figstyle.despine(axis)
    axis.tick_params(axis="y", length=0)


def main() -> None:
    figstyle.use_paper_style()

    data = load_json(TRANSFER_PATH)
    matrices = data["matrices"]
    pairs = {name: set(cells) for name, cells in data["pairs"].items()}
    cells = list(DISPLAY_NAMES)

    def shares_cell(source: str, target: str) -> bool:
        return bool(pairs[source] & pairs[target])

    grid = margin_matrix(matrices, cells)
    rows = subset_summary(matrices, cells, pairs, shares_cell)
    for label, count, probe, bow, margin, sd in rows:
        print(f"{label:12s} n={count:2d} probe={probe:.5f} bow={bow:.5f} "
              f"margin={margin:+.5f} +/- {sd:.5f}")

    fig = plt.figure(figsize=(figstyle.TEXT_WIDTH_IN, 2.55))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.62)
    axis_heatmap = fig.add_subplot(outer[0])
    axis_subsets = fig.add_subplot(outer[1])

    panel_heatmap(fig, axis_heatmap, grid, cells, shares_cell)
    panel_subsets(axis_subsets, rows)

    figstyle.save(fig, "disjoint_transfer")


if __name__ == "__main__":
    main()
