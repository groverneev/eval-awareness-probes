"""Figure 3: signed cross-construction transfer against unigram counts.

The primary grid is the math-free one: GSM8K-derived items are removed from
every cell first, because digit density alone nearly solves some contrasts.
Signed target AUROC is primary because the source labels fix which direction
means evaluation-like. Absolute distance from chance is retained only as a
secondary measure of target separability when orientation is discarded.
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


def signed_probe_matrix(matrices: list[dict], cells: list[str]) -> np.ndarray:
    """Seed-averaged signed probe AUROC for every ordered transfer."""
    grid = np.full((len(cells), len(cells)), np.nan)
    for row, source in enumerate(cells):
        for column, target in enumerate(cells):
            if source == target:
                continue
            per_seed = [matrix[source]["to"][target]["probe"]["auroc"]
                        for matrix in matrices]
            grid[row, column] = float(np.mean(per_seed))
    return grid


def subset_summary(matrices, cells, shares_cell) -> list[dict]:
    """Signed AUROC, orientation, and secondary strength for each subset."""
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
                float(np.mean([e["probe"]["auroc"] for e in entries]))
            )
            per_seed_bow.append(
                float(np.mean([e["bow"]["auroc"] for e in entries]))
            )
        probe_by_transfer = {
            transfer: [
                matrix[transfer[0]]["to"][transfer[1]]["probe"]["auroc"]
                for matrix in matrices
            ]
            for transfer in transfers
        }
        bow_by_transfer = {
            transfer: [
                matrix[transfer[0]]["to"][transfer[1]]["bow"]["auroc"]
                for matrix in matrices
            ]
            for transfer in transfers
        }
        majority = len(matrices) // 2 + 1
        probe_oriented = sum(
            sum(value > 0.5 for value in values) >= majority
            for values in probe_by_transfer.values()
        )
        bow_oriented = sum(
            sum(value > 0.5 for value in values) >= majority
            for values in bow_by_transfer.values()
        )
        probe_all_seeds = sum(
            all(value > 0.5 for value in values)
            for values in probe_by_transfer.values()
        )
        bow_all_seeds = sum(
            all(value > 0.5 for value in values)
            for values in bow_by_transfer.values()
        )
        signed_margins = np.array(per_seed_probe) - np.array(per_seed_bow)
        strength_probe = [
            float(np.mean([
                strength(matrix[s]["to"][t]["probe"]["auroc"])
                for s, t in transfers
            ]))
            for matrix in matrices
        ]
        strength_bow = [
            float(np.mean([
                strength(matrix[s]["to"][t]["bow"]["auroc"])
                for s, t in transfers
            ]))
            for matrix in matrices
        ]
        strength_margins = np.array(strength_probe) - np.array(strength_bow)
        rows.append({
            "label": label,
            "count": len(transfers),
            "probe": float(np.mean(per_seed_probe)),
            "probe_sd": float(np.std(per_seed_probe, ddof=0)),
            "bow": float(np.mean(per_seed_bow)),
            "bow_sd": float(np.std(per_seed_bow, ddof=0)),
            "margin": float(signed_margins.mean()),
            "margin_sd": float(signed_margins.std(ddof=0)),
            "probe_oriented": probe_oriented,
            "bow_oriented": bow_oriented,
            "probe_all_seeds": probe_all_seeds,
            "bow_all_seeds": bow_all_seeds,
            "probe_strength": float(np.mean(strength_probe)),
            "bow_strength": float(np.mean(strength_bow)),
            "strength_margin": float(strength_margins.mean()),
            "strength_margin_sd": float(strength_margins.std(ddof=0)),
        })
    return rows


def panel_heatmap(fig, axis, grid, cells, shares_cell) -> None:
    colormap = LinearSegmentedColormap.from_list(
        "signed_auroc", [PURPLE, "#FFFFFF", GREEN], N=256
    )
    colormap.set_bad("#F2F2F2")

    image = axis.imshow(
        np.ma.masked_invalid(grid),
        cmap=colormap,
        vmin=0.0,
        vmax=1.0,
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

            if source != target:
                value = grid[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.2,
                    color=NEUTRAL,
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
    bar.set_label("Probe signed target AUROC", labelpad=2)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2, width=0.6)

    figstyle.panel_label(axis, "a", "Every transfer direction")


def panel_subsets(axis, rows) -> None:
    y = np.arange(len(rows))
    for index, row in enumerate(rows):
        probe, bow = row["probe"], row["bow"]
        axis.plot(
            [bow, probe], [y[index], y[index]], color=MUTED, linewidth=1.1, zorder=1
        )
        axis.errorbar(
            bow, y[index], xerr=row["bow_sd"], fmt="o", markersize=4.4,
            color="white", markeredgecolor=NEUTRAL, markeredgewidth=1.0,
            ecolor=NEUTRAL, elinewidth=0.7, capsize=1.5, zorder=2,
            label="Unigram" if index == 0 else None,
        )
        axis.errorbar(
            probe, y[index], xerr=row["probe_sd"], fmt="o", markersize=4.4,
            color=GREEN, ecolor=GREEN, elinewidth=0.7, capsize=1.5, zorder=2,
            label="Probe" if index == 0 else None,
        )
        axis.text(
            0.995,
            y[index] + 0.22,
            f"oriented: P {row['probe_oriented']}/{row['count']}, "
            f"U {row['bow_oriented']}/{row['count']}",
            fontsize=5.7,
            color=NEUTRAL,
            ha="right",
            va="top",
        )

    axis.set_yticks(y, [f"{row['label']}\n({row['count']})" for row in rows])
    axis.set_ylim(len(rows) - 0.35, -0.65)
    axis.set_xlim(0.45, 1.02)
    axis.set_xticks([0.5, 0.75, 1.0])
    axis.axvline(0.5, color=MUTED, linestyle="--", linewidth=0.7)
    axis.set_xlabel(r"Mean signed target AUROC ($\pm$ seed SD)")
    axis.grid(axis="x", color=GRID, linewidth=0.5)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right", bbox_to_anchor=(1.03, 1.03))
    figstyle.panel_label(axis, "b", "Signed transfer and orientation")
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

    grid = signed_probe_matrix(matrices, cells)
    rows = subset_summary(matrices, cells, shares_cell)
    for row in rows:
        print(
            f"{row['label']:12s} n={row['count']:2d} "
            f"signed probe={row['probe']:.5f} unigram={row['bow']:.5f} "
            f"margin={row['margin']:+.5f} +/- {row['margin_sd']:.5f}; "
            f"majority-oriented probe={row['probe_oriented']}/{row['count']} "
            f"unigram={row['bow_oriented']}/{row['count']}; "
            f"all-seed probe={row['probe_all_seeds']}/{row['count']} "
            f"unigram={row['bow_all_seeds']}/{row['count']}"
        )
        print(
            f"{'':12s} secondary strength probe={row['probe_strength']:.5f} "
            f"unigram={row['bow_strength']:.5f} "
            f"margin={row['strength_margin']:+.5f} "
            f"+/- {row['strength_margin_sd']:.5f}"
        )

    fig = plt.figure(figsize=(figstyle.TEXT_WIDTH_IN, 2.55))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.66)
    axis_heatmap = fig.add_subplot(outer[0])
    axis_subsets = fig.add_subplot(outer[1])

    panel_heatmap(fig, axis_heatmap, grid, cells, shares_cell)
    panel_subsets(axis_subsets, rows)

    figstyle.save(fig, "disjoint_transfer")


if __name__ == "__main__":
    main()
