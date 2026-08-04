"""Slide-ready figures for the slice-2b transfer results.

Produces three PNGs in figures/ (committed, so teammates can use them):
  fig3_transfer_matrix.png  - probe AUROC, train x test, 6x6
  fig4_asymmetry.png        - theirs->ours vs ours->theirs, the headline
  fig5_probe_vs_bow.png     - probe minus bag-of-words margin per training set

Numbers are read from the multi-seed transfer results file, so the figures
cannot drift from the run. No prompt text is used.

DEFAULT SOURCE IS THE MATH-FREE RUN (arc + mmlu). GSM8K is excluded because
digit density alone separates its eval items from their deploy counterparts at
0.949 — identical to the probe's score there — so those items measure numeral
counting, not evaluation recognition. See CLAUDE.md, slice 2b results.

Usage: .venv\\Scripts\\python scripts/make_2b_figures.py
       .venv\\Scripts\\python scripts/make_2b_figures.py --all-sources
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--all-sources", action="store_true",
                help="use the gsm8k-contaminated run instead of the math-free one")
cli = ap.parse_args()

OUT = Path("figures")
OUT.mkdir(exist_ok=True)
SUFFIX = "" if cli.all_sources else "_mathfree"
RESULTS = Path(f"results/s2b_Llama-3.1-8B-Instruct_transfer_multiseed{SUFFIX}.json")
SOURCE_NOTE = ("all sources incl. gsm8k (contaminated)" if cli.all_sources
               else "ARC + MMLU only; math items excluded as digit-confounded")
FIG_SUFFIX = "_allsources" if cli.all_sources else ""

LABELS = {
    "devb_standard": "standard\n(bench vs real)",
    "devb_fmt_matched": "format-matched\n(bench vs bench)",
    "ours_a_wc": "ours A\nvs WildChat",
    "ours_a_tm": "ours A\nvs topic-matched",
    "ours_b_wc": "ours B\nvs WildChat",
    "ours_b_tm": "ours B\nvs topic-matched",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})

with open(RESULTS) as f:
    data = json.load(f)
mats = data["matrices"]
pairs = list(data["pairs"])
n_seeds = len(data["seeds"])


def vals(train: str, test: str, method: str) -> np.ndarray:
    return np.array([m[train]["to"][test][method]["auroc"] for m in mats])


def strength(a):
    return np.abs(np.asarray(a) - 0.5)


# ---------------------------------------------------------------- fig 3
mean = np.array([[vals(tr, te, "probe").mean() for te in pairs] for tr in pairs])
sd = np.array([[vals(tr, te, "probe").std() for te in pairs] for tr in pairs])

fig, ax = plt.subplots(figsize=(9.5, 7.2))
im = ax.imshow(mean, cmap="RdYlBu_r", vmin=0.35, vmax=1.0)
ax.set_xticks(range(len(pairs)), [LABELS[p] for p in pairs], fontsize=8.5,
              rotation=30, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(pairs)), [LABELS[p] for p in pairs], fontsize=8.5)
ax.set_xlabel("tested on", fontsize=12, labelpad=10)
ax.set_ylabel("trained on", fontsize=12, labelpad=10)
for i in range(len(pairs)):
    for j in range(len(pairs)):
        on_diag = i == j
        ax.text(j, i, f"{mean[i, j]:.2f}\n±{sd[i, j]:.2f}", ha="center", va="center",
                fontsize=8.5, fontweight="bold" if on_diag else "normal",
                color="white" if mean[i, j] > 0.85 or mean[i, j] < 0.45 else "black")
        if on_diag:
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                       edgecolor="black", lw=2.5))
fig.colorbar(im, ax=ax, label="probe AUROC", shrink=0.8)
ax.set_title("Cross-cell transfer: what the probe learned, tested elsewhere\n"
             f"boxed = in-distribution; mean ± sd over {n_seeds} seeds\n"
             f"{SOURCE_NOTE}", fontsize=11.5, pad=14)
fig.tight_layout()
fig.savefig(OUT / f"fig3_transfer_matrix{FIG_SUFFIX}.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 4
fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.8),
                               gridspec_kw={"width_ratios": [1, 1.25]})

# Panel A: the asymmetry. Two reciprocal transfers between the same two cells.
a_m = [vals("devb_standard", "ours_a_wc", "probe").mean(),
       vals("ours_a_wc", "devb_standard", "probe").mean()]
a_e = [vals("devb_standard", "ours_a_wc", "probe").std(),
       vals("ours_a_wc", "devb_standard", "probe").std()]
axa.bar([0, 1], a_m, 0.55, yerr=a_e, capsize=5, color=["#eb6834", "#2a78d6"])
for xi, mi in zip([0, 1], a_m):
    axa.text(xi, mi + 0.03, f"{mi:.2f}", ha="center", fontsize=11, fontweight="bold")
axa.axhline(0.5, color="#888780", ls="--", lw=1)
axa.set_xticks([0, 1], ["their cell →\nscored on ours", "our cell →\nscored on theirs"],
               fontsize=10)
axa.set_ylabel("probe AUROC on the target cell")
axa.set_ylim(0.3, 1.1)
axa.set_title("The direction only travels one way", fontsize=11.5, pad=10)

# Panel B: like-for-like. Same target cell, two different training sets.
targets = ["ours_b_wc", "ours_a_tm", "devb_fmt_matched"]
x = np.arange(len(targets))
w = 0.38
for k, (label, tr, colour) in enumerate([
    ("trained on the standard confounded contrast", "devb_standard", "#eb6834"),
    ("trained on our format-matched cell", "ours_a_wc", "#2a78d6"),
]):
    m = [vals(tr, te, "probe").mean() for te in targets]
    e = [vals(tr, te, "probe").std() for te in targets]
    axb.bar(x + (k - 0.5) * w, m, w, yerr=e, capsize=4, label=label, color=colour)
    for xi, mi in zip(x + (k - 0.5) * w, m):
        axb.text(xi, mi + 0.03, f"{mi:.2f}", ha="center", fontsize=9)
axb.axhline(0.5, color="#888780", ls="--", lw=1)
axb.text(len(targets) - 0.55, 0.515, "chance", fontsize=9, color="#888780", ha="right")
axb.set_xticks(x, ["ours B\nvs WildChat", "ours A\nvs topic-matched",
                   "format-matched\n(bench vs bench)"], fontsize=9.5)
axb.set_ylim(0.3, 1.1)
axb.legend(fontsize=9, loc="lower center", frameon=False, ncol=1)
axb.set_title("Same target cell, two training sets", fontsize=11.5, pad=10)

fig.suptitle("Realistic training data generalizes; benchmark training data does not\n"
             f"({SOURCE_NOTE})", fontsize=12.5, y=1.04)
fig.tight_layout()
fig.savefig(OUT / f"fig4_asymmetry{FIG_SUFFIX}.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 5
margins, errs = [], []
for tr in pairs:
    per_seed = np.array([
        np.mean([strength(vals(tr, te, "probe")[i]) - strength(vals(tr, te, "bow")[i])
                 for te in pairs if te != tr])
        for i in range(n_seeds)
    ])
    margins.append(per_seed.mean())
    errs.append(per_seed.std())

order = np.argsort(margins)
fig, ax = plt.subplots(figsize=(8, 4.6))
colours = ["#eb6834" if margins[i] < 0 else "#2a78d6" for i in order]
ax.barh(range(len(pairs)), [margins[i] for i in order],
        xerr=[errs[i] for i in order], capsize=4, color=colours)
ax.set_yticks(range(len(pairs)),
              [LABELS[pairs[i]].replace("\n", " ") for i in order], fontsize=9.5)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("probe advantage over bag-of-words  (mean |AUROC−0.5| difference)")
ax.set_title("Only the standard confounded setup makes activations\n"
             f"worse than word counts at generalizing\n({SOURCE_NOTE})",
             fontsize=11.5, pad=12)
fig.tight_layout()
fig.savefig(OUT / f"fig5_probe_vs_bow{FIG_SUFFIX}.png", bbox_inches="tight")
plt.close(fig)

print(f"wrote figs with suffix {FIG_SUFFIX!r} from {RESULTS.name}")
