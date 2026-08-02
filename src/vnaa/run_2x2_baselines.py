"""Surface baselines on Devbunova's 2x2 cells — slice 2a, step 1.

CPU-only. Independently verifies the dataset's format-matching claims with OUR
baseline hierarchy before any activation extraction. Label convention: class 1
is the first-named cell of a contrast.

Contrasts:
- bench_eval vs casual_deploy_matched: the standard confounded contrast.
  Surface baselines should be high; this is what slice 1 measured on SAD.
- bench_eval vs bench_deploy_rewritten: format held at benchmark, content
  differs. Their matching claim says format/length ~0.5 here.
- casual_eval_mutual vs casual_deploy_matched: format held at casual. Their
  paired-training result leans on this cell; if surface baselines separate it,
  the MuTual proxy is surface-distinguishable and the "restored invariance"
  claim is weaker than reported.
- bench_deploy_rewritten vs casual_deploy_matched: same content source,
  format flipped. A format detector should be high; sanity check.
- casual_deploy_formal vs casual_deploy_matched: genre control — formal tone,
  no benchmark structure. Separates style from structure.

For each baseline: AUROC on seed-0 split with a bootstrap 95% CI, plus
mean +/- sd across split seeds.
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

from vnaa.baselines import (
    bag_of_words_scores,
    bootstrap_auroc_ci,
    format_scores,
    make_split,
    strength,
)
from vnaa.data_2x2 import load_cell

CONTRASTS = [
    ("bench_eval", "casual_deploy_matched"),
    ("bench_eval", "bench_deploy_rewritten"),
    ("casual_eval_mutual", "casual_deploy_matched"),
    ("bench_deploy_rewritten", "casual_deploy_matched"),
    ("casual_deploy_formal", "casual_deploy_matched"),
]


def run_contrast(
    pos_texts: list[str],
    neg_texts: list[str],
    seeds: list[int],
    test_frac: float = 0.3,
) -> list[tuple[str, float, tuple[float, float], float, float]]:
    """Rows of (baseline, seed0 AUROC, seed0 CI, mean, sd) for one contrast."""
    texts = pos_texts + neg_texts
    labels = np.array([1] * len(pos_texts) + [0] * len(neg_texts))

    per_seed: dict[str, list[float]] = {}
    seed0: dict[str, tuple[float, tuple[float, float]]] = {}
    for seed in seeds:
        train_idx, test_idx = make_split(labels, test_frac, seed)
        y = labels[test_idx]
        char_len = np.array([len(texts[i]) for i in test_idx], dtype=float)
        word_len = np.array([len(texts[i].split()) for i in test_idx], dtype=float)
        scored = {
            "length (char)": char_len,
            "length (word)": word_len,
            "format (structure)": format_scores(texts, labels, train_idx, test_idx, seed),
            "bag-of-words": bag_of_words_scores(texts, labels, train_idx, test_idx, seed=seed),
        }
        for name, scores in scored.items():
            auroc = roc_auc_score(y, scores)
            per_seed.setdefault(name, []).append(auroc)
            if seed == seeds[0]:
                seed0[name] = (auroc, bootstrap_auroc_ci(y, scores, seed=seed))

    rows = []
    for name, vals in per_seed.items():
        a0, ci = seed0[name]
        rows.append((name, a0, ci, float(np.mean(vals)), float(np.std(vals))))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Surface baselines on the 2x2 cells")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--test-frac", type=float, default=0.3)
    args = ap.parse_args()

    cells = sorted({c for pair in CONTRASTS for c in pair})
    texts = {c: load_cell(c) for c in cells}
    for c in cells:
        print(f"loaded {c}: n={len(texts[c])}")

    for pos, neg in CONTRASTS:
        print(f"\n=== {pos} (1) vs {neg} (0) ===")
        rows = run_contrast(texts[pos], texts[neg], args.seeds, args.test_frac)
        print(f"{'baseline':<20} {'AUROC':>7} {'95% CI':>17} {'mean+/-sd':>15} {'strength':>9}")
        for name, a0, ci, mean, sd in rows:
            print(
                f"{name:<20} {a0:>7.4f} [{ci[0]:.4f}, {ci[1]:.4f}] "
                f"{mean:>9.4f}+/-{sd:.4f} {strength(a0):>9.4f}"
            )


if __name__ == "__main__":
    main()
