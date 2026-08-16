"""Stronger text baselines and direct probe-vs-baseline inference.

This CPU-only audit targets the two decisive, byte-identical-prefix Slice-3
controls.  It improves on the historical five repeated holdouts in two ways:

1. grouped outer-fold predictions cover every matched scenario exactly once;
2. probe layers are selected by grouped cross-validation nested inside each
   outer training fold; and
3. uncertainty and approximate-randomization tests directly compare the probe
   with the strongest text-only baseline on the same predictions.

The shared prefix is excluded from additive text baselines by construction, as
in the corrected main analysis.  Each baseline sees the differing final user
message only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from vnaa.baselines import (
    bag_of_words_scores,
    bootstrap_auroc_difference_ci,
    format_scores,
    paired_prediction_permutation_test,
    text_logistic_scores,
)
from vnaa.estimators import auroc_columns, dom_directions
from vnaa.probe import fit_layer
from vnaa.run_slice3 import MODEL_TAG, differing_final_messages
from vnaa.run_slice3_transfer import load_pairs


RESULTS = Path("results")
BASELINES = ("bow", "word_tfidf", "char_tfidf", "format", "length")


def pair_folds(n_pairs: int, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic folds that keep both members of a pair together."""
    if n_folds < 2 or n_folds > n_pairs:
        raise ValueError("n_folds must be between 2 and n_pairs")
    order = np.random.default_rng(seed).permutation(n_pairs)
    held_out = np.array_split(order, n_folds)
    all_pairs = np.arange(n_pairs)
    return [
        (np.setdiff1d(all_pairs, test, assume_unique=True), np.sort(test))
        for test in held_out
    ]


def grouped_dom_layer_scores(
    acts_pos: np.ndarray,
    acts_neg: np.ndarray,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """Grouped out-of-fold direction-of-means AUROC for every layer.

    Each matched scenario stays intact.  The returned curve is suitable for
    layer selection inside an outer training fold; no score is obtained from a
    direction fitted on the row it scores.
    """
    if acts_pos.shape != acts_neg.shape or acts_pos.ndim != 3:
        raise ValueError(
            "paired activations must have matching (pair, layer, dim) shapes"
        )
    n_pairs = len(acts_pos)
    labels = np.r_[np.ones(n_pairs), np.zeros(n_pairs)]
    held_scores = np.empty((2 * n_pairs, acts_pos.shape[1]), dtype=np.float64)
    for train_pairs, test_pairs in pair_folds(n_pairs, n_folds, seed):
        x_train = np.r_[acts_pos[train_pairs], acts_neg[train_pairs]]
        y_train = labels[np.r_[train_pairs, n_pairs + train_pairs]]
        directions = dom_directions(x_train, y_train)
        x_test = np.r_[acts_pos[test_pairs], acts_neg[test_pairs]]
        test_idx = np.r_[test_pairs, n_pairs + test_pairs]
        held_scores[test_idx] = np.einsum("nld,ld->nl", x_test, directions)
    return auroc_columns(labels, held_scores)


def fit_selected_dom_grouped_cv(
    acts_pos: np.ndarray,
    acts_neg: np.ndarray,
    n_folds: int,
    seed: int,
):
    """Fit a probe after pair-grouped, training-only layer selection."""
    cv_aurocs = grouped_dom_layer_scores(acts_pos, acts_neg, n_folds, seed)
    layer = int(np.argmax(cv_aurocs))
    x_train = np.r_[acts_pos, acts_neg]
    labels = np.r_[np.ones(len(acts_pos)), np.zeros(len(acts_neg))]
    return fit_layer(x_train[:, layer], labels), layer, cv_aurocs


def _text_scores(all_texts, labels, train_idx, test_idx, seed):
    """All fixed text-only competitors on one train/test partition."""
    return {
        "bow": bag_of_words_scores(
            all_texts, labels, train_idx, test_idx, min_df=2, seed=seed
        ),
        "word_tfidf": text_logistic_scores(
            all_texts, labels, train_idx, test_idx, "word_tfidf", seed=seed
        ),
        "char_tfidf": text_logistic_scores(
            all_texts, labels, train_idx, test_idx, "char_tfidf", seed=seed
        ),
        "format": format_scores(all_texts, labels, train_idx, test_idx, seed=seed),
        "length": np.asarray([len(all_texts[i]) for i in test_idx], dtype=float),
    }


def grouped_oof_predictions(pair, n_folds: int, seed: int) -> dict:
    """Honest grouped out-of-fold predictions for one matched contrast."""
    pos, neg, acts_pos, acts_neg = pair
    if not (len(pos) == len(neg) == len(acts_pos) == len(acts_neg)):
        raise ValueError("matched pair arrays have inconsistent lengths")
    n = len(pos)
    pos_text, neg_text = differing_final_messages(pos, neg)
    texts = pos_text + neg_text
    labels = np.r_[np.ones(n), np.zeros(n)]
    predictions = {name: np.empty(2 * n, dtype=float)
                   for name in ("probe", *BASELINES)}
    selected_layers = []
    selected_layer_aurocs = []

    for fold, (train_pairs, test_pairs) in enumerate(pair_folds(n, n_folds, seed)):
        train_idx = np.r_[train_pairs, n + train_pairs]
        test_idx = np.r_[test_pairs, n + test_pairs]
        x_test = np.r_[acts_pos[test_pairs], acts_neg[test_pairs]]
        probe, layer, inner_aurocs = fit_selected_dom_grouped_cv(
            acts_pos[train_pairs],
            acts_neg[train_pairs],
            n_folds,
            seed + fold,
        )
        predictions["probe"][test_idx] = probe.project(x_test[:, layer])
        selected_layers.append(layer)
        selected_layer_aurocs.append(float(inner_aurocs[layer]))

        for name, scores in _text_scores(
            texts, labels, train_idx, test_idx, seed + fold
        ).items():
            predictions[name][test_idx] = scores

    return {
        "labels": labels,
        "cluster_ids": np.r_[np.arange(n), np.arange(n)],
        "predictions": predictions,
        "selected_layers": selected_layers,
        "layer_selection": {
            "method": "nested_pair_grouped_cv_auroc",
            "inner_folds": n_folds,
            "inner_seed_rule": "outer_seed_plus_fold_index",
            "selected_layer_inner_aurocs": selected_layer_aurocs,
        },
    }


def _source_cv_layer(acts_pos, acts_neg, n_folds: int, seed: int):
    """Choose layer and orientation using grouped source-only CV."""
    aurocs = grouped_dom_layer_scores(acts_pos, acts_neg, n_folds, seed)
    layer = int(np.argmax(np.abs(aurocs - 0.5)))
    sign = 1 if aurocs[layer] >= 0.5 else -1
    return layer, sign, aurocs


def transfer_predictions(source, target, n_folds: int, seed: int) -> dict:
    """Fit on all source pairs after source-only layer selection; score target."""
    pos, neg, acts_pos, acts_neg = source
    target_pos, target_neg, target_acts_pos, target_acts_neg = target
    if len(pos) != len(neg) or len(target_pos) != len(target_neg):
        raise ValueError("transfer inputs must contain matched pairs")

    layer, sign, source_cv_aurocs = _source_cv_layer(
        acts_pos, acts_neg, n_folds, seed
    )
    source_labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    source_acts = np.r_[acts_pos, acts_neg]
    probe = fit_layer(source_acts[:, layer], source_labels)
    target_acts = np.r_[target_acts_pos, target_acts_neg]

    source_pos_text, source_neg_text = differing_final_messages(pos, neg)
    target_pos_text, target_neg_text = differing_final_messages(target_pos, target_neg)
    texts = source_pos_text + source_neg_text + target_pos_text + target_neg_text
    n_source = len(pos) + len(neg)
    n_target = len(target_pos) + len(target_neg)
    labels = np.r_[source_labels, np.ones(len(target_pos)), np.zeros(len(target_neg))]
    train_idx = np.arange(n_source)
    test_idx = np.arange(n_source, n_source + n_target)

    predictions = {
        "probe": sign * probe.project(target_acts[:, layer]),
        **_text_scores(texts, labels, train_idx, test_idx, seed),
    }
    target_labels = labels[test_idx]
    return {
        "labels": target_labels,
        "cluster_ids": np.r_[np.arange(len(target_pos)), np.arange(len(target_pos))],
        "predictions": predictions,
        "selected_layers": [layer],
        "source_cv_auroc": float(source_cv_aurocs[layer]),
        "source_orientation": int(sign),
    }


def summarize_predictions(run: dict, n_boot: int, n_permutations: int, seed: int) -> dict:
    labels = run["labels"]
    clusters = run["cluster_ids"]
    predictions = run["predictions"]
    aurocs = {
        name: float(roc_auc_score(labels, scores))
        for name, scores in predictions.items()
    }
    strongest = max(BASELINES, key=lambda name: aurocs[name])
    comparisons = {}
    for name in BASELINES:
        bootstrap = bootstrap_auroc_difference_ci(
            labels, predictions["probe"], predictions[name], clusters,
            n_boot=n_boot, seed=seed,
        )
        permutation = paired_prediction_permutation_test(
            labels, predictions["probe"], predictions[name], clusters,
            n_permutations=n_permutations, seed=seed,
        )
        comparisons[name] = {"bootstrap": bootstrap, "permutation": permutation}
    result = {
        "n_pairs": int(len(np.unique(clusters))),
        "aurocs": aurocs,
        "strongest_text_baseline": strongest,
        "probe_minus_strongest": comparisons[strongest],
        "probe_minus_each": comparisons,
        "selected_layers": [int(x) for x in run["selected_layers"]],
    }
    if "source_cv_auroc" in run:
        result["source_cv_auroc"] = run["source_cv_auroc"]
        result["source_orientation"] = run["source_orientation"]
    if "layer_selection" in run:
        result["layer_selection"] = run["layer_selection"]
    return result


def _print_summary(name: str, summary: dict) -> None:
    print(f"\n{name} (n={summary['n_pairs']} matched pairs)")
    for method, auroc in summary["aurocs"].items():
        marker = "  <- strongest text" if method == summary["strongest_text_baseline"] else ""
        print(f"  {method:<12} {auroc:.4f}{marker}")
    inference = summary["probe_minus_strongest"]
    boot = inference["bootstrap"]
    perm = inference["permutation"]
    print(
        f"  probe - {summary['strongest_text_baseline']}: "
        f"{boot['difference']:+.4f} "
        f"95% cluster-bootstrap CI [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}], "
        f"one-sided p={perm['p_value']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=2000)
    args = parser.parse_args()

    pairs = load_pairs()
    output = {
        "model": MODEL_TAG,
        "read_point": "last_user_token",
        "paired_surface_text_view": "differing_final_message_only",
        "inference_unit": "matched_scenario_pair",
        "folds": args.folds,
        "seed": args.seed,
        "layer_selection": {
            "method": "nested_pair_grouped_cv_auroc",
            "inner_folds": args.folds,
            "inner_seed_rule": "outer_seed_plus_fold_index",
        },
        "n_boot": args.n_boot,
        "n_permutations": args.n_permutations,
        "baselines": {
            "bow": "word unigram counts",
            "word_tfidf": "word 1-2 gram TF-IDF logistic",
            "char_tfidf": "character 3-5 gram TF-IDF logistic",
            "format": "13 structural features",
            "length": "character length",
        },
        "within": {},
        "transfer": {},
    }

    for name, pair in pairs.items():
        run = grouped_oof_predictions(pair, args.folds, args.seed)
        summary = summarize_predictions(
            run, args.n_boot, args.n_permutations, args.seed
        )
        output["within"][name] = summary
        _print_summary(f"within/{name}", summary)

    for source, target in (
        ("eval_shared", "coherent_pairs"),
        ("coherent_pairs", "eval_shared"),
    ):
        run = transfer_predictions(
            pairs[source], pairs[target], args.folds, args.seed
        )
        summary = summarize_predictions(
            run, args.n_boot, args.n_permutations, args.seed
        )
        key = f"{source}_to_{target}"
        output["transfer"][key] = summary
        _print_summary(f"transfer/{key}", summary)

    out = RESULTS / f"strong_baseline_audit_{MODEL_TAG}_slice3.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
