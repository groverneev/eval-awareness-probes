"""Text-baseline and paired-margin statistics on planted examples."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from vnaa.baselines import (
    bootstrap_auroc_difference_ci,
    paired_prediction_permutation_test,
    text_logistic_scores,
)
from vnaa.run_strong_baseline_audit import pair_folds
from vnaa.prep_authorship_factorial import select_two_turn_items


def test_tfidf_text_baselines_learn_held_out_signal():
    texts = (["ordinary practical request please" for _ in range(40)]
             + ["canonical benchmark answer option" for _ in range(40)])
    labels = np.array([0] * 40 + [1] * 40)
    train = np.r_[np.arange(30), np.arange(40, 70)]
    test = np.r_[np.arange(30, 40), np.arange(70, 80)]
    for kind in ("word_tfidf", "char_tfidf"):
        scores = text_logistic_scores(texts, labels, train, test, kind, min_df=1)
        assert roc_auc_score(labels[test], scores) == pytest.approx(1.0)


def test_tfidf_empty_causal_text_is_chance():
    texts = [""] * 20
    labels = np.array([0] * 10 + [1] * 10)
    train = np.r_[np.arange(8), np.arange(10, 18)]
    test = np.array([8, 9, 18, 19])
    scores = text_logistic_scores(texts, labels, train, test, "char_tfidf")
    assert np.all(scores == 0)
    assert roc_auc_score(labels[test], scores) == pytest.approx(0.5)


def test_cluster_bootstrap_detects_paired_auc_advantage():
    # Each scenario contributes one positive and one negative prediction.
    n = 80
    labels = np.r_[np.ones(n), np.zeros(n)]
    clusters = np.r_[np.arange(n), np.arange(n)]
    strong = np.r_[np.linspace(0.6, 1.0, n), np.linspace(0.0, 0.4, n)]
    weak = np.tile([0.45, 0.55], n)[: 2 * n]
    result = bootstrap_auroc_difference_ci(
        labels, strong, weak, clusters, n_boot=300, seed=0
    )
    assert result["difference"] > 0.3
    assert result["ci_low"] > 0
    assert result["n_clusters"] == n


def test_paired_prediction_permutation_finds_clear_winner():
    n = 60
    labels = np.r_[np.ones(n), np.zeros(n)]
    clusters = np.r_[np.arange(n), np.arange(n)]
    strong = np.r_[np.linspace(0.6, 1.0, n), np.linspace(0.0, 0.4, n)]
    weak = np.full(2 * n, 0.5)
    result = paired_prediction_permutation_test(
        labels, strong, weak, clusters, n_permutations=500, seed=0
    )
    assert result["difference"] > 0.4
    assert result["p_value"] <= 0.01
    assert result["n_clusters"] == n


def test_paired_prediction_permutation_rejects_bad_alternative():
    labels = np.array([0, 1])
    with pytest.raises(ValueError):
        paired_prediction_permutation_test(
            labels, labels, labels, alternative="less"
        )


def test_pair_folds_cover_each_pair_once_without_leakage():
    folds = pair_folds(23, 5, seed=4)
    held = []
    for train, test in folds:
        assert not set(train) & set(test)
        assert sorted(np.r_[train, test].tolist()) == list(range(23))
        held.extend(test.tolist())
    assert sorted(held) == list(range(23))


def test_authorship_pilot_selection_uses_only_id_hash_and_turn_count():
    def row(item_id, n_users):
        prefix = []
        for turn in range(n_users - 1):
            prefix.extend([
                {"role": "user", "content": f"u{turn}"},
                {"role": "assistant", "content": f"a{turn}"},
            ])
        return {
            "item_id": item_id,
            "eval": prefix + [{"role": "user", "content": "eval"}],
            "deploy": prefix + [{"role": "user", "content": "deploy"}],
        }

    rows = [row("c", 2), row("a", 2), row("b", 3), row("d", 2)]
    first = select_two_turn_items(rows, 2)
    second = select_two_turn_items(list(reversed(rows)), 2)
    assert first == second
    assert "b" not in first
