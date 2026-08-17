import numpy as np
from sklearn.metrics import roc_auc_score

from vnaa.run_nonlinear_followup import (
    contextual_rbf_configurations,
    grouped_combined_text_rbf_oof_predictions,
    grouped_contextual_rbf_config_scores,
    grouped_contextual_rbf_oof_predictions,
    nonlinear_activation_transfer,
)
from vnaa.run_nonlinear_probe_audit import (
    rbf_configurations,
    text_rbf_configurations,
)


def _conversations(n_pairs: int, positive: bool) -> list[list[dict]]:
    label_word = "evaluation" if positive else "casual"
    return [
        [
            {"role": "user", "content": f"shared context {index}"},
            {"role": "assistant", "content": "shared reply"},
            {"role": "user", "content": f"{label_word} ending {index}"},
        ]
        for index in range(n_pairs)
    ]


def _contextual_features(n_pairs: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    result = {}
    for representation, dimension in (
        ("full_context", 4),
        ("ending_only", 4),
        ("prefix_ending_interaction", 8),
    ):
        positive = rng.normal(size=(n_pairs, dimension))
        negative = rng.normal(size=(n_pairs, dimension))
        if representation == "prefix_ending_interaction":
            positive[:, 0] += 3.0
            negative[:, 0] -= 3.0
        result[representation] = np.r_[positive, negative]
    return result


def test_contextual_rbf_scores_are_deterministic_and_find_signal():
    features = _contextual_features(24)
    configs = contextual_rbf_configurations((0.1, 1.0), (0.5, 1.0))
    first = grouped_contextual_rbf_config_scores(features, configs, 3, 2)
    second = grouped_contextual_rbf_config_scores(features, configs, 3, 2)
    np.testing.assert_allclose(first, second)
    assert first.max() > 0.95
    assert configs[int(np.argmax(first))]["representation"] == (
        "prefix_ending_interaction"
    )


def test_fixed_contextual_oof_selection_covers_every_pair():
    n_pairs = 24
    features = _contextual_features(n_pairs)
    configs = [
        config
        for config in contextual_rbf_configurations((0.1, 1.0), (0.5, 1.0))
        if config["representation"] == "prefix_ending_interaction"
    ]
    run = grouped_contextual_rbf_oof_predictions(features, configs, 3, 0)
    labels = np.r_[np.ones(n_pairs), np.zeros(n_pairs)]
    assert np.isfinite(run["predictions"]).all()
    assert roc_auc_score(labels, run["predictions"]) > 0.95
    assert len(run["selected"]) == 3


def test_combined_text_selection_is_nested_and_pair_complete():
    n_pairs = 24
    positive = _conversations(n_pairs, True)
    negative = _conversations(n_pairs, False)
    features = _contextual_features(n_pairs)
    run = grouped_combined_text_rbf_oof_predictions(
        positive,
        negative,
        features,
        n_folds=3,
        seed=0,
        sparse_configurations=text_rbf_configurations((1.0,), (1.0,)),
        contextual_configurations=contextual_rbf_configurations((1.0,), (1.0,)),
    )
    labels = np.r_[np.ones(n_pairs), np.zeros(n_pairs)]
    assert np.isfinite(run["predictions"]).all()
    assert roc_auc_score(labels, run["predictions"]) > 0.95
    assert len(run["selected"]) == 3


def test_nonlinear_transfer_uses_source_selected_layer_and_scores_all_targets():
    rng = np.random.default_rng(4)
    n_pairs = 30
    source_pos = rng.normal(size=(n_pairs, 2, 5))
    source_neg = rng.normal(size=(n_pairs, 2, 5))
    target_pos = rng.normal(size=(n_pairs, 2, 5))
    target_neg = rng.normal(size=(n_pairs, 2, 5))
    for matrix, direction in (
        (source_pos, 2.5),
        (source_neg, -2.5),
        (target_pos, 2.5),
        (target_neg, -2.5),
    ):
        matrix[:, 1, 0] += direction
    conversations = _conversations(n_pairs, True)
    source = (conversations, conversations, source_pos, source_neg)
    target = (conversations, conversations, target_pos, target_neg)
    run = nonlinear_activation_transfer(
        source,
        target,
        n_folds=3,
        seed=0,
        configurations=rbf_configurations((1.0,), (1.0,)),
    )
    assert run["selected_layer"] == 1
    assert len(run["predictions"]["rbf"]) == 2 * n_pairs
    assert roc_auc_score(run["labels"], run["predictions"]["rbf"]) > 0.95
