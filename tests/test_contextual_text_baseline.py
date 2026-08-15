"""Local tests for the contextual baseline; no model download required."""

import numpy as np
from sklearn.metrics import roc_auc_score

from vnaa.cross_model_suite import PairedContrast
from vnaa.run_contextual_text_baseline import (
    contrast_text_views,
    feature_matrices,
    grouped_embedding_predictions,
    render_messages,
)


def test_render_messages_preserves_roles_and_order():
    messages = [
        {"role": "user", "content": " first "},
        {"role": "assistant", "content": "second"},
    ]
    assert render_messages(messages) == "[USER]\nfirst\n\n[ASSISTANT]\nsecond"


def test_contrast_views_are_positive_then_negative():
    prefix = [
        {"role": "user", "content": "context"},
        {"role": "assistant", "content": "answer"},
    ]
    contrast = PairedContrast(
        "tiny",
        ["x"],
        [prefix + [{"role": "user", "content": "positive"}]],
        [prefix + [{"role": "user", "content": "negative"}]],
    )
    views = contrast_text_views(contrast)
    assert views["prefix"][0] == views["prefix"][1]
    assert views["ending"] == ["positive", "negative"]
    assert views["full"][0].endswith("positive")
    assert views["full"][1].endswith("negative")


def test_interaction_feature_shape_and_contents():
    prefix = np.array([[1.0, 2.0], [3.0, 4.0]])
    ending = np.array([[2.0, 0.0], [1.0, 5.0]])
    full = np.array([[0.1, 0.2], [0.3, 0.4]])
    features = feature_matrices(prefix, ending, full)
    assert features["full_context"] is full
    assert features["ending_only"] is ending
    assert features["prefix_ending_interaction"].shape == (2, 8)
    np.testing.assert_allclose(
        features["prefix_ending_interaction"][0],
        [1.0, 2.0, 2.0, 0.0, 1.0, 2.0, 2.0, 0.0],
    )


def test_grouped_predictions_learn_held_out_embedding_signal():
    rng = np.random.default_rng(0)
    n_pairs = 20
    positive = rng.normal(size=(n_pairs, 4))
    negative = rng.normal(size=(n_pairs, 4))
    positive[:, 0] += 5.0
    negative[:, 0] -= 5.0
    features = np.r_[positive, negative]
    predictions = grouped_embedding_predictions(features, n_pairs, 5, 0)
    labels = np.r_[np.ones(n_pairs), np.zeros(n_pairs)]
    assert roc_auc_score(labels, predictions) == 1.0
