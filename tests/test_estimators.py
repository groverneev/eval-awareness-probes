"""Secondary estimator and null baselines, checked on planted-answer data.

No model involved. The point of these is that the nulls answer the question they
claim to answer, because the three of them disagree by design on real
activations and it would be easy to wire the wrong one into a p-value.
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from vnaa.estimators import (
    auroc_columns,
    dom_directions,
    fit_layer_logistic,
    logistic_probe_auroc,
    permutation_null,
    random_direction_null,
    select_layer_cv,
    train_shuffled_direction_null,
)
from tests.test_probe import make_data


def test_auroc_columns_matches_sklearn():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 40 + [1] * 40)
    scores = rng.standard_normal((80, 5))
    got = auroc_columns(labels, scores)
    for column in range(scores.shape[1]):
        assert got[column] == pytest.approx(
            roc_auc_score(labels, scores[:, column]))


def test_auroc_columns_handles_ties():
    """Constant scores must give exactly 0.5, not a rank-order artifact.

    The embedding layer produces identical activations whenever the read token is
    the same for every prompt, so this case is real, not hypothetical.
    """
    labels = np.array([0] * 10 + [1] * 10)
    constant = np.zeros((20, 2))
    assert auroc_columns(labels, constant) == pytest.approx([0.5, 0.5])


def test_dom_directions_matches_per_layer_fit():
    from vnaa.probe import fit_layer

    rng = np.random.default_rng(1)
    acts, labels = make_data(rng, 30, 4, 8, [0.0, 3.0, 0.0, 1.0])
    stacked = dom_directions(acts, labels)
    for layer in range(acts.shape[1]):
        expected = fit_layer(acts[:, layer, :], labels).direction
        assert stacked[layer] == pytest.approx(expected)


def test_logistic_estimator_recovers_planted_layer():
    rng = np.random.default_rng(2)
    sep = [0.0, 0.0, 6.0, 0.0]
    tr_a, tr_y = make_data(rng, 80, 4, 8, sep)
    te_a, te_y = make_data(rng, 80, 4, 8, sep)

    auroc, layer = logistic_probe_auroc(tr_a, tr_y, te_a, te_y, seed=0)

    assert layer == 2
    assert auroc > 0.95


def test_select_layer_cv_ignores_test_split():
    """Layer choice must be a function of training data alone."""
    rng = np.random.default_rng(3)
    tr_a, tr_y = make_data(rng, 60, 4, 8, [0.0, 0.0, 5.0, 0.0])
    first, scores = select_layer_cv(tr_a, tr_y, seed=0)
    again, scores_again = select_layer_cv(tr_a, tr_y, seed=0)
    assert first == again == 2
    assert scores == pytest.approx(scores_again)


def test_permutation_null_is_near_chance_on_real_separation():
    """The significance null must sit near 0.5 even when the signal is strong.

    This is the property the whole baseline hierarchy rests on: a null that
    inherits the class structure would make a real result look unremarkable.
    """
    rng = np.random.default_rng(4)
    acts, labels = make_data(rng, 60, 3, 8, [0.0, 6.0, 0.0])
    n_train = 80
    null = permutation_null(acts, labels, n_train, observed_strength=0.49,
                            n_permutations=100, seed=0)
    assert null["mean_strength"] < 0.12
    assert null["p_value"] <= 0.02


def test_permutation_null_p_value_is_uninformative_on_noise():
    """With no planted signal the observed strength must not look significant."""
    rng = np.random.default_rng(5)
    acts, labels = make_data(rng, 60, 3, 8, [0.0, 0.0, 0.0])
    null = permutation_null(acts, labels, 80, observed_strength=0.05,
                            n_permutations=100, seed=0)
    assert null["p_value"] > 0.05


def test_permutation_p_value_has_a_floor():
    """No finite permutation set can report p = 0."""
    rng = np.random.default_rng(6)
    acts, labels = make_data(rng, 40, 2, 8, [0.0, 8.0])
    null = permutation_null(acts, labels, 50, observed_strength=0.5,
                            n_permutations=50, seed=0)
    assert null["p_value"] == pytest.approx(1 / 51)


def test_train_shuffled_null_is_not_a_significance_test():
    """It keeps true test labels, so a dominant class axis inflates it.

    Planting the separation on a single axis reproduces the real-activation
    situation that made this distinction necessary: an uninformed fitted
    direction lands on that axis often enough to separate, while signed AUROC
    still averages to chance because the sign is random.
    """
    rng = np.random.default_rng(7)
    n = 120
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    acts = rng.standard_normal((n, 2, 8)) * 0.15
    acts[n // 2:, :, 0] += 6.0  # one dominant axis carries the label

    # Both classes must land in both splits, so shuffle before slicing.
    order = rng.permutation(n)
    train, test = order[:80], order[80:]
    null = train_shuffled_direction_null(
        acts[train], labels[train], acts[test], labels[test],
        layer=0, n_shuffles=100, seed=0)

    assert null["mean_strength"] > 0.15
    assert null["mean_auroc"] == pytest.approx(0.5, abs=0.15)


def test_random_direction_null_reports_strength():
    rng = np.random.default_rng(8)
    acts, labels = make_data(rng, 50, 2, 16, [0.0, 0.0])
    null = random_direction_null(acts, labels, layer=0, n_draws=100, seed=0)
    assert null["mean_strength"] < 0.12
    assert null["p95_strength"] >= null["mean_strength"]


def test_logistic_probe_rejects_single_class():
    rng = np.random.default_rng(9)
    acts = rng.standard_normal((10, 4))
    with pytest.raises(ValueError):
        fit_layer_logistic(acts, np.ones(10))
