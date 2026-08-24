import numpy as np
from sklearn.metrics import roc_auc_score

from vnaa.run_nonlinear_probe_audit import (
    grouped_nonlinear_oof_predictions,
    grouped_text_rbf_oof_predictions,
    logistic_configurations,
    rbf_configurations,
    select_logistic_config_grouped_cv,
    select_rbf_config_grouped_cv,
    text_rbf_configurations,
)


def _conversations(n):
    return [
        [
            {"role": "user", "content": "same prefix"},
            {"role": "assistant", "content": "same reply"},
            {"role": "user", "content": f"ending {index}"},
        ]
        for index in range(n)
    ]


def test_rbf_grid_rejects_empty_or_nonpositive_values():
    for c_grid, gamma_grid in (((), (1.0,)), ((1.0,), ()), ((0.0,), (1.0,))):
        try:
            rbf_configurations(c_grid, gamma_grid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid RBF grid was accepted")


def test_logistic_grid_rejects_empty_or_nonpositive_values():
    for c_grid in ((), (0.0,), (-1.0,)):
        try:
            logistic_configurations(c_grid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid logistic grid was accepted")


def test_grouped_logistic_selection_is_deterministic():
    rng = np.random.default_rng(7)
    pos = rng.normal(loc=0.5, size=(30, 6))
    neg = rng.normal(loc=-0.5, size=(30, 6))
    configs = logistic_configurations((0.001, 0.01, 0.1))

    first, first_scores = select_logistic_config_grouped_cv(
        pos, neg, configs, n_folds=3, seed=2
    )
    second, second_scores = select_logistic_config_grouped_cv(
        pos, neg, configs, n_folds=3, seed=2
    )

    assert first == second
    np.testing.assert_allclose(first_scores, second_scores)


def test_grouped_rbf_selection_is_deterministic():
    rng = np.random.default_rng(0)
    pos = rng.normal(loc=1.0, scale=0.2, size=(30, 6))
    neg = rng.normal(loc=-1.0, scale=0.2, size=(30, 6))
    configs = rbf_configurations((0.1, 1.0), (0.5, 1.0))

    first, first_scores = select_rbf_config_grouped_cv(
        pos, neg, configs, n_folds=3, seed=2
    )
    second, second_scores = select_rbf_config_grouped_cv(
        pos, neg, configs, n_folds=3, seed=2
    )

    assert first == second
    np.testing.assert_allclose(first_scores, second_scores)


def test_rbf_recovers_xor_missed_by_direction_of_means():
    rng = np.random.default_rng(1)
    n_pairs = 80
    repeats = n_pairs // 4
    positive_centers = np.repeat([[1.0, 1.0], [-1.0, -1.0]], repeats * 2, axis=0)
    negative_centers = np.repeat([[1.0, -1.0], [-1.0, 1.0]], repeats * 2, axis=0)
    rng.shuffle(positive_centers)
    rng.shuffle(negative_centers)
    pos = positive_centers + rng.normal(scale=0.12, size=(n_pairs, 2))
    neg = negative_centers + rng.normal(scale=0.12, size=(n_pairs, 2))
    pos = pos[:, None, :]
    neg = neg[:, None, :]
    conversations = _conversations(n_pairs)

    run = grouped_nonlinear_oof_predictions(
        (conversations, conversations, pos, neg),
        n_folds=4,
        seed=0,
        configurations=rbf_configurations((0.1, 1.0, 10.0), (0.5, 1.0)),
        include_text=False,
        include_l2=True,
    )
    labels = run["labels"]
    dom = roc_auc_score(labels, run["predictions"]["direction_of_means"])
    l2 = roc_auc_score(labels, run["predictions"]["l2_logistic"])
    rbf = roc_auc_score(labels, run["predictions"]["rbf"])

    assert 0.35 < dom < 0.65
    assert 0.35 < l2 < 0.65
    assert rbf > 0.95
    assert len(run["selected_logistic_configs"]) == 4
    assert len(run["selected_rbf_configs"]) == 4


def test_outer_predictions_cover_every_row_once_and_preserve_pairs():
    rng = np.random.default_rng(3)
    n_pairs = 24
    pos = rng.normal(loc=0.5, size=(n_pairs, 2, 5))
    neg = rng.normal(loc=-0.5, size=(n_pairs, 2, 5))
    conversations = _conversations(n_pairs)
    run = grouped_nonlinear_oof_predictions(
        (conversations, conversations, pos, neg),
        n_folds=3,
        seed=0,
        configurations=rbf_configurations((1.0,), (1.0,)),
        include_text=False,
        include_l2=True,
    )

    assert np.isfinite(run["predictions"]["direction_of_means"]).all()
    assert np.isfinite(run["predictions"]["l2_logistic"]).all()
    assert np.isfinite(run["predictions"]["rbf"]).all()
    np.testing.assert_array_equal(
        run["cluster_ids"][:n_pairs], run["cluster_ids"][n_pairs:]
    )
    assert len(run["selected_layers"]) == 3
    assert len(run["selected_logistic_configs"]) == 3


def test_text_rbf_is_nested_deterministic_and_pair_complete():
    n_pairs = 24
    pos = _conversations(n_pairs)
    neg = _conversations(n_pairs)
    for index in range(n_pairs):
        pos[index][-1]["content"] = f"formal evaluation rubric score {index}"
        neg[index][-1]["content"] = f"casual chat message friend {index}"
    configs = text_rbf_configurations((0.1, 1.0), (0.5, 1.0))

    first = grouped_text_rbf_oof_predictions(pos, neg, 3, 0, configs)
    second = grouped_text_rbf_oof_predictions(pos, neg, 3, 0, configs)

    np.testing.assert_allclose(first["predictions"], second["predictions"])
    assert np.isfinite(first["predictions"]).all()
    assert roc_auc_score(
        np.r_[np.ones(n_pairs), np.zeros(n_pairs)], first["predictions"]
    ) > 0.95
    assert len(first["selected_configs"]) == 3
