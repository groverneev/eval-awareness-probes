"""Secondary probe estimator and null baselines.

Two gaps between the proposal and the code lived here until now.

1. The proposal says "Logistic-regression probes serve as a secondary
   estimator", but every reported number came from the difference-of-means
   direction.  A result that only holds for one estimator is a result about the
   estimator.  ``fit_layer_logistic`` is the second opinion.

2. The proposal's baseline hierarchy starts with "a random probe near 0.5".
   That existed only as a synthetic-data assertion in the test suite, never as a
   measured number on real activations.

Three nulls, and the difference between them matters
----------------------------------------------------
``permutation_null`` is the significance test.  It permutes the *entire* label
vector, then reruns the whole pipeline — layer selection included — so the labels
carry no information anywhere.  Measured on real activations it sits where the
proposal assumed a random probe would: mean strength ~0.04, 95th percentile
~0.10.

``train_shuffled_direction_null`` permutes the training labels only and scores
against true test labels.  This is *not* a significance test, and reading it as
one is a trap worth naming: because the eval/deploy axis carries a large share of
the total activation variance, any direction fitted from this data partially
recovers that axis by accident.  On the matched-coherence cell it reaches mean
strength 0.26 with an unfitted direction.  What it measures is how easy the
geometry is, not whether the label is real.  Reported because the answer is
surprising and belongs in the limitations, never as the p-value.

``random_direction_null`` projects onto random unit vectors, fitting nothing at
all.  It lands between the two (~0.11) for the same geometric reason.

Layer selection
---------------
The historical repeated-holdout direction-of-means runner picked its layer by
train-split AUROC.  A later cross-model audit showed that even this estimator
can saturate in high dimensions and make the argmax unstable.  Current grouped
comparisons therefore select direction-of-means layers by pair-grouped CV
nested inside each outer training fold.  Logistic regression also separates a
training split perfectly at essentially every layer, so ``select_layer_cv``
picks its layer by cross-validated AUROC inside the training split.  No outer
test split is consulted for layer choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vnaa.probe import fit_layer

# Regularization is deliberately strong.  hidden_dim (4096) far exceeds the
# number of training rows, so an underregularized LR fits the split exactly and
# generalizes on whatever direction the noise happened to favour.
DEFAULT_C = 0.01


def auroc_columns(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """AUROC for each column of ``scores`` against one label vector.

    The rank identity, so a permutation null over dozens of layers and hundreds
    of draws stays cheap.  ``rankdata`` averages ties, which is what makes this
    agree with ``roc_auc_score`` on constant or duplicated scores — the
    embedding layer produces exactly that.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[:, None]
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs both classes present")
    ranks = rankdata(scores, axis=0)
    pos_rank_sum = ranks[labels == 1].sum(axis=0)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def dom_directions(acts: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Difference-of-means direction for every layer at once, unit-normalized.

    acts: (n_samples, n_layers, hidden_dim).  Returns (n_layers, hidden_dim).
    """
    labels = np.asarray(labels)
    pos = acts[labels == 1].mean(axis=0)
    neg = acts[labels == 0].mean(axis=0)
    diff = pos - neg
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    return np.divide(diff, norms, out=np.zeros_like(diff), where=norms > 0)


def _pipeline_strength(x_train, y_train, x_test, y_test) -> float:
    """Full difference-of-means pipeline: select layer on train, score on test."""
    directions = dom_directions(x_train, y_train)
    train_scores = np.einsum("nld,ld->nl", x_train, directions)
    layer = int(np.argmax(np.abs(auroc_columns(y_train, train_scores) - 0.5)))
    test_scores = x_test[:, layer, :] @ directions[layer]
    return float(abs(auroc_columns(y_test, test_scores)[0] - 0.5))


def permutation_null(
    acts: np.ndarray,
    labels: np.ndarray,
    n_train: int,
    observed_strength: float | None = None,
    n_permutations: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Significance null: permute every label and rerun the whole pipeline.

    ``acts`` and ``labels`` are the pooled train-then-test rows in that order,
    with ``n_train`` marking the boundary, so each draw reuses the real split
    geometry and only the labels move.  Layer selection is repeated inside every
    draw; fixing the layer chosen under the true labels would understate a
    pipeline that searches 33 of them.

    Returns the null distribution summary and, when ``observed_strength`` is
    given, the one-sided permutation p-value.  The p-value is floored at
    1/(n_permutations + 1) since no finite permutation set can prove less.
    """
    acts = np.asarray(acts, dtype=np.float64)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    strengths = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        drawn = rng.permutation(labels)
        y_train, y_test = drawn[:n_train], drawn[n_train:]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            strengths[i] = 0.0
            continue
        strengths[i] = _pipeline_strength(
            acts[:n_train], y_train, acts[n_train:], y_test)
    out = {
        "mean_strength": float(strengths.mean()),
        "p95_strength": float(np.percentile(strengths, 95)),
        "max_strength": float(strengths.max()),
        "n_permutations": int(n_permutations),
    }
    if observed_strength is not None:
        exceed = int((strengths >= observed_strength).sum())
        out["p_value"] = float((exceed + 1) / (n_permutations + 1))
    return out


def train_shuffled_direction_null(
    acts_train: np.ndarray,
    labels_train: np.ndarray,
    acts_test: np.ndarray,
    labels_test: np.ndarray,
    layer: int,
    n_shuffles: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """How well an *uninformed* fitted direction separates the true test labels.

    Permutes training labels but scores against real test labels, so a direction
    that accidentally aligns with the genuine class axis still scores.  Read it
    as a property of the activation geometry, never as a p-value: see the module
    docstring.  Signed AUROC averages to ~0.5 because the accidental alignment is
    equally often inverted, which is exactly why the sign-invariant strength is
    the wrong statistic to compare against a nominal 0.5.
    """
    rng = np.random.default_rng(seed)
    labels_train = np.asarray(labels_train)
    aurocs = np.empty(n_shuffles, dtype=np.float64)
    for i in range(n_shuffles):
        probe = fit_layer(acts_train[:, layer, :], rng.permutation(labels_train))
        aurocs[i] = roc_auc_score(labels_test, probe.project(acts_test[:, layer, :]))
    strengths = np.abs(aurocs - 0.5)
    return {
        "mean_auroc": float(aurocs.mean()),
        "mean_strength": float(strengths.mean()),
        "p95_strength": float(np.percentile(strengths, 95)),
        "n_shuffles": int(n_shuffles),
    }


def random_direction_null(
    acts_test: np.ndarray,
    labels_test: np.ndarray,
    layer: int,
    n_draws: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Projection onto random unit directions, fitting nothing.

    The proposal's "random probe" read literally.  Like the shuffled-direction
    null it is inflated wherever one axis dominates the activation variance, and
    for the same reason.
    """
    rng = np.random.default_rng(seed)
    acts = np.asarray(acts_test, dtype=np.float64)[:, layer, :]
    directions = rng.normal(size=(acts.shape[1], n_draws))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    strengths = np.abs(auroc_columns(labels_test, acts @ directions) - 0.5)
    return {
        "mean_strength": float(strengths.mean()),
        "p95_strength": float(np.percentile(strengths, 95)),
        "n_draws": int(n_draws),
    }


@dataclass
class LogisticLayerProbe:
    """A fitted logistic-regression probe for a single layer.

    ``project`` returns the signed distance from the decision boundary, so it is
    interchangeable with :class:`vnaa.probe.LayerProbe` for AUROC scoring.
    """

    pipeline: object

    def project(self, acts: np.ndarray) -> np.ndarray:
        return self.pipeline.decision_function(np.asarray(acts, dtype=np.float64))


def fit_layer_logistic(
    acts: np.ndarray,
    labels: np.ndarray,
    C: float = DEFAULT_C,
    seed: int = 0,
) -> LogisticLayerProbe:
    """L2 logistic regression on standardized activations for one layer."""
    acts = np.asarray(acts, dtype=np.float64)
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        raise ValueError("fit_layer_logistic needs both classes present")
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=C, random_state=seed),
    )
    pipeline.fit(acts, labels)
    return LogisticLayerProbe(pipeline=pipeline)


def select_layer_cv(
    acts: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    C: float = DEFAULT_C,
    seed: int = 0,
) -> tuple[int, np.ndarray]:
    """Pick a layer by cross-validated AUROC inside the training split.

    Returns (best_layer, per_layer_cv_auroc).  Selection maximizes
    |AUROC - 0.5| so a direction pointing the other way still counts, matching
    the strength metric used everywhere else.
    """
    acts = np.asarray(acts, dtype=np.float64)
    labels = np.asarray(labels)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(folds.split(np.zeros(len(labels)), labels))

    scores = np.empty(acts.shape[1], dtype=np.float64)
    for layer in range(acts.shape[1]):
        held = np.empty(len(labels), dtype=np.float64)
        for train_idx, test_idx in splits:
            probe = fit_layer_logistic(acts[train_idx, layer, :], labels[train_idx], C, seed)
            held[test_idx] = probe.project(acts[test_idx, layer, :])
        scores[layer] = roc_auc_score(labels, held)
    return int(np.argmax(np.abs(scores - 0.5))), scores


def logistic_probe_auroc(
    acts_train: np.ndarray,
    labels_train: np.ndarray,
    acts_test: np.ndarray,
    labels_test: np.ndarray,
    n_folds: int = 5,
    C: float = DEFAULT_C,
    seed: int = 0,
) -> tuple[float, int]:
    """Held-out AUROC for the logistic estimator, layer picked on train."""
    best, _ = select_layer_cv(acts_train, labels_train, n_folds, C, seed)
    probe = fit_layer_logistic(acts_train[:, best, :], labels_train, C, seed)
    return float(roc_auc_score(labels_test, probe.project(acts_test[:, best, :]))), best
