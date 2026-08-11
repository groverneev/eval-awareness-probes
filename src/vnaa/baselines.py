"""Surface baselines for the probe to beat.

The probe reads the model's internal activations. These baselines read only the
prompt text. If a dumb text-surface classifier matches the probe, then a high
probe AUROC does not establish that the model internally recognizes evaluation —
the surface cue and the "this is an evaluation" fact are confounded.

The reported result of a cell is the GAP between the probe and the best surface
baseline, not the probe's raw AUROC.

Baselines here:
- length: char count and word count as a single score (no fitting).
- bag_of_words: logistic regression on word counts, vocabulary fit on TRAIN ONLY
  (fitting the vectorizer on all data leaks test vocabulary and inflates it).

Canary discipline: prompt text is read in memory and never written out.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_split(labels: np.ndarray, test_frac: float = 0.3, seed: int = 0):
    """Stratified train/test split. Must match run_slice1 so baselines and the
    probe are scored on exactly the same held-out prompts."""
    idx = np.arange(len(labels))
    return train_test_split(
        idx, test_size=test_frac, random_state=seed, stratify=labels
    )


def strength(auroc: float) -> float:
    """Sign-invariant separation strength."""
    return abs(auroc - 0.5)


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-bootstrap 95% CI for AUROC over the test set.

    Resamples (label, score) pairs with replacement; draws that lose one class
    entirely are skipped. This is the slice-2 statistics upgrade: at slice 1's
    n=120 a 0.03 strength gap was uninterpretable without an interval.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    while len(vals) < n_boot:
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        if yb.min() == yb.max():
            continue
        vals.append(roc_auc_score(yb, scores[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def length_auroc(texts: list[str], labels: np.ndarray, test_idx) -> dict[str, float]:
    """Length used directly as the score. No fitting, so no split needed to fit."""
    char_len = np.array([len(t) for t in texts], dtype=float)
    word_len = np.array([len(t.split()) for t in texts], dtype=float)
    y = labels[test_idx]
    return {
        "char_length": roc_auc_score(y, char_len[test_idx]),
        "word_length": roc_auc_score(y, word_len[test_idx]),
    }


def bag_of_words_scores(
    texts: list[str],
    labels: np.ndarray,
    train_idx,
    test_idx,
    min_df: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Test-set scores from logistic regression on word counts.

    Vocabulary fit on TRAIN ONLY (fitting on all data leaks test vocabulary).
    Word order and meaning are discarded; only counts remain.
    """
    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]

    vec = CountVectorizer(lowercase=True, min_df=min_df)
    x_train = vec.fit_transform(train_texts)  # fit on TRAIN only
    x_test = vec.transform(test_texts)

    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(x_train, labels[train_idx])
    return clf.predict_proba(x_test)[:, 1]


def bag_of_words_auroc(
    texts: list[str],
    labels: np.ndarray,
    train_idx,
    test_idx,
    min_df: int = 2,
    seed: int = 0,
) -> float:
    scores = bag_of_words_scores(texts, labels, train_idx, test_idx, min_df, seed)
    return roc_auc_score(labels[test_idx], scores)


def text_logistic_scores(
    texts: list[str],
    labels: np.ndarray,
    train_idx,
    test_idx,
    kind: str,
    min_df: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Scores from stronger sparse text-only classifiers.

    ``word_tfidf`` includes word unigrams and bigrams. ``char_tfidf`` uses
    character 3--5 grams inside word boundaries, capturing spelling, affixes,
    punctuation-adjacent patterns, and short formatting cues that word counts
    miss.  Every vocabulary and IDF statistic is fitted on training text only.

    Empty causal text is a real condition for identical-prefix pre-final
    controls.  In that case there is no text signal, so return a constant score
    rather than allowing a vectorizer's empty-vocabulary error to obscure the
    intended chance result.
    """
    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    if not any(t.strip() for t in train_texts):
        return np.zeros(len(test_texts), dtype=float)

    if kind == "word_tfidf":
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=min_df,
            sublinear_tf=True,
            max_features=50_000,
        )
    elif kind == "char_tfidf":
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            lowercase=True,
            ngram_range=(3, 5),
            min_df=min_df,
            sublinear_tf=True,
            max_features=100_000,
        )
    else:
        raise ValueError(f"unknown text baseline kind: {kind}")

    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    classifier = LogisticRegression(max_iter=3000, random_state=seed)
    classifier.fit(x_train, labels[train_idx])
    return classifier.predict_proba(x_test)[:, 1]


def bootstrap_auroc_difference_ci(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    cluster_ids: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Cluster bootstrap for the paired AUROC difference ``A - B``.

    Matched eval/deploy examples share a scenario and must be resampled as one
    unit.  Passing their pair ids in ``cluster_ids`` preserves that dependence.
    With no ids, each prediction is treated as an independent singleton.
    """
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    if not (len(y_true) == len(scores_a) == len(scores_b)):
        raise ValueError("labels and score arrays must have equal length")
    if len(np.unique(y_true)) < 2:
        raise ValueError("AUROC needs both classes present")

    clusters = (np.arange(len(y_true)) if cluster_ids is None
                else np.asarray(cluster_ids))
    if len(clusters) != len(y_true):
        raise ValueError("cluster_ids must match labels")
    unique = np.unique(clusters)
    members = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    while len(values) < n_boot:
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([members[cluster] for cluster in sampled])
        labels = y_true[index]
        if len(np.unique(labels)) < 2:
            continue
        values.append(
            roc_auc_score(labels, scores_a[index])
            - roc_auc_score(labels, scores_b[index])
        )
    observed = roc_auc_score(y_true, scores_a) - roc_auc_score(y_true, scores_b)
    return {
        "difference": float(observed),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "n_boot": int(n_boot),
        "n_clusters": int(len(unique)),
    }


def paired_prediction_permutation_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    cluster_ids: np.ndarray | None = None,
    n_permutations: int = 5000,
    seed: int = 0,
    alternative: str = "greater",
) -> dict[str, float]:
    """Approximate-randomization test for a paired AUROC difference.

    Under the null that methods A and B are exchangeable, swap their prediction
    vectors independently per scenario cluster and recompute AUROC(A)-AUROC(B).
    This directly tests the paper's central margin, unlike a label-permutation
    test which only asks whether either method separates the labels at all.
    """
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    if not (len(y_true) == len(scores_a) == len(scores_b)):
        raise ValueError("labels and score arrays must have equal length")
    if alternative not in {"greater", "two-sided"}:
        raise ValueError("alternative must be 'greater' or 'two-sided'")

    clusters = (np.arange(len(y_true)) if cluster_ids is None
                else np.asarray(cluster_ids))
    if len(clusters) != len(y_true):
        raise ValueError("cluster_ids must match labels")
    unique, inverse = np.unique(clusters, return_inverse=True)
    observed = roc_auc_score(y_true, scores_a) - roc_auc_score(y_true, scores_b)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=np.float64)
    for draw in range(n_permutations):
        row_swap = rng.integers(0, 2, size=len(unique), dtype=np.int8)[inverse].astype(bool)
        perm_a = np.where(row_swap, scores_b, scores_a)
        perm_b = np.where(row_swap, scores_a, scores_b)
        null[draw] = (
            roc_auc_score(y_true, perm_a) - roc_auc_score(y_true, perm_b)
        )
    if alternative == "greater":
        exceed = int((null >= observed).sum())
    else:
        exceed = int((np.abs(null) >= abs(observed)).sum())
    return {
        "difference": float(observed),
        "p_value": float((exceed + 1) / (n_permutations + 1)),
        "n_permutations": int(n_permutations),
        "n_clusters": int(len(unique)),
        "alternative": alternative,
    }


_OPTION_MARKER = re.compile(r"(?m)(^\s*\(?[A-Da-d][\)\.]\s)|(\([A-D]\))")


def format_features(text: str) -> list[float]:
    """Structural / typographic features only — no content words.

    Raw length is deliberately excluded so this measures what *shape* adds beyond
    the length baseline. A benchmark item looks like a benchmark item: option
    markers, digits, colons, line structure.
    """
    n = max(len(text), 1)
    lines = text.split("\n")
    words = text.split()
    n_alpha = sum(c.isalpha() for c in text)
    return [
        len(lines),                                        # line count
        sum(c == "\n" for c in text) / n,                  # newline density
        sum(c.isdigit() for c in text) / n,                # digit density
        sum(not c.isalnum() and not c.isspace() for c in text) / n,  # punct density
        sum(c.isupper() for c in text) / max(n_alpha, 1),  # uppercase ratio
        np.mean([len(w) for w in words]) if words else 0.0,  # mean word length
        len(_OPTION_MARKER.findall(text)),                 # "A)" / "(B)" markers
        text.count("?") / n,
        text.count(":") / n,
        text.count("(") / n,
        float(text.rstrip().endswith("?")),
        sum(c.isspace() for c in text) / n,                # whitespace ratio
        float(np.mean([len(l) for l in lines])) if lines else 0.0,  # mean line len
    ]


def format_scores(
    texts: list[str],
    labels: np.ndarray,
    train_idx,
    test_idx,
    seed: int = 0,
) -> np.ndarray:
    """Test-set scores from logistic regression on structural features only."""
    x = np.array([format_features(t) for t in texts], dtype=float)
    clf = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
    )
    clf.fit(x[train_idx], labels[train_idx])
    return clf.predict_proba(x[test_idx])[:, 1]


def format_auroc(
    texts: list[str],
    labels: np.ndarray,
    train_idx,
    test_idx,
    seed: int = 0,
) -> float:
    scores = format_scores(texts, labels, train_idx, test_idx, seed)
    return roc_auc_score(labels[test_idx], scores)


def main() -> None:
    import argparse

    from vnaa.data import load_stages_oversight

    ap = argparse.ArgumentParser(description="Surface baselines for slice 1")
    ap.add_argument("--sad-repo", default="data/sad/oversight_structs.zip")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-auroc", type=float, default=None,
                    help="probe AUROC to compare against, e.g. 0.9944 for the 8B")
    args = ap.parse_args()

    data = load_stages_oversight(args.sad_repo, seed=args.seed)
    texts = [c[0]["content"] for c in data.conversations]
    labels = np.array(data.labels)
    train_idx, test_idx = make_split(labels, args.test_frac, args.seed)
    print(f"{len(data)} prompts | train {len(train_idx)} / test {len(test_idx)}")

    rows: list[tuple[str, float]] = []
    for name, val in length_auroc(texts, labels, test_idx).items():
        rows.append((f"length ({name.split('_')[0]})", val))
    rows.append(("format (structure)", format_auroc(
        texts, labels, train_idx, test_idx, seed=args.seed)))
    rows.append(("bag-of-words", bag_of_words_auroc(
        texts, labels, train_idx, test_idx, seed=args.seed)))

    print(f"\n{'baseline':<22} {'AUROC':>7} {'|AUROC-.5|':>11}")
    for name, val in rows:
        print(f"{name:<22} {val:>7.4f} {strength(val):>11.4f}")

    if args.probe_auroc is not None:
        best = max(rows, key=lambda r: strength(r[1]))
        print(f"\n{'probe (activations)':<22} {args.probe_auroc:>7.4f} "
              f"{strength(args.probe_auroc):>11.4f}")
        print(f"\nbest surface baseline: {best[0]} "
              f"(strength {strength(best[1]):.4f})")
        print(f"probe - best baseline (strength): "
              f"{strength(args.probe_auroc) - strength(best[1]):+.4f}")


if __name__ == "__main__":
    main()
