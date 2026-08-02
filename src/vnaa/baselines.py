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
from sklearn.feature_extraction.text import CountVectorizer
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
