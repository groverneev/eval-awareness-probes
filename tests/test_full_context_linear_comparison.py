"""Unit tests for the cached full-context linear comparison."""

import numpy as np

from vnaa.run_full_context_linear_comparison import summarize_comparison


def test_summarize_comparison_reports_signed_probe_minus_text() -> None:
    labels = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    clusters = np.array([0, 1, 2, 0, 1, 2])
    probe = np.array([0.8, 0.7, 0.4, 0.2, 0.3, 0.6])
    text = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])

    result = summarize_comparison(
        labels,
        clusters,
        probe,
        text,
        n_boot=20,
        n_permutations=20,
        seed=0,
    )

    assert result["probe_auroc"] < result["text_auroc"]
    assert result["probe_minus_text"]["bootstrap"]["difference"] < 0


def test_summarize_comparison_rejects_mismatched_vectors() -> None:
    labels = np.array([1, 0], dtype=float)
    with np.testing.assert_raises(ValueError):
        summarize_comparison(
            labels,
            np.array([0, 0]),
            np.array([1.0]),
            np.array([1.0, 0.0]),
            n_boot=2,
            n_permutations=2,
            seed=0,
        )
