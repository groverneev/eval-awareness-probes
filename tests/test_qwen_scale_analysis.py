import numpy as np
from sklearn.metrics import roc_auc_score

from vnaa.run_qwen_scale_analysis import (
    PRIMARY_ENDPOINTS,
    paired_bootstrap_slope,
    primary_trend_decision,
)


def _run(probe):
    n = len(probe) // 2
    return {
        "labels": np.r_[np.ones(n), np.zeros(n)],
        "cluster_ids": np.r_[np.arange(n), np.arange(n)],
        "probe": np.asarray(probe, dtype=float),
        "baseline": np.r_[np.linspace(0.55, 0.95, n), np.linspace(0.05, 0.45, n)],
    }


def test_paired_bootstrap_slope_is_deterministic_and_matches_point_fit():
    rng = np.random.default_rng(4)
    n = 40
    latent = np.r_[np.ones(n), -np.ones(n)] + rng.normal(0, 1.4, 2 * n)
    runs = [_run(scale * latent) for scale in (0.35, 0.8, 1.4)]
    x = np.log10([750_000_000, 4_000_000_000, 32_000_000_000])
    first = paired_bootstrap_slope(x, runs, n_boot=80, seed=9)
    second = paired_bootstrap_slope(x, runs, n_boot=80, seed=9)
    margins = [
        roc_auc_score(run["labels"], run["probe"])
        - roc_auc_score(run["labels"], run["baseline"])
        for run in runs
    ]

    assert first == second
    assert np.isclose(
        first["slope_auroc_per_log10_parameter"], np.polyfit(x, margins, 1)[0]
    )
    assert first["paired_cluster_count"] == n


def test_primary_decision_requires_both_intervals_same_nonzero_direction():
    positive = {"ci_low": 0.01, "ci_high": 0.03}
    negative = {"ci_low": -0.03, "ci_high": -0.01}
    uncertain = {"ci_low": -0.01, "ci_high": 0.02}

    aligned = primary_trend_decision(
        {name: positive for name in PRIMARY_ENDPOINTS}
    )
    mixed = primary_trend_decision(
        {PRIMARY_ENDPOINTS[0]: positive, PRIMARY_ENDPOINTS[1]: uncertain}
    )
    reversed_result = primary_trend_decision(
        {name: negative for name in PRIMARY_ENDPOINTS}
    )

    assert aligned["reporting_classification"] == "consistent_positive_within_family_trend"
    assert mixed["reporting_classification"] == "no_consistent_within_family_scale_trend"
    assert reversed_result["consistent_nonzero_same_direction"] is True
