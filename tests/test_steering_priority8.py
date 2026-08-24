"""CPU-only tests for the post-result steering robustness audit."""

import numpy as np
import pytest

from vnaa import run_steering_priority8 as audit


def test_random_controls_are_seeded_unit_and_orthogonal() -> None:
    target = np.array([1.0, -2.0, 0.5, 3.0])
    first = audit.random_orthogonal_directions(target, [11, 12, 13])
    second = audit.random_orthogonal_directions(target, [11, 12, 13])
    unit_target = target / np.linalg.norm(target)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(first @ unit_target, 0.0, atol=1e-12)
    assert not np.array_equal(first[0], first[1])


def test_random_controls_reject_zero_target() -> None:
    with pytest.raises(ValueError, match="zero norm"):
        audit.random_orthogonal_directions(np.zeros(4), [1])


def test_paired_binary_contrast_uses_prompt_differences() -> None:
    high = np.array([1, 1, 0, 1])
    low = np.array([0, 1, 0, 0])
    result = audit.paired_binary_contrast(
        high,
        low,
        high_alpha=2.0,
        low_alpha=-2.0,
        n_bootstrap=100,
        seed=7,
    )

    assert result["high_rate"] == 0.75
    assert result["low_rate"] == 0.25
    assert result["risk_difference"] == 0.5
    assert result["discordant_high_only"] == 2
    assert result["discordant_low_only"] == 0
    assert result["mcnemar_exact_two_sided_p"] == 0.5


def test_paired_binary_contrast_rejects_nonbinary_values() -> None:
    with pytest.raises(ValueError, match="binary"):
        audit.paired_binary_contrast(
            np.array([0, 2]),
            np.array([0, 1]),
            high_alpha=2.0,
            low_alpha=-2.0,
            n_bootstrap=10,
        )


def test_prompt_slope_is_computed_within_prompt() -> None:
    outcomes = {
        -1.0: np.array([0, 0, 1]),
        0.0: np.array([0, 1, 1]),
        1.0: np.array([1, 1, 1]),
    }
    result = audit.prompt_slope_inference(outcomes, n_bootstrap=100, seed=3)

    # Prompt slopes are 0.5, 0.5, and 0.0.
    assert result["mean_within_prompt_slope"] == pytest.approx(1 / 3)
    assert result["n_prompts"] == 3


def test_outcome_table_aligns_rows_by_prompt_index() -> None:
    items = [
        {"alpha": -2.0, "index": 4, "reply": "This is a real user interaction."},
        {"alpha": 2.0, "index": 2, "reply": "This is an AI evaluation."},
        {"alpha": 2.0, "index": 4, "reply": "This is an AI evaluation."},
        {"alpha": -2.0, "index": 2, "reply": "This is a real user interaction."},
    ]
    indices, outcomes = audit.outcome_table("self_report", items, [-2.0, 2.0])

    np.testing.assert_array_equal(indices, [2, 4])
    np.testing.assert_array_equal(outcomes[-2.0], [0, 0])
    np.testing.assert_array_equal(outcomes[2.0], [1, 1])


def test_outcome_table_rejects_missing_prompt_condition() -> None:
    items = [
        {"alpha": -2.0, "index": 0, "reply": "No refusal."},
        {"alpha": 2.0, "index": 0, "reply": "No refusal."},
        {"alpha": 2.0, "index": 1, "reply": "No refusal."},
    ]
    with pytest.raises(ValueError, match="same prompt indices"):
        audit.outcome_table("safety", items, [-2.0, 2.0])
