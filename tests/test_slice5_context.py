"""Guard the Slice 5 context-ablation construction."""

import numpy as np
import pytest

from vnaa.run_slice5 import generation_prefix, shared_terminated_mask


CONVERSATION = [
    {"role": "user", "content": "first question"},
    {"role": "assistant", "content": "first answer"},
    {"role": "user", "content": "final question"},
    {"role": "assistant", "content": "genuine final answer"},
]


def test_full_context_generation_uses_every_pre_final_message():
    assert generation_prefix(CONVERSATION, "full") == CONVERSATION[:-1]


def test_context_ablation_generates_from_only_final_user_message():
    assert generation_prefix(CONVERSATION, "final_user_only") == [CONVERSATION[-2]]


def test_shared_terminated_mask_keeps_only_pairs_valid_in_every_condition():
    assert np.array_equal(
        shared_terminated_mask(np.array([True, True, False]), np.array([True, False, True])),
        np.array([True, False, False]),
    )


def test_shared_terminated_mask_rejects_incompatible_conditions():
    with pytest.raises(ValueError, match="different shapes"):
        shared_terminated_mask(np.array([True]), np.array([True, False]))
