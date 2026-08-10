"""Regression tests for surface views on byte-identical-prefix pairs."""

import pytest

from vnaa.run_slice3 import differing_final_messages


def test_differing_final_messages_removes_identical_prefix():
    prefix = [
        {"role": "user", "content": "shared question"},
        {"role": "assistant", "content": "shared answer"},
    ]
    positive = [prefix + [{"role": "user", "content": "evaluation ending"}]]
    negative = [prefix + [{"role": "user", "content": "deployment ending"}]]

    assert differing_final_messages(positive, negative) == (
        ["evaluation ending"],
        ["deployment ending"],
    )


def test_differing_final_messages_rejects_nonidentical_prefix():
    positive = [[
        {"role": "user", "content": "positive prefix"},
        {"role": "user", "content": "evaluation ending"},
    ]]
    negative = [[
        {"role": "user", "content": "negative prefix"},
        {"role": "user", "content": "deployment ending"},
    ]]

    with pytest.raises(ValueError, match="identical prefix"):
        differing_final_messages(positive, negative)
