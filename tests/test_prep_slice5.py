"""Source normalization for the slice-5 deployment corpora.

WildChat and LMSYS-Chat-1M carry the same conversations in different shapes.
Getting the difference wrong would not raise: it would quietly admit
non-English or PII-redacted conversations into one corpus and not the other,
which is exactly the kind of asymmetry a probe can read as a label.
"""

import pytest

from vnaa.prep_slice5 import SOURCES, normalize, usable


def turns(n_user: int, tail: str = "assistant") -> list[dict]:
    """A shape-valid conversation: alternating, question-form final user turn."""
    messages = []
    for _ in range(n_user):
        messages.append({"role": "user", "content": "What is a good way to store bread?"})
        messages.append({"role": "assistant", "content": "Keep it in a paper bag, cut side down."})
    if tail != "assistant":
        messages = messages[:-1]
    return messages


def test_both_named_sources_are_available():
    """The proposal names WildChat and LMSYS; only WildChat had been used."""
    assert set(SOURCES) == {"wildchat", "lmsys"}


def test_lmsys_language_is_read_from_the_record():
    record = {"language": "English", "redacted": False, "conversation": turns(2)}
    assert normalize(record, "lmsys") == record["conversation"]
    assert normalize({**record, "language": "Portuguese"}, "lmsys") is None


def test_lmsys_redacted_conversations_are_dropped():
    """NAME_1-style placeholders are a surface artifact, not neutral text."""
    record = {"language": "English", "redacted": True, "conversation": turns(2)}
    assert normalize(record, "lmsys") is None


def test_wildchat_language_stays_per_message():
    """WildChat tags each message, so a record-level check would pass everything."""
    conv = turns(2)
    conv[0] = {**conv[0], "language": "Portuguese"}
    record = {"conversation": conv}
    assert normalize(record, "wildchat") == conv  # record-level gate does not apply
    assert usable(conv) is None                   # the per-message gate catches it


def test_usable_requires_an_assistant_ending():
    assert usable(turns(2)) is not None
    assert usable(turns(2, tail="user")) is None


def test_usable_rejects_a_non_question_final_prompt():
    """Slice 3's rule: otherwise imperative-vs-question form becomes the label."""
    conv = turns(2)
    conv[-2] = {"role": "user", "content": "Write me a paragraph about storing bread."}
    assert usable(conv) is None


@pytest.mark.parametrize("n_user", [1, 5])
def test_usable_enforces_the_turn_count_window(n_user):
    assert usable(turns(n_user)) is None
