"""Multi-turn read-point index maths, checked against a known-answer template.

These indices decide which token every slice-3 number is computed from. A wrong
span would not crash — it would quietly probe the wrong position and produce
plausible AUROCs, so it is worth pinning precisely.

The stub tokenizer mimics a Llama-3 style chat template with one token per
piece, so expected indices can be worked out by hand. A separate test runs the
real Llama tokenizer if its files are already cached locally, and is skipped
otherwise so the suite stays offline.
"""

from __future__ import annotations

import pytest
import torch

from vnaa.extract import READ_POINTS, read_slice, template_ids, turn_spans


class StubTokenizer:
    """One token per piece: BOS, then <role>, each content word, <eot>."""

    def apply_chat_template(self, conv, add_generation_prompt=False,
                            return_dict=False, return_tensors=None):
        ids = [0]  # BOS
        for msg in conv:
            ids.append(1)  # role header
            ids.extend([100 + len(w) for w in msg["content"].split()])
            ids.append(2)  # eot
        t = torch.tensor([ids])
        return {"input_ids": t} if return_dict else t


class NonPrefixTokenizer(StubTokenizer):
    """Appends a trailing token, so growing prefixes are not prefixes."""

    def apply_chat_template(self, conv, add_generation_prompt=False,
                            return_dict=False, return_tensors=None):
        out = super().apply_chat_template(conv, add_generation_prompt, True, None)
        ids = out["input_ids"][0].tolist() + [9]
        t = torch.tensor([ids])
        return {"input_ids": t} if return_dict else t


CONV = [
    {"role": "system", "content": "be helpful"},        # 2 content tokens
    {"role": "user", "content": "hi there friend"},     # 3
    {"role": "assistant", "content": "hello"},          # 1
    {"role": "user", "content": "one two"},             # 2
]


def test_turn_spans_partition_the_sequence():
    spans = turn_spans(StubTokenizer(), CONV)
    # BOS then each message contributes 1 + n_words + 1 tokens.
    assert spans == [(0, 5), (5, 10), (10, 13), (13, 17)]
    assert spans[0][0] == 0
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end == start, "spans must tile without gaps or overlap"


def test_last_user_token_is_final_token_when_conv_ends_on_user():
    spans = turn_spans(StubTokenizer(), CONV)
    start, end = read_slice("last_user_token", CONV, spans)
    assert (start, end) == (16, 17)
    assert end == spans[-1][1], "must be the very last token of the sequence"


def test_last_user_token_skips_a_trailing_assistant_turn():
    conv = CONV + [{"role": "assistant", "content": "bye now"}]
    spans = turn_spans(StubTokenizer(), conv)
    start, end = read_slice("last_user_token", conv, spans)
    # Still the final user turn, not the assistant turn that follows it.
    assert (start, end) == (16, 17)
    assert end < spans[-1][1]


def test_post_instruction_token_is_end_of_system_message():
    spans = turn_spans(StubTokenizer(), CONV)
    assert read_slice("post_instruction_token", CONV, spans) == (4, 5)


def test_post_instruction_token_falls_back_to_first_user_turn():
    conv = CONV[1:]  # no system message
    spans = turn_spans(StubTokenizer(), conv)
    start, end = read_slice("post_instruction_token", conv, spans)
    assert (start, end) == (spans[0][1] - 1, spans[0][1])


def test_final_turn_mean_spans_the_whole_last_user_turn():
    spans = turn_spans(StubTokenizer(), CONV)
    assert read_slice("final_turn_mean", CONV, spans) == (13, 17)


def test_non_prefix_template_is_rejected_not_silently_wrong():
    with pytest.raises(ValueError, match="prefix-consistent"):
        turn_spans(NonPrefixTokenizer(), CONV)


def test_every_declared_read_point_is_implemented():
    spans = turn_spans(StubTokenizer(), CONV)
    for rp in READ_POINTS:
        start, end = read_slice(rp, CONV, spans)
        assert 0 <= start < end <= spans[-1][1]


def test_real_llama_template_if_cached():
    """The stub proves the arithmetic; this proves the assumption about the
    actual template we run on."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct", local_files_only=True
        )
    except Exception:
        pytest.skip("Llama tokenizer not cached locally")

    ids = template_ids(tok, CONV)
    spans = turn_spans(tok, CONV)
    assert spans[0][0] == 0 and spans[-1][1] == len(ids)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end == start

    start, end = read_slice("last_user_token", CONV, spans)
    assert end == len(ids), "last user turn must end the sequence here"
    # The read token is the turn delimiter, not a content token.
    assert tok.decode([ids[start]]) == "<|eot_id|>"

    # The final-turn span must cover the last user turn's words and nothing
    # from the assistant turn before it.
    s, e = read_slice("final_turn_mean", CONV, spans)
    text = tok.decode(ids[s:e])
    assert "one two" in text and "hello" not in text
