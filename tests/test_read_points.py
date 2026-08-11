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

from vnaa.extract import (
    ASSISTANT_READ_POINTS,
    READ_POINTS,
    LoadedModel,
    _text_model_and_config,
    _tokenizer_kwargs,
    extract_trajectories_prefixwise,
    read_slice,
    template_ids,
    turn_spans,
)


def test_multimodal_adapters_select_only_the_causal_text_backbone():
    text_model = object()
    text_config = object()
    wrapper = type(
        "Wrapper", (),
        {"model": type("MultimodalModel", (), {"language_model": text_model})()},
    )()
    for model_type in ("gemma3", "mistral3"):
        config = type(
            "Config", (), {"model_type": model_type, "text_config": text_config}
        )()
        assert _text_model_and_config(wrapper, config) == (text_model, text_config)


def test_standard_causal_model_is_not_unwrapped():
    model = object()
    config = type("Config", (), {"model_type": "qwen3"})()
    assert _text_model_and_config(model, config) == (model, config)


def test_ministral_tokenizer_regex_fix_is_narrow():
    assert _tokenizer_kwargs("mistral3") == {"fix_mistral_regex": True}
    assert _tokenizer_kwargs("gemma3") == {}
    assert _tokenizer_kwargs("qwen3") == {}


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


class QwenStyleChatMLTokenizer:
    """ChatML stub whose final assistant prefix gains an empty think block."""

    pieces = {
        10: "<|im_start|>", 11: "<|im_end|>", 12: "\n",
        20: "user", 21: "assistant", 30: "<think>", 31: "</think>",
        100: "x",
    }

    def get_vocab(self):
        return {piece: token for token, piece in self.pieces.items()}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces.get(int(token), "x") for token in ids)

    def apply_chat_template(self, conv, add_generation_prompt=False,
                            return_dict=False, return_tensors=None):
        ids = []
        for index, message in enumerate(conv):
            role = 20 if message["role"] == "user" else 21
            ids.extend([10, role, 12])
            if message["role"] == "assistant" and index == len(conv) - 1:
                ids.extend([30, 12, 31, 12])
            ids.extend([100, 11, 12])
        tensor = torch.tensor([ids])
        return {"input_ids": tensor} if return_dict else tensor


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


def test_final_assistant_reads_select_the_last_assistant_turn():
    conv = CONV + [{"role": "assistant", "content": "bye now"}]
    spans = turn_spans(StubTokenizer(), conv)
    assert read_slice("final_assistant_token", conv, spans) == (20, 21)
    assert read_slice("final_assistant_mean", conv, spans) == (17, 21)


def test_non_prefix_template_is_rejected_not_silently_wrong():
    with pytest.raises(ValueError, match="prefix-consistent"):
        turn_spans(NonPrefixTokenizer(), CONV)


def test_chatml_delimiters_recover_qwen_style_nonprefix_template():
    conv = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]
    tokenizer = QwenStyleChatMLTokenizer()
    assert template_ids(tokenizer, conv[:2]) != template_ids(tokenizer, conv)[:10]
    spans = turn_spans(tokenizer, conv)
    assert spans == [(0, 6), (6, 12), (12, 18)]
    assert read_slice("last_user_token", conv, spans) == (17, 18)


def test_real_qwen3_template_if_cached():
    """Qwen3's empty thinking block triggers the ChatML fallback."""
    transformers = pytest.importorskip("transformers")
    revision = "b968826d9c46dd6066d109eabc6255188de91218"
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-8B", revision=revision, local_files_only=True
        )
    except Exception:
        pytest.skip("Qwen3 tokenizer not cached locally")
    conv = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]
    ids = template_ids(tokenizer, conv)
    spans = turn_spans(tokenizer, conv)
    assert spans[0][0] == 0 and spans[-1][1] == len(ids)
    assert [tokenizer.decode(ids[start:end]).count("<|im_start|>")
            for start, end in spans] == [1, 1, 1]
    start, end = read_slice("last_user_token", conv, spans)
    assert end == len(ids)


def test_real_ministral3_template_if_cached():
    """The native template must tile into exact causal message prefixes."""
    transformers = pytest.importorskip("transformers")
    revision = "f6fae9795746f63c9be8344932f01275f3c63734"
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "mistralai/Ministral-3-8B-Instruct-2512-BF16",
            revision=revision,
            local_files_only=True,
            fix_mistral_regex=True,
        )
    except Exception:
        pytest.skip("Ministral 3 tokenizer not cached locally")
    conv = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]
    ids = template_ids(tokenizer, conv)
    spans = turn_spans(tokenizer, conv)
    assert spans[0][0] == 0 and spans[-1][1] == len(ids)
    decoded = [tokenizer.decode(ids[start:end]) for start, end in spans]
    assert decoded[0].endswith("[INST]one[/INST]")
    assert decoded[1] == "two</s>"
    assert decoded[2] == "[INST]three[/INST]"


def test_every_declared_read_point_is_implemented():
    spans = turn_spans(StubTokenizer(), CONV)
    for rp in READ_POINTS + ASSISTANT_READ_POINTS:
        start, end = read_slice(rp, CONV, spans)
        assert 0 <= start < end <= spans[-1][1]


def test_prefixwise_trajectory_runs_exact_causal_user_prefixes():
    class RecordingModel:
        def __init__(self):
            self.lengths = []

        def __call__(self, input_ids, output_hidden_states):
            self.lengths.append(input_ids.shape[1])
            base = input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
            return type("Output", (), {"hidden_states": (base, base + 1)})()

    tokenizer = StubTokenizer()
    model = RecordingModel()
    lm = LoadedModel(tokenizer, model, "cpu", n_layers=1, hidden_size=3)
    trajectories = extract_trajectories_prefixwise(
        lm, [CONV], show_progress=False
    )
    first_user_end = len(template_ids(tokenizer, CONV[:2]))
    final_user_end = len(template_ids(tokenizer, CONV))
    assert model.lengths == [first_user_end, final_user_end]
    assert trajectories[0].shape == (2, 2, 3)
    assert trajectories[0][0, 0, 0] == template_ids(tokenizer, CONV[:2])[-1]
    assert trajectories[0][1, 1, 0] == template_ids(tokenizer, CONV)[-1] + 1


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
