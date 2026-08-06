"""Guard the Slice 5 context-ablation construction."""

from vnaa.run_slice5 import generation_prefix


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
