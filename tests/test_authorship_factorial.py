import hashlib
import json

from vnaa.build_authorship_gate import opaque_id
from vnaa.run_authorship_generation import sha256_token_ids, termination_ids


def test_token_hash_is_canonical_json_sha256():
    expected = hashlib.sha256(json.dumps([1, 20, 3], separators=(",", ":")).encode()).hexdigest()
    assert sha256_token_ids([1, 20, 3]) == expected


def test_opaque_ids_are_deterministic_and_condition_specific():
    assert opaque_id("row_1", "target_on_policy") == opaque_id(
        "row_1", "target_on_policy"
    )
    assert opaque_id("row_1", "target_on_policy") != opaque_id(
        "row_1", "cross_family_off_policy"
    )


def test_termination_ids_union_and_named_eot():
    class Tokenizer:
        eos_token_id = 2
        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            return {"<|eot_id|>": 7, "<|im_end|>": 0}[token]

    class Config:
        eos_token_id = [2, 3]

    class GenerationConfig:
        eos_token_id = 4

    class Model:
        config = Config()
        generation_config = GenerationConfig()

    assert termination_ids(Tokenizer(), Model()) == [2, 3, 4, 7]
