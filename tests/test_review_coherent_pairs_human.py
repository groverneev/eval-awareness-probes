import json

from vnaa.review_coherent_pairs_human import ask_choice, prepare, summarize


def _conversation(ending):
    return [
        {"role": "user", "content": "I am working on a display."},
        {"role": "assistant", "content": "What is it about?"},
        {"role": "user", "content": ending},
    ]


def _write_source(path):
    path.write_text(json.dumps({
        "items": [
            {"item_id": "one", "eval": _conversation("What is gravity?"),
             "deploy": _conversation("Can you write a caption?")},
            {"item_id": "two", "eval": _conversation("Which answer is correct?"),
             "deploy": _conversation("What should I do next?")},
        ]
    }), encoding="utf-8")


def test_prepare_is_deterministic_and_reviewer_file_is_blinded(tmp_path):
    source = tmp_path / "source.json"
    _write_source(source)
    first_rows, first_key = prepare(source, tmp_path / "first", 17)
    second_rows, second_key = prepare(source, tmp_path / "second", 17)

    assert json.loads(first_rows.read_text()) == json.loads(second_rows.read_text())
    assert json.loads(first_key.read_text()) == json.loads(second_key.read_text())
    reviewer_blob = json.loads(first_rows.read_text())
    assert all("item_id" not in row for row in reviewer_blob["rows"])
    assert all("eval" not in row and "deploy" not in row for row in reviewer_blob["rows"])


def test_ask_choice_retries_invalid_input():
    answers = iter(["bad", "Y"])
    assert ask_choice("question", {"y": True, "n": False}, lambda _: next(answers)) is True


def test_summarize_unblinds_conditions_and_builds_joint_pass_subset(tmp_path):
    source = tmp_path / "source.json"
    _write_source(source)
    rows_path, key_path = prepare(source, tmp_path / "audit", 17)
    rows = json.loads(rows_path.read_text())["rows"]

    labels = []
    for row in rows:
        labels.append({
            "id": row["id"],
            "ending_a_coherent": True,
            "ending_a_topic_match": True,
            "ending_a_naturalness": 4,
            "ending_a_speech_act": "factual_or_closed_answer",
            "ending_b_coherent": True,
            "ending_b_topic_match": True,
            "ending_b_naturalness": 4,
            "ending_b_speech_act": "practical_advice_or_action",
            "more_benchmark_like": "a",
        })
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    for path, reviewer in ((first, "r1"), (second, "r2")):
        path.write_text(json.dumps({"reviewer": reviewer, "labels": labels}), encoding="utf-8")
    output = tmp_path / "summary.json"
    summarize(first, second, key_path, output)

    result = json.loads(output.read_text())
    assert result["joint_pass_n"] == 2
    assert result["agreement"]["coherent"]["agreement_rate"] == 1.0
    assert result["first_condition_summary"]["eval"]["n"] == 2
