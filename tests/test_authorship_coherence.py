import json

import pytest

from vnaa.authorship_coherence import read_label_array, validate_labels


def labels():
    return [{"id": "x", "ending_a_pass": True, "ending_b_pass": False,
             "reason": "brief"}]


def test_read_claude_wrapped_labels(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"result": json.dumps(labels())}), encoding="utf-8")
    assert read_label_array(path) == labels()


def test_read_claude_fenced_labels_and_bom(tmp_path):
    path = tmp_path / "labels.json"
    result = "```json\n" + json.dumps(labels()) + "\n```"
    path.write_text(json.dumps({"result": result}), encoding="utf-8-sig")
    assert read_label_array(path) == labels()


def test_validate_labels_rejects_non_boolean():
    bad = labels()
    bad[0]["ending_a_pass"] = "true"
    with pytest.raises(ValueError, match="non-boolean"):
        validate_labels([{"id": "x"}], bad, "test")


def test_validate_labels_rejects_id_mismatch():
    with pytest.raises(ValueError, match="id mismatch"):
        validate_labels([{"id": "y"}], labels(), "test")
