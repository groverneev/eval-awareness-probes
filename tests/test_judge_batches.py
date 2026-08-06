"""Build/merge round trip for the verbalized-awareness judge harness.

The judge itself is a model, so what can be pinned down here is the harness
around it: that every response reaches exactly one batch, that the blinding
holds, and that a judge which answers "none" to everything is reported as having
failed its planted controls rather than as having found nothing.
"""

import json

import pytest

from vnaa import judge_batches


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """A tiny response cache wired through the module's real paths."""
    responses = tmp_path / "responses.json"
    responses.write_text(json.dumps({
        "dataset_sha256": "0" * 64,
        "model": "test-model",
        "decoding": {"do_sample": False, "max_new_tokens": 256},
        "items": [
            {"id": f"pair_{i:03d}",
             "eval": f"eval reply {i}", "deploy": f"deploy reply {i}",
             "eval_generated_tokens": 10, "deploy_generated_tokens": 10,
             "eval_terminated": i % 5 != 0, "deploy_terminated": True}
            for i in range(20)
        ],
    }), encoding="utf-8")
    judge_dir = tmp_path / "judge"
    monkeypatch.setattr(judge_batches, "JUDGE_DIR", judge_dir)
    return responses, judge_dir


def build(responses, n_batches=3, overlap=0.1, seed=0):
    import argparse

    judge_batches.build(argparse.Namespace(
        responses=responses, n_batches=n_batches, overlap=overlap, seed=seed))


def test_every_response_is_issued_exactly_once(staged):
    responses, judge_dir = staged
    build(responses)
    key = json.loads((judge_dir / "key.json").read_text(encoding="utf-8"))["key"]
    real = [v for v in key.values()
            if v["source"] == "real" and "duplicate_of_index" not in v]
    assert len(real) == 40  # 20 pairs x 2 cells
    assert len({(v["pair_id"], v["cell"]) for v in real}) == 40


def test_batches_partition_the_pool(staged):
    responses, judge_dir = staged
    build(responses)
    key = json.loads((judge_dir / "key.json").read_text(encoding="utf-8"))["key"]
    seen = []
    for path in sorted(judge_dir.glob("batch_*.json")):
        seen += [row["id"] for row in
                 json.loads(path.read_text(encoding="utf-8"))["rows"]]
    assert sorted(seen) == sorted(key)
    assert len(seen) == len(set(seen))


def test_batch_rows_carry_no_provenance(staged):
    """A judge must not be able to see the cell, the pair, or the controls."""
    responses, judge_dir = staged
    build(responses)
    for path in sorted(judge_dir.glob("batch_*.json")):
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            assert set(row) == {"id", "reply"}
            assert row["id"].startswith("r_")


def test_merge_reports_a_lazy_judge_as_uncalibrated(staged):
    """Answering "none" to everything must not read as a clean null result."""
    import argparse

    responses, judge_dir = staged
    build(responses)
    for path in sorted(judge_dir.glob("batch_*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
        (judge_dir / f"labels_{path.stem.split('_')[1]}.json").write_text(
            json.dumps([{"id": r["id"], "label": "none",
                         "confidence": "high", "quote": ""} for r in rows]),
            encoding="utf-8")

    judge_batches.merge(argparse.Namespace())
    out = json.loads((judge_dir / "judge_labels.json").read_text(encoding="utf-8"))

    caught, total = out["calibration"]["planted_positive_recall"]
    assert total == 14
    assert caught == 0  # the failure is visible, not hidden by a quiet dataset
    assert out["calibration"]["planted_negative_false_alarms"] == [0, 10]
    assert out["n_pairs"] == 20


def test_merge_rejects_an_incomplete_labelling(staged):
    import argparse

    responses, judge_dir = staged
    build(responses)
    paths = sorted(judge_dir.glob("batch_*.json"))
    rows = json.loads(paths[0].read_text(encoding="utf-8"))["rows"]
    (judge_dir / "labels_00.json").write_text(
        json.dumps([{"id": r["id"], "label": "none", "confidence": "high",
                     "quote": ""} for r in rows]), encoding="utf-8")

    with pytest.raises(ValueError, match="unlabelled"):
        judge_batches.merge(argparse.Namespace())


def test_merge_rejects_an_unknown_label(staged):
    import argparse

    responses, judge_dir = staged
    build(responses)
    for path in sorted(judge_dir.glob("batch_*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
        (judge_dir / f"labels_{path.stem.split('_')[1]}.json").write_text(
            json.dumps([{"id": r["id"], "label": "maybe",
                         "confidence": "high", "quote": ""} for r in rows]),
            encoding="utf-8")

    with pytest.raises(ValueError, match="label"):
        judge_batches.merge(argparse.Namespace())


def test_controls_include_both_sensitivity_and_specificity_probes():
    expects = [c["expect"] for c in judge_batches.CONTROLS]
    assert expects.count("positive") >= 10
    assert expects.count("none") >= 5
    assert len({c["key"] for c in judge_batches.CONTROLS}) == len(expects)
