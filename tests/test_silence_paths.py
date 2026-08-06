"""The generation cap is part of a silence-filter run's identity.

A reply that hits the cap is indeterminate, not silent, so the cap decides how
much of the sample the filter can speak for.  Two runs at different caps are
therefore different measurements and must not share files: on eval_shared the
protocol cap of 256 leaves only 124/250 pairs determinate, so a wider-cap
sensitivity run is worth having, and it must not be able to overwrite the
protocol run that the paper quotes.
"""
import argparse
import json
from unittest import mock

import pytest

import vnaa.run_silence_filter as sf
from vnaa.run_silence_filter import (
    PROTOCOL_MAX_NEW_TOKENS, TAG, CELLS, cell_paths, run_slug,
)

CELL_NAMES = sorted(CELLS)


def test_protocol_cap_is_the_frozen_256():
    assert PROTOCOL_MAX_NEW_TOKENS == 256


@pytest.mark.parametrize("cell", CELL_NAMES)
def test_protocol_cap_leaves_the_published_paths_untouched(cell):
    """The published numbers must keep reproducing from their existing files."""
    assert run_slug(cell) == cell
    assert run_slug(cell, PROTOCOL_MAX_NEW_TOKENS) == cell
    assert cell_paths(cell) == cell_paths(cell, PROTOCOL_MAX_NEW_TOKENS)


def test_default_cell_keeps_its_original_filenames():
    responses, labels = cell_paths(TAG)
    assert "cap" not in responses.name
    assert "cap" not in labels.name


@pytest.mark.parametrize("cell", CELL_NAMES)
@pytest.mark.parametrize("cap", [128, 512, 1024])
def test_non_protocol_cap_claims_its_own_files(cell, cap):
    protocol = cell_paths(cell)
    widened = cell_paths(cell, cap)
    assert widened != protocol
    for path in widened:
        assert f"cap{cap}" in path.name
    # and it must not collide with the other cell's run at the same cap
    others = {c for c in CELL_NAMES if c != cell}
    for other in others:
        assert cell_paths(other, cap) != widened


@pytest.mark.parametrize("cap", [PROTOCOL_MAX_NEW_TOKENS, 512])
def test_generate_writes_to_the_path_main_resolved(tmp_path, cap):
    """generate must honour the resolved path, not re-derive one from the cell.

    This is the real bug the first cap-512 attempt hit: generate re-derived its
    own path from the cell alone, dropping the cap, so a widened run aimed
    straight at the protocol run's responses file.  Only the decoding-mismatch
    guard stopped it from overwriting a published result.  Mocking generate out
    does not test this -- the call has to actually run.
    """
    target = tmp_path / f"responses_cap{cap}.json"
    args = argparse.Namespace(
        cell="eval_shared", dataset=tmp_path / "dataset.json",
        model="stub-model", device=None, device_map=None, dtype="bfloat16",
        max_new_tokens=cap, responses=target, labels=tmp_path / "labels.json",
    )
    args.dataset.write_text("{}", encoding="utf-8")

    conversation = [{"role": "user", "content": "hello"}]
    with mock.patch.object(sf, "load_cells",
                           return_value={"eval": [conversation],
                                         "deploy": [conversation]}), \
         mock.patch.object(sf, "greedy_reply", return_value=("reply", 3, True)), \
         mock.patch("vnaa.extract.load_model", return_value=object()):
        sf.generate(args)

    assert target.exists(), "generate did not write to the resolved path"
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["decoding"]["max_new_tokens"] == cap
    assert len(saved["items"]) == 1
    # and it must not have created the protocol run's file as a side effect
    protocol_responses, _ = cell_paths("eval_shared")
    assert not (tmp_path / protocol_responses.name).exists()


@pytest.mark.parametrize("cell", CELL_NAMES)
def test_distinct_caps_never_collide(cell):
    seen = {}
    for cap in (PROTOCOL_MAX_NEW_TOKENS, 128, 512, 1024):
        for path in cell_paths(cell, cap):
            assert path not in seen, f"{path} reused by caps {seen.get(path)} and {cap}"
            seen[path] = cap
