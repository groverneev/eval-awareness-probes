import argparse
import json

import numpy as np

from vnaa.cross_model_suite import DEFAULT_MANIFEST, load_contrasts
from vnaa.run_readout_audit import (
    AUDIT_ID,
    expected_mean_provenance,
    extract,
    mean_cache_path,
)


def test_mean_cache_is_readpoint_specific_and_provenance_checked(tmp_path, monkeypatch):
    class StubModel:
        n_layers = 2
        hidden_size = 3

    calls = []

    def fake_extract(_model, conversations, read_point, show_progress):
        calls.append((len(conversations), read_point, show_progress))
        return np.zeros((len(conversations), 3, 3), dtype=np.float32)

    monkeypatch.setattr("vnaa.extract.load_model", lambda *args, **kwargs: StubModel())
    monkeypatch.setattr("vnaa.extract.extract_activations", fake_extract)
    args = argparse.Namespace(
        manifest=DEFAULT_MANIFEST,
        results=tmp_path,
        model="Qwen/Qwen3-8B",
        device=None,
        device_map="auto",
        dtype="bfloat16",
        limit=2,
        quiet=True,
    )
    extract(args)
    assert calls == [(2, "final_turn_mean", False)] * 4

    contrasts = load_contrasts()
    for contrast in contrasts.values():
        for condition in ("positive", "negative"):
            path = mean_cache_path(
                args.model, contrast.name, condition, args.limit, tmp_path
            )
            assert path.exists()
            with np.load(path, allow_pickle=False) as cached:
                provenance = json.loads(str(cached["provenance"].item()))
            expected = expected_mean_provenance(
                args.model,
                DEFAULT_MANIFEST,
                contrast,
                condition,
                args.dtype,
                args.limit,
            )
            assert provenance == expected
            assert provenance["audit_id"] == AUDIT_ID
            assert provenance["read_point"] == "final_turn_mean"

    extract(args)
    assert calls == [(2, "final_turn_mean", False)] * 4
