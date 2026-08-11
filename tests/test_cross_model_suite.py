import argparse
import copy
import json

import numpy as np
import pytest

from vnaa.cross_model_suite import (
    DEFAULT_MANIFEST,
    _pack_trajectories,
    analysis_path,
    cache_path,
    extract,
    frozen_revision,
    load_contrasts,
    load_manifest,
    model_loader_adapter,
    model_slug,
    temporal_diagnostic,
    unpack_trajectories,
    validate_suite,
)


def test_frozen_suite_and_real_datasets_validate():
    manifest = load_manifest(DEFAULT_MANIFEST)
    fingerprints = validate_suite(manifest, load_contrasts())
    assert manifest["status"] == "frozen_before_qwen3_activation_extraction"
    assert manifest["amends_suite_id"] == "cross_model_confirmatory_v1"
    assert set(fingerprints) == {"eval_shared", "coherent_pairs"}
    assert all(len(value) == 64 for value in fingerprints.values())


def test_suite_contains_only_valid_paired_controls():
    manifest = load_manifest(DEFAULT_MANIFEST)
    assert set(manifest["contrasts"]) == {"eval_shared", "coherent_pairs"}
    excluded = " ".join(manifest["explicitly_excluded"])
    assert "authorship_factorial_v2" in excluded
    assert "failed_surface_gate" in excluded


def test_primary_panel_is_fixed_and_cross_family():
    manifest = load_manifest(DEFAULT_MANIFEST)
    models = manifest["primary_replication_models"]
    assert [model["repository"] for model in models] == [
        "Qwen/Qwen3-8B",
        "google/gemma-3-12b-it",
        "mistralai/Ministral-3-8B-Instruct-2512-BF16",
    ]
    assert len({model["family"] for model in models}) == len(models)
    assert "Do not choose" in manifest["stopping_rule"]
    assert all(frozen_revision(manifest, model["repository"])
               for model in models)


def test_unplanned_model_is_refused():
    with pytest.raises(ValueError, match="not in the frozen suite panel"):
        frozen_revision(load_manifest(DEFAULT_MANIFEST), "Qwen/Qwen2.5-7B-Instruct")


def test_content_change_breaks_frozen_fingerprint():
    manifest = copy.deepcopy(load_manifest(DEFAULT_MANIFEST))
    manifest["contrasts"]["eval_shared"]["expected_content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="content hash mismatch"):
        validate_suite(manifest, load_contrasts())


def test_model_artifacts_cannot_collide(tmp_path):
    qwen = "Qwen/Qwen3-8B"
    gemma = "google/gemma-3-12b-it"
    assert model_slug(qwen) != model_slug(gemma)
    assert cache_path(qwen, "eval_shared", "positive", results=tmp_path) != cache_path(
        gemma, "eval_shared", "positive", results=tmp_path
    )
    assert cache_path(qwen, "eval_shared", "positive", results=tmp_path) != cache_path(
        qwen, "eval_shared", "positive", limit=8, results=tmp_path
    )
    assert analysis_path(qwen, results=tmp_path) != analysis_path(
        qwen, limit=8, results=tmp_path
    )


def test_model_slug_uses_full_repository_identity():
    first = model_slug("org-a/shared-name")
    second = model_slug("org-b/shared-name")
    assert first != second
    assert first == model_slug("org-a/shared-name")


def test_multimodal_loader_adapters_are_explicit_and_model_specific():
    assert model_loader_adapter("Qwen/Qwen3-8B") is None
    assert model_loader_adapter("google/gemma-3-12b-it") == (
        "gemma3_wrapper_language_model"
    )
    assert model_loader_adapter(
        "mistralai/Ministral-3-8B-Instruct-2512-BF16"
    ) == "mistral3_wrapper_language_model_fix_mistral_regex"


def test_trajectory_cache_round_trip():
    trajectories = [
        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        np.arange(12, dtype=np.float32).reshape(1, 3, 4),
    ]
    flat, n_turns = _pack_trajectories(trajectories)
    restored = unpack_trajectories(flat, n_turns)
    assert n_turns.tolist() == [2, 1]
    for expected, found in zip(trajectories, restored):
        np.testing.assert_array_equal(found, expected)


def test_temporal_diagnostic_requires_identical_prefix_but_allows_final_change():
    negative = [np.zeros((2, 3, 4)), np.ones((2, 3, 4))]
    positive = [item.copy() for item in negative]
    positive[0][-1] += 2.0
    positive[1][-1] -= 3.0
    result = temporal_diagnostic(positive, negative)
    assert result["pre_final_numerically_identical"] is True
    assert result["max_absolute_pre_final_difference"] == 0.0
    assert result["mean_final_l2_difference"] > 0


def test_temporal_diagnostic_flags_prefinal_leakage():
    negative = [np.zeros((2, 2, 3))]
    positive = [negative[0].copy()]
    positive[0][0, 1, 2] = 1e-3
    result = temporal_diagnostic(positive, negative)
    assert result["pre_final_numerically_identical"] is False


def test_extract_writes_four_isolated_provenance_checked_caches(tmp_path, monkeypatch):
    class StubModel:
        n_layers = 2
        hidden_size = 3

    calls = []

    def fake_extract(_model, conversations, show_progress):
        calls.append((len(conversations), show_progress))
        return [np.zeros((len(conv[::2]), 3, 3), dtype=np.float32)
                for conv in conversations]

    monkeypatch.setattr("vnaa.extract.load_model", lambda *args, **kwargs: StubModel())
    monkeypatch.setattr("vnaa.extract.extract_trajectories_prefixwise", fake_extract)
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
    assert calls == [(2, False)] * 4
    for contrast in ("eval_shared", "coherent_pairs"):
        for condition in ("positive", "negative"):
            path = cache_path(
                args.model, contrast, condition, limit=2, results=tmp_path
            )
            assert path.exists()
            with np.load(path, allow_pickle=False) as cached:
                provenance = json.loads(str(cached["provenance"].item()))
                assert provenance["model_repository"] == args.model
                assert provenance["model_revision"] == frozen_revision(
                    load_manifest(DEFAULT_MANIFEST), args.model
                )
                assert provenance["contrast"] == contrast
                assert provenance["condition"] == condition
                assert provenance["sample"] == "smoke_n2"
                assert provenance["extraction_method"] == (
                    "independent_causal_prefix_per_user_turn"
                )

    # A resume verifies provenance and skips completed work.
    extract(args)
    assert calls == [(2, False)] * 4
