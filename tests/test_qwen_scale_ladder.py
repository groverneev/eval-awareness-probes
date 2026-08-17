from pathlib import Path

from vnaa.cross_model_suite import (
    frozen_revision,
    load_contrasts,
    load_manifest,
    panel_model_specs,
    validate_suite,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets/qwen_scale/scale_ladder_v1.json"


def test_qwen_scale_ladder_is_frozen_complete_and_monotonic():
    manifest = load_manifest(MANIFEST)
    fingerprints = validate_suite(manifest, load_contrasts())
    models = panel_model_specs(manifest)

    assert manifest["status"] == "frozen_before_new_scale_activation_extraction"
    assert set(fingerprints) == {"eval_shared", "coherent_pairs"}
    assert [model["repository"] for model in models] == [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
    ]
    counts = [model["parameter_count"] for model in models]
    assert counts == sorted(counts)
    assert len({model["revision"] for model in models}) == 6
    assert all(model["family"] == "qwen3_dense" for model in models)
    assert all(
        frozen_revision(manifest, model["repository"]) == model["revision"]
        for model in models
    )


def test_qwen_scale_ladder_prespecifies_reporting_and_uniform_rerun():
    manifest = load_manifest(MANIFEST)
    assert manifest["reference_model"] == "Qwen/Qwen3-8B"
    assert "same single 80GB GPU pod" in manifest["hardware_rule"]
    assert "not reused" in manifest["hardware_rule"]
    assert "both primary" in manifest["analysis"]["trend_reporting_rule"]
    assert "Do not stop early" in manifest["stopping_rule"]
    assert "Qwen3.8-27B_cross_generation_architecture" in manifest[
        "explicitly_excluded"
    ]
