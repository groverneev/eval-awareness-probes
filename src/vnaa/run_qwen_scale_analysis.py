"""Prediction export and prespecified inference for the Qwen3 scale ladder.

The per-checkpoint confirmatory runner intentionally writes only summaries.
Scale-slope inference additionally needs the paired out-of-fold predictions so
that every bootstrap draw resamples the same scenario IDs at every model size.
This module reconstructs and validates those predictions from the frozen
activation caches, then performs the preregistered slope analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from vnaa.cross_model_suite import (
    ROOT,
    _provenance_matches,
    analysis_path,
    cache_path,
    expected_provenance,
    load_contrasts,
    load_manifest,
    manifest_sha256,
    model_slug,
    panel_model_specs,
    validate_suite,
)
from vnaa.run_strong_baseline_audit import (
    BASELINES,
    grouped_oof_predictions,
    transfer_predictions,
)


DEFAULT_MANIFEST = ROOT / "datasets/qwen_scale/scale_ladder_v1.json"
DEFAULT_RESULTS = ROOT / "results/qwen_scale/v1"
PREDICTION_FILE = "scale_predictions.npz"
ENDPOINTS = (
    "within_eval_shared",
    "within_coherent_pairs",
    "transfer_eval_shared_to_coherent_pairs",
    "transfer_coherent_pairs_to_eval_shared",
)
PRIMARY_ENDPOINTS = ENDPOINTS[:2]


def _load_final_cache(
    repository: str,
    contrast,
    condition: str,
    manifest_path: Path,
    results: Path,
    dtype: str = "bfloat16",
) -> np.ndarray:
    """Load only final-user-token rows, validating frozen cache provenance."""
    path = cache_path(repository, contrast.name, condition, None, results)
    expected = expected_provenance(
        repository, manifest_path, contrast, condition, dtype, None
    )
    with np.load(path, allow_pickle=False) as cached:
        found = json.loads(str(cached["provenance"].item()))
        if not _provenance_matches(found, expected, manifest_path):
            raise ValueError(f"cache provenance mismatch: {path}")
        n_turns = cached["n_turns"].astype(np.int64)
        final_indices = np.cumsum(n_turns) - 1
        # Index before converting: the compressed cache contains every user
        # turn, whereas slope inference needs only one row per conversation.
        return cached["acts"][final_indices].astype(np.float64)


def _endpoint_runs(loaded: dict, folds: int, seed: int) -> dict[str, dict]:
    return {
        "within_eval_shared": grouped_oof_predictions(
            loaded["eval_shared"], folds, seed
        ),
        "within_coherent_pairs": grouped_oof_predictions(
            loaded["coherent_pairs"], folds, seed
        ),
        "transfer_eval_shared_to_coherent_pairs": transfer_predictions(
            loaded["eval_shared"], loaded["coherent_pairs"], folds, seed
        ),
        "transfer_coherent_pairs_to_eval_shared": transfer_predictions(
            loaded["coherent_pairs"], loaded["eval_shared"], folds, seed
        ),
    }


def _analysis_section(analysis: dict, endpoint: str) -> dict:
    if endpoint.startswith("within_"):
        return analysis["within"][endpoint.removeprefix("within_")]
    return analysis["transfer"][endpoint.removeprefix("transfer_")]


def export_model_predictions(
    repository: str,
    manifest_path: Path,
    results: Path,
    folds: int,
    seed: int,
) -> Path:
    """Reconstruct, validate, and save paired predictions for one checkpoint."""
    manifest = load_manifest(manifest_path)
    contrasts = load_contrasts(ROOT)
    validate_suite(manifest, contrasts)
    allowed = {model["repository"] for model in panel_model_specs(manifest)}
    if repository not in allowed:
        raise ValueError(f"{repository}: not in frozen Qwen scale panel")

    loaded = {}
    layer_counts = {}
    for name, contrast in contrasts.items():
        pos = _load_final_cache(
            repository, contrast, "positive", manifest_path, results
        )
        neg = _load_final_cache(
            repository, contrast, "negative", manifest_path, results
        )
        if pos.shape != neg.shape or pos.ndim != 3:
            raise ValueError(f"{repository}/{name}: invalid final cache shapes")
        layer_counts[name] = int(pos.shape[1] - 1)
        loaded[name] = (contrast.positive, contrast.negative, pos, neg)

    analysis_file = analysis_path(repository, None, results)
    analysis = json.loads(analysis_file.read_text(encoding="utf-8"))
    runs = _endpoint_runs(loaded, folds, seed)
    arrays: dict[str, np.ndarray] = {}
    metadata = {
        "suite_id": manifest["suite_id"],
        "manifest_sha256": manifest_sha256(manifest_path),
        "model_repository": repository,
        "model_slug": model_slug(repository),
        "folds": folds,
        "seed": seed,
        "layer_counts_excluding_embedding": layer_counts,
        "endpoints": {},
    }
    for endpoint, run in runs.items():
        section = _analysis_section(analysis, endpoint)
        observed = {
            name: float(roc_auc_score(run["labels"], scores))
            for name, scores in run["predictions"].items()
        }
        for name, value in observed.items():
            if not np.isclose(value, section["aurocs"][name], atol=1e-12, rtol=0):
                raise ValueError(
                    f"{repository}/{endpoint}/{name}: reconstructed AUROC "
                    f"{value} != analysis {section['aurocs'][name]}"
                )
        if [int(x) for x in run["selected_layers"]] != section["selected_layers"]:
            raise ValueError(f"{repository}/{endpoint}: selected layers changed")
        strongest = section["strongest_text_baseline"]
        arrays[f"{endpoint}__labels"] = np.asarray(run["labels"], dtype=np.int8)
        arrays[f"{endpoint}__cluster_ids"] = np.asarray(
            run["cluster_ids"], dtype=np.int64
        )
        for name, scores in run["predictions"].items():
            arrays[f"{endpoint}__{name}"] = np.asarray(scores, dtype=np.float64)
        metadata["endpoints"][endpoint] = {
            "strongest_text_baseline": strongest,
            "selected_layers": section["selected_layers"],
            "probe_auroc": observed["probe"],
            "strongest_text_auroc": observed[strongest],
            "probe_minus_strongest_text": observed["probe"] - observed[strongest],
        }

    destination = analysis_file.parent / PREDICTION_FILE
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        **arrays,
    )
    temporary.replace(destination)
    print(f"saved and validated {destination}")
    return destination


def _load_prediction_endpoint(path: Path, endpoint: str) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as cached:
        metadata = json.loads(str(cached["metadata"].item()))
        baseline = metadata["endpoints"][endpoint]["strongest_text_baseline"]
        run = {
            "labels": cached[f"{endpoint}__labels"].astype(np.int8),
            "cluster_ids": cached[f"{endpoint}__cluster_ids"].astype(np.int64),
            "probe": cached[f"{endpoint}__probe"].astype(np.float64),
            "baseline": cached[f"{endpoint}__{baseline}"].astype(np.float64),
            "baseline_name": baseline,
        }
    return metadata, run


def paired_bootstrap_slope(
    log10_parameter_counts: np.ndarray,
    runs: list[dict],
    n_boot: int = 5000,
    seed: int = 0,
) -> dict:
    """Bootstrap the OLS slope while pairing scenario draws across models."""
    x = np.asarray(log10_parameter_counts, dtype=np.float64)
    if len(x) != len(runs) or len(x) < 2:
        raise ValueError("one prediction run is required for every scale point")
    reference_labels = np.asarray(runs[0]["labels"])
    reference_clusters = np.asarray(runs[0]["cluster_ids"])
    unique_clusters = np.unique(reference_clusters)
    members = {
        cluster: np.flatnonzero(reference_clusters == cluster)
        for cluster in unique_clusters
    }
    margins = []
    for run in runs:
        labels = np.asarray(run["labels"])
        clusters = np.asarray(run["cluster_ids"])
        if not np.array_equal(labels, reference_labels):
            raise ValueError("endpoint labels differ across model sizes")
        if not np.array_equal(clusters, reference_clusters):
            raise ValueError("endpoint cluster IDs differ across model sizes")
        margins.append(
            roc_auc_score(labels, run["probe"])
            - roc_auc_score(labels, run["baseline"])
        )
    point_slope = float(np.polyfit(x, np.asarray(margins), 1)[0])

    rng = np.random.default_rng(seed)
    slopes = np.empty(n_boot, dtype=np.float64)
    for draw in range(n_boot):
        sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        index = np.concatenate([members[cluster] for cluster in sampled])
        labels = reference_labels[index]
        draw_margins = [
            roc_auc_score(labels, run["probe"][index])
            - roc_auc_score(labels, run["baseline"][index])
            for run in runs
        ]
        slopes[draw] = np.polyfit(x, np.asarray(draw_margins), 1)[0]
    return {
        "slope_auroc_per_log10_parameter": point_slope,
        "ci_low": float(np.percentile(slopes, 2.5)),
        "ci_high": float(np.percentile(slopes, 97.5)),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "paired_cluster_count": int(len(unique_clusters)),
    }


def _interval_direction(result: dict) -> str:
    if result["ci_low"] > 0:
        return "positive"
    if result["ci_high"] < 0:
        return "negative"
    return "includes_zero"


def primary_trend_decision(trends: dict[str, dict]) -> dict:
    directions = [_interval_direction(trends[name]) for name in PRIMARY_ENDPOINTS]
    consistent = directions[0] == directions[1] and directions[0] in {
        "positive", "negative"
    }
    if consistent:
        statement = f"consistent_{directions[0]}_within_family_trend"
    else:
        statement = "no_consistent_within_family_scale_trend"
    return {
        "endpoint_directions": dict(zip(PRIMARY_ENDPOINTS, directions)),
        "consistent_nonzero_same_direction": consistent,
        "reporting_classification": statement,
    }


def _reproducibility_check(
    uniform_analysis: dict,
    historical_path: Path,
    threshold: float,
) -> dict:
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    comparisons = {}
    trigger = False
    for contrast in ("eval_shared", "coherent_pairs"):
        new = uniform_analysis["within"][contrast]
        old = historical["within"][contrast]
        new_baseline = new["strongest_text_baseline"]
        old_baseline = old["strongest_text_baseline"]
        values = {
            "probe_auroc_difference": new["aurocs"]["probe"] - old["aurocs"]["probe"],
            "strongest_text_auroc_difference": (
                new["aurocs"][new_baseline] - old["aurocs"][old_baseline]
            ),
            "margin_difference": (
                new["aurocs"]["probe"] - new["aurocs"][new_baseline]
                - old["aurocs"]["probe"] + old["aurocs"][old_baseline]
            ),
        }
        cell_trigger = any(abs(float(value)) > threshold for value in values.values())
        trigger = trigger or cell_trigger
        comparisons[contrast] = {
            "uniform_strongest_text": new_baseline,
            "historical_strongest_text": old_baseline,
            **{name: float(value) for name, value in values.items()},
            "audit_trigger": cell_trigger,
        }
    return {
        "historical_analysis": str(historical_path),
        "absolute_difference_threshold": threshold,
        "comparisons": comparisons,
        "audit_trigger": trigger,
    }


def aggregate(
    manifest_path: Path,
    results: Path,
    historical_8b_analysis: Path,
) -> Path:
    manifest = load_manifest(manifest_path)
    models = panel_model_specs(manifest)
    validate_suite(manifest, load_contrasts(ROOT))
    x = np.log10([model["parameter_count"] for model in models])
    n_boot = int(manifest["analysis"]["trend_bootstrap_resamples"])
    seed = int(manifest["analysis"]["trend_seed"])

    per_model = []
    endpoint_runs = {endpoint: [] for endpoint in ENDPOINTS}
    analyses = {}
    for model in models:
        repository = model["repository"]
        root = analysis_path(repository, None, results).parent
        analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
        analyses[repository] = analysis
        prediction_path = root / PREDICTION_FILE
        model_record = {
            "repository": repository,
            "parameter_count": int(model["parameter_count"]),
            "parameter_billions": float(model["parameter_billions"]),
            "endpoints": {},
            "normalized_selected_layer_depth": {},
        }
        for endpoint in ENDPOINTS:
            metadata, run = _load_prediction_endpoint(prediction_path, endpoint)
            if metadata["manifest_sha256"] != manifest_sha256(manifest_path):
                raise ValueError(f"prediction manifest mismatch: {prediction_path}")
            endpoint_runs[endpoint].append(run)
            section = _analysis_section(analysis, endpoint)
            strongest = section["strongest_text_baseline"]
            comparison = section["probe_minus_strongest"]
            model_record["endpoints"][endpoint] = {
                "probe_auroc": float(section["aurocs"]["probe"]),
                "strongest_text_baseline": strongest,
                "strongest_text_auroc": float(section["aurocs"][strongest]),
                "probe_minus_strongest_text": float(
                    section["aurocs"]["probe"] - section["aurocs"][strongest]
                ),
                "margin_ci_low": float(comparison["bootstrap"]["ci_low"]),
                "margin_ci_high": float(comparison["bootstrap"]["ci_high"]),
                "one_sided_randomization_p": float(
                    comparison["permutation"]["p_value"]
                ),
                "selected_layers": section["selected_layers"],
            }
        layer_counts = metadata["layer_counts_excluding_embedding"]
        for contrast in ("eval_shared", "coherent_pairs"):
            layers = analysis["within"][contrast]["selected_layers"]
            model_record["normalized_selected_layer_depth"][contrast] = float(
                np.mean(layers) / layer_counts[contrast]
            )
        per_model.append(model_record)

    trends = {
        endpoint: paired_bootstrap_slope(x, endpoint_runs[endpoint], n_boot, seed)
        for endpoint in ENDPOINTS
    }
    reference = manifest["reference_model"]
    reproducibility = _reproducibility_check(
        analyses[reference], historical_8b_analysis, threshold=0.01
    )
    output = {
        "suite_id": manifest["suite_id"],
        "manifest_sha256": manifest_sha256(manifest_path),
        "scale_predictor": "log10_exact_parameter_count",
        "models": per_model,
        "paired_bootstrap_slopes": trends,
        "primary_trend_decision": primary_trend_decision(trends),
        "uniform_hardware_8b_reproducibility": reproducibility,
        "claim_boundary": (
            "Six prespecified dense Qwen3 checkpoints test an internal-family "
            "pattern; they do not establish a universal scaling law."
        ),
    }
    destination = results / "scale_analysis.json"
    temporary = destination.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(f"saved {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(dest="command", required=True)
    export_parser = sub.add_parser("export")
    export_parser.add_argument("--model", action="append")
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument(
        "--historical-8b-analysis",
        type=Path,
        default=ROOT
        / "results/cross_model/v2/Qwen-Qwen3-8B-a82fd54762/full/analysis.json",
    )
    args = parser.parse_args()

    if args.command == "export":
        manifest = load_manifest(args.manifest)
        repositories = args.model or [
            model["repository"] for model in panel_model_specs(manifest)
        ]
        for repository in repositories:
            export_model_predictions(
                repository, args.manifest, args.results, args.folds, args.seed
            )
    else:
        aggregate(args.manifest, args.results, args.historical_8b_analysis)


if __name__ == "__main__":
    main()
