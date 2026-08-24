"""CPU-only primary linear-probe comparison with the strongest E5 text view.

This post-result reprioritization audit reuses the frozen cross-model activation
caches and the immutable E5 feature cache.  For every model and paired control,
it recreates the primary nested direction-of-means out-of-fold predictions and
compares them with grouped L2-logistic predictions from the E5 prefix--ending
interaction.  Pair-cluster bootstrap intervals and paired prediction swaps use
the same scenario grouping as the primary analysis.

No target model, activation extractor, or text encoder is loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from vnaa.baselines import (
    bootstrap_auroc_difference_ci,
    paired_prediction_permutation_test,
)
from vnaa.cross_model_suite import (
    DEFAULT_MANIFEST,
    DEFAULT_RESULTS,
    analysis_path,
    load_contrasts,
    load_manifest,
    manifest_sha256,
    panel_model_specs,
    validate_suite,
)
from vnaa.run_qwen_scale_analysis import (
    DEFAULT_MANIFEST as DEFAULT_SCALE_MANIFEST,
    DEFAULT_RESULTS as DEFAULT_SCALE_RESULTS,
    PREDICTION_FILE,
    paired_bootstrap_slope,
    primary_trend_decision,
)
from vnaa.run_contextual_text_baseline import grouped_embedding_predictions
from vnaa.run_readout_audit import _load_acts
from vnaa.run_strong_baseline_audit import grouped_oof_predictions


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_E5_FEATURES = (
    ROOT / "results/nonlinear_followup/e5_contextual_features.npz"
)
DEFAULT_CONTEXTUAL_RESULT = (
    ROOT / "results/contextual_text_baseline/contextual_text_baseline.json"
)
DEFAULT_OUTPUT = ROOT / "results/full_context_linear_comparison/analysis.json"
AUDIT_ID = "postresult_full_context_linear_comparison_v1"
REPRESENTATION = "prefix_ending_interaction"


def _repositories(manifest: dict, requested: list[str] | None) -> list[str]:
    available = [manifest["reference_model"]] + [
        item["repository"] for item in manifest["primary_replication_models"]
    ]
    if not requested or requested == ["all"]:
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"models are not in the frozen panel: {unknown}")
    return requested


def summarize_comparison(
    labels: np.ndarray,
    clusters: np.ndarray,
    probe_predictions: np.ndarray,
    text_predictions: np.ndarray,
    n_boot: int,
    n_permutations: int,
    seed: int,
) -> dict:
    """Return signed probe-minus-text inference on paired predictions."""
    if not (
        labels.shape
        == clusters.shape
        == probe_predictions.shape
        == text_predictions.shape
    ):
        raise ValueError("comparison vectors must have identical shapes")
    probe_auroc = float(roc_auc_score(labels, probe_predictions))
    text_auroc = float(roc_auc_score(labels, text_predictions))
    return {
        "probe_auroc": probe_auroc,
        "text_auroc": text_auroc,
        "probe_minus_text": {
            "bootstrap": bootstrap_auroc_difference_ci(
                labels,
                probe_predictions,
                text_predictions,
                clusters,
                n_boot=n_boot,
                seed=seed,
            ),
            "permutation": paired_prediction_permutation_test(
                labels,
                probe_predictions,
                text_predictions,
                clusters,
                n_permutations=n_permutations,
                seed=seed,
            ),
        },
    }


def _load_text_predictions(
    path: Path,
    contrasts: dict,
    folds: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict]:
    predictions = {}
    with np.load(path, allow_pickle=False) as cached:
        metadata = json.loads(str(cached["metadata_json"].item()))
        if REPRESENTATION not in metadata["representations"]:
            raise ValueError(f"missing {REPRESENTATION} in E5 feature cache")
        for name, contrast in contrasts.items():
            key = f"{name}__{REPRESENTATION}"
            features = cached[key].astype(np.float64)
            n_pairs = len(contrast.item_ids)
            predictions[name] = grouped_embedding_predictions(
                features, n_pairs, folds, seed
            )
    return predictions, metadata


def analyze(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    contrasts = load_contrasts(ROOT)
    fingerprints = validate_suite(manifest, contrasts)
    repositories = _repositories(manifest, args.model)
    text_predictions, e5_metadata = _load_text_predictions(
        args.e5_features, contrasts, args.folds, args.seed
    )

    contextual_result = json.loads(args.contextual_result.read_text("utf-8"))
    if e5_metadata["dataset_sha256"] != contextual_result["dataset_sha256"]:
        raise ValueError("E5 feature cache does not match contextual audit")
    for name, predictions in text_predictions.items():
        n_pairs = len(contrasts[name].item_ids)
        labels = np.r_[np.ones(n_pairs), np.zeros(n_pairs)]
        expected = contextual_result["representations"][name][REPRESENTATION][
            "auroc"
        ]
        found = float(roc_auc_score(labels, predictions))
        if not np.isclose(found, expected, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"E5 prediction mismatch for {name}: {found} != {expected}"
            )

    output = {
        "audit_id": AUDIT_ID,
        "status": "post_result_strongest_full_input_baseline_reprioritization",
        "manifest_sha256": manifest_sha256(args.manifest),
        "dataset_fingerprints": fingerprints,
        "e5_feature_cache": str(args.e5_features.relative_to(ROOT)),
        "e5_metadata": e5_metadata,
        "text_representation": {
            "name": REPRESENTATION,
            "description": (
                "frozen E5 prefix and ending embeddings concatenated with "
                "absolute difference and elementwise product"
            ),
            "classifier": "StandardScaler + L2 LogisticRegression(C=1)",
            "original_protocol_status": "prespecified_secondary",
            "current_role": "strongest_tested_full_input_linear_text_baseline",
        },
        "inference": {
            "outer_folds": args.folds,
            "seed": args.seed,
            "grouping_unit": "matched_scenario_pair",
            "n_boot": args.n_boot,
            "n_permutations": args.n_permutations,
            "difference_orientation": "probe_minus_text",
        },
        "models": {},
    }

    for repository in repositories:
        print(f"\nmodel={repository}", flush=True)
        model_result = {}
        for name, contrast in contrasts.items():
            pos_acts, pos_source = _load_acts(
                repository,
                contrast,
                "positive",
                "last_user_token",
                args.manifest,
                None,
                args.primary_results,
                args.readout_results,
                args.dtype,
            )
            neg_acts, neg_source = _load_acts(
                repository,
                contrast,
                "negative",
                "last_user_token",
                args.manifest,
                None,
                args.primary_results,
                args.readout_results,
                args.dtype,
            )
            pair = (
                contrast.positive,
                contrast.negative,
                pos_acts,
                neg_acts,
            )
            run = grouped_oof_predictions(pair, args.folds, args.seed)
            summary = summarize_comparison(
                run["labels"],
                run["cluster_ids"],
                run["predictions"]["probe"],
                text_predictions[name],
                args.n_boot,
                args.n_permutations,
                args.seed,
            )
            summary["selected_layers"] = run["selected_layers"]
            summary["layer_selection"] = run["layer_selection"]
            summary["activation_sources"] = {
                "positive": pos_source,
                "negative": neg_source,
            }
            model_result[name] = summary
            boot = summary["probe_minus_text"]["bootstrap"]
            print(
                f"  {name}: probe={summary['probe_auroc']:.5f} "
                f"E5={summary['text_auroc']:.5f} "
                f"difference={boot['difference']:+.5f} "
                f"95% CI [{boot['ci_low']:+.5f}, {boot['ci_high']:+.5f}]",
                flush=True,
            )
        output["models"][repository] = model_result

    scale_manifest = load_manifest(args.scale_manifest)
    validate_suite(scale_manifest, contrasts)
    scale_models = panel_model_specs(scale_manifest)
    scale_output = {
        "manifest_sha256": manifest_sha256(args.scale_manifest),
        "models": [],
        "paired_bootstrap_slopes": {},
    }
    slope_runs = {name: [] for name in contrasts}
    for model in scale_models:
        repository = model["repository"]
        prediction_path = (
            analysis_path(repository, None, args.scale_results).parent
            / PREDICTION_FILE
        )
        model_output = {
            "repository": repository,
            "parameter_count": model["parameter_count"],
            "parameter_billions": model["parameter_billions"],
            "comparisons": {},
        }
        with np.load(prediction_path, allow_pickle=False) as cached:
            for name, contrast in contrasts.items():
                endpoint = f"within_{name}"
                labels = cached[f"{endpoint}__labels"].astype(np.int8)
                clusters = cached[f"{endpoint}__cluster_ids"].astype(np.int64)
                probe = cached[f"{endpoint}__probe"].astype(np.float64)
                text = text_predictions[name]
                summary = summarize_comparison(
                    labels,
                    clusters,
                    probe,
                    text,
                    args.n_boot,
                    args.n_permutations,
                    args.seed,
                )
                model_output["comparisons"][name] = summary
                slope_runs[name].append(
                    {
                        "labels": labels,
                        "cluster_ids": clusters,
                        "probe": probe,
                        "baseline": text,
                    }
                )
        scale_output["models"].append(model_output)

    log10_counts = np.log10(
        [model["parameter_count"] for model in scale_models]
    )
    for name in contrasts:
        endpoint = f"within_{name}"
        scale_output["paired_bootstrap_slopes"][endpoint] = paired_bootstrap_slope(
            log10_counts,
            slope_runs[name],
            n_boot=scale_manifest["analysis"]["trend_bootstrap_resamples"],
            seed=scale_manifest["analysis"]["trend_seed"],
        )
    scale_output["primary_decision"] = primary_trend_decision(
        scale_output["paired_bootstrap_slopes"]
    )
    output["qwen_scale"] = scale_output

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"saved {args.output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--primary-results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--readout-results", type=Path, default=ROOT / "results/readout_audit"
    )
    parser.add_argument("--e5-features", type=Path, default=DEFAULT_E5_FEATURES)
    parser.add_argument(
        "--contextual-result", type=Path, default=DEFAULT_CONTEXTUAL_RESULT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", action="append")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--scale-manifest", type=Path, default=DEFAULT_SCALE_MANIFEST)
    parser.add_argument("--scale-results", type=Path, default=DEFAULT_SCALE_RESULTS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=2000)
    return parser


def main() -> None:
    analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
