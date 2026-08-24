"""CPU-only same-layer comparison of DoM, L2-logistic, and RBF probes.

This post-result audit reuses the frozen cross-model activation caches.  Within
each pair-grouped outer fold, the primary direction-of-means pipeline selects a
layer using only the outer-training pairs.  Direction of means, L2 logistic
regression, and RBF SVC are then fit at that same layer.  L2 regularization and
RBF hyperparameters are selected in the same pair-grouped inner folds.

The comparison therefore isolates decoder flexibility at a common learned
representation; it performs no model loading, activation extraction, or text
generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from vnaa.cross_model_suite import (
    DEFAULT_MANIFEST,
    DEFAULT_RESULTS,
    load_contrasts,
    load_manifest,
    manifest_sha256,
    model_slug,
    validate_suite,
)
from vnaa.run_nonlinear_probe_audit import (
    DEFAULT_C_GRID,
    DEFAULT_GAMMA_MULTIPLIERS,
    DEFAULT_LOGISTIC_C_GRID,
    _paired_comparison,
    grouped_nonlinear_oof_predictions,
    logistic_configurations,
    rbf_configurations,
)
from vnaa.run_readout_audit import _load_acts


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/decoder_comparison"
AUDIT_ID = "postresult_same_layer_decoder_comparison_v1"
ESTIMATORS = ("direction_of_means", "l2_logistic", "rbf")


def _repositories(manifest: dict, requested: list[str] | None) -> list[str]:
    available = [manifest["reference_model"]] + [
        model["repository"] for model in manifest["primary_replication_models"]
    ]
    if not requested or requested == ["all"]:
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"models are not in the frozen panel: {unknown}")
    return requested


def _aggregate(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
        "values": [float(value) for value in array],
    }


def _seed_summary(run: dict, args: argparse.Namespace, seed: int) -> dict:
    labels = run["labels"]
    predictions = run["predictions"]
    aurocs = {
        name: float(roc_auc_score(labels, predictions[name]))
        for name in ESTIMATORS
    }
    result = {
        "aurocs": aurocs,
        "selected_layers": run["selected_layers"],
        "selected_layer_inner_aurocs": run["selected_layer_inner_aurocs"],
        "selected_logistic_configs": run["selected_logistic_configs"],
        "selected_logistic_inner_aurocs": run[
            "selected_logistic_inner_aurocs"
        ],
        "selected_rbf_configs": run["selected_rbf_configs"],
        "selected_rbf_inner_aurocs": run["selected_rbf_inner_aurocs"],
        "differences": {
            "l2_minus_dom": aurocs["l2_logistic"] - aurocs["direction_of_means"],
            "rbf_minus_l2": aurocs["rbf"] - aurocs["l2_logistic"],
            "rbf_minus_dom": aurocs["rbf"] - aurocs["direction_of_means"],
        },
    }
    if seed == args.inference_seed:
        clusters = run["cluster_ids"]
        result["paired_inference"] = {
            "l2_minus_dom": _paired_comparison(
                labels,
                predictions["l2_logistic"],
                predictions["direction_of_means"],
                clusters,
                args.n_boot,
                args.n_permutations,
                seed,
            ),
            "rbf_minus_l2": _paired_comparison(
                labels,
                predictions["rbf"],
                predictions["l2_logistic"],
                clusters,
                args.n_boot,
                args.n_permutations,
                seed,
            ),
            "rbf_minus_dom": _paired_comparison(
                labels,
                predictions["rbf"],
                predictions["direction_of_means"],
                clusters,
                args.n_boot,
                args.n_permutations,
                seed,
            ),
        }
    return result


def _aggregate_model(seed_runs: dict[str, dict], contrast_names: list[str]) -> dict:
    aggregate = {}
    for contrast in contrast_names:
        records = [seed_runs[str(seed)][contrast] for seed in sorted(
            int(value) for value in seed_runs
        )]
        aggregate[contrast] = {
            "aurocs": {
                estimator: _aggregate(
                    [record["aurocs"][estimator] for record in records]
                )
                for estimator in ESTIMATORS
            },
            "differences": {
                name: _aggregate(
                    [record["differences"][name] for record in records]
                )
                for name in ("l2_minus_dom", "rbf_minus_l2", "rbf_minus_dom")
            },
        }
    return aggregate


def analyze(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    contrasts = load_contrasts(ROOT)
    fingerprints = validate_suite(manifest, contrasts)
    rbf_configs = rbf_configurations(
        tuple(args.rbf_c_grid), tuple(args.gamma_multipliers)
    )
    logistic_configs = logistic_configurations(tuple(args.logistic_c_grid))
    repositories = _repositories(manifest, args.model)

    combined = {
        "audit_id": AUDIT_ID,
        "status": "post_result_estimator_sensitivity_audit",
        "manifest_sha256": manifest_sha256(args.manifest),
        "dataset_fingerprints": fingerprints,
        "read_point": "last_user_token",
        "sample": "full",
        "dtype": args.dtype,
        "seeds": args.seeds,
        "design": {
            "outer_folds": args.folds,
            "pair_grouped_outer_and_inner_folds": True,
            "shared_fold_specific_layer": (
                "primary_direction_of_means_nested_grouped_cv_on_outer_train"
            ),
            "estimators": {
                "direction_of_means": "normalized class-mean difference",
                "l2_logistic": "StandardScaler + L2 LogisticRegression",
                "rbf": "StandardScaler + RBF SVC",
            },
            "logistic_configurations": logistic_configs,
            "rbf_configurations": rbf_configs,
            "scope": "decoder_comparison_at_identical_fold_selected_layers",
            "post_result": True,
        },
        "inference": {
            "seed": args.inference_seed,
            "n_boot": args.n_boot,
            "n_permutations": args.n_permutations,
            "grouping_unit": "matched_scenario_pair",
        },
        "models": {},
    }

    for repository in repositories:
        print(f"\nmodel={repository}", flush=True)
        cached_pairs = {}
        for contrast in contrasts.values():
            pos_acts, _ = _load_acts(
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
            neg_acts, _ = _load_acts(
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
            cached_pairs[contrast.name] = (
                contrast.positive,
                contrast.negative,
                pos_acts,
                neg_acts,
            )
            print(
                f"  cache {contrast.name}: pairs={len(pos_acts)} "
                f"layers={pos_acts.shape[1]} dim={pos_acts.shape[2]}",
                flush=True,
            )

        seed_runs = {}
        for seed in args.seeds:
            seed_result = {}
            for contrast in contrasts.values():
                run = grouped_nonlinear_oof_predictions(
                    cached_pairs[contrast.name],
                    args.folds,
                    seed,
                    rbf_configs,
                    include_text=False,
                    logistic_configs=logistic_configs,
                    include_l2=True,
                )
                summary = _seed_summary(run, args, seed)
                seed_result[contrast.name] = summary
                aurocs = summary["aurocs"]
                print(
                    f"  seed={seed} {contrast.name}: "
                    f"DoM={aurocs['direction_of_means']:.5f} "
                    f"L2={aurocs['l2_logistic']:.5f} "
                    f"RBF={aurocs['rbf']:.5f} "
                    f"RBF-L2={summary['differences']['rbf_minus_l2']:+.5f}",
                    flush=True,
                )
            seed_runs[str(seed)] = seed_result

        aggregate = _aggregate_model(seed_runs, list(contrasts))
        model_output = {
            "audit_id": AUDIT_ID,
            "model_repository": repository,
            "seeds": seed_runs,
            "aggregate": aggregate,
        }
        destination = args.results / model_slug(repository) / "analysis.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp.json")
        temporary.write_text(
            json.dumps(model_output, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
        combined["models"][repository] = aggregate
        print(f"saved {destination}", flush=True)

    combined_path = args.results / "combined_analysis.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = combined_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    temporary.replace(combined_path)
    print(f"saved {combined_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--primary-results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--readout-results",
        type=Path,
        default=ROOT / "results/readout_audit",
    )
    parser.add_argument("--model", action="append")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--inference-seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument(
        "--logistic-c-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_LOGISTIC_C_GRID),
    )
    parser.add_argument(
        "--rbf-c-grid", type=float, nargs="+", default=list(DEFAULT_C_GRID)
    )
    parser.add_argument(
        "--gamma-multipliers",
        type=float,
        nargs="+",
        default=list(DEFAULT_GAMMA_MULTIPLIERS),
    )
    parser.set_defaults(func=analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.inference_seed not in args.seeds:
        raise ValueError("inference seed must be included in --seeds")
    args.func(args)


if __name__ == "__main__":
    main()
