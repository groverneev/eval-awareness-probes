"""Post-result audit of final-user-token versus final-user-turn mean readouts.

The primary cross-model suite reads the residual stream at the closing token of
the final user message.  This audit tests whether the Gemma matched-coherence
result is specific to that delimiter readout.  It reuses the frozen paired
contrasts and existing last-token caches, extracts only the missing mean-pooled
final-turn activations, and applies the same nested pair-grouped inference to
both readouts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vnaa.baselines import (
    bootstrap_auroc_difference_ci,
    paired_prediction_permutation_test,
)
from vnaa.cross_model_suite import (
    DEFAULT_MANIFEST,
    DEFAULT_RESULTS,
    _atomic_npz,
    _load_cache,
    _provenance_matches,
    contrast_sha256,
    frozen_revision,
    load_contrasts,
    load_manifest,
    manifest_sha256,
    model_loader_adapter,
    model_slug,
    validate_suite,
)
from vnaa.run_strong_baseline_audit import (
    grouped_oof_predictions,
    summarize_predictions,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_RESULTS = ROOT / "results/readout_audit"
AUDIT_ID = "postresult_final_user_turn_mean_v1"
READ_POINTS = ("last_user_token", "final_turn_mean")
REFERENCE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
REFERENCE_TAG = "Llama-3.1-8B-Instruct"


def audit_root(
    repository: str,
    limit: int | None = None,
    results: Path = DEFAULT_AUDIT_RESULTS,
) -> Path:
    sample = "full" if limit is None else f"smoke_n{limit}"
    return results / model_slug(repository) / sample


def mean_cache_path(
    repository: str,
    contrast: str,
    condition: str,
    limit: int | None = None,
    results: Path = DEFAULT_AUDIT_RESULTS,
) -> Path:
    return audit_root(repository, limit, results) / (
        f"{contrast}_{condition}_final_turn_mean.npz"
    )


def audit_analysis_path(
    repository: str,
    limit: int | None = None,
    results: Path = DEFAULT_AUDIT_RESULTS,
    minimum_layer: int = 0,
) -> Path:
    name = "analysis.json" if minimum_layer == 0 else f"analysis_min_layer_{minimum_layer}.json"
    return audit_root(repository, limit, results) / name


def expected_mean_provenance(
    repository: str,
    manifest_path: Path,
    contrast,
    condition: str,
    dtype: str,
    limit: int | None,
) -> dict:
    provenance = {
        "audit_id": AUDIT_ID,
        "manifest_sha256": manifest_sha256(manifest_path),
        "model_repository": repository,
        "model_revision": frozen_revision(load_manifest(manifest_path), repository),
        "model_slug": model_slug(repository),
        "contrast": contrast.name,
        "condition": condition,
        "contrast_sha256": contrast_sha256(contrast),
        "read_point": "final_turn_mean",
        "dtype": dtype,
        "quantized": False,
        "extraction_method": "chat_template_final_user_message_span_mean",
        "n_conversations": len(contrast.item_ids) if limit is None else limit,
        "sample": "full" if limit is None else f"smoke_n{limit}",
    }
    adapter = model_loader_adapter(repository)
    if adapter is not None:
        provenance["model_loader_adapter"] = adapter
    return provenance


def _cache_provenance(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as cached:
        return json.loads(str(cached["provenance"].item()))


def extract(args: argparse.Namespace) -> None:
    import torch

    from vnaa.extract import extract_activations, load_model

    manifest = load_manifest(args.manifest)
    contrasts = load_contrasts(ROOT)
    validate_suite(manifest, contrasts)
    frozen_revision(manifest, args.model)
    if args.limit is not None and not 1 <= args.limit <= manifest["n_pairs_per_contrast"]:
        raise ValueError("--limit must be between 1 and the frozen pair count")
    if args.dtype != "bfloat16" and args.limit is None:
        raise ValueError("full readout-audit extraction is fixed to bfloat16")

    tasks = []
    for contrast in contrasts.values():
        for condition in ("positive", "negative"):
            path = mean_cache_path(
                args.model, contrast.name, condition, args.limit, args.results
            )
            provenance = expected_mean_provenance(
                args.model,
                args.manifest,
                contrast,
                condition,
                args.dtype,
                args.limit,
            )
            if path.exists():
                if not _provenance_matches(
                    _cache_provenance(path), provenance, args.manifest
                ):
                    raise ValueError(f"refusing incompatible existing cache: {path}")
                print(f"verified existing {path}")
            else:
                tasks.append((contrast, condition, path, provenance))
    if not tasks:
        print("all final-turn-mean caches exist and passed provenance checks")
        return

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    lm = load_model(
        args.model,
        device=args.device,
        dtype=dtype,
        device_map=args.device_map,
        revision=frozen_revision(manifest, args.model),
    )
    print(
        f"model={args.model} layers={lm.n_layers} hidden={lm.hidden_size} "
        f"sample={'full' if args.limit is None else args.limit}"
    )
    for contrast, condition, path, provenance in tasks:
        conversations = getattr(contrast, condition)
        if args.limit is not None:
            conversations = conversations[: args.limit]
        acts = extract_activations(
            lm,
            conversations,
            read_point="final_turn_mean",
            show_progress=not args.quiet,
        )
        if not np.isfinite(acts).all():
            raise ValueError(
                f"non-finite activations in {contrast.name}/{condition}"
            )
        _atomic_npz(
            path,
            acts=acts.astype(np.float32),
            provenance=json.dumps(provenance, sort_keys=True),
        )
        print(f"saved {path} acts={acts.shape}")


def _reference_cache_path(contrast: str, condition: str, read_point: str) -> Path:
    cell = {
        ("eval_shared", "positive"): "eval_shared",
        ("eval_shared", "negative"): "deploy",
        ("coherent_pairs", "positive"): "coherent_pairs_eval",
        ("coherent_pairs", "negative"): "coherent_pairs_deploy",
    }[(contrast, condition)]
    return ROOT / "results" / f"s3_{REFERENCE_TAG}_{cell}_{read_point}.npz"


def _load_acts(
    repository: str,
    contrast,
    condition: str,
    read_point: str,
    manifest_path: Path,
    limit: int | None,
    primary_results: Path,
    audit_results: Path,
    dtype: str,
) -> tuple[np.ndarray, str]:
    if repository == REFERENCE_MODEL:
        path = _reference_cache_path(contrast.name, condition, read_point)
        with np.load(path, allow_pickle=False) as cached:
            acts = cached["acts"].astype(np.float64)
        source = str(path)
    elif read_point == "last_user_token":
        trajectories = _load_cache(
            repository,
            contrast,
            condition,
            manifest_path,
            limit,
            primary_results,
            dtype,
        )
        acts = np.stack([trajectory[-1] for trajectory in trajectories])
        source = "frozen_cross_model_final_trajectory"
    else:
        path = mean_cache_path(
            repository, contrast.name, condition, limit, audit_results
        )
        expected = expected_mean_provenance(
            repository, manifest_path, contrast, condition, dtype, limit
        )
        with np.load(path, allow_pickle=False) as cached:
            found = json.loads(str(cached["provenance"].item()))
            if not _provenance_matches(found, expected, manifest_path):
                raise ValueError(f"cache provenance mismatch: {path}")
            acts = cached["acts"].astype(np.float64)
        source = str(path)
    if limit is not None:
        acts = acts[:limit]
    expected_n = len(contrast.item_ids) if limit is None else limit
    if acts.ndim != 3 or len(acts) != expected_n or not np.isfinite(acts).all():
        raise ValueError(
            f"invalid {repository}/{contrast.name}/{condition}/{read_point} "
            f"activation shape {acts.shape}"
        )
    return acts, source


def _compare_readouts(
    last_run: dict,
    mean_run: dict,
    n_boot: int,
    n_permutations: int,
    seed: int,
) -> dict:
    labels = last_run["labels"]
    clusters = last_run["cluster_ids"]
    if not np.array_equal(labels, mean_run["labels"]):
        raise ValueError("readout runs use different labels")
    last_scores = last_run["predictions"]["probe"]
    mean_scores = mean_run["predictions"]["probe"]
    return {
        "bootstrap": bootstrap_auroc_difference_ci(
            labels,
            mean_scores,
            last_scores,
            clusters,
            n_boot=n_boot,
            seed=seed,
        ),
        "permutation": paired_prediction_permutation_test(
            labels,
            mean_scores,
            last_scores,
            clusters,
            n_permutations=n_permutations,
            seed=seed,
        ),
    }


def analyze(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    contrasts = load_contrasts(ROOT)
    fingerprints = validate_suite(manifest, contrasts)
    frozen_revision(manifest, args.model)
    output = {
        "audit_id": AUDIT_ID,
        "status": "post_result_readout_sensitivity_audit",
        "model_repository": args.model,
        "model_revision": frozen_revision(manifest, args.model),
        "manifest_sha256": manifest_sha256(args.manifest),
        "dataset_fingerprints": fingerprints,
        "sample": "full" if args.limit is None else f"smoke_n{args.limit}",
        "dtype": args.dtype,
        "read_points": list(READ_POINTS),
        "layer_window": {
            "minimum_layer": args.minimum_layer,
            "includes_embedding_layer": args.minimum_layer == 0,
        },
        "inference": {
            "folds": args.folds,
            "seed": args.seed,
            "layer_selection": "nested_pair_grouped_cv_auroc",
            "n_boot": args.n_boot,
            "n_permutations": args.n_permutations,
        },
        "cache_sources": {},
        "within": {},
        "mean_minus_last_token": {},
    }

    for contrast in contrasts.values():
        runs = {}
        output["cache_sources"][contrast.name] = {}
        for read_point in READ_POINTS:
            pos_acts, pos_source = _load_acts(
                args.model,
                contrast,
                "positive",
                read_point,
                args.manifest,
                args.limit,
                args.primary_results,
                args.results,
                args.dtype,
            )
            neg_acts, neg_source = _load_acts(
                args.model,
                contrast,
                "negative",
                read_point,
                args.manifest,
                args.limit,
                args.primary_results,
                args.results,
                args.dtype,
            )
            if args.minimum_layer >= pos_acts.shape[1]:
                raise ValueError(
                    f"--minimum-layer {args.minimum_layer} exceeds the "
                    f"available layer axis {pos_acts.shape[1]}"
                )
            pos_acts = pos_acts[:, args.minimum_layer :]
            neg_acts = neg_acts[:, args.minimum_layer :]
            n = len(pos_acts)
            pair = (
                contrast.positive[:n],
                contrast.negative[:n],
                pos_acts,
                neg_acts,
            )
            run = grouped_oof_predictions(pair, args.folds, args.seed)
            runs[read_point] = run
            summary = summarize_predictions(
                run, args.n_boot, args.n_permutations, args.seed
            )
            summary["selected_layers"] = [
                layer + args.minimum_layer for layer in summary["selected_layers"]
            ]
            output["within"].setdefault(read_point, {})[contrast.name] = summary
            output["cache_sources"][contrast.name][read_point] = {
                "positive": pos_source,
                "negative": neg_source,
            }
        output["mean_minus_last_token"][contrast.name] = _compare_readouts(
            runs["last_user_token"],
            runs["final_turn_mean"],
            args.n_boot,
            args.n_permutations,
            args.seed,
        )

    destination = audit_analysis_path(
        args.model, args.limit, args.results, args.minimum_layer
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(f"saved {destination}")
    for contrast in contrasts:
        last = output["within"]["last_user_token"][contrast]
        mean = output["within"]["final_turn_mean"][contrast]
        change = output["mean_minus_last_token"][contrast]["bootstrap"]
        print(
            f"{contrast}: last={last['aurocs']['probe']:.5f} "
            f"mean={mean['aurocs']['probe']:.5f} "
            f"mean-last={change['difference']:+.5f} "
            f"[{change['ci_low']:+.5f}, {change['ci_high']:+.5f}]"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, default=DEFAULT_AUDIT_RESULTS)
    parser.add_argument("--primary-results", type=Path, default=DEFAULT_RESULTS)
    sub = parser.add_subparsers(dest="command", required=True)

    extraction = sub.add_parser("extract")
    extraction.add_argument("--model", required=True)
    extraction.add_argument("--device", default=None)
    extraction.add_argument("--device-map", default=None)
    extraction.add_argument(
        "--dtype", choices=["bfloat16", "float32"], default="bfloat16"
    )
    extraction.add_argument("--limit", type=int, default=None)
    extraction.add_argument("--quiet", action="store_true")
    extraction.set_defaults(func=extract)

    analysis = sub.add_parser("analyze")
    analysis.add_argument("--model", required=True)
    analysis.add_argument("--limit", type=int, default=None)
    analysis.add_argument(
        "--dtype", choices=["bfloat16", "float32"], default="bfloat16"
    )
    analysis.add_argument("--folds", type=int, default=5)
    analysis.add_argument("--seed", type=int, default=0)
    analysis.add_argument("--n-boot", type=int, default=2000)
    analysis.add_argument("--n-permutations", type=int, default=2000)
    analysis.add_argument(
        "--minimum-layer",
        type=int,
        default=0,
        help="Exclude layers below this absolute hidden-state index; 1 excludes embeddings.",
    )
    analysis.set_defaults(func=analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
