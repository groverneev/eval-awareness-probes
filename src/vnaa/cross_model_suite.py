"""Frozen, model-aware cross-model confirmatory suite.

The historical Slice-3 runners use a fixed Llama artifact tag. This module
keeps cross-model artifacts in an isolated directory and records the exact
model repository, suite hash, dataset content hash, precision, and sample size
inside every cache. It deliberately includes only the two valid paired
controls; failed exploratory constructions cannot enter through a CLI flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "datasets/cross_model/confirmatory_suite_v2.json"
DEFAULT_RESULTS = ROOT / "results/cross_model/v2"


@dataclass(frozen=True)
class PairedContrast:
    name: str
    item_ids: list[str]
    positive: list[list[dict]]
    negative: list[list[dict]]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_sha256(path: Path = DEFAULT_MANIFEST) -> str:
    """Hash a manifest independently of platform checkout line endings."""
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return sha256_bytes(payload)


def _manifest_sha256_variants(path: Path) -> set[str]:
    """Accepted raw hashes for pre-normalization LF/CRLF artifacts."""
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return {
        sha256_bytes(payload),
        sha256_bytes(payload.replace(b"\n", b"\r\n")),
    }


def _provenance_matches(
    found: dict, expected: dict, manifest_path: Path
) -> bool:
    """Compare provenance while accepting only the legacy EOL hash variant."""
    if found == expected:
        return True
    found_without_hash = {k: v for k, v in found.items() if k != "manifest_sha256"}
    expected_without_hash = {
        k: v for k, v in expected.items() if k != "manifest_sha256"
    }
    return (
        found_without_hash == expected_without_hash
        and found.get("manifest_sha256") in _manifest_sha256_variants(manifest_path)
    )


def frozen_revision(manifest: dict, repository: str) -> str | None:
    """Return the prespecified immutable revision for an allowed model."""
    if repository == manifest["reference_model"]:
        return manifest.get("reference_revision")
    for model in manifest["primary_replication_models"]:
        if model["repository"] == repository:
            revision = model.get("revision")
            if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise ValueError(f"{repository}: missing immutable 40-hex revision")
            return revision
    raise ValueError(f"{repository}: model is not in the frozen suite panel")


def model_slug(repository: str) -> str:
    """Readable collision-resistant artifact tag for a full model ID."""
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", repository).strip("-")
    digest = sha256_bytes(repository.encode("utf-8"))[:10]
    return f"{readable}-{digest}"


def model_loader_adapter(repository: str) -> str | None:
    """Frozen text-only adapter identity for multimodal top-level checkpoints."""
    return {
        "google/gemma-3-12b-it": "gemma3_wrapper_language_model",
        "mistralai/Ministral-3-8B-Instruct-2512-BF16": (
            "mistral3_wrapper_language_model_fix_mistral_regex"
        ),
    }.get(repository)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_contrasts(root: Path = ROOT) -> dict[str, PairedContrast]:
    shared_blob = _load_json(root / "datasets/slice3/eval_shared.json")
    shared_ids = [row["item_id"] for row in shared_blob["items"]]
    shared_positive = [row["eval_shared"] for row in shared_blob["items"]]

    original_rows = []
    for path in sorted((root / "datasets/slice3/batches").glob("out_*.json")):
        original_rows.extend(_load_json(path)["items"])
    original_by_id = {row["item_id"]: row for row in original_rows}
    if len(original_by_id) != len(original_rows):
        raise ValueError("duplicate item_id in Slice-3 batch outputs")
    missing = [item_id for item_id in shared_ids if item_id not in original_by_id]
    if missing:
        raise ValueError(f"eval_shared items missing deploy histories: {missing[:3]}")
    shared_negative = [original_by_id[item_id]["deploy"] for item_id in shared_ids]

    coherent_blob = _load_json(
        root / "datasets/slice3/coherent_pairs/validated.json"
    )
    coherent_ids = [row["item_id"] for row in coherent_blob["items"]]
    coherent_positive = [row["eval"] for row in coherent_blob["items"]]
    coherent_negative = [row["deploy"] for row in coherent_blob["items"]]

    return {
        "eval_shared": PairedContrast(
            "eval_shared", shared_ids, shared_positive, shared_negative
        ),
        "coherent_pairs": PairedContrast(
            "coherent_pairs", coherent_ids, coherent_positive, coherent_negative
        ),
    }


def contrast_sha256(contrast: PairedContrast) -> str:
    canonical = json.dumps(
        {
            "item_ids": contrast.item_ids,
            "positive": contrast.positive,
            "negative": contrast.negative,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(canonical)


def validate_conversation(conversation: list[dict], label: str) -> None:
    if not conversation:
        raise ValueError(f"{label}: empty conversation")
    roles = [message.get("role") for message in conversation]
    expected = ["user" if index % 2 == 0 else "assistant"
                for index in range(len(conversation))]
    if len(conversation) < 3 or roles != expected or roles[-1] != "user":
        raise ValueError(
            f"{label}: expected alternating user/assistant turns ending in user, "
            f"got {roles}"
        )
    if any(not isinstance(message.get("content"), str) or not message["content"].strip()
           for message in conversation):
        raise ValueError(f"{label}: empty or non-string message content")


def validate_suite(
    manifest: dict, contrasts: dict[str, PairedContrast]
) -> dict[str, str]:
    expected_names = set(manifest["contrasts"])
    if set(contrasts) != expected_names:
        raise ValueError("loaded contrasts do not match the frozen manifest")
    expected_n = int(manifest["n_pairs_per_contrast"])
    fingerprints = {}
    for name, contrast in contrasts.items():
        n = len(contrast.item_ids)
        if not (n == len(contrast.positive) == len(contrast.negative) == expected_n):
            raise ValueError(f"{name}: expected {expected_n} matched pairs, got {n}")
        if len(set(contrast.item_ids)) != n:
            raise ValueError(f"{name}: duplicate item IDs")
        for index, (positive, negative) in enumerate(
            zip(contrast.positive, contrast.negative)
        ):
            validate_conversation(positive, f"{name}/{index}/positive")
            validate_conversation(negative, f"{name}/{index}/negative")
            if positive[:-1] != negative[:-1]:
                raise ValueError(f"{name}/{index}: prefix is not byte-identical")
            if positive[-1]["role"] != negative[-1]["role"]:
                raise ValueError(f"{name}/{index}: final roles differ")
        fingerprints[name] = contrast_sha256(contrast)
        expected = manifest["contrasts"][name]["expected_content_sha256"]
        if expected == "TO_BE_FILLED":
            raise ValueError(f"{name}: manifest fingerprint has not been frozen")
        if fingerprints[name] != expected:
            raise ValueError(
                f"{name}: content hash mismatch; expected {expected}, "
                f"got {fingerprints[name]}"
            )

    repositories = [manifest["reference_model"]]
    repositories.extend(
        model["repository"] for model in manifest["primary_replication_models"]
    )
    if manifest.get("deferred_scale_model"):
        repositories.append(manifest["deferred_scale_model"]["repository"])
    slugs = [model_slug(repository) for repository in repositories]
    if len(slugs) != len(set(slugs)):
        raise ValueError("model artifact slugs collide")
    if manifest["precision"] != "bfloat16_unquantized":
        raise ValueError("confirmatory precision must remain unquantized bfloat16")
    for repository in repositories[1:]:
        frozen_revision(manifest, repository)
    return fingerprints


def run_root(
    repository: str, limit: int | None = None, results: Path = DEFAULT_RESULTS
) -> Path:
    sample = "full" if limit is None else f"smoke_n{limit}"
    return results / model_slug(repository) / sample


def cache_path(
    repository: str,
    contrast: str,
    condition: str,
    limit: int | None = None,
    results: Path = DEFAULT_RESULTS,
) -> Path:
    return run_root(repository, limit, results) / f"{contrast}_{condition}_trajectory.npz"


def analysis_path(
    repository: str, limit: int | None = None, results: Path = DEFAULT_RESULTS
) -> Path:
    return run_root(repository, limit, results) / "analysis.json"


def _pack_trajectories(trajectories: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate(trajectories, axis=0),
        np.asarray([len(trajectory) for trajectory in trajectories], dtype=np.int64),
    )


def unpack_trajectories(flat: np.ndarray, n_turns: np.ndarray) -> list[np.ndarray]:
    ends = np.cumsum(n_turns)
    starts = np.r_[0, ends[:-1]]
    return [flat[start:end] for start, end in zip(starts, ends)]


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _cache_provenance(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as cached:
        return json.loads(str(cached["provenance"].item()))


def expected_provenance(
    repository: str,
    manifest_path: Path,
    contrast: PairedContrast,
    condition: str,
    dtype: str,
    limit: int | None,
) -> dict:
    provenance = {
        "suite_id": load_manifest(manifest_path)["suite_id"],
        "manifest_sha256": manifest_sha256(manifest_path),
        "model_repository": repository,
        "model_revision": frozen_revision(load_manifest(manifest_path), repository),
        "model_slug": model_slug(repository),
        "contrast": contrast.name,
        "condition": condition,
        "contrast_sha256": contrast_sha256(contrast),
        "dtype": dtype,
        "quantized": False,
        "extraction_method": "independent_causal_prefix_per_user_turn",
        "n_conversations": len(contrast.item_ids) if limit is None else limit,
        "sample": "full" if limit is None else f"smoke_n{limit}",
    }
    adapter = model_loader_adapter(repository)
    if adapter is not None:
        provenance["model_loader_adapter"] = adapter
    return provenance


def extract(args: argparse.Namespace) -> None:
    import torch

    from vnaa.extract import extract_trajectories_prefixwise, load_model

    manifest = load_manifest(args.manifest)
    contrasts = load_contrasts(ROOT)
    validate_suite(manifest, contrasts)
    if args.limit is not None and not 1 <= args.limit <= manifest["n_pairs_per_contrast"]:
        raise ValueError("--limit must be between 1 and the frozen pair count")
    if args.dtype != "bfloat16" and args.limit is None:
        raise ValueError("full confirmatory extraction is frozen to bfloat16")
    tasks = []
    for name, contrast in contrasts.items():
        for condition in ("positive", "negative"):
            path = cache_path(args.model, name, condition, args.limit, args.results)
            provenance = expected_provenance(
                args.model, args.manifest, contrast, condition, args.dtype, args.limit
            )
            if path.exists():
                if not _provenance_matches(
                    _cache_provenance(path), provenance, args.manifest
                ):
                    raise ValueError(f"refusing incompatible existing cache: {path}")
                print(f"verified existing {path}")
            else:
                tasks.append((name, contrast, condition, path, provenance))
    if not tasks:
        print("all extraction caches already exist and passed provenance checks")
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
    for name, contrast, condition, path, provenance in tasks:
        conversations = getattr(contrast, condition)
        if args.limit is not None:
            conversations = conversations[: args.limit]
        trajectories = extract_trajectories_prefixwise(
            lm, conversations, show_progress=not args.quiet
        )
        flat, n_turns = _pack_trajectories(trajectories)
        if not np.isfinite(flat).all():
            raise ValueError(f"non-finite activations in {name}/{condition}")
        _atomic_npz(
            path,
            acts=flat.astype(np.float32),
            n_turns=n_turns,
            provenance=json.dumps(provenance, sort_keys=True),
        )
        print(f"saved {path} acts={flat.shape}")


def _load_cache(
    repository: str,
    contrast: PairedContrast,
    condition: str,
    manifest_path: Path,
    limit: int | None,
    results: Path,
    dtype: str = "bfloat16",
) -> list[np.ndarray]:
    path = cache_path(repository, contrast.name, condition, limit, results)
    expected = expected_provenance(
        repository, manifest_path, contrast, condition, dtype, limit
    )
    with np.load(path, allow_pickle=False) as cached:
        found = json.loads(str(cached["provenance"].item()))
        if not _provenance_matches(found, expected, manifest_path):
            raise ValueError(f"cache provenance mismatch: {path}")
        return unpack_trajectories(
            cached["acts"].astype(np.float64), cached["n_turns"]
        )


def temporal_diagnostic(
    positive: list[np.ndarray], negative: list[np.ndarray]
) -> dict:
    if len(positive) != len(negative):
        raise ValueError("temporal cells have different sample counts")
    pre_final_differences = []
    for index, (pos, neg) in enumerate(zip(positive, negative)):
        if len(pos) != len(neg):
            raise ValueError(f"pair {index} has different trajectory lengths")
        if len(pos) < 2:
            raise ValueError(f"pair {index} has no pre-final user turn")
        pre_final_differences.append(np.max(np.abs(pos[:-1] - neg[:-1])))
    final_pos = np.stack([trajectory[-1] for trajectory in positive])
    final_neg = np.stack([trajectory[-1] for trajectory in negative])
    return {
        "n_pairs": len(positive),
        "max_absolute_pre_final_difference": float(max(pre_final_differences)),
        "mean_final_l2_difference": float(
            np.linalg.norm(final_pos - final_neg, axis=2).mean()
        ),
        "pre_final_numerically_identical": bool(max(pre_final_differences) <= 1e-6),
    }


def analyze(args: argparse.Namespace) -> None:
    from vnaa.run_strong_baseline_audit import (
        grouped_oof_predictions,
        summarize_predictions,
        transfer_predictions,
    )

    manifest = load_manifest(args.manifest)
    contrasts = load_contrasts(ROOT)
    fingerprints = validate_suite(manifest, contrasts)
    loaded = {}
    trajectories = {}
    for name, contrast in contrasts.items():
        n = len(contrast.item_ids) if args.limit is None else args.limit
        pos_conversations = contrast.positive[:n]
        neg_conversations = contrast.negative[:n]
        pos_trajectories = _load_cache(
            args.model, contrast, "positive", args.manifest, args.limit,
            args.results, args.dtype
        )
        neg_trajectories = _load_cache(
            args.model, contrast, "negative", args.manifest, args.limit,
            args.results, args.dtype
        )
        trajectories[name] = (pos_trajectories, neg_trajectories)
        pos_final = np.stack([trajectory[-1] for trajectory in pos_trajectories])
        neg_final = np.stack([trajectory[-1] for trajectory in neg_trajectories])
        loaded[name] = (
            pos_conversations, neg_conversations, pos_final, neg_final
        )

    output = {
        "suite_id": manifest["suite_id"],
        "manifest_sha256": manifest_sha256(args.manifest),
        "model_repository": args.model,
        "model_slug": model_slug(args.model),
        "sample": "full" if args.limit is None else f"smoke_n{args.limit}",
        "dtype": args.dtype,
        "dataset_fingerprints": fingerprints,
        "inference": {
            "folds": args.folds,
            "seed": args.seed,
            "layer_selection": {
                "within_contrast": "nested_pair_grouped_cv_auroc",
                "inner_folds": args.folds,
                "inner_seed_rule": "outer_seed_plus_fold_index",
                "transfer": "grouped_source_cv_auroc",
            },
            "n_boot": args.n_boot,
            "n_permutations": args.n_permutations,
        },
        "temporal": {},
        "within": {},
        "transfer": {},
    }
    for name, contrast in contrasts.items():
        pos_trajectories, neg_trajectories = trajectories[name]
        output["temporal"][name] = temporal_diagnostic(
            pos_trajectories, neg_trajectories
        )
        run = grouped_oof_predictions(loaded[name], args.folds, args.seed)
        output["within"][name] = summarize_predictions(
            run, args.n_boot, args.n_permutations, args.seed
        )

    for source, target in (
        ("eval_shared", "coherent_pairs"),
        ("coherent_pairs", "eval_shared"),
    ):
        run = transfer_predictions(loaded[source], loaded[target], args.folds, args.seed)
        output["transfer"][f"{source}_to_{target}"] = summarize_predictions(
            run, args.n_boot, args.n_permutations, args.seed
        )
    destination = analysis_path(args.model, args.limit, args.results)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(f"saved {destination}")


def validate(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    fingerprints = validate_suite(manifest, load_contrasts(ROOT))
    print(json.dumps({"status": "pass", "fingerprints": fingerprints}, indent=2))


def plan(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    contrasts = load_contrasts(ROOT)
    validate_suite(manifest, contrasts)
    forward_passes = sum(
        sum(message["role"] == "user" for message in conversation)
        for contrast in contrasts.values()
        for conversation in contrast.positive + contrast.negative
    )
    rows = [
        {
            "repository": model["repository"],
            "revision": model["revision"],
            "role": model["role"],
            "parameter_billions": model["parameter_billions"],
            "bf16_weight_gb": round(2 * model["parameter_billions"], 1),
            "full_forward_passes": forward_passes,
            "artifact_root": str(run_root(model["repository"], results=args.results)),
        }
        for model in manifest["primary_replication_models"]
    ]
    print(json.dumps({"suite_id": manifest["suite_id"], "models": rows}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate").set_defaults(func=validate)
    sub.add_parser("plan").set_defaults(func=plan)

    extraction = sub.add_parser("extract")
    extraction.add_argument("--model", required=True)
    extraction.add_argument("--device", default=None)
    extraction.add_argument("--device-map", default=None)
    extraction.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    extraction.add_argument("--limit", type=int, default=None)
    extraction.add_argument("--quiet", action="store_true")
    extraction.set_defaults(func=extract)

    analysis = sub.add_parser("analyze")
    analysis.add_argument("--model", required=True)
    analysis.add_argument("--limit", type=int, default=None)
    analysis.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    analysis.add_argument("--folds", type=int, default=5)
    analysis.add_argument("--seed", type=int, default=0)
    analysis.add_argument("--n-boot", type=int, default=2000)
    analysis.add_argument("--n-permutations", type=int, default=2000)
    analysis.set_defaults(func=analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
