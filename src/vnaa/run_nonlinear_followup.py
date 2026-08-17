"""Complete the post-result nonlinear audit on the frozen paired controls.

This follow-up adds three checks to ``run_nonlinear_probe_audit``:

1. a nested RBF classifier over the frozen E5 contextual representations;
2. bidirectional nonlinear transfer with every choice made on the source only;
3. repetition over several deterministic pair-fold seeds.

The activation RBF remains a capacity-only audit: its layer is selected by the
primary direction-of-means pipeline, and only the RBF hyperparameters are then
selected at that layer.  The nonlinear text comparator honestly selects among
word/character TF--IDF and the three frozen E5 views inside training folds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from vnaa.baselines import (
    bootstrap_auroc_difference_ci,
    paired_prediction_permutation_test,
)
from vnaa.cross_model_suite import (
    DEFAULT_MANIFEST,
    DEFAULT_RESULTS,
    load_contrasts,
    load_manifest,
    manifest_sha256,
    model_slug,
    validate_suite,
)
from vnaa.probe import fit_layer
from vnaa.run_contextual_text_baseline import (
    MODEL_ID as E5_MODEL_ID,
    MODEL_REVISION as E5_MODEL_REVISION,
    REPRESENTATIONS as E5_REPRESENTATIONS,
    _fingerprint as contextual_fingerprint,
    contrast_text_views,
    encode_texts,
    feature_matrices,
)
from vnaa.run_nonlinear_probe_audit import (
    DEFAULT_C_GRID,
    DEFAULT_GAMMA_MULTIPLIERS,
    DEFAULT_TEXT_GAMMA_GRID,
    _fit_rbf,
    _fit_sparse_rbf,
    _text_vectorizer,
    grouped_nonlinear_oof_predictions,
    grouped_rbf_config_scores,
    grouped_text_rbf_config_scores,
    rbf_configurations,
    text_rbf_configurations,
)
from vnaa.run_readout_audit import _load_acts
from vnaa.run_slice3 import differing_final_messages
from vnaa.run_strong_baseline_audit import (
    fit_selected_dom_grouped_cv,
    grouped_dom_layer_scores,
    pair_folds,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT / "results/nonlinear_followup"
DEFAULT_E5_CACHE = DEFAULT_RESULTS_DIR / "e5_contextual_features.npz"
AUDIT_ID = "postresult_nonlinear_context_transfer_multiseed_v1"
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def contextual_rbf_configurations(
    c_grid: tuple[float, ...] = DEFAULT_C_GRID,
    gamma_multipliers: tuple[float, ...] = DEFAULT_GAMMA_MULTIPLIERS,
) -> list[dict[str, str | float]]:
    """Return the fixed representation-by-RBF grid for frozen E5 features."""
    base = rbf_configurations(c_grid, gamma_multipliers)
    return [
        {"representation": representation, **config}
        for representation in E5_REPRESENTATIONS
        for config in base
    ]


def _cache_key(contrast: str, representation: str) -> str:
    return f"{contrast}__{representation}"


def prepare_contextual_features(
    contrasts,
    cache_path: Path,
    batch_size: int,
    device: str,
    refresh: bool = False,
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """Load or create a fingerprinted cache of the three frozen E5 views."""
    expected_fingerprint = contextual_fingerprint(contrasts)
    if cache_path.exists() and not refresh:
        with np.load(cache_path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata_json"].item()))
            if metadata["dataset_sha256"] != expected_fingerprint:
                raise ValueError("cached E5 features have the wrong dataset fingerprint")
            if metadata["model_id"] != E5_MODEL_ID:
                raise ValueError("cached E5 features have the wrong encoder")
            features = {
                name: {
                    representation: saved[_cache_key(name, representation)].astype(
                        np.float64
                    )
                    for representation in E5_REPRESENTATIONS
                }
                for name in contrasts
            }
        return features, metadata

    all_views = {
        name: contrast_text_views(contrast) for name, contrast in contrasts.items()
    }
    flat_texts: list[str] = []
    slices: dict[str, dict[str, tuple[int, int]]] = {}
    for name, views in all_views.items():
        slices[name] = {}
        for view_name in ("prefix", "ending", "full"):
            start = len(flat_texts)
            flat_texts.extend(views[view_name])
            slices[name][view_name] = (start, len(flat_texts))

    embeddings, encoder_metadata = encode_texts(
        flat_texts, batch_size=batch_size, device=device
    )
    features = {}
    for name in contrasts:
        arrays = {
            view: embeddings[start:end]
            for view, (start, end) in slices[name].items()
        }
        features[name] = feature_matrices(
            arrays["prefix"], arrays["ending"], arrays["full"]
        )

    metadata = {
        "dataset_sha256": expected_fingerprint,
        "model_id": E5_MODEL_ID,
        "model_revision": E5_MODEL_REVISION,
        "encoder": encoder_metadata,
        "representations": list(E5_REPRESENTATIONS),
    }
    payload = {
        _cache_key(name, representation): matrix.astype(np.float32)
        for name, by_representation in features.items()
        for representation, matrix in by_representation.items()
    }
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return features, metadata


def _validate_contextual_features(
    features: dict[str, np.ndarray], n_pairs: int
) -> None:
    if set(features) != set(E5_REPRESENTATIONS):
        raise ValueError("contextual feature views do not match the frozen design")
    for representation, matrix in features.items():
        if matrix.ndim != 2 or matrix.shape[0] != 2 * n_pairs:
            raise ValueError(
                f"{representation} must have positive-then-negative paired rows"
            )


def grouped_contextual_rbf_config_scores(
    features: dict[str, np.ndarray],
    configurations: list[dict[str, str | float]],
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """Pair-grouped OOF scores for every frozen-E5 RBF configuration."""
    first = next(iter(features.values()))
    n_pairs = first.shape[0] // 2
    _validate_contextual_features(features, n_pairs)
    labels = np.r_[np.ones(n_pairs), np.zeros(n_pairs)]
    held_scores = np.empty((2 * n_pairs, len(configurations)), dtype=np.float64)

    for train_pairs, test_pairs in pair_folds(n_pairs, n_folds, seed):
        train_idx = np.r_[train_pairs, n_pairs + train_pairs]
        test_idx = np.r_[test_pairs, n_pairs + test_pairs]
        for index, config in enumerate(configurations):
            representation = str(config["representation"])
            model = _fit_rbf(
                features[representation][train_idx], labels[train_idx], config
            )
            held_scores[test_idx, index] = model.decision_function(
                features[representation][test_idx]
            )

    return np.asarray(
        [
            roc_auc_score(labels, held_scores[:, index])
            for index in range(len(configurations))
        ],
        dtype=np.float64,
    )


def grouped_contextual_rbf_oof_predictions(
    features: dict[str, np.ndarray],
    configurations: list[dict[str, str | float]],
    n_folds: int,
    seed: int,
) -> dict:
    """Nested OOF predictions for a fixed set of contextual candidates."""
    first = next(iter(features.values()))
    n_pairs = first.shape[0] // 2
    _validate_contextual_features(features, n_pairs)
    labels = np.r_[np.ones(n_pairs), np.zeros(n_pairs)]
    predictions = np.empty(2 * n_pairs, dtype=np.float64)
    selected = []
    for fold, (train_pairs, test_pairs) in enumerate(
        pair_folds(n_pairs, n_folds, seed)
    ):
        train_idx = np.r_[train_pairs, n_pairs + train_pairs]
        test_idx = np.r_[test_pairs, n_pairs + test_pairs]
        inner_features = _subset_contextual_features(features, train_pairs)
        scores = grouped_contextual_rbf_config_scores(
            inner_features, configurations, n_folds, seed + fold
        )
        best = int(np.argmax(scores))
        config = configurations[best]
        representation = str(config["representation"])
        model = _fit_rbf(features[representation][train_idx], labels[train_idx], config)
        predictions[test_idx] = model.decision_function(
            features[representation][test_idx]
        )
        selected.append(
            {"config": dict(config), "inner_auroc": float(scores[best])}
        )
    return {"predictions": predictions, "selected": selected}


def _subset_contextual_features(
    features: dict[str, np.ndarray], pairs: np.ndarray
) -> dict[str, np.ndarray]:
    n_pairs = next(iter(features.values())).shape[0] // 2
    return {
        representation: np.r_[matrix[pairs], matrix[n_pairs + pairs]]
        for representation, matrix in features.items()
    }


def _fit_predict_sparse(
    train_text: list[str],
    train_labels: np.ndarray,
    test_text: list[str],
    config: dict[str, str | float],
) -> np.ndarray:
    vectorizer = _text_vectorizer(str(config["kind"]))
    x_train = vectorizer.fit_transform(train_text)
    classifier = _fit_sparse_rbf(x_train, train_labels, config)
    return classifier.decision_function(vectorizer.transform(test_text))


def grouped_combined_text_rbf_oof_predictions(
    positive: list[list[dict]],
    negative: list[list[dict]],
    contextual_features: dict[str, np.ndarray],
    n_folds: int,
    seed: int,
    sparse_configurations: list[dict[str, str | float]],
    contextual_configurations: list[dict[str, str | float]],
) -> dict:
    """Nested OOF predictions selecting sparse or contextual nonlinear text."""
    if len(positive) != len(negative):
        raise ValueError("nonlinear text inputs must contain matched pairs")
    n_pairs = len(positive)
    _validate_contextual_features(contextual_features, n_pairs)
    pos_text, neg_text = differing_final_messages(positive, negative)
    texts = pos_text + neg_text
    labels = np.r_[np.ones(n_pairs), np.zeros(n_pairs)]
    predictions = np.empty(2 * n_pairs, dtype=np.float64)
    selected = []

    for fold, (train_pairs, test_pairs) in enumerate(
        pair_folds(n_pairs, n_folds, seed)
    ):
        train_idx = np.r_[train_pairs, n_pairs + train_pairs]
        test_idx = np.r_[test_pairs, n_pairs + test_pairs]
        inner_pos_text = [pos_text[index] for index in train_pairs]
        inner_neg_text = [neg_text[index] for index in train_pairs]
        sparse_scores = grouped_text_rbf_config_scores(
            inner_pos_text,
            inner_neg_text,
            sparse_configurations,
            n_folds,
            seed + fold,
        )
        inner_contextual = _subset_contextual_features(
            contextual_features, train_pairs
        )
        contextual_scores = grouped_contextual_rbf_config_scores(
            inner_contextual,
            contextual_configurations,
            n_folds,
            seed + fold,
        )
        sparse_best = int(np.argmax(sparse_scores))
        contextual_best = int(np.argmax(contextual_scores))

        if contextual_scores[contextual_best] > sparse_scores[sparse_best]:
            config = contextual_configurations[contextual_best]
            representation = str(config["representation"])
            model = _fit_rbf(
                contextual_features[representation][train_idx],
                labels[train_idx],
                config,
            )
            predictions[test_idx] = model.decision_function(
                contextual_features[representation][test_idx]
            )
            selected.append(
                {
                    "family": "contextual_e5",
                    "config": dict(config),
                    "inner_auroc": float(contextual_scores[contextual_best]),
                }
            )
        else:
            config = sparse_configurations[sparse_best]
            predictions[test_idx] = _fit_predict_sparse(
                [texts[index] for index in train_idx],
                labels[train_idx],
                [texts[index] for index in test_idx],
                config,
            )
            selected.append(
                {
                    "family": "sparse_tfidf",
                    "config": dict(config),
                    "inner_auroc": float(sparse_scores[sparse_best]),
                }
            )

    return {"predictions": predictions, "selected": selected}


def nonlinear_activation_transfer(
    source,
    target,
    n_folds: int,
    seed: int,
    configurations: list[dict[str, float]],
) -> dict:
    """Source-only activation layer/configuration selection and target scoring."""
    _, _, source_pos, source_neg = source
    _, _, target_pos, target_neg = target
    source_layer_scores = grouped_dom_layer_scores(
        source_pos, source_neg, n_folds, seed
    )
    layer = int(np.argmax(np.abs(source_layer_scores - 0.5)))
    dom_orientation = 1 if source_layer_scores[layer] >= 0.5 else -1
    config_scores = grouped_rbf_config_scores(
        source_pos[:, layer], source_neg[:, layer], configurations, n_folds, seed
    )
    config_index = int(np.argmax(np.abs(config_scores - 0.5)))
    config = configurations[config_index]
    rbf_orientation = 1 if config_scores[config_index] >= 0.5 else -1
    source_labels = np.r_[np.ones(len(source_pos)), np.zeros(len(source_neg))]
    source_matrix = np.r_[source_pos[:, layer], source_neg[:, layer]]
    target_matrix = np.r_[target_pos[:, layer], target_neg[:, layer]]
    rbf = _fit_rbf(source_matrix, source_labels, config)
    dom = fit_layer(source_matrix, source_labels)
    return {
        "labels": np.r_[np.ones(len(target_pos)), np.zeros(len(target_neg))],
        "cluster_ids": np.r_[
            np.arange(len(target_pos)), np.arange(len(target_pos))
        ],
        "predictions": {
            "direction_of_means": dom_orientation * dom.project(target_matrix),
            "rbf": rbf_orientation * rbf.decision_function(target_matrix),
        },
        "selected_layer": layer,
        "source_layer_cv_auroc": float(source_layer_scores[layer]),
        "selected_rbf_config": dict(config),
        "source_rbf_cv_auroc": float(config_scores[config_index]),
        "source_rbf_orientation": int(rbf_orientation),
    }


def nonlinear_text_transfer(
    source,
    target,
    source_contextual: dict[str, np.ndarray],
    target_contextual: dict[str, np.ndarray],
    n_folds: int,
    seed: int,
    sparse_configurations: list[dict[str, str | float]],
    contextual_configurations: list[dict[str, str | float]],
) -> dict:
    """Select the nonlinear text family/configuration on source CV only."""
    source_positive, source_negative = source[:2]
    target_positive, target_negative = target[:2]
    source_pos_text, source_neg_text = differing_final_messages(
        source_positive, source_negative
    )
    sparse_scores = grouped_text_rbf_config_scores(
        source_pos_text,
        source_neg_text,
        sparse_configurations,
        n_folds,
        seed,
    )
    contextual_scores = grouped_contextual_rbf_config_scores(
        source_contextual, contextual_configurations, n_folds, seed
    )
    sparse_best = int(np.argmax(np.abs(sparse_scores - 0.5)))
    contextual_best = int(np.argmax(np.abs(contextual_scores - 0.5)))
    source_labels = np.r_[
        np.ones(len(source_positive)), np.zeros(len(source_negative))
    ]

    if abs(contextual_scores[contextual_best] - 0.5) > abs(
        sparse_scores[sparse_best] - 0.5
    ):
        config = contextual_configurations[contextual_best]
        representation = str(config["representation"])
        model = _fit_rbf(
            source_contextual[representation], source_labels, config
        )
        predictions = model.decision_function(target_contextual[representation])
        source_cv_auroc = float(contextual_scores[contextual_best])
        selected = {"family": "contextual_e5", "config": dict(config)}
    else:
        config = sparse_configurations[sparse_best]
        target_pos_text, target_neg_text = differing_final_messages(
            target_positive, target_negative
        )
        predictions = _fit_predict_sparse(
            source_pos_text + source_neg_text,
            source_labels,
            target_pos_text + target_neg_text,
            config,
        )
        source_cv_auroc = float(sparse_scores[sparse_best])
        selected = {"family": "sparse_tfidf", "config": dict(config)}

    orientation = 1 if source_cv_auroc >= 0.5 else -1
    return {
        "predictions": orientation * predictions,
        "selected": selected,
        "source_cv_auroc": source_cv_auroc,
        "source_orientation": int(orientation),
    }


def _comparison(
    labels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    clusters: np.ndarray,
    n_boot: int,
    n_permutations: int,
    seed: int,
) -> dict:
    return {
        "bootstrap": bootstrap_auroc_difference_ci(
            labels, first, second, clusters, n_boot=n_boot, seed=seed
        ),
        "permutation": paired_prediction_permutation_test(
            labels,
            first,
            second,
            clusters,
            n_permutations=n_permutations,
            seed=seed,
        ),
    }


def _aggregate(records: list[dict], keys: tuple[str, ...]) -> dict:
    result = {}
    for key in keys:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "sd": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "values": values.tolist(),
        }
    return result


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


def analyze(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    contrasts = load_contrasts(ROOT)
    fingerprints = validate_suite(manifest, contrasts)
    activation_configs = rbf_configurations(
        tuple(args.c_grid), tuple(args.gamma_multipliers)
    )
    sparse_configs = text_rbf_configurations(
        tuple(args.c_grid), tuple(args.text_gamma_grid)
    )
    contextual_configs = contextual_rbf_configurations(
        tuple(args.c_grid), tuple(args.gamma_multipliers)
    )
    contextual_features, e5_metadata = prepare_contextual_features(
        contrasts,
        args.e5_cache,
        args.batch_size,
        args.device,
        args.refresh_e5,
    )
    if args.prepare_e5_only:
        print(f"saved/validated {args.e5_cache}")
        return
    if args.contextual_only:
        selected_representation = args.contextual_representation
        fixed_configs = [
            config
            for config in contextual_configs
            if config["representation"] == selected_representation
        ]
        contextual_output = {
            "audit_id": AUDIT_ID,
            "dataset_fingerprint": contextual_fingerprint(contrasts),
            "representation": selected_representation,
            "seeds": {},
            "aggregate": {},
            "configurations": fixed_configs,
            "selection": "RBF hyperparameters nested inside outer pair folds",
            "e5": e5_metadata,
        }
        for seed in args.seeds:
            contextual_output["seeds"][str(seed)] = {}
            for name, contrast in contrasts.items():
                run = grouped_contextual_rbf_oof_predictions(
                    contextual_features[name],
                    fixed_configs,
                    args.folds,
                    seed,
                )
                labels = np.r_[
                    np.ones(len(contrast.item_ids)),
                    np.zeros(len(contrast.item_ids)),
                ]
                auroc = float(roc_auc_score(labels, run["predictions"]))
                contextual_output["seeds"][str(seed)][name] = {
                    "auroc": auroc,
                    "selected": run["selected"],
                }
                print(
                    f"seed={seed} {name} fixed {selected_representation}: "
                    f"{auroc:.5f}",
                    flush=True,
                )
        for name in contrasts:
            records = [
                contextual_output["seeds"][str(seed)][name]
                for seed in args.seeds
            ]
            contextual_output["aggregate"][name] = _aggregate(records, ("auroc",))
        destination = args.results / f"e5_{selected_representation}_analysis.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(contextual_output, indent=2) + "\n", encoding="utf-8"
        )
        print(f"saved {destination}", flush=True)
        return
    if args.fixed_e5_inference_only:
        seed = args.seeds[0]
        selected_representation = args.contextual_representation
        fixed_configs = [
            config
            for config in contextual_configs
            if config["representation"] == selected_representation
        ]
        text_runs = {}
        for name, contrast in contrasts.items():
            text_runs[name] = grouped_contextual_rbf_oof_predictions(
                contextual_features[name], fixed_configs, args.folds, seed
            )
        inference_output = {
            "audit_id": AUDIT_ID,
            "seed": seed,
            "text_comparator": {
                "encoder": E5_MODEL_ID,
                "representation": selected_representation,
                "estimator": "nested StandardScaler + RBF SVC",
            },
            "n_boot": args.n_boot,
            "n_permutations": args.n_permutations,
            "models": {},
        }
        for repository in _repositories(manifest, args.model):
            inference_output["models"][repository] = {}
            for name, contrast in contrasts.items():
                positive_acts, _ = _load_acts(
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
                negative_acts, _ = _load_acts(
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
                activation_run = grouped_nonlinear_oof_predictions(
                    (
                        contrast.positive,
                        contrast.negative,
                        positive_acts,
                        negative_acts,
                    ),
                    args.folds,
                    seed,
                    activation_configs,
                    include_text=False,
                )
                labels = activation_run["labels"]
                activation_predictions = activation_run["predictions"]["rbf"]
                text_predictions = text_runs[name]["predictions"]
                activation_auroc = float(
                    roc_auc_score(labels, activation_predictions)
                )
                text_auroc = float(roc_auc_score(labels, text_predictions))
                inference_output["models"][repository][name] = {
                    "activation_rbf_auroc": activation_auroc,
                    "e5_rbf_auroc": text_auroc,
                    "activation_minus_e5_rbf": _comparison(
                        labels,
                        activation_predictions,
                        text_predictions,
                        activation_run["cluster_ids"],
                        args.n_boot,
                        args.n_permutations,
                        seed,
                    ),
                }
                print(
                    f"{repository} {name}: {activation_auroc:.5f} - "
                    f"{text_auroc:.5f} = {activation_auroc-text_auroc:+.5f}",
                    flush=True,
                )
        destination = args.results / "fixed_e5_seed0_inference.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(inference_output, indent=2) + "\n", encoding="utf-8"
        )
        print(f"saved {destination}", flush=True)
        return

    output = {
        "audit_id": AUDIT_ID,
        "manifest_sha256": manifest_sha256(args.manifest),
        "dataset_fingerprints": fingerprints,
        "seeds": args.seeds,
        "read_point": "last_user_token",
        "design": {
            "outer_and_inner_pair_grouped": True,
            "activation_layer": (
                "primary direction-of-means grouped CV; no nonlinear layer search"
            ),
            "activation_configurations": activation_configs,
            "nonlinear_text_candidates": {
                "sparse": sparse_configs,
                "contextual_e5": contextual_configs,
            },
            "within_text_selection": "nested inside each outer training fold",
            "transfer_selection": "source control only",
            "e5": e5_metadata,
        },
        "text": {"seeds": {}, "aggregate": {}},
        "models": {},
    }

    print("nonlinear text baselines", flush=True)
    text_runs: dict[int, dict] = {}
    contrast_names = list(contrasts)
    transfer_directions = [
        (contrast_names[0], contrast_names[1]),
        (contrast_names[1], contrast_names[0]),
    ]
    for seed in args.seeds:
        text_runs[seed] = {"within": {}, "transfer": {}}
        output["text"]["seeds"][str(seed)] = {"within": {}, "transfer": {}}
        for name, contrast in contrasts.items():
            run = grouped_combined_text_rbf_oof_predictions(
                contrast.positive,
                contrast.negative,
                contextual_features[name],
                args.folds,
                seed,
                sparse_configs,
                contextual_configs,
            )
            labels = np.r_[
                np.ones(len(contrast.item_ids)), np.zeros(len(contrast.item_ids))
            ]
            auroc = float(roc_auc_score(labels, run["predictions"]))
            text_runs[seed]["within"][name] = run
            output["text"]["seeds"][str(seed)]["within"][name] = {
                "auroc": auroc,
                "selected": run["selected"],
            }
            print(f"  seed={seed} {name}: {auroc:.5f}", flush=True)
        for source_name, target_name in transfer_directions:
            run = nonlinear_text_transfer(
                (
                    contrasts[source_name].positive,
                    contrasts[source_name].negative,
                ),
                (
                    contrasts[target_name].positive,
                    contrasts[target_name].negative,
                ),
                contextual_features[source_name],
                contextual_features[target_name],
                args.folds,
                seed,
                sparse_configs,
                contextual_configs,
            )
            labels = np.r_[
                np.ones(len(contrasts[target_name].item_ids)),
                np.zeros(len(contrasts[target_name].item_ids)),
            ]
            key = f"{source_name}_to_{target_name}"
            strength = float(abs(roc_auc_score(labels, run["predictions"]) - 0.5))
            text_runs[seed]["transfer"][key] = run
            output["text"]["seeds"][str(seed)]["transfer"][key] = {
                "auroc": float(roc_auc_score(labels, run["predictions"])),
                "strength": strength,
                "selected": run["selected"],
                "source_cv_auroc": run["source_cv_auroc"],
                "source_orientation": run["source_orientation"],
            }

    for name in contrasts:
        records = [
            output["text"]["seeds"][str(seed)]["within"][name]
            for seed in args.seeds
        ]
        output["text"]["aggregate"][name] = _aggregate(records, ("auroc",))
    for source_name, target_name in transfer_directions:
        key = f"{source_name}_to_{target_name}"
        records = [
            output["text"]["seeds"][str(seed)]["transfer"][key]
            for seed in args.seeds
        ]
        output["text"]["aggregate"][key] = _aggregate(
            records, ("auroc", "strength")
        )

    for repository in _repositories(manifest, args.model):
        print(f"\nmodel={repository}", flush=True)
        loaded = {}
        for name, contrast in contrasts.items():
            positive_acts, _ = _load_acts(
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
            negative_acts, _ = _load_acts(
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
            loaded[name] = (
                contrast.positive,
                contrast.negative,
                positive_acts,
                negative_acts,
            )

        model_output = {"seeds": {}, "aggregate": {"within": {}, "transfer": {}}}
        for seed in args.seeds:
            seed_output = {"within": {}, "transfer": {}}
            for name in contrasts:
                run = grouped_nonlinear_oof_predictions(
                    loaded[name],
                    args.folds,
                    seed,
                    activation_configs,
                    include_text=False,
                )
                text_run = text_runs[seed]["within"][name]
                labels = run["labels"]
                clusters = run["cluster_ids"]
                dom_auroc = float(
                    roc_auc_score(labels, run["predictions"]["direction_of_means"])
                )
                rbf_auroc = float(roc_auc_score(labels, run["predictions"]["rbf"]))
                text_auroc = float(roc_auc_score(labels, text_run["predictions"]))
                record = {
                    "direction_of_means_auroc": dom_auroc,
                    "activation_rbf_auroc": rbf_auroc,
                    "nonlinear_text_auroc": text_auroc,
                    "rbf_minus_text": rbf_auroc - text_auroc,
                    "selected_layers": run["selected_layers"],
                    "selected_rbf_configs": run["selected_rbf_configs"],
                }
                if seed == args.seeds[0]:
                    record["rbf_minus_text_bootstrap"] = _comparison(
                        labels,
                        run["predictions"]["rbf"],
                        text_run["predictions"],
                        clusters,
                        args.n_boot,
                        args.n_permutations,
                        seed,
                    )
                seed_output["within"][name] = record
                print(
                    f"  seed={seed} {name}: RBF={rbf_auroc:.5f} "
                    f"text={text_auroc:.5f} margin={rbf_auroc-text_auroc:+.5f}",
                    flush=True,
                )

            for source_name, target_name in transfer_directions:
                key = f"{source_name}_to_{target_name}"
                run = nonlinear_activation_transfer(
                    loaded[source_name],
                    loaded[target_name],
                    args.folds,
                    seed,
                    activation_configs,
                )
                text_run = text_runs[seed]["transfer"][key]
                labels = run["labels"]
                activation_auroc = float(
                    roc_auc_score(labels, run["predictions"]["rbf"])
                )
                dom_auroc = float(
                    roc_auc_score(labels, run["predictions"]["direction_of_means"])
                )
                text_auroc = float(roc_auc_score(labels, text_run["predictions"]))
                activation_strength = abs(activation_auroc - 0.5)
                text_strength = abs(text_auroc - 0.5)
                record = {
                    "direction_of_means_auroc": dom_auroc,
                    "activation_rbf_auroc": activation_auroc,
                    "activation_rbf_strength": activation_strength,
                    "nonlinear_text_auroc": text_auroc,
                    "nonlinear_text_strength": text_strength,
                    "rbf_minus_text_strength": activation_strength - text_strength,
                    "selected_layer": run["selected_layer"],
                    "source_layer_cv_auroc": run["source_layer_cv_auroc"],
                    "selected_rbf_config": run["selected_rbf_config"],
                    "source_rbf_cv_auroc": run["source_rbf_cv_auroc"],
                    "source_rbf_orientation": run["source_rbf_orientation"],
                }
                if seed == args.seeds[0]:
                    record["rbf_minus_text_auroc_bootstrap"] = _comparison(
                        labels,
                        run["predictions"]["rbf"],
                        text_run["predictions"],
                        run["cluster_ids"],
                        args.n_boot,
                        args.n_permutations,
                        seed,
                    )
                seed_output["transfer"][key] = record
                print(
                    f"  seed={seed} {key}: RBF={activation_auroc:.5f} "
                    f"text={text_auroc:.5f} strength margin="
                    f"{activation_strength-text_strength:+.5f}",
                    flush=True,
                )
            model_output["seeds"][str(seed)] = seed_output

        for name in contrasts:
            records = [
                model_output["seeds"][str(seed)]["within"][name]
                for seed in args.seeds
            ]
            model_output["aggregate"]["within"][name] = _aggregate(
                records,
                (
                    "direction_of_means_auroc",
                    "activation_rbf_auroc",
                    "nonlinear_text_auroc",
                    "rbf_minus_text",
                ),
            )
        for source_name, target_name in transfer_directions:
            key = f"{source_name}_to_{target_name}"
            records = [
                model_output["seeds"][str(seed)]["transfer"][key]
                for seed in args.seeds
            ]
            model_output["aggregate"]["transfer"][key] = _aggregate(
                records,
                (
                    "direction_of_means_auroc",
                    "activation_rbf_auroc",
                    "activation_rbf_strength",
                    "nonlinear_text_auroc",
                    "nonlinear_text_strength",
                    "rbf_minus_text_strength",
                ),
            )
        output["models"][repository] = model_output

        destination = args.results / model_slug(repository) / "analysis.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "audit_id": AUDIT_ID,
                    "model_repository": repository,
                    **model_output,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"saved {destination}", flush=True)

    combined = args.results / "combined_analysis.json"
    combined.parent.mkdir(parents=True, exist_ok=True)
    combined.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"saved {combined}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--primary-results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--readout-results", type=Path, default=ROOT / "results/readout_audit"
    )
    parser.add_argument("--e5-cache", type=Path, default=DEFAULT_E5_CACHE)
    parser.add_argument("--refresh-e5", action="store_true")
    parser.add_argument("--prepare-e5-only", action="store_true")
    parser.add_argument("--contextual-only", action="store_true")
    parser.add_argument("--fixed-e5-inference-only", action="store_true")
    parser.add_argument(
        "--contextual-representation",
        choices=list(E5_REPRESENTATIONS),
        default="prefix_ending_interaction",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--model", action="append")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--c-grid", type=float, nargs="+", default=list(DEFAULT_C_GRID))
    parser.add_argument(
        "--gamma-multipliers",
        type=float,
        nargs="+",
        default=list(DEFAULT_GAMMA_MULTIPLIERS),
    )
    parser.add_argument(
        "--text-gamma-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_TEXT_GAMMA_GRID),
    )
    return parser


def main() -> None:
    analyze(build_parser().parse_args())


if __name__ == "__main__":
    main()
