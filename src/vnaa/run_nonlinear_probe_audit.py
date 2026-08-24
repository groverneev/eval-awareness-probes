"""Post-result nonlinear-capacity audit on the frozen cross-model caches.

The primary analysis uses a direction-of-means probe.  This audit asks the
narrow follow-up question: at the layer selected by that primary pipeline using
outer-training data only, does a regularized RBF-kernel classifier recover
materially more held-out signal?  It deliberately does not search for a
different layer with the nonlinear estimator, so the comparison changes probe
capacity without opening a second architecture-by-layer search.

Every scenario pair remains grouped in both the outer evaluation folds and the
inner hyperparameter folds.  Scaling, layer selection, and RBF hyperparameter
selection are all fit without access to the outer test fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
from vnaa.run_readout_audit import REFERENCE_MODEL, _load_acts
from vnaa.run_strong_baseline_audit import (
    BASELINES,
    _text_scores,
    fit_selected_dom_grouped_cv,
    pair_folds,
)
from vnaa.run_slice3 import differing_final_messages


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_RESULTS = ROOT / "results/nonlinear_probe_audit"
AUDIT_ID = "postresult_rbf_capacity_v1"
DEFAULT_C_GRID = (0.1, 1.0, 10.0)
DEFAULT_GAMMA_MULTIPLIERS = (0.25, 1.0, 4.0)
DEFAULT_LOGISTIC_C_GRID = (0.001, 0.01, 0.1, 1.0)
TEXT_KINDS = ("word_tfidf", "char_tfidf")
DEFAULT_TEXT_GAMMA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)


def rbf_configurations(
    c_grid: tuple[float, ...] = DEFAULT_C_GRID,
    gamma_multipliers: tuple[float, ...] = DEFAULT_GAMMA_MULTIPLIERS,
) -> list[dict[str, float]]:
    """Return the fixed, deterministic RBF grid in conservative-first order."""
    if not c_grid or not gamma_multipliers:
        raise ValueError("RBF grids must be non-empty")
    if any(value <= 0 for value in (*c_grid, *gamma_multipliers)):
        raise ValueError("RBF grid values must be positive")
    return [
        {"C": float(c_value), "gamma_multiplier": float(gamma_multiplier)}
        for c_value in c_grid
        for gamma_multiplier in gamma_multipliers
    ]


def _fit_rbf(
    acts: np.ndarray,
    labels: np.ndarray,
    config: dict[str, float],
) -> object:
    """Fit scaling and the RBF classifier on one training partition."""
    acts = np.asarray(acts, dtype=np.float64)
    labels = np.asarray(labels)
    if acts.ndim != 2 or len(acts) != len(labels):
        raise ValueError("RBF inputs must have shape (sample, feature)")
    if len(np.unique(labels)) < 2:
        raise ValueError("RBF fitting needs both classes")
    gamma = config["gamma_multiplier"] / acts.shape[1]
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(
            C=config["C"],
            gamma=gamma,
            kernel="rbf",
            cache_size=1024,
        ),
    )
    pipeline.fit(acts, labels)
    return pipeline


def logistic_configurations(
    c_grid: tuple[float, ...] = DEFAULT_LOGISTIC_C_GRID,
) -> list[dict[str, float]]:
    """Return the fixed L2 grid in strong-to-weak regularization order."""
    if not c_grid or any(value <= 0 for value in c_grid):
        raise ValueError("logistic C values must be non-empty and positive")
    return [{"C": float(value)} for value in c_grid]


def _fit_l2_logistic(
    acts: np.ndarray,
    labels: np.ndarray,
    config: dict[str, float],
    seed: int,
) -> object:
    """Fit standardized L2 logistic regression on one training partition."""
    acts = np.asarray(acts, dtype=np.float64)
    labels = np.asarray(labels)
    if acts.ndim != 2 or len(acts) != len(labels):
        raise ValueError("logistic inputs must have shape (sample, feature)")
    if len(np.unique(labels)) < 2:
        raise ValueError("logistic fitting needs both classes")
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=config["C"],
            max_iter=5000,
            random_state=seed,
        ),
    )
    pipeline.fit(acts, labels)
    return pipeline


def text_rbf_configurations(
    c_grid: tuple[float, ...] = DEFAULT_C_GRID,
    gamma_grid: tuple[float, ...] = DEFAULT_TEXT_GAMMA_GRID,
) -> list[dict[str, str | float]]:
    """Fixed text-kernel grid; TF--IDF rows are already L2 normalized."""
    if not c_grid or not gamma_grid:
        raise ValueError("text RBF grids must be non-empty")
    if any(value <= 0 for value in (*c_grid, *gamma_grid)):
        raise ValueError("text RBF grid values must be positive")
    return [
        {"kind": kind, "C": float(c_value), "gamma": float(gamma)}
        for kind in TEXT_KINDS
        for c_value in c_grid
        for gamma in gamma_grid
    ]


def _text_vectorizer(kind: str) -> TfidfVectorizer:
    if kind == "word_tfidf":
        return TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=50_000,
        )
    if kind == "char_tfidf":
        return TfidfVectorizer(
            analyzer="char_wb",
            lowercase=True,
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            max_features=100_000,
        )
    raise ValueError(f"unknown text representation: {kind}")


def _fit_sparse_rbf(x_train, labels, config: dict[str, str | float]) -> SVC:
    classifier = SVC(
        C=float(config["C"]),
        gamma=float(config["gamma"]),
        kernel="rbf",
        cache_size=1024,
    )
    classifier.fit(x_train, labels)
    return classifier


def grouped_text_rbf_config_scores(
    pos_text: list[str],
    neg_text: list[str],
    configurations: list[dict[str, str | float]],
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """Nested pair-grouped scores for word/character TF--IDF RBF models."""
    if len(pos_text) != len(neg_text):
        raise ValueError("text RBF inputs must contain matched pairs")
    if not configurations:
        raise ValueError("at least one text RBF configuration is required")
    n_pairs = len(pos_text)
    texts = pos_text + neg_text
    labels = np.r_[np.ones(n_pairs), np.zeros(n_pairs)]
    held_scores = np.empty(
        (2 * n_pairs, len(configurations)), dtype=np.float64
    )

    for train_pairs, test_pairs in pair_folds(n_pairs, n_folds, seed):
        train_idx = np.r_[train_pairs, n_pairs + train_pairs]
        test_idx = np.r_[test_pairs, n_pairs + test_pairs]
        train_text = [texts[index] for index in train_idx]
        test_text = [texts[index] for index in test_idx]
        for kind in TEXT_KINDS:
            vectorizer = _text_vectorizer(kind)
            x_train = vectorizer.fit_transform(train_text)
            x_test = vectorizer.transform(test_text)
            for index, config in enumerate(configurations):
                if config["kind"] != kind:
                    continue
                classifier = _fit_sparse_rbf(x_train, labels[train_idx], config)
                held_scores[test_idx, index] = classifier.decision_function(x_test)

    return np.asarray(
        [
            roc_auc_score(labels, held_scores[:, index])
            for index in range(len(configurations))
        ],
        dtype=np.float64,
    )


def grouped_text_rbf_oof_predictions(
    pos: list[list[dict]],
    neg: list[list[dict]],
    n_folds: int,
    seed: int,
    configurations: list[dict[str, str | float]],
) -> dict:
    """Outer predictions for a capacity-matched nonlinear text comparator."""
    if len(pos) != len(neg):
        raise ValueError("text RBF inputs must contain matched conversations")
    pos_text, neg_text = differing_final_messages(pos, neg)
    n_pairs = len(pos_text)
    texts = pos_text + neg_text
    labels = np.r_[np.ones(n_pairs), np.zeros(n_pairs)]
    predictions = np.empty(2 * n_pairs, dtype=np.float64)
    selected_configs = []
    selected_inner_aurocs = []

    for fold, (train_pairs, test_pairs) in enumerate(
        pair_folds(n_pairs, n_folds, seed)
    ):
        train_idx = np.r_[train_pairs, n_pairs + train_pairs]
        test_idx = np.r_[test_pairs, n_pairs + test_pairs]
        inner_pos = [pos_text[index] for index in train_pairs]
        inner_neg = [neg_text[index] for index in train_pairs]
        config_scores = grouped_text_rbf_config_scores(
            inner_pos,
            inner_neg,
            configurations,
            n_folds,
            seed + fold,
        )
        config = configurations[int(np.argmax(config_scores))]
        vectorizer = _text_vectorizer(str(config["kind"]))
        x_train = vectorizer.fit_transform([texts[index] for index in train_idx])
        x_test = vectorizer.transform([texts[index] for index in test_idx])
        classifier = _fit_sparse_rbf(x_train, labels[train_idx], config)
        predictions[test_idx] = classifier.decision_function(x_test)
        selected_configs.append(dict(config))
        selected_inner_aurocs.append(float(np.max(config_scores)))

    return {
        "predictions": predictions,
        "selected_configs": selected_configs,
        "selected_inner_aurocs": selected_inner_aurocs,
    }


def grouped_rbf_config_scores(
    acts_pos: np.ndarray,
    acts_neg: np.ndarray,
    configurations: list[dict[str, float]],
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """Pair-grouped inner-CV AUROC for each fixed RBF configuration."""
    acts_pos = np.asarray(acts_pos, dtype=np.float64)
    acts_neg = np.asarray(acts_neg, dtype=np.float64)
    if acts_pos.shape != acts_neg.shape or acts_pos.ndim != 2:
        raise ValueError("paired RBF activations must share (pair, feature) shape")
    if not configurations:
        raise ValueError("at least one RBF configuration is required")
    n_pairs = len(acts_pos)
    labels = np.r_[np.ones(n_pairs), np.zeros(n_pairs)]
    held_scores = np.empty(
        (2 * n_pairs, len(configurations)), dtype=np.float64
    )

    for train_pairs, test_pairs in pair_folds(n_pairs, n_folds, seed):
        train_idx = np.r_[train_pairs, n_pairs + train_pairs]
        test_idx = np.r_[test_pairs, n_pairs + test_pairs]
        pooled = np.r_[acts_pos, acts_neg]
        x_train = pooled[train_idx]
        x_test = pooled[test_idx]
        y_train = labels[train_idx]
        for index, config in enumerate(configurations):
            model = _fit_rbf(x_train, y_train, config)
            held_scores[test_idx, index] = model.decision_function(x_test)

    return np.asarray(
        [
            roc_auc_score(labels, held_scores[:, index])
            for index in range(len(configurations))
        ],
        dtype=np.float64,
    )


def select_rbf_config_grouped_cv(
    acts_pos: np.ndarray,
    acts_neg: np.ndarray,
    configurations: list[dict[str, float]],
    n_folds: int,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    """Select the highest inner-CV AUROC; stable order breaks exact ties."""
    scores = grouped_rbf_config_scores(
        acts_pos, acts_neg, configurations, n_folds, seed
    )
    return configurations[int(np.argmax(scores))], scores


def grouped_logistic_config_scores(
    acts_pos: np.ndarray,
    acts_neg: np.ndarray,
    configurations: list[dict[str, float]],
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """Pair-grouped inner-CV AUROC for each L2 configuration."""
    acts_pos = np.asarray(acts_pos, dtype=np.float64)
    acts_neg = np.asarray(acts_neg, dtype=np.float64)
    if acts_pos.shape != acts_neg.shape or acts_pos.ndim != 2:
        raise ValueError("paired logistic activations must share shape")
    if not configurations:
        raise ValueError("at least one logistic configuration is required")
    n_pairs = len(acts_pos)
    labels = np.r_[np.ones(n_pairs), np.zeros(n_pairs)]
    pooled = np.r_[acts_pos, acts_neg]
    held_scores = np.empty((2 * n_pairs, len(configurations)), dtype=np.float64)

    for fold, (train_pairs, test_pairs) in enumerate(
        pair_folds(n_pairs, n_folds, seed)
    ):
        train_idx = np.r_[train_pairs, n_pairs + train_pairs]
        test_idx = np.r_[test_pairs, n_pairs + test_pairs]
        for index, config in enumerate(configurations):
            model = _fit_l2_logistic(
                pooled[train_idx], labels[train_idx], config, seed + fold
            )
            held_scores[test_idx, index] = model.decision_function(
                pooled[test_idx]
            )

    return np.asarray(
        [
            roc_auc_score(labels, held_scores[:, index])
            for index in range(len(configurations))
        ],
        dtype=np.float64,
    )


def select_logistic_config_grouped_cv(
    acts_pos: np.ndarray,
    acts_neg: np.ndarray,
    configurations: list[dict[str, float]],
    n_folds: int,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    """Select L2 regularization by grouped inner CV with stable tie-breaking."""
    scores = grouped_logistic_config_scores(
        acts_pos, acts_neg, configurations, n_folds, seed
    )
    return configurations[int(np.argmax(scores))], scores


def grouped_nonlinear_oof_predictions(
    pair,
    n_folds: int,
    seed: int,
    configurations: list[dict[str, float]],
    include_text: bool = True,
    logistic_configs: list[dict[str, float]] | None = None,
    include_l2: bool = False,
) -> dict:
    """Honest same-layer outer predictions for DoM, optional L2, and RBF."""
    pos, neg, acts_pos, acts_neg = pair
    acts_pos = np.asarray(acts_pos, dtype=np.float64)
    acts_neg = np.asarray(acts_neg, dtype=np.float64)
    if not (
        len(pos) == len(neg) == len(acts_pos) == len(acts_neg)
        and acts_pos.shape == acts_neg.shape
        and acts_pos.ndim == 3
    ):
        raise ValueError("matched pair arrays have inconsistent shapes")

    n_pairs = len(pos)
    labels = np.r_[np.ones(n_pairs), np.zeros(n_pairs)]
    if include_l2 and logistic_configs is None:
        logistic_configs = logistic_configurations()
    prediction_names = ["direction_of_means", "rbf"]
    if include_l2:
        prediction_names.insert(1, "l2_logistic")
    if include_text:
        prediction_names.extend(BASELINES)
        pos_text, neg_text = differing_final_messages(pos, neg)
        texts = pos_text + neg_text
    predictions = {
        name: np.empty(2 * n_pairs, dtype=np.float64)
        for name in prediction_names
    }
    selected_layers = []
    selected_layer_inner_aurocs = []
    selected_logistic_configs = []
    selected_logistic_inner_aurocs = []
    selected_rbf_configs = []
    selected_rbf_inner_aurocs = []

    for fold, (train_pairs, test_pairs) in enumerate(
        pair_folds(n_pairs, n_folds, seed)
    ):
        train_idx = np.r_[train_pairs, n_pairs + train_pairs]
        test_idx = np.r_[test_pairs, n_pairs + test_pairs]
        x_test = np.r_[acts_pos[test_pairs], acts_neg[test_pairs]]

        dom_probe, layer, layer_scores = fit_selected_dom_grouped_cv(
            acts_pos[train_pairs],
            acts_neg[train_pairs],
            n_folds,
            seed + fold,
        )
        predictions["direction_of_means"][test_idx] = dom_probe.project(
            x_test[:, layer]
        )
        selected_layers.append(int(layer))
        selected_layer_inner_aurocs.append(float(layer_scores[layer]))

        if include_l2:
            logistic_config, logistic_scores = select_logistic_config_grouped_cv(
                acts_pos[train_pairs, layer],
                acts_neg[train_pairs, layer],
                logistic_configs,
                n_folds,
                seed + fold,
            )

        config, config_scores = select_rbf_config_grouped_cv(
            acts_pos[train_pairs, layer],
            acts_neg[train_pairs, layer],
            configurations,
            n_folds,
            seed + fold,
        )
        x_train = np.r_[
            acts_pos[train_pairs, layer], acts_neg[train_pairs, layer]
        ]
        y_train = labels[train_idx]
        if include_l2:
            logistic = _fit_l2_logistic(
                x_train, y_train, logistic_config, seed + fold
            )
            predictions["l2_logistic"][test_idx] = logistic.decision_function(
                x_test[:, layer]
            )
            selected_logistic_configs.append(dict(logistic_config))
            selected_logistic_inner_aurocs.append(float(np.max(logistic_scores)))

        rbf = _fit_rbf(x_train, y_train, config)
        predictions["rbf"][test_idx] = rbf.decision_function(x_test[:, layer])
        selected_rbf_configs.append(dict(config))
        selected_rbf_inner_aurocs.append(float(np.max(config_scores)))

        if include_text:
            for name, scores in _text_scores(
                texts, labels, train_idx, test_idx, seed + fold
            ).items():
                predictions[name][test_idx] = scores

    return {
        "labels": labels,
        "cluster_ids": np.r_[np.arange(n_pairs), np.arange(n_pairs)],
        "predictions": predictions,
        "selected_layers": selected_layers,
        "selected_layer_inner_aurocs": selected_layer_inner_aurocs,
        "selected_logistic_configs": selected_logistic_configs,
        "selected_logistic_inner_aurocs": selected_logistic_inner_aurocs,
        "selected_rbf_configs": selected_rbf_configs,
        "selected_rbf_inner_aurocs": selected_rbf_inner_aurocs,
    }


def _paired_comparison(
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


def summarize_nonlinear_predictions(
    run: dict,
    n_boot: int,
    n_permutations: int,
    seed: int,
) -> dict:
    labels = run["labels"]
    clusters = run["cluster_ids"]
    predictions = run["predictions"]
    aurocs = {
        name: float(roc_auc_score(labels, scores))
        for name, scores in predictions.items()
    }
    result = {
        "n_pairs": int(len(np.unique(clusters))),
        "aurocs": aurocs,
        "selected_layers": run["selected_layers"],
        "selected_layer_inner_aurocs": run["selected_layer_inner_aurocs"],
        "selected_rbf_configs": run["selected_rbf_configs"],
        "selected_rbf_inner_aurocs": run["selected_rbf_inner_aurocs"],
        "rbf_minus_direction_of_means": _paired_comparison(
            labels,
            predictions["rbf"],
            predictions["direction_of_means"],
            clusters,
            n_boot,
            n_permutations,
            seed,
        ),
    }
    if "l2_logistic" in predictions:
        result["selected_logistic_configs"] = run["selected_logistic_configs"]
        result["selected_logistic_inner_aurocs"] = run[
            "selected_logistic_inner_aurocs"
        ]
        result["l2_minus_direction_of_means"] = _paired_comparison(
            labels,
            predictions["l2_logistic"],
            predictions["direction_of_means"],
            clusters,
            n_boot,
            n_permutations,
            seed,
        )
        result["rbf_minus_l2"] = _paired_comparison(
            labels,
            predictions["rbf"],
            predictions["l2_logistic"],
            clusters,
            n_boot,
            n_permutations,
            seed,
        )
    available_text = [
        name for name in (*BASELINES, "text_rbf") if name in predictions
    ]
    if available_text:
        strongest = max(available_text, key=lambda name: aurocs[name])
        result["strongest_text_baseline"] = strongest
        result["rbf_minus_strongest_text"] = _paired_comparison(
            labels,
            predictions["rbf"],
            predictions[strongest],
            clusters,
            n_boot,
            n_permutations,
            seed,
        )
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


def _analysis_path(results: Path, repository: str) -> Path:
    return results / model_slug(repository) / "analysis.json"


def analyze(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    contrasts = load_contrasts(ROOT)
    fingerprints = validate_suite(manifest, contrasts)
    configurations = rbf_configurations(
        tuple(args.c_grid), tuple(args.gamma_multipliers)
    )
    text_configurations = text_rbf_configurations(
        tuple(args.c_grid), tuple(args.text_gamma_grid)
    )
    text_rbf_runs = {}
    print("computing capacity-matched nonlinear text baselines", flush=True)
    for contrast in contrasts.values():
        text_rbf_runs[contrast.name] = grouped_text_rbf_oof_predictions(
            contrast.positive,
            contrast.negative,
            args.folds,
            args.seed,
            text_configurations,
        )
        text_auroc = roc_auc_score(
            np.r_[np.ones(len(contrast.item_ids)), np.zeros(len(contrast.item_ids))],
            text_rbf_runs[contrast.name]["predictions"],
        )
        print(f"  {contrast.name}: text RBF={text_auroc:.5f}", flush=True)

    for repository in _repositories(manifest, args.model):
        output = {
            "audit_id": AUDIT_ID,
            "status": "post_result_nonlinear_capacity_sensitivity_audit",
            "model_repository": repository,
            "model_slug": model_slug(repository),
            "manifest_sha256": manifest_sha256(args.manifest),
            "dataset_fingerprints": fingerprints,
            "read_point": "last_user_token",
            "sample": "full",
            "dtype": args.dtype,
            "design": {
                "outer_folds": args.folds,
                "seed": args.seed,
                "pair_grouped_outer_and_inner_folds": True,
                "layer_selection": (
                    "primary_direction_of_means_nested_grouped_cv_on_outer_train"
                ),
                "nonlinear_estimator": "StandardScaler + RBF SVC",
                "gamma_definition": "gamma_multiplier / hidden_dimension",
                "configurations": configurations,
                "scope": "decoder_capacity_at_primary_pipeline_selected_layer",
                "nonlinear_text_comparator": {
                    "estimator": "word-or-character TF-IDF + RBF SVC",
                    "configurations": text_configurations,
                    "selection": "nested_pair_grouped_cv_on_outer_train",
                },
            },
            "inference": {
                "n_boot": args.n_boot,
                "n_permutations": args.n_permutations,
                "grouping_unit": "matched_scenario_pair",
            },
            "within": {},
        }
        print(f"\nmodel={repository}", flush=True)
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
            pair = (contrast.positive, contrast.negative, pos_acts, neg_acts)
            run = grouped_nonlinear_oof_predictions(
                pair,
                args.folds,
                args.seed,
                configurations,
                include_text=True,
            )
            text_run = text_rbf_runs[contrast.name]
            run["predictions"]["text_rbf"] = text_run["predictions"]
            summary = summarize_nonlinear_predictions(
                run, args.n_boot, args.n_permutations, args.seed
            )
            summary["selected_text_rbf_configs"] = text_run["selected_configs"]
            summary["selected_text_rbf_inner_aurocs"] = text_run[
                "selected_inner_aurocs"
            ]
            output["within"][contrast.name] = summary
            dom = summary["aurocs"]["direction_of_means"]
            rbf = summary["aurocs"]["rbf"]
            delta = summary["rbf_minus_direction_of_means"]["bootstrap"]
            print(
                f"  {contrast.name}: DoM={dom:.5f} RBF={rbf:.5f} "
                f"delta={delta['difference']:+.5f} "
                f"[{delta['ci_low']:+.5f}, {delta['ci_high']:+.5f}]",
                flush=True,
            )

        destination = _analysis_path(args.results, repository)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
        print(f"saved {destination}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results", type=Path, default=DEFAULT_AUDIT_RESULTS)
    parser.add_argument("--primary-results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--readout-results",
        type=Path,
        default=ROOT / "results/readout_audit",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Frozen repository ID; repeat for multiple models (default: all).",
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument(
        "--c-grid", type=float, nargs="+", default=list(DEFAULT_C_GRID)
    )
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
    parser.set_defaults(func=analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
