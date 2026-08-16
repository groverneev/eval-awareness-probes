"""Frozen-encoder contextual baseline for byte-identical-prefix controls.

Unlike the additive sparse baselines, this audit lets a text-only encoder see
the shared conversation prefix and the differing final user message together.
The protocol is frozen in ``datasets/contextual_text_baseline/prespec.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from vnaa.baselines import (
    bootstrap_auroc_difference_ci,
    paired_prediction_permutation_test,
)
from vnaa.cross_model_suite import PairedContrast, load_contrasts
from vnaa.run_strong_baseline_audit import (
    fit_selected_dom_grouped_cv,
    pair_folds,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/contextual_text_baseline"
MODEL_ID = "intfloat/e5-base-v2"
MODEL_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
MAX_LENGTH = 512
REPRESENTATIONS = (
    "full_context",
    "ending_only",
    "prefix_ending_interaction",
)


def render_messages(messages: list[dict]) -> str:
    """Serialize visible chat text without a target-model chat template."""
    role_names = {"user": "USER", "assistant": "ASSISTANT"}
    rendered = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in role_names or not isinstance(content, str) or not content.strip():
            raise ValueError("messages must contain nonempty user/assistant content")
        rendered.append(f"[{role_names[role]}]\n{content.strip()}")
    return "\n\n".join(rendered)


def contrast_text_views(contrast: PairedContrast) -> dict[str, list[str]]:
    """Return prefix, ending, and full text in positive-then-negative order."""
    prefix, ending, full = [], [], []
    for conversations in (contrast.positive, contrast.negative):
        for conversation in conversations:
            if len(conversation) < 2:
                raise ValueError("contextual baseline requires a nonempty prefix")
            prefix.append(render_messages(conversation[:-1]))
            ending.append(conversation[-1]["content"].strip())
            full.append(render_messages(conversation))
    return {"prefix": prefix, "ending": ending, "full": full}


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def encode_texts(
    texts: list[str],
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, dict]:
    """Encode unique texts with the immutable frozen E5 checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.truncation_side = "left"
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model.to(device)
    model.eval()

    unique = list(dict.fromkeys(texts))
    index = {text: position for position, text in enumerate(unique)}
    batches = []
    for start in tqdm(range(0, len(unique), batch_size), desc="E5 encoding"):
        batch_text = ["query: " + text for text in unique[start:start + batch_size]]
        tokens = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
            pooled = _mean_pool(hidden, tokens["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        batches.append(pooled.float().cpu().numpy())
    unique_embeddings = np.concatenate(batches)
    embeddings = unique_embeddings[[index[text] for text in texts]]
    metadata = {
        "model_id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "loaded_commit_hash": getattr(model.config, "_commit_hash", None),
        "max_length": MAX_LENGTH,
        "truncation_side": tokenizer.truncation_side,
        "pooling": "attention_mask_mean_then_l2_normalize",
        "input_prefix": "query: ",
    }
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return embeddings, metadata


def feature_matrices(
    prefix: np.ndarray, ending: np.ndarray, full: np.ndarray
) -> dict[str, np.ndarray]:
    if prefix.shape != ending.shape or prefix.shape != full.shape:
        raise ValueError("embedding views must have identical shapes")
    return {
        "full_context": full,
        "ending_only": ending,
        "prefix_ending_interaction": np.concatenate(
            [prefix, ending, np.abs(prefix - ending), prefix * ending], axis=1
        ),
    }


def grouped_embedding_predictions(
    features: np.ndarray,
    n_pairs: int,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """Fixed logistic model with scenario-grouped out-of-fold predictions."""
    if features.shape[0] != 2 * n_pairs:
        raise ValueError("expected positive-then-negative rows for every pair")
    labels = np.r_[np.ones(n_pairs), np.zeros(n_pairs)]
    predictions = np.empty(2 * n_pairs, dtype=float)
    for fold, (train_pairs, test_pairs) in enumerate(
        pair_folds(n_pairs, n_folds, seed)
    ):
        train_idx = np.r_[train_pairs, n_pairs + train_pairs]
        test_idx = np.r_[test_pairs, n_pairs + test_pairs]
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                solver="liblinear",
                max_iter=5000,
                random_state=seed + fold,
            ),
        )
        classifier.fit(features[train_idx], labels[train_idx])
        predictions[test_idx] = classifier.decision_function(features[test_idx])
    return predictions


def probe_predictions(pair, n_folds: int, seed: int) -> dict:
    """Recreate the reference probe's grouped OOF prediction vector."""
    positive, negative, acts_positive, acts_negative = pair
    n_pairs = len(positive)
    predictions = np.empty(2 * n_pairs, dtype=float)
    selected_layers = []
    selected_layer_aurocs = []
    for fold, (train_pairs, test_pairs) in enumerate(
        pair_folds(n_pairs, n_folds, seed)
    ):
        test_idx = np.r_[test_pairs, n_pairs + test_pairs]
        x_test = np.r_[acts_positive[test_pairs], acts_negative[test_pairs]]
        probe, layer, inner_aurocs = fit_selected_dom_grouped_cv(
            acts_positive[train_pairs],
            acts_negative[train_pairs],
            n_folds,
            seed + fold,
        )
        predictions[test_idx] = probe.project(x_test[:, layer])
        selected_layers.append(layer)
        selected_layer_aurocs.append(float(inner_aurocs[layer]))
    return {
        "predictions": predictions,
        "selected_layers": selected_layers,
        "layer_selection": {
            "method": "nested_pair_grouped_cv_auroc",
            "inner_folds": n_folds,
            "inner_seed_rule": "outer_seed_plus_fold_index",
            "selected_layer_inner_aurocs": selected_layer_aurocs,
        },
    }


def _fingerprint(contrasts: dict[str, PairedContrast]) -> str:
    payload = {
        name: {
            "ids": contrast.item_ids,
            "positive": contrast.positive,
            "negative": contrast.negative,
        }
        for name, contrast in contrasts.items()
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument(
        "--compare-probe",
        action="store_true",
        help="load reference-model activations for paired probe comparisons",
    )
    args = parser.parse_args()

    contrasts = load_contrasts(ROOT)
    all_views = {name: contrast_text_views(value) for name, value in contrasts.items()}
    flat_texts = []
    slices = {}
    for name, views in all_views.items():
        slices[name] = {}
        for view_name in ("prefix", "ending", "full"):
            start = len(flat_texts)
            flat_texts.extend(views[view_name])
            slices[name][view_name] = (start, len(flat_texts))

    embeddings, encoder_metadata = encode_texts(
        flat_texts, batch_size=args.batch_size, device=args.device
    )
    predictions_by_contrast = {}
    output = {
        "protocol": "datasets/contextual_text_baseline/prespec.md",
        "dataset_sha256": _fingerprint(contrasts),
        "encoder": encoder_metadata,
        "folds": args.folds,
        "seed": args.seed,
        "classifier": "StandardScaler + L2 logistic regression (C=1, liblinear)",
        "representations": {},
    }
    for name, contrast in contrasts.items():
        arrays = {
            view: embeddings[start:end]
            for view, (start, end) in slices[name].items()
        }
        feature_views = feature_matrices(
            arrays["prefix"], arrays["ending"], arrays["full"]
        )
        n_pairs = len(contrast.item_ids)
        labels = np.r_[np.ones(n_pairs), np.zeros(n_pairs)]
        predictions_by_contrast[name] = {}
        output["representations"][name] = {}
        print(f"\n{name} ({n_pairs} matched pairs)")
        for representation in REPRESENTATIONS:
            predictions = grouped_embedding_predictions(
                feature_views[representation], n_pairs, args.folds, args.seed
            )
            predictions_by_contrast[name][representation] = predictions
            auroc = float(roc_auc_score(labels, predictions))
            output["representations"][name][representation] = {"auroc": auroc}
            print(f"  {representation:<27} {auroc:.4f}")

    if args.compare_probe:
        from vnaa.run_slice3_transfer import load_pairs

        activation_pairs = load_pairs()
        output["probe_comparisons"] = {}
        for name, pair in activation_pairs.items():
            n_pairs = len(pair[0])
            labels = np.r_[np.ones(n_pairs), np.zeros(n_pairs)]
            clusters = np.r_[np.arange(n_pairs), np.arange(n_pairs)]
            probe_run = probe_predictions(pair, args.folds, args.seed)
            activation_predictions = probe_run["predictions"]
            result = {
                "probe_auroc": float(roc_auc_score(labels, activation_predictions)),
                "selected_layers": probe_run["selected_layers"],
                "layer_selection": probe_run["layer_selection"],
                "comparisons": {},
            }
            for representation in REPRESENTATIONS:
                contextual_predictions = predictions_by_contrast[name][representation]
                bootstrap = bootstrap_auroc_difference_ci(
                    labels,
                    activation_predictions,
                    contextual_predictions,
                    clusters,
                    n_boot=args.n_boot,
                    seed=args.seed,
                )
                permutation = paired_prediction_permutation_test(
                    labels,
                    activation_predictions,
                    contextual_predictions,
                    clusters,
                    n_permutations=args.n_permutations,
                    seed=args.seed,
                )
                contextual_auroc = float(
                    roc_auc_score(labels, contextual_predictions)
                )
                result["comparisons"][representation] = {
                    "analysis_status": (
                        "prespecified_primary"
                        if representation == "full_context"
                        else "prespecified_secondary"
                    ),
                    "contextual_auroc": contextual_auroc,
                    "probe_minus_contextual": {
                        "bootstrap": bootstrap,
                        "permutation": permutation,
                    },
                }
                print(
                    f"\nprobe comparison/{name}/{representation}: "
                    f"{result['probe_auroc']:.4f} - {contextual_auroc:.4f} = "
                    f"{bootstrap['difference']:+.4f}, "
                    f"95% CI [{bootstrap['ci_low']:+.4f}, "
                    f"{bootstrap['ci_high']:+.4f}], "
                    f"p={permutation['p_value']:.4f}"
                )
            output["probe_comparisons"][name] = result

    RESULTS.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS / "contextual_text_baseline.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nsaved {output_path}")


if __name__ == "__main__":
    main()
