"""Slice 5: exact-label deployment reconstruction on genuine conversations.

``reconstruct`` replaces each original final assistant reply with a greedy
reply.  Its generation context is either the complete prefix (the on-policy
and model-swap conditions) or just the final user message (the context-ablation
control).  ``extract`` always probes the complete original conversation with
the replacement inserted, so a context-ablation difference is not caused by
showing the probe a shorter conversation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from vnaa.extract import ASSISTANT_READ_POINTS, extract_activations, load_model
from vnaa.run_silence_filter import atomic_json, greedy_reply
from vnaa.run_slice3 import MODEL_TAG, RESULTS, differing_final_messages, score_pair

DATASET = Path("data/slice5_genuine.json")
RECONSTRUCTIONS = Path("data/slice5_reconstructed.json")
DEFAULT_TAG = "reconstruction_pilot50"


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_genuine(path: Path) -> list[list[dict]]:
    conversations = json.loads(path.read_text(encoding="utf-8"))
    if not conversations or any(c[-1]["role"] != "assistant" for c in conversations):
        raise ValueError("genuine dataset must contain assistant-ending conversations")
    return conversations


def generation_prefix(conversation: list[dict], context: str) -> list[dict]:
    """Return the messages available to the response generator.

    ``final_user_only`` removes every earlier turn *only while generating* the
    replacement.  The full prefix is restored before activation extraction.
    This isolates a reply's dependence on conversational context from the
    evaluator's access to that same context.
    """
    prefix = conversation[:-1]
    if context == "full":
        return prefix
    if context == "final_user_only":
        if not prefix or prefix[-1]["role"] != "user":
            raise ValueError("assistant-ending conversations must have a final user message")
        return [prefix[-1]]
    raise ValueError(f"unknown generation context {context!r}")


def reconstruct(args) -> None:
    genuine = load_genuine(args.dataset)
    expected = {"dataset_sha256": fingerprint(args.dataset), "model": args.model,
                "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
                "generation_context": args.generation_context, "items": []}
    if args.output.exists():
        saved = json.loads(args.output.read_text(encoding="utf-8"))
        if any(saved.get(k) != expected[k]
               for k in ("dataset_sha256", "model", "decoding", "generation_context")):
            raise ValueError("existing reconstruction cache belongs to different inputs/settings")
    else:
        saved = expected
    done = {item["id"] for item in saved["items"]}
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    lm = load_model(args.model, device=args.device, dtype=dtype, device_map=args.device_map)
    for index, conversation in enumerate(genuine):
        item_id = f"pair_{index:03d}"
        if item_id in done:
            continue
        reply, tokens, terminated = greedy_reply(
            lm, generation_prefix(conversation, args.generation_context), args.max_new_tokens
        )
        saved["items"].append({"id": item_id, "reply": reply, "generated_tokens": tokens,
                               "terminated": terminated})
        atomic_json(args.output, saved)
        print(f"saved {index + 1}/{len(genuine)}: {item_id}", flush=True)


def load_pairs(args) -> tuple[list[list[dict]], list[list[dict]], np.ndarray]:
    genuine = load_genuine(args.dataset)
    saved = json.loads(args.reconstructions.read_text(encoding="utf-8"))
    if saved.get("dataset_sha256") != fingerprint(args.dataset):
        raise ValueError("reconstructions were made from a different genuine dataset")
    if len(saved.get("items", [])) != len(genuine):
        raise ValueError("reconstruction cache is incomplete")
    rebuilt, keep = [], []
    for i, (conversation, item) in enumerate(zip(genuine, saved["items"])):
        if item.get("id") != f"pair_{i:03d}":
            raise ValueError("reconstruction cache order does not match dataset")
        keep.append(bool(item["terminated"]))
        rebuilt.append(conversation[:-1] + [{"role": "assistant", "content": item["reply"]}])
    return genuine, rebuilt, np.array(keep)


def acts_path(tag: str, kind: str, read_point: str) -> Path:
    return RESULTS / f"s5_{MODEL_TAG}_{tag}_{kind}_{read_point}.npz"


def shared_terminated_mask(*masks: np.ndarray) -> np.ndarray:
    """Intersection of the naturally terminated pairs across conditions."""
    if not masks:
        raise ValueError("need at least one termination mask")
    if any(mask.shape != masks[0].shape for mask in masks[1:]):
        raise ValueError("termination masks have different shapes")
    return np.logical_and.reduce(masks)


def extract(args) -> None:
    genuine, rebuilt, keep = load_pairs(args)
    RESULTS.mkdir(exist_ok=True)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    lm = load_model(args.model, device=args.device, dtype=dtype, device_map=args.device_map)
    for kind, conversations in (("genuine", genuine), ("resampled", rebuilt)):
        for read_point in ASSISTANT_READ_POINTS:
            output = acts_path(args.tag, kind, read_point)
            if output.exists():
                continue
            acts = extract_activations(lm, conversations, read_point=read_point, show_progress=True)
            np.savez_compressed(output, acts=acts)
            print(f"saved {output} {acts.shape}")


def analyze(args) -> None:
    genuine, rebuilt, keep = load_pairs(args)
    if keep.sum() < args.min_pairs:
        raise ValueError(f"only {keep.sum()} naturally terminated reconstructions; need {args.min_pairs}")
    print(f"naturally terminated pairs: {keep.sum()}/{len(keep)}")
    output = {
        "tag": args.tag,
        "genuine_activations_tag": args.genuine_tag or args.tag,
        "n_total_pairs": len(keep),
        "n_naturally_terminated_pairs": int(keep.sum()),
        "paired_surface_text_view": "differing_final_message_only",
        "read_points": {},
    }
    for read_point in ASSISTANT_READ_POINTS:
        original = np.load(
            acts_path(args.genuine_tag or args.tag, "genuine", read_point)
        )["acts"].astype(np.float64)[keep]
        resampled = np.load(acts_path(args.tag, "resampled", read_point))["acts"].astype(np.float64)[keep]
        rebuilt_text, genuine_text = differing_final_messages(rebuilt, genuine)
        result = score_pair(
            resampled,
            original,
            [t for t, k in zip(rebuilt_text, keep) if k],
            [t for t, k in zip(genuine_text, keep) if k],
            args.seeds,
            args.test_frac,
        )
        summary = {key: (float(np.mean(value)), float(np.std(value))) for key, value in result.items()}
        surfaces = {key: summary[key][0] for key in ("bow", "format", "length")}
        best_surface, surface_auroc = max(surfaces.items(), key=lambda item: abs(item[1] - .5))
        margin = abs(summary["probe"][0] - .5) - abs(surface_auroc - .5)
        print(f"{read_point}: probe {summary['probe'][0]:.3f}+/-{summary['probe'][1]:.3f} "
              f"BoW {summary['bow'][0]:.3f}+/-{summary['bow'][1]:.3f} "
              f"format {summary['format'][0]:.3f}+/-{summary['format'][1]:.3f} "
              f"best surface {best_surface} ({surface_auroc:.3f}); "
              f"probe-best strength {margin:+.3f}")
        output["read_points"][read_point] = {
            "summary": {key: {"mean": val[0], "sd": val[1]} for key, val in summary.items()},
            "best_surface": best_surface,
            "probe_minus_best_surface_strength": margin,
            "per_seed": result,
        }
    destination = RESULTS / f"s5_{MODEL_TAG}_{args.tag}_analysis.json"
    atomic_json(destination, output)
    print(f"saved {destination}")


def compare(args) -> None:
    """Score two reconstruction conditions on their exact shared pair subset."""
    left_args = argparse.Namespace(dataset=args.dataset, reconstructions=args.left_reconstructions)
    right_args = argparse.Namespace(dataset=args.dataset, reconstructions=args.right_reconstructions)
    genuine, left_rebuilt, left_keep = load_pairs(left_args)
    right_genuine, right_rebuilt, right_keep = load_pairs(right_args)
    if genuine != right_genuine:
        raise ValueError("conditions were not built from the same genuine conversations")
    keep = shared_terminated_mask(left_keep, right_keep)
    if keep.sum() < args.min_pairs:
        raise ValueError(f"only {keep.sum()} shared naturally terminated pairs; need {args.min_pairs}")
    print(f"shared naturally terminated pairs: {keep.sum()}/{len(keep)}")
    output = {
        "left_tag": args.left_tag,
        "right_tag": args.right_tag,
        "n_total_pairs": len(keep),
        "n_shared_naturally_terminated_pairs": int(keep.sum()),
        "paired_surface_text_view": "differing_final_message_only",
        "read_points": {},
    }
    for read_point in ASSISTANT_READ_POINTS:
        original = np.load(acts_path(args.left_tag, "genuine", read_point))["acts"].astype(np.float64)[keep]
        conditions = {}
        for name, tag, rebuilt in (
            ("left", args.left_tag, left_rebuilt),
            ("right", args.right_tag, right_rebuilt),
        ):
            replacement = np.load(acts_path(tag, "resampled", read_point))["acts"].astype(np.float64)[keep]
            rebuilt_text, genuine_text = differing_final_messages(rebuilt, genuine)
            result = score_pair(
                replacement, original,
                [t for t, chosen in zip(rebuilt_text, keep) if chosen],
                [t for t, chosen in zip(genuine_text, keep) if chosen],
                args.seeds, args.test_frac,
            )
            summary = {key: (float(np.mean(value)), float(np.std(value))) for key, value in result.items()}
            surfaces = {key: summary[key][0] for key in ("bow", "format", "length")}
            best_surface, surface_auroc = max(surfaces.items(), key=lambda item: abs(item[1] - .5))
            margin = abs(summary["probe"][0] - .5) - abs(surface_auroc - .5)
            print(f"{read_point} {name}: probe {summary['probe'][0]:.3f}+/-{summary['probe'][1]:.3f} "
                  f"best surface {best_surface} ({surface_auroc:.3f}); margin {margin:+.3f}")
            conditions[name] = {
                "summary": {key: {"mean": val[0], "sd": val[1]} for key, val in summary.items()},
                "best_surface": best_surface,
                "probe_minus_best_surface_strength": margin,
                "per_seed": result,
            }
        output["read_points"][read_point] = conditions
    destination = RESULTS / f"s5_{MODEL_TAG}_{args.left_tag}_vs_{args.right_tag}_shared_analysis.json"
    atomic_json(destination, output)
    print(f"saved {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("reconstruct", "extract"):
        command = sub.add_parser(name)
        command.add_argument("--dataset", type=Path, default=DATASET)
        command.add_argument("--reconstructions", "--output", dest="output", type=Path,
                             default=RECONSTRUCTIONS)
        command.add_argument("--model", default=f"meta-llama/{MODEL_TAG}")
        command.add_argument("--tag", default=DEFAULT_TAG,
                             help="cache namespace; change it for a new frozen dataset")
        command.add_argument("--device", default=None)
        command.add_argument("--device-map", default=None)
        command.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
        command.add_argument("--max-new-tokens", type=int, default=256)
        command.add_argument(
            "--generation-context", choices=["full", "final_user_only"], default="full",
            help="messages available while generating a replacement; extraction always uses full context",
        )
    ana = sub.add_parser("analyze")
    ana.add_argument("--dataset", type=Path, default=DATASET)
    ana.add_argument("--reconstructions", type=Path, default=RECONSTRUCTIONS)
    ana.add_argument("--tag", default=DEFAULT_TAG,
                     help="cache namespace matching the extraction run")
    ana.add_argument(
        "--genuine-tag",
        default=None,
        help="tag holding shared genuine activations when this tag cached only replacements",
    )
    ana.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ana.add_argument("--test-frac", type=float, default=.3)
    ana.add_argument("--min-pairs", type=int, default=30)
    cmp = sub.add_parser("compare", help="score two conditions on their shared terminated pairs")
    cmp.add_argument("--dataset", type=Path, default=DATASET)
    cmp.add_argument("--left-reconstructions", type=Path, required=True)
    cmp.add_argument("--left-tag", required=True)
    cmp.add_argument("--right-reconstructions", type=Path, required=True)
    cmp.add_argument("--right-tag", required=True)
    cmp.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    cmp.add_argument("--test-frac", type=float, default=.3)
    cmp.add_argument("--min-pairs", type=int, default=30)
    args = parser.parse_args()
    if args.command == "reconstruct":
        reconstruct(args)
    elif args.command == "extract":
        args.reconstructions = args.output
        extract(args)
    elif args.command == "compare":
        compare(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
