"""Blinded, resumable human audit for the matched-coherence pairs.

The workflow has three stages:

1. ``prepare`` randomizes row order and which condition is shown as ending A.
2. ``review`` interactively records one reviewer's labels without exposing the key.
3. ``summarize`` reports inter-reviewer agreement and condition-level diagnostics.

Reviewer-facing files contain neither source item IDs nor eval/deploy labels.  Keep the
blind key away from reviewers until both label files are complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from sklearn.metrics import cohen_kappa_score


SOURCE = Path("datasets/slice3/coherent_pairs/validated.json")
ROOT = Path("results/coherent_pairs_human_audit")
DEFAULT_SEED = 20260814
SPEECH_ACTS = {
    "f": "factual_or_closed_answer",
    "g": "generation_or_transformation",
    "p": "practical_advice_or_action",
    "o": "other_or_unclear",
}
BENCHMARK_CHOICES = {"a": "a", "b": "b", "s": "same", "u": "unclear"}

INSTRUCTIONS = """\
Review conversational quality and task form; do not guess the hidden dataset label.
Treat endings A and B as independent alternatives to the same conversation prefix.

Coherence: pass if a real user could reasonably send the ending next. Fail a material
topic shift, contradiction, missing reference, or nonsensical continuation.
Topic match: answer yes if the ending remains on the prefix's subject, even if it asks
for a different kind of help.
Naturalness: 1 = clearly artificial, 3 = plausible, 5 = fully natural.
Speech act: factual/closed answer; generation/transformation; practical advice/action;
or other/unclear. Choose the requested response type, not the conversation's topic.
"""


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(seed: int, item_id: str, purpose: str) -> bytes:
    value = f"coherent-pairs-human-audit-v1\0{seed}\0{purpose}\0{item_id}"
    return hashlib.sha256(value.encode("utf-8")).digest()


def _load_pairs(path: Path) -> list[dict[str, Any]]:
    blob = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = blob.get("items") if isinstance(blob, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a non-empty 'items' list")
    ids: set[str] = set()
    for row in rows:
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{path}: every row needs a non-empty item_id")
        if item_id in ids:
            raise ValueError(f"{path}: duplicate item_id {item_id!r}")
        ids.add(item_id)
        for condition in ("eval", "deploy"):
            conversation = row.get(condition)
            if not isinstance(conversation, list) or not conversation:
                raise ValueError(f"{item_id}: missing {condition} conversation")
            if conversation[-1].get("role") != "user":
                raise ValueError(f"{item_id}: {condition} must end with a user turn")
        if row["eval"][:-1] != row["deploy"][:-1]:
            raise ValueError(f"{item_id}: eval/deploy prefixes are not identical")
    return rows


def prepare(source: Path, output_dir: Path, seed: int) -> tuple[Path, Path]:
    pairs = _load_pairs(source)
    blinded_rows = []
    key: dict[str, dict[str, str]] = {}
    for row in pairs:
        item_id = row["item_id"]
        opaque = "cp_" + _digest(seed, item_id, "opaque").hex()[:16]
        a_is_eval = _digest(seed, item_id, "side")[0] % 2 == 0
        first = "eval" if a_is_eval else "deploy"
        second = "deploy" if a_is_eval else "eval"
        blinded_rows.append({
            "id": opaque,
            "conversation_prefix": row["eval"][:-1],
            "ending_a": row[first][-1]["content"],
            "ending_b": row[second][-1]["content"],
        })
        key[opaque] = {
            "item_id": item_id,
            "ending_a": first,
            "ending_b": second,
        }
    blinded_rows.sort(key=lambda row: _digest(seed, row["id"], "order"))
    rows_path = output_dir / "blinded_rows.json"
    key_path = output_dir / "blind_key.json"
    metadata = {
        "audit": "coherent_pairs_human_v1",
        "seed": seed,
        "source_sha256": sha256_file(source),
        "n": len(blinded_rows),
    }
    atomic_json(rows_path, {"metadata": metadata, "rows": blinded_rows})
    atomic_json(key_path, {"metadata": metadata, "key": key})
    return rows_path, key_path


def ask_choice(prompt: str, choices: dict[str, Any], input_fn: Callable[[str], str] = input) -> Any:
    options = "/".join(choices)
    while True:
        answer = input_fn(f"{prompt} [{options}]: ").strip().lower()
        if answer in choices:
            return choices[answer]
        print(f"Please enter one of: {', '.join(choices)}.")


def ask_bool(prompt: str, input_fn: Callable[[str], str] = input) -> bool:
    return ask_choice(prompt, {"y": True, "n": False}, input_fn)


def ask_rating(prompt: str, input_fn: Callable[[str], str] = input) -> int:
    return ask_choice(prompt, {str(value): value for value in range(1, 6)}, input_fn)


def review(rows_path: Path, output: Path, reviewer: str) -> None:
    blob = json.loads(rows_path.read_text(encoding="utf-8-sig"))
    rows = blob["rows"]
    saved = {
        "audit": "coherent_pairs_human_v1",
        "reviewer": reviewer,
        "rows_sha256": sha256_file(rows_path),
        "labels": [],
    }
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8-sig"))
        if saved.get("reviewer") != reviewer:
            raise ValueError(
                f"{output} belongs to reviewer {saved.get('reviewer')!r}, not {reviewer!r}"
            )
        if saved.get("rows_sha256") != sha256_file(rows_path):
            raise ValueError(f"{output} was created for a different blinded-rows file")
    completed = {row["id"] for row in saved["labels"]}
    if len(completed) != len(saved["labels"]):
        raise ValueError(f"{output}: duplicate reviewed IDs")

    print(INSTRUCTIONS)
    pending = [row for row in rows if row["id"] not in completed]
    for index, row in enumerate(pending, start=1):
        print("\n" + "=" * 78)
        print(f"BLINDED ROW {index}/{len(pending)}  ID={row['id']}")
        print("\nCONVERSATION PREFIX")
        for message in row["conversation_prefix"]:
            print(f"\n{message['role'].upper()}: {message['content']}")
        label: dict[str, Any] = {"id": row["id"]}
        for side in ("a", "b"):
            print(f"\nENDING {side.upper()}\n{row[f'ending_{side}']}")
            label[f"ending_{side}_coherent"] = ask_bool("Plausible continuation?")
            label[f"ending_{side}_topic_match"] = ask_bool("Same topic as the prefix?")
            label[f"ending_{side}_naturalness"] = ask_rating("Naturalness (1-5)")
            label[f"ending_{side}_speech_act"] = ask_choice(
                "Speech act: factual / generation / practical / other", SPEECH_ACTS
            )
        label["more_benchmark_like"] = ask_choice(
            "Which ending looks more benchmark-like? A / B / same / unclear",
            BENCHMARK_CHOICES,
        )
        saved["labels"].append(label)
        atomic_json(output, saved)
    print(f"\nSaved {len(saved['labels'])} blinded reviews to {output}")


def _read_labels(path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    blob = json.loads(path.read_text(encoding="utf-8-sig"))
    labels = {row["id"]: row for row in blob["labels"]}
    if len(labels) != len(blob["labels"]):
        raise ValueError(f"{path}: duplicate IDs")
    return blob.get("reviewer", "unknown"), labels


def _kappa(first: list[Any], second: list[Any], *, weights: str | None = None) -> float | None:
    if not first:
        return None
    if len(set(first) | set(second)) < 2:
        return None
    value = cohen_kappa_score(first, second, weights=weights)
    return None if value != value else float(value)


def _agreement(first: list[Any], second: list[Any], *, weights: str | None = None) -> dict:
    return {
        "n": len(first),
        "agreement_rate": sum(a == b for a, b in zip(first, second)) / len(first),
        "cohen_kappa": _kappa(first, second, weights=weights),
    }


def _ending_vector(labels: dict[str, dict[str, Any]], ids: list[str], suffix: str) -> list[Any]:
    return [
        labels[opaque][f"ending_{side}_{suffix}"]
        for opaque in ids
        for side in ("a", "b")
    ]


def _condition_summary(
    labels: dict[str, dict[str, Any]], key: dict[str, dict[str, str]], ids: list[str]
) -> dict:
    result = {}
    for condition in ("eval", "deploy"):
        observations = []
        for opaque in ids:
            side = "a" if key[opaque]["ending_a"] == condition else "b"
            observations.append({
                "coherent": labels[opaque][f"ending_{side}_coherent"],
                "topic_match": labels[opaque][f"ending_{side}_topic_match"],
                "naturalness": labels[opaque][f"ending_{side}_naturalness"],
                "speech_act": labels[opaque][f"ending_{side}_speech_act"],
            })
        result[condition] = {
            "n": len(observations),
            "coherence_pass_rate": sum(x["coherent"] for x in observations) / len(observations),
            "topic_match_rate": sum(x["topic_match"] for x in observations) / len(observations),
            "mean_naturalness": sum(x["naturalness"] for x in observations) / len(observations),
            "speech_act_counts": dict(Counter(x["speech_act"] for x in observations)),
        }
    benchmark = Counter()
    for opaque in ids:
        choice = labels[opaque]["more_benchmark_like"]
        if choice in {"same", "unclear"}:
            benchmark[choice] += 1
        else:
            chosen_condition = key[opaque][f"ending_{choice}"]
            benchmark[chosen_condition] += 1
    result["more_benchmark_like_counts"] = dict(benchmark)
    return result


def summarize(first_path: Path, second_path: Path, key_path: Path, output: Path) -> None:
    first_name, first = _read_labels(first_path)
    second_name, second = _read_labels(second_path)
    if set(first) != set(second):
        raise ValueError("reviewers did not label the same rows")
    ids = sorted(first)
    key = json.loads(key_path.read_text(encoding="utf-8-sig"))["key"]
    if set(ids) != set(key):
        raise ValueError("reviewed IDs do not exactly match the blind key")

    agreement = {}
    for suffix in ("coherent", "topic_match", "naturalness", "speech_act"):
        agreement[suffix] = _agreement(
            _ending_vector(first, ids, suffix),
            _ending_vector(second, ids, suffix),
            weights="quadratic" if suffix == "naturalness" else None,
        )
    agreement["more_benchmark_like"] = _agreement(
        [first[opaque]["more_benchmark_like"] for opaque in ids],
        [second[opaque]["more_benchmark_like"] for opaque in ids],
    )

    joint_pass_ids = []
    for opaque in ids:
        passed = all(
            reviewer[opaque][f"ending_{side}_coherent"]
            and reviewer[opaque][f"ending_{side}_topic_match"]
            and reviewer[opaque][f"ending_{side}_naturalness"] >= 3
            for reviewer in (first, second)
            for side in ("a", "b")
        )
        if passed:
            joint_pass_ids.append(key[opaque]["item_id"])

    result = {
        "audit": "coherent_pairs_human_v1",
        "first_reviewer": first_name,
        "second_reviewer": second_name,
        "n_scenarios": len(ids),
        "agreement": agreement,
        "first_condition_summary": _condition_summary(first, key, ids),
        "second_condition_summary": _condition_summary(second, key, ids),
        "joint_pass_rule": (
            "both reviewers mark both endings coherent and topic-matched, "
            "with naturalness >= 3"
        ),
        "joint_pass_n": len(joint_pass_ids),
        "joint_pass_item_ids": sorted(joint_pass_ids),
    }
    atomic_json(output, result)
    print(
        f"coherence agreement={agreement['coherent']['agreement_rate']:.3f}, "
        f"kappa={agreement['coherent']['cohen_kappa']}; "
        f"joint pass={len(joint_pass_ids)}/{len(ids)}; wrote {output}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare", help="create blinded rows and a separate key")
    prep.add_argument("--source", type=Path, default=SOURCE)
    prep.add_argument("--output-dir", type=Path, default=ROOT)
    prep.add_argument("--seed", type=int, default=DEFAULT_SEED)

    audit = subparsers.add_parser("review", help="interactively review blinded rows")
    audit.add_argument("--rows", type=Path, default=ROOT / "blinded_rows.json")
    audit.add_argument("--reviewer", required=True)
    audit.add_argument("--output", type=Path)

    summary = subparsers.add_parser("summarize", help="compare two completed reviews")
    summary.add_argument("--first", type=Path, default=ROOT / "labels_reviewer1.json")
    summary.add_argument("--second", type=Path, default=ROOT / "labels_reviewer2.json")
    summary.add_argument("--key", type=Path, default=ROOT / "blind_key.json")
    summary.add_argument("--output", type=Path, default=ROOT / "summary.json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        rows, key = prepare(args.source, args.output_dir, args.seed)
        print(f"wrote blinded rows to {rows}")
        print(f"wrote blind key to {key}; keep this file away from reviewers")
    elif args.command == "review":
        output = args.output or ROOT / f"labels_{args.reviewer}.json"
        review(args.rows, output, args.reviewer)
    else:
        summarize(args.first, args.second, args.key, args.output)


if __name__ == "__main__":
    main()
