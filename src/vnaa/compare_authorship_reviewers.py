"""Diagnose agreement among blinded human and model coherence reviewers.

This is an instrument diagnostic. It never changes the frozen v2 gate result.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from vnaa.authorship_coherence import atomic_json


ROOT = Path("results/authorship/v2/coherence")


def read_human(path: Path) -> tuple[str, dict[str, dict]]:
    blob = json.loads(path.read_text(encoding="utf-8-sig"))
    labels = {row["id"]: row for row in blob["labels"]}
    if len(labels) != len(blob["labels"]):
        raise ValueError(f"{path}: duplicate ids")
    return blob.get("reviewer", "unknown"), labels


def ending_vector(labels: dict[str, dict], ids: list[str]) -> list[bool]:
    vector = []
    for opaque in ids:
        vector.extend([
            labels[opaque]["ending_a_pass"],
            labels[opaque]["ending_b_pass"],
        ])
    return vector


def agreement(first: list[bool], second: list[bool]) -> dict:
    if len(first) != len(second):
        raise ValueError("label vectors have different lengths")
    confusion = Counter(zip(first, second))
    return {
        "n": len(first),
        "n_agree": sum(a == b for a, b in zip(first, second)),
        "agreement_rate": sum(a == b for a, b in zip(first, second)) / len(first),
        "cohen_kappa": float(cohen_kappa_score(first, second)),
        "first_pass_rate": sum(first) / len(first),
        "second_pass_rate": sum(second) / len(second),
        "confusion": {
            "first_pass_second_pass": confusion[(True, True)],
            "first_pass_second_fail": confusion[(True, False)],
            "first_fail_second_pass": confusion[(False, True)],
            "first_fail_second_fail": confusion[(False, False)],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, default=ROOT / "human_labels.json")
    parser.add_argument(
        "--second", type=Path, default=ROOT / "human_labels_reviewer2.json"
    )
    parser.add_argument(
        "--model", type=Path, default=ROOT / "judge_labels_merged.json"
    )
    parser.add_argument("--key", type=Path, default=ROOT / "blind_key.json")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reviewer_agreement_diagnostic.json"
    )
    args = parser.parse_args()

    first_name, first = read_human(args.first)
    second_name, second = read_human(args.second)
    if set(first) != set(second):
        raise ValueError("human reviewers did not label the same frozen rows")
    ids = sorted(first)
    model_blob = json.loads(args.model.read_text(encoding="utf-8"))
    model = {row["id"]: row for row in model_blob["labels"]}
    key = json.loads(args.key.read_text(encoding="utf-8"))["key"]

    first_vector = ending_vector(first, ids)
    second_vector = ending_vector(second, ids)
    model_vector = ending_vector(model, ids)
    row_disagreements = [
        opaque for opaque in ids
        if (
            first[opaque]["ending_a_pass"] != second[opaque]["ending_a_pass"]
            or first[opaque]["ending_b_pass"] != second[opaque]["ending_b_pass"]
        )
    ]
    result = {
        "status": "diagnostic_only_does_not_change_v2_gate",
        "first_reviewer": first_name,
        "second_reviewer": second_name,
        "human_human": agreement(first_vector, second_vector),
        "first_human_model": agreement(first_vector, model_vector),
        "second_human_model": agreement(second_vector, model_vector),
        "row_disagreements": row_disagreements,
        "n_row_disagreements": len(row_disagreements),
        "disagreements_by_history_authorship": dict(Counter(
            key[opaque]["history_authorship"] for opaque in row_disagreements
        )),
    }
    atomic_json(args.output, result)
    print(
        f"human-human agreement={result['human_human']['agreement_rate']:.3f}, "
        f"kappa={result['human_human']['cohen_kappa']:.3f}; wrote {args.output}"
    )


if __name__ == "__main__":
    main()
