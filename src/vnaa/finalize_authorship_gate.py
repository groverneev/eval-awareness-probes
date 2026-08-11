"""Validate and finalize the probe-blind authorship-pilot gate."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from vnaa.authorship_coherence import atomic_json
from vnaa.run_authorship_generation import GENERATORS


ROOT = Path("results/authorship/v2")


def validate_cells(cells: list[dict]) -> list[str]:
    expected_conditions = set(GENERATORS)
    by_item: dict[str, dict[tuple[str, str], dict]] = defaultdict(dict)
    for cell in cells:
        key = (cell["history_authorship"], cell["status"])
        if key in by_item[cell["item_id"]]:
            raise ValueError(f"duplicate cell {cell['item_id']} {key}")
        if [message["role"] for message in cell["conversation"]] != [
            "user", "assistant", "user"
        ]:
            raise ValueError(f"{cell['item_id']} {key}: invalid role sequence")
        if not cell["terminated_naturally"]:
            raise ValueError(f"{cell['item_id']} {key}: history did not terminate")
        by_item[cell["item_id"]][key] = cell

    expected_keys = {
        (condition, status)
        for condition in expected_conditions
        for status in ("eval", "deploy")
    }
    for item_id, item_cells in by_item.items():
        if set(item_cells) != expected_keys:
            raise ValueError(f"{item_id}: incomplete factorial cells")
        for condition in expected_conditions:
            eval_conv = item_cells[(condition, "eval")]["conversation"]
            deploy_conv = item_cells[(condition, "deploy")]["conversation"]
            if eval_conv[:-1] != deploy_conv[:-1]:
                raise ValueError(f"{item_id} {condition}: eval/deploy prefixes differ")
        eval_finals = {
            item_cells[(condition, "eval")]["conversation"][-1]["content"]
            for condition in expected_conditions
        }
        deploy_finals = {
            item_cells[(condition, "deploy")]["conversation"][-1]["content"]
            for condition in expected_conditions
        }
        if len(eval_finals) != 1 or len(deploy_finals) != 1:
            raise ValueError(f"{item_id}: final endings differ across authorship")
    return sorted(by_item)


def load_human_labels(path: Path, expected_ids: set[str]) -> dict[str, dict]:
    blob = json.loads(path.read_text(encoding="utf-8-sig"))
    labels = blob["labels"]
    by_id = {}
    for label in labels:
        if set(label) != {"id", "ending_a_pass", "ending_b_pass"}:
            raise ValueError(f"invalid human-label fields for {label.get('id')}")
        if not isinstance(label["ending_a_pass"], bool) or not isinstance(
            label["ending_b_pass"], bool
        ):
            raise ValueError(f"non-boolean human label for {label['id']}")
        if label["id"] in by_id:
            raise ValueError(f"duplicate human label for {label['id']}")
        by_id[label["id"]] = label
    if set(by_id) != expected_ids:
        raise ValueError("human labels do not exactly cover the frozen review sample")
    return by_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--allow-pending-human", action="store_true")
    args = parser.parse_args()
    coherence = args.root / "coherence"

    cells_blob = json.loads((args.root / "pilot_cells.json").read_text(encoding="utf-8"))
    item_ids = validate_cells(cells_blob["cells"])
    if len(item_ids) != 50 or len(cells_blob["cells"]) != 300:
        raise ValueError("expected exactly 50 scenarios and 300 factorial cells")

    key = json.loads((coherence / "blind_key.json").read_text(encoding="utf-8"))["key"]
    model_blob = json.loads(
        (coherence / "judge_labels_merged.json").read_text(encoding="utf-8")
    )
    model_labels = {label["id"]: label for label in model_blob["labels"]}
    if set(model_labels) != set(key) or len(model_labels) != 150:
        raise ValueError("model labels do not cover the 150 blinded rows")

    human_rows = json.loads(
        (coherence / "human_review_rows.json").read_text(encoding="utf-8")
    )["rows"]
    expected_human = {row["id"] for row in human_rows}
    if len(expected_human) != 30:
        raise ValueError("expected exactly 30 frozen human-review rows")
    human_path = coherence / "human_labels.json"
    human = None
    if human_path.exists():
        human = load_human_labels(human_path, expected_human)
    elif not args.allow_pending_human:
        raise ValueError(
            "human review is pending; run `python -m vnaa.review_authorship_human` first"
        )

    failed_by_item: dict[str, list[dict]] = defaultdict(list)
    model_rows_passing = 0
    for opaque, label in model_labels.items():
        model_pass = label["ending_a_pass"] and label["ending_b_pass"]
        model_rows_passing += int(model_pass)
        human_pass = None
        if human is not None and opaque in human:
            human_pass = (
                human[opaque]["ending_a_pass"] and human[opaque]["ending_b_pass"]
            )
        row_pass = model_pass and human_pass is not False
        if not row_pass:
            decoded = key[opaque]
            failed_by_item[decoded["item_id"]].append({
                "id": opaque,
                "history_authorship": decoded["history_authorship"],
                "model_ending_a_pass": label["ending_a_pass"],
                "model_ending_b_pass": label["ending_b_pass"],
                "human_ending_a_pass": None if human is None or opaque not in human
                else human[opaque]["ending_a_pass"],
                "human_ending_b_pass": None if human is None or opaque not in human
                else human[opaque]["ending_b_pass"],
            })

    retained = [item_id for item_id in item_ids if item_id not in failed_by_item]
    status = "pass" if human is not None and len(retained) >= 40 else (
        "fail" if human is not None else "pending_human_review"
    )
    result = {
        "status": status,
        "mechanical_gate": {
            "passed": True,
            "n_scenarios": len(item_ids),
            "n_cells": len(cells_blob["cells"]),
            "all_histories_terminated_naturally": True,
            "role_sequence": ["user", "assistant", "user"],
            "paired_prefixes_identical": True,
            "endings_identical_across_authorship": True,
        },
        "coherence_gate": {
            "model_rows_passing_both": model_rows_passing,
            "model_rows_total": len(model_labels),
            "human_rows_complete": 0 if human is None else len(human),
            "human_rows_required": len(expected_human),
        },
        "retained_scenarios": retained,
        "n_retained": len(retained),
        "pass_threshold": 40,
        "excluded": dict(sorted(failed_by_item.items())),
        "generation_metadata": cells_blob["generation_metadata"],
        "model_judge_runs": model_blob["model_runs"],
    }
    output = coherence / (
        "gate_result.json" if human is not None else "gate_pending_human.json"
    )
    atomic_json(output, result)
    print(f"{status}: {len(retained)}/50 scenarios currently retained; wrote {output}")


if __name__ == "__main__":
    main()
