"""Prepare and merge blinded authorship-pilot coherence judgments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RUBRIC = Path("datasets/authorship/coherence_judge_v1.md")
ROOT = Path("results/authorship/v2/coherence")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_batches(args: argparse.Namespace) -> None:
    rows_path = args.root / "judge_rows.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))["rows"]
    batches = [[] for _ in range(args.n_batches)]
    for index, row in enumerate(rows):
        batches[index % args.n_batches].append(row)
    for index, batch in enumerate(batches):
        atomic_json(args.root / f"judge_batch_{index:02d}.json", {"rows": batch})
    atomic_json(args.root / "judge_batch_manifest.json", {
        "rubric": args.rubric.as_posix(),
        "rubric_sha256": sha256(args.rubric),
        "rows": rows_path.as_posix(),
        "rows_sha256": sha256(rows_path),
        "n_batches": args.n_batches,
        "batch_sizes": [len(batch) for batch in batches],
    })
    print(f"wrote {args.n_batches} blinded batches for {len(rows)} rows")


def read_label_array(path: Path) -> list[dict]:
    # Windows PowerShell's UTF-8 writer prepends a BOM; accept both forms.
    blob = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(blob, list):
        return blob
    # Claude Code --output-format json wraps the model text in `result`.
    if isinstance(blob, dict) and isinstance(blob.get("result"), str):
        result = blob["result"].strip()
        if result.startswith("```") and result.endswith("```"):
            result = result.split("\n", 1)[1].rsplit("\n", 1)[0]
        return json.loads(result)
    raise ValueError(f"{path} does not contain a label array or Claude JSON result")


def validate_labels(rows: list[dict], labels: list[dict], source: str) -> dict[str, dict]:
    expected = {row["id"] for row in rows}
    by_id = {}
    for label in labels:
        if set(label) != {"id", "ending_a_pass", "ending_b_pass", "reason"}:
            raise ValueError(f"{source}: invalid fields for {label.get('id')}")
        if not isinstance(label["ending_a_pass"], bool) or not isinstance(
            label["ending_b_pass"], bool
        ):
            raise ValueError(f"{source}: non-boolean label for {label['id']}")
        if label["id"] in by_id:
            raise ValueError(f"{source}: duplicate id {label['id']}")
        by_id[label["id"]] = label
    if set(by_id) != expected:
        missing = sorted(expected - set(by_id))
        extra = sorted(set(by_id) - expected)
        raise ValueError(f"{source}: id mismatch; missing={missing[:3]}, extra={extra[:3]}")
    return by_id


def merge(args: argparse.Namespace) -> None:
    rows = json.loads((args.root / "judge_rows.json").read_text(encoding="utf-8"))["rows"]
    batch_rows = [[] for _ in range(args.n_batches)]
    for index, row in enumerate(rows):
        batch_rows[index % args.n_batches].append(row)

    labels = {}
    model_runs = []
    for index in range(args.n_batches):
        path = args.root / f"judge_labels_{index:02d}.json"
        blob = json.loads(path.read_text(encoding="utf-8-sig"))
        batch_labels = validate_labels(
            batch_rows[index], read_label_array(path), path.name
        )
        labels.update(batch_labels)
        if isinstance(blob, dict):
            model_runs.append({
                "file": path.name,
                "model_usage": blob.get("modelUsage"),
                "duration_ms": blob.get("duration_ms"),
                "total_cost_usd": blob.get("total_cost_usd"),
            })

    atomic_json(args.root / "judge_labels_merged.json", {
        "rubric": args.rubric.as_posix(),
        "rubric_sha256": sha256(args.rubric),
        "model_runs": model_runs,
        "labels": [labels[row["id"]] for row in rows],
    })
    passed = sum(
        row["ending_a_pass"] and row["ending_b_pass"] for row in labels.values()
    )
    print(f"merged {len(labels)} labels; {passed}/{len(labels)} rows pass both endings")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "merge"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--rubric", type=Path, default=RUBRIC)
    parser.add_argument("--n-batches", type=int, default=5)
    args = parser.parse_args()
    if args.command == "build":
        build_batches(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
