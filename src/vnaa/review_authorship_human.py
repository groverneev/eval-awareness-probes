"""Interactive blinded review of the prespecified 20% coherence sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


INPUT = Path("results/authorship/v2/coherence/human_review_rows.json")
OUTPUT = Path("results/authorship/v2/coherence/human_labels.json")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def ask(label: str) -> bool:
    while True:
        answer = input(f"Is ending {label} a plausible continuation? [y/n]: ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please enter y or n.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))["rows"]
    saved = {"reviewer": "human", "labels": []}
    if args.output.exists():
        saved = json.loads(args.output.read_text(encoding="utf-8"))
    completed = {row["id"] for row in saved["labels"]}
    pending = [row for row in rows if row["id"] not in completed]
    for index, row in enumerate(pending, start=1):
        print("\n" + "=" * 78)
        print(f"BLINDED ROW {index}/{len(pending)}  ID={row['id']}")
        print("\nCONVERSATION PREFIX")
        for message in row["conversation_prefix"]:
            print(f"\n{message['role'].upper()}: {message['content']}")
        print(f"\nENDING A\n{row['ending_a']}")
        pass_a = ask("A")
        print(f"\nENDING B\n{row['ending_b']}")
        pass_b = ask("B")
        saved["labels"].append({"id": row["id"], "ending_a_pass": pass_a,
                                "ending_b_pass": pass_b})
        atomic_json(args.output, saved)
    print(f"\nSaved {len(saved['labels'])} blinded human labels to {args.output}")


if __name__ == "__main__":
    main()
