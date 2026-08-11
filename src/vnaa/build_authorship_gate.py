"""Build probe-blind coherence rows and the six frozen pilot cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from vnaa.prep_authorship_factorial import sha256_file
from vnaa.run_authorship_generation import GENERATORS, load_frozen_inputs


GENERATION_DIR = Path("results/authorship/generations")
OUTPUT_DIR = Path("results/authorship")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def opaque_id(item_id: str, condition: str) -> str:
    value = f"authorship-coherence-v1\0{item_id}\0{condition}".encode()
    return "c_" + hashlib.sha256(value).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path,
        default=Path("datasets/slice3/coherent_pairs/validated.json"),
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("datasets/authorship/pilot_manifest.json"),
    )
    parser.add_argument("--generation-dir", type=Path, default=GENERATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    frozen = load_frozen_inputs(args.source, args.manifest)
    by_id = {row["item_id"]: row for row in frozen}
    generations: dict[str, dict[str, dict]] = {}
    metadata = {}
    for condition in GENERATORS:
        path = args.generation_dir / f"{condition}.json"
        blob = json.loads(path.read_text(encoding="utf-8"))
        if len(blob["items"]) != len(frozen):
            raise ValueError(f"{path} has {len(blob['items'])} items, expected {len(frozen)}")
        metadata[condition] = {**blob["metadata"], "file_sha256": sha256_file(path)}
        generations[condition] = {row["item_id"]: row for row in blob["items"]}

    cells = []
    judge_rows = []
    key = {}
    for item in frozen:
        item_id = item["item_id"]
        for condition in GENERATORS:
            generated = generations[condition][item_id]
            assistant = {"role": "assistant", "content": generated["response"]}
            prefix = item["generation_messages"] + [assistant]
            for status in ("eval", "deploy"):
                final = item[f"{status}_final"]
                cells.append({
                    "item_id": item_id,
                    "history_authorship": condition,
                    "status": status,
                    "conversation": prefix + [final],
                    "terminated_naturally": generated["terminated_naturally"],
                })

            opaque = opaque_id(item_id, condition)
            judge_rows.append({
                "id": opaque,
                "conversation_prefix": prefix,
                "ending_a": item["eval_final"]["content"],
                "ending_b": item["deploy_final"]["content"],
            })
            key[opaque] = {"item_id": item_id, "history_authorship": condition,
                           "ending_a": "eval", "ending_b": "deploy"}

    judge_rows.sort(key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())
    human_n = (len(judge_rows) + 4) // 5
    human_rows = judge_rows[:human_n]
    atomic_json(args.output_dir / "pilot_cells.json", {
        "generation_metadata": metadata, "cells": cells,
    })
    atomic_json(args.output_dir / "coherence" / "judge_rows.json", {"rows": judge_rows})
    atomic_json(args.output_dir / "coherence" / "blind_key.json", {"key": key})
    atomic_json(args.output_dir / "coherence" / "human_review_rows.json", {
        "selection": "first ceil(20%) rows by sha256(opaque id)",
        "n": human_n,
        "rows": human_rows,
    })
    print(f"wrote {len(cells)} cells, {len(judge_rows)} coherence rows, "
          f"{human_n} human-review rows")


if __name__ == "__main__":
    main()
