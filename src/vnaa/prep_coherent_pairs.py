"""Prepare all 250 items for the paired matched-coherence control."""

from __future__ import annotations

import json
from pathlib import Path

from vnaa.prep_coherent_pair_pilot import INSTRUCTIONS

OUT = Path("datasets/slice3/coherent_pairs/input.json")


def main() -> None:
    items = []
    for path in sorted(Path("datasets/slice3/batches").glob("batch_*.json")):
        batch = json.load(open(path, encoding="utf-8"))
        items.extend({"item_id": it["item_id"], "roles": it["roles"],
                      "target_chars": it["target_chars"],
                      "benchmark_question": it["question"]}
                     for it in batch["items"])
    if len(items) != 250 or len({it["item_id"] for it in items}) != len(items):
        raise SystemExit(f"expected 250 unique items, found {len(items)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"instructions": INSTRUCTIONS, "items": items}, f,
                  ensure_ascii=False, indent=1)
    print(f"wrote {len(items)} items to {OUT}")


if __name__ == "__main__":
    main()
