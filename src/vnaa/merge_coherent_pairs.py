"""Validate and merge the 250-item paired matched-coherence control."""

from __future__ import annotations

import json
from pathlib import Path

from vnaa.merge_coherent_pair_pilot import TEST_CUE
from vnaa.build_eval_shared import final_user_index

ROOT = Path("datasets/slice3/coherent_pairs")


def main() -> None:
    expected = {x["item_id"]: x for x in json.load(open(ROOT / "input.json", encoding="utf-8"))["items"]}
    parts = sorted(ROOT.glob("part_*.json"))
    if not parts:
        raise SystemExit("no writer parts")
    rows = [x for path in parts for x in json.load(open(path, encoding="utf-8"))["items"]]
    by_id = {x["item_id"]: x for x in rows}
    if len(rows) != len(by_id) or set(by_id) != set(expected):
        raise SystemExit("writer item IDs do not exactly match input")
    checked = []
    for item_id, spec in expected.items():
        row = by_id[item_id]
        ev, de = row["eval"], row["deploy"]
        for name, conv in (("eval", ev), ("deploy", de)):
            if [m["role"] for m in conv] != spec["roles"]:
                raise SystemExit(f"{item_id}/{name}: role sequence differs")
            if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in conv):
                raise SystemExit(f"{item_id}/{name}: empty message")
        k = final_user_index(ev)
        if ev[:k] != de[:k]:
            raise SystemExit(f"{item_id}: prefix is not byte-identical")
        if TEST_CUE.search(ev[k]["content"]):
            raise SystemExit(f"{item_id}: explicit evaluation framing")
        checked.append({"item_id": item_id, "eval": ev, "deploy": de})
    with open(ROOT / "validated.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"items": checked}, f, ensure_ascii=False)
    print(f"validated {len(checked)} paired conversations")


if __name__ == "__main__":
    main()
