"""Validate writer output for the paired matched-coherence pilot."""

from __future__ import annotations

import json
import re
from pathlib import Path

from vnaa.build_eval_shared import final_user_index

IN = Path("datasets/slice3/coherent_pair_pilot/input.json")
WRITTEN = Path("datasets/slice3/coherent_pair_pilot/output.json")
OUT = Path("datasets/slice3/coherent_pair_pilot/validated.json")
TEST_CUE = re.compile(r"\b(test|quiz|exam|assessment|answer key|multiple choice)\b", re.I)


def main() -> None:
    spec = json.load(open(IN, encoding="utf-8"))
    if WRITTEN.exists():
        rows = json.load(open(WRITTEN, encoding="utf-8"))["items"]
    else:
        parts = sorted(IN.parent.glob("part_*.json"))
        if not parts:
            raise SystemExit(f"missing {WRITTEN} or part_*.json")
        rows = [row for part in parts
                for row in json.load(open(part, encoding="utf-8"))["items"]]
    by_id = {r["item_id"]: r for r in rows}
    expected = {r["item_id"]: r for r in spec["items"]}
    if set(by_id) != set(expected):
        raise SystemExit("writer IDs do not exactly match pilot input")
    checked = []
    for item_id, want in expected.items():
        row = by_id[item_id]
        ev, de = row["eval"], row["deploy"]
        for name, conv in (("eval", ev), ("deploy", de)):
            if [m["role"] for m in conv] != want["roles"]:
                raise SystemExit(f"{item_id}/{name}: role sequence differs")
            if any(not isinstance(m.get("content"), str) or not m["content"].strip() for m in conv):
                raise SystemExit(f"{item_id}/{name}: empty message")
        k = final_user_index(ev)
        if ev[:k] != de[:k]:
            raise SystemExit(f"{item_id}: prefix is not byte-identical")
        if TEST_CUE.search(ev[k]["content"]):
            raise SystemExit(f"{item_id}: explicit evaluation framing")
        checked.append({"item_id": item_id, "eval": ev, "deploy": de})
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"items": checked}, f, ensure_ascii=False)
    print(f"validated {len(checked)} paired conversations -> {OUT}")


if __name__ == "__main__":
    main()
