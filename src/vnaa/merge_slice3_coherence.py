"""Validate coherent-final-turn writer output and build ``eval_coherent``.

Writer output format per batch is ``{"batch": N, "items": [{"item_id": ...,
"final_user": ...}]}``.  The deploy prefix is never accepted from writer output;
it is always copied from the frozen slice-3 deploy cell.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from vnaa.build_eval_shared import final_user_index
from vnaa.prep_slice3_batches import BATCH_DIR

INPUT_DIR = Path("datasets/slice3/coherence_batches")
DEFAULT_OUT = Path("datasets/slice3/eval_coherent.json")
TEST_CUE = re.compile(r"\b(test|quiz|exam|assessment|answer key|multiple choice)\b", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    merged = []
    for source in sorted(BATCH_DIR.glob("out_*.json")):
        batch = int(source.stem.split("_")[1])
        reply_path = args.input_dir / source.name
        if not reply_path.exists():
            raise SystemExit(f"missing writer output: {reply_path}")
        deploy_items = {it["item_id"]: it["deploy"]
                        for it in json.load(open(source, encoding="utf-8"))["items"]}
        reply = json.load(open(reply_path, encoding="utf-8"))
        if reply.get("batch") != batch:
            raise SystemExit(f"{reply_path}: expected batch {batch}")
        got = {it["item_id"]: it.get("final_user", "") for it in reply["items"]}
        if set(got) != set(deploy_items):
            raise SystemExit(f"{reply_path}: item IDs do not exactly match frozen batch")
        for item_id, final_user in got.items():
            if not isinstance(final_user, str) or not final_user.strip():
                raise SystemExit(f"{item_id}: final_user is empty")
            if TEST_CUE.search(final_user):
                raise SystemExit(f"{item_id}: explicit evaluation cue in final_user")
            deploy = deploy_items[item_id]
            k = final_user_index(deploy)
            conv = [dict(message) for message in deploy]
            conv[k] = {"role": "user", "content": final_user.strip()}
            assert conv[:k] == deploy[:k], f"{item_id}: prefix changed"
            merged.append({"item_id": item_id, "eval_coherent": conv})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"items": merged}, f, ensure_ascii=False)
    print(f"wrote {len(merged)} coherent conversations to {args.out}")


if __name__ == "__main__":
    main()
