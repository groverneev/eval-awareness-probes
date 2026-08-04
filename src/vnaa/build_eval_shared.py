"""Build the `eval_shared` control cell by recombination.

Take each `deploy` conversation and replace ONLY its final user turn with that
item's `eval_disguised` final user turn. Everything before the final turn is
then byte-identical between the two cells, so:

- any read point before the final turn MUST score exactly chance. That is the
  sanity check the whole construction exists to provide, and it is what caught
  the slice-3 dataset failure: on the as-written cells the probe scored 0.994
  at `post_instruction_token`, a position that cannot see the eval question,
  because writers' invented lead-ins differed from real conversation openings.
- whatever separation remains at the final turn is attributable to that turn
  alone, not to the surrounding conversation.

Costs no writer time. The tradeoff is coherence: the eval question no longer
follows from the preceding turns, so these conversations are deliberately
topic-shifted. Real users do change subject, but a matched-coherence version is
the obvious next control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vnaa.prep_slice3_batches import BATCH_DIR, OUT_DIR


def final_user_index(conv: list[dict]) -> int:
    for i in range(len(conv) - 1, -1, -1):
        if conv[i]["role"] == "user":
            return i
    raise ValueError("conversation has no user turn")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the eval_shared control cell")
    ap.add_argument("--out", default=str(OUT_DIR / "eval_shared.json"))
    args = ap.parse_args()

    items = []
    for p in sorted(BATCH_DIR.glob("out_*.json")):
        for it in json.load(open(p, encoding="utf-8"))["items"]:
            deploy, disguised = it["deploy"], it["eval_disguised"]
            conv = [dict(m) for m in deploy]
            conv[final_user_index(conv)] = {
                "role": "user",
                "content": disguised[final_user_index(disguised)]["content"],
            }
            # The construction is only meaningful if everything before the final
            # turn really is identical; assert rather than hope.
            k = final_user_index(deploy)
            assert [m["content"] for m in conv[:k]] == [
                m["content"] for m in deploy[:k]
            ], f"{it['item_id']}: prefix diverged from deploy"
            assert len(conv) == len(deploy), f"{it['item_id']}: turn count changed"
            items.append({"item_id": it["item_id"], "eval_shared": conv})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"items": items}, f, ensure_ascii=False)
    print(f"wrote {len(items)} conversations to {out}")


if __name__ == "__main__":
    main()
