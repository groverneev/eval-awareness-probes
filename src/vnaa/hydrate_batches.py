"""Write writer-ready batch files containing the real WildChat prompt text.

The committed batch files carry `wildchat_index` instead of the prompt text,
because WildChat-1M is gated (AI2 ImpACT) and the entries are real user
messages — neither belongs in a public repo. This script rehydrates them into
a scratch directory for the rewriting pass. That directory must never be
committed; the default lives outside the repo.

Anyone with WildChat access can reproduce the exact inputs by running
prep_2b_batches.py and then this, because the pool filtering and matching are
deterministic given the seed.

    python -m vnaa.hydrate_batches --out <scratch dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vnaa.data_2x2 import load_cell  # noqa: F401  (kept for symmetry of imports)
from vnaa.prep_2b_batches import BATCH_DIR, is_question_form, looks_like_quiz


def load_pool(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        pool = json.load(f)
    pool = [t for t in pool if 40 < len(t) < 600]
    return [t for t in pool if not looks_like_quiz(t) and is_question_form(t)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Rehydrate batch files for writers")
    ap.add_argument("--wildchat", default="data/wildchat_first_turns.json")
    ap.add_argument("--out", required=True, help="scratch dir; must not be committed")
    args = ap.parse_args()

    pool = load_pool(Path(args.wildchat))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in sorted(BATCH_DIR.glob("batch_*.json")):
        with open(p, encoding="utf-8") as f:
            spec = json.load(f)
        for it in spec["deploy_items"]:
            if "wildchat_index" not in it:
                raise SystemExit(f"{p.name}: no wildchat_index; regenerate batches")
            it["original"] = pool[it["wildchat_index"]]
        dest = out_dir / p.name
        with open(dest, "w", encoding="utf-8", newline="\n") as f:
            json.dump(spec, f, ensure_ascii=False, indent=1)
        print(f"wrote {dest}")

    print("\nPoint the writers at this directory, not at datasets/.")


if __name__ == "__main__":
    main()
