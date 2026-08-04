"""Slice 3: build the writing batches for the multi-turn cells.

Three cells, all model-authored so authorship cannot carry the label (slice 2b
lesson 4 — the deployment conversations carry REAL assistant turns from
WildChat, so if we authored only the eval side the probe could read assistant
authorship instead of eval-ness):

  mt_deploy_rw       a rewrite of a real multi-turn conversation, same shape
  mt_eval_disguised  a conversation of the SAME shape whose final user turn
                     poses a benchmark question, reached through natural
                     build-up with no test framing
  mt_eval_obvious    the same benchmark question with explicit test framing,
                     which is the obvious cell of the turn-structure x disguise
                     2x2

Shape is matched per item, not per pool: every cell for a given batch item is
written to the same skeleton (turn count and per-turn target lengths taken from
the real conversation). Slice 2b nearly lost its headline to pool-level
matching that hid a 13x per-item gap.

Benchmark items are ARC and MMLU only. GSM8K is excluded from the start because
slice 2b showed digit density alone separates maths items from their
counterparts at 0.949, which made the probe look good for the wrong reason.

The committed batch files reference the real conversation by index into the
gitignored local sample; --scratch writes hydrated copies containing the text
for the writers, which must never be committed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from vnaa.build_2b import OUT_DIR as S2B_DIR
from vnaa.build_2b import read_jsonl
from vnaa.prep_2b_batches import is_malformed

OUT_DIR = Path("datasets/slice3")
BATCH_DIR = OUT_DIR / "batches"
KEEP_SOURCES = ("arc", "mmlu")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare slice-3 writing batches")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scratch", default=None,
                    help="also write hydrated copies (with real text) here")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    with open("data/slice3_skeletons.json", encoding="utf-8") as f:
        skeletons = json.load(f)
    with open("data/wildchat_multiturn.json", encoding="utf-8") as f:
        convs = json.load(f)
    assert len(skeletons) == len(convs)

    # Draw fresh items from the full benchmark pool rather than reusing the
    # slice-2b sample, for two reasons: that sample only has 200 well-formed
    # arc+mmlu items, so 250 skeletons would force repeats (repeated content
    # inflates within-cell similarity and lets a probe memorize rather than
    # generalize), and reusing them would make slice-3 results partly a retest
    # on slice-2b content.
    import pandas as pd
    from huggingface_hub import hf_hub_download

    from vnaa.data_2x2 import REPO

    path = hf_hub_download(REPO, "bench_eval_full/train-00000-of-00001.parquet",
                           repo_type="dataset")
    df = pd.read_parquet(path)
    df = df[df["source"].isin(KEEP_SOURCES)]
    used = {it["question"] for it in read_jsonl(S2B_DIR / "bench_eval_sample.jsonl")}

    def norm_choices(raw: str) -> list[str]:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = parsed["text"]
        return [str(t) for t in parsed]

    # A handful of stems carry their own answer inline ("...attributed to (A)
    # hydrogen bonding between..."). 1 in 2000, but it would put the answer in
    # the eval cell's final user turn, so drop them.
    leaks_answer = re.compile(r"\(\s*[A-E]\s*\)")

    items = []
    for r in df.itertuples():
        if r.question in used or leaks_answer.search(r.question):
            continue
        it = {"source": r.source, "question": r.question,
              "choices": norm_choices(r.choices)}
        if is_malformed(it):
            continue
        items.append(it)
    print(f"fresh arc+mmlu items (unused in slice 2b, well-formed): {len(items)}")
    if len(items) < len(skeletons):
        raise SystemExit(f"need {len(skeletons)} items, only {len(items)} available")
    order = rng.permutation(len(items))[: len(skeletons)]
    assigned = [items[i] for i in order]

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for old in BATCH_DIR.glob("batch_*.json"):
        old.unlink()

    n_batches = (len(skeletons) + args.batch_size - 1) // args.batch_size
    scratch = Path(args.scratch) if args.scratch else None
    if scratch:
        scratch.mkdir(parents=True, exist_ok=True)

    for b in range(n_batches):
        lo = b * args.batch_size
        hi = min(lo + args.batch_size, len(skeletons))
        spec_items = []
        for k in range(lo, hi):
            sk, it = skeletons[k], assigned[k]
            spec_items.append({
                "item_id": sk["skeleton_id"],
                "conversation_index": k,
                "roles": sk["roles"],
                "target_chars": sk["target_chars"],
                "source": it["source"],
                "question": it["question"],
                "choices": it["choices"],
            })
        spec = {"batch": b, "items": spec_items}
        with open(BATCH_DIR / f"batch_{b:02d}.json", "w",
                  encoding="utf-8", newline="\n") as f:
            json.dump(spec, f, ensure_ascii=False, indent=1)

        if scratch:
            hydrated = json.loads(json.dumps(spec))
            for it in hydrated["items"]:
                it["original_conversation"] = convs[it["conversation_index"]]
            with open(scratch / f"batch_{b:02d}.json", "w",
                      encoding="utf-8", newline="\n") as f:
                json.dump(hydrated, f, ensure_ascii=False, indent=1)
        print(f"batch {b:02d}: {len(spec_items)} items")

    turns = [sum(r == "user" for r in s["roles"]) for s in skeletons]
    print(f"\nuser turns per item: { {k: turns.count(k) for k in sorted(set(turns))} }")
    print(f"committed specs (no conversation text) -> {BATCH_DIR}")
    if scratch:
        print(f"hydrated specs for writers (DO NOT COMMIT) -> {scratch}")


if __name__ == "__main__":
    main()
