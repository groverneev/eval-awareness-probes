"""Probe-blind gate for the slice-3 multi-turn cells.

Checks, in order of how badly each would corrupt the result:

1. Shape: every written conversation must have exactly the role sequence of its
   skeleton. A turn-count difference between cells would let the probe read
   conversation length instead of eval-ness, and would also shift every
   multi-turn read point.
2. Length parity ACROSS CELLS, both whole-conversation and final-user-turn.
   Cross-cell parity is what the surface baselines actually measure; adherence
   to the original skeleton's own numbers matters less, which is the tradeoff
   the pilot writer made deliberately.
3. Surface baselines on the serialized conversation and, separately, on the
   final user turn alone — that is where the eval/deploy difference lives, so
   it is where leakage would hide.
4. Test-framing leakage: explicit cues must be near-absent from the disguised
   cell while being common in the obvious cell, otherwise the two eval cells
   are not actually different conditions.

As in slice 2b, length and format are gated; bag-of-words is reported as the
bar the probe must beat, because eval and deploy content genuinely differ.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from vnaa.merge_2b import oof_scores
from vnaa.prep_slice3_batches import BATCH_DIR

CELLS = ("deploy", "eval_disguised", "eval_obvious")
GATE_MAX_STRENGTH = 0.10

TEST_CUE = re.compile(
    r"\b(test(ing)?|quiz(zing)?|assess(ing|ment)?|exam|score|grade|"
    r"question \d|how many questions|check your knowledge|see if you)\b", re.I
)


def serialize(conv: list[dict]) -> str:
    return " ".join(m["content"] for m in conv)


def final_user(conv: list[dict]) -> str:
    for m in reversed(conv):
        if m["role"] == "user":
            return m["content"]
    raise ValueError("no user turn")


def contrast(name: str, pos: list[str], neg: list[str], seed: int) -> bool:
    texts = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))
    rows = {
        "length (char)": np.array([len(t) for t in texts], dtype=float),
        "format (structure)": oof_scores(texts, y, "format", seed),
        "bag-of-words": oof_scores(texts, y, "bow", seed),
    }
    print(f"\n--- {name} (n={len(pos)} vs {len(neg)}) ---")
    ok = True
    for bname, scores in rows.items():
        a = roc_auc_score(y, scores)
        s = abs(a - 0.5)
        if bname.startswith("bag"):
            verdict = "  (reported, probe must beat this)"
        else:
            passed = s <= GATE_MAX_STRENGTH
            ok = ok and passed
            verdict = "  PASS" if passed else "  FAIL"
        print(f"{bname:<20} {a:>7.4f}  strength {s:.4f}{verdict}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Slice-3 gate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    specs, outs = {}, {}
    for p in sorted(BATCH_DIR.glob("batch_*.json")):
        spec = json.load(open(p, encoding="utf-8"))
        op = BATCH_DIR / f"out_{spec['batch']:02d}.json"
        if not op.exists():
            continue
        specs[spec["batch"]] = {it["item_id"]: it for it in spec["items"]}
        outs[spec["batch"]] = json.load(open(op, encoding="utf-8"))

    if not outs:
        raise SystemExit("no out_*.json found")

    cells: dict[str, list[list[dict]]] = {c: [] for c in CELLS}
    shape_errors = []
    for b, out in outs.items():
        for it in out["items"]:
            want = specs[b][it["item_id"]]["roles"]
            for c in CELLS:
                conv = it[c]
                got = [m["role"] for m in conv]
                if got != want:
                    shape_errors.append((it["item_id"], c, got, want))
                cells[c].append(conv)

    n = len(cells["deploy"])
    print(f"items: {n}   shape errors: {len(shape_errors)}")
    for e in shape_errors[:5]:
        print("   ", e)

    print(f"\n{'cell':<16} {'conv chars':>12} {'final-turn chars':>18} {'turns':>7}")
    for c in CELLS:
        conv_len = np.mean([len(serialize(v)) for v in cells[c]])
        fin_len = np.mean([len(final_user(v)) for v in cells[c]])
        turns = np.mean([len(v) for v in cells[c]])
        print(f"{c:<16} {conv_len:>12.1f} {fin_len:>18.1f} {turns:>7.2f}")

    cue = {c: np.mean([bool(TEST_CUE.search(serialize(v))) for v in cells[c]])
           for c in CELLS}
    print("\nexplicit test-framing cue present:")
    for c in CELLS:
        print(f"  {c:<16} {cue[c]:.2f}")
    framing_ok = cue["eval_disguised"] <= 0.10 and cue["eval_obvious"] >= 0.70
    print(f"  framing separation: {'PASS' if framing_ok else 'FAIL'} "
          f"(need disguised <= 0.10, obvious >= 0.70)")

    ok = True
    for view, fn in (("whole conversation", serialize), ("final user turn", final_user)):
        print(f"\n================ {view} ================")
        ok &= contrast(f"eval_disguised vs deploy [{view}]",
                       [fn(v) for v in cells["eval_disguised"]],
                       [fn(v) for v in cells["deploy"]], args.seed)
        contrast(f"eval_obvious vs deploy [{view}] (obvious cell, not gated)",
                 [fn(v) for v in cells["eval_obvious"]],
                 [fn(v) for v in cells["deploy"]], args.seed)

    print(f"\nGATE {'PASSED' if ok and framing_ok and not shape_errors else 'NOT PASSED'}")


if __name__ == "__main__":
    main()
