"""Quantify how far the deploy rewrites drifted from their real originals.

The caveat this measures: several writers reported that some real WildChat
prompts were themselves single-answer factual questions (a physics problem, a
stats true/false), and reframed them as help-requests to satisfy the "deploy
items stay requests" rule. That is defensible, but it pushes the deploy cell
further along exactly the speech-act axis that the slice-2b gate identified as
the residual signal — so it could inflate our own eval-vs-deploy separation.

Two measurements:

1. Feature drift: speech-act markers in the ORIGINAL WildChat prompt vs in our
   rewrite of it, paired per item. A large shift means our rewriting, not real
   traffic, is carrying the distinction.
2. Held-out effect: restrict to the subset of deploy items whose originals were
   NOT factual questions (so no reframing was needed) and re-score the surface
   contrasts there. If separation on that subset matches the full cell, the
   reframing is not doing the work.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from vnaa.build_2b import OUT_DIR, read_jsonl
from vnaa.merge_2b import oof_scores
from vnaa.prep_2b_batches import BATCH_DIR

FIRST_PERSON = re.compile(r"\b(i|i'm|im|my|we|our|me|i've|i'd|us|mine)\b", re.I)
REQUEST_CUE = re.compile(
    r"\b(help me|walk me|show me|can you|could you|please|i need|i want|"
    r"how do i|how can i|explain to me|give me|make me|write me)\b", re.I
)
# A prompt that poses a question with one canonical answer, i.e. the kind a
# writer would have had to reframe.
FACTUAL_Q = re.compile(
    r"\b(what is the|which of|how many|how much|calculate|true or false|"
    r"what are the|define |name the|state the)\b", re.I
)


def markers(text: str) -> dict[str, float]:
    return {
        "first_person": float(bool(FIRST_PERSON.search(text))),
        "request_cue": float(bool(REQUEST_CUE.search(text))),
        "factual_q": float(bool(FACTUAL_Q.search(text))),
        "ends_q": float(text.rstrip().endswith("?")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Deploy rewrite drift check")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    originals: dict[str, str] = {}
    for p in sorted(BATCH_DIR.glob("batch_*.json")):
        with open(p, encoding="utf-8") as f:
            spec = json.load(f)
        for it in spec["deploy_items"]:
            originals[it["uid"]] = it["original"]

    rewrites = {r["uid"]: r["text"] for r in read_jsonl(OUT_DIR / "casual_deploy_wc.jsonl")}
    uids = [u for u in rewrites if u in originals]
    print(f"paired deploy items: {len(uids)}\n")

    print(f"{'marker':<15} {'original':>10} {'rewrite':>10} {'shift':>8}")
    for key in ("first_person", "request_cue", "factual_q", "ends_q"):
        o = np.mean([markers(originals[u])[key] for u in uids])
        r = np.mean([markers(rewrites[u])[key] for u in uids])
        print(f"{key:<15} {o:>10.2f} {r:>10.2f} {r - o:>+8.2f}")

    # Items whose originals were already request-shaped need no reframing.
    clean = [u for u in uids if not FACTUAL_Q.search(originals[u])]
    reframed = [u for u in uids if FACTUAL_Q.search(originals[u])]
    print(f"\noriginals that were factual questions (reframing needed): "
          f"{len(reframed)}/{len(uids)} = {len(reframed) / len(uids):.0%}")

    eval_a = [r["text"] for r in read_jsonl(OUT_DIR / "casual_eval_a.jsonl")]
    eval_b = [r["text"] for r in read_jsonl(OUT_DIR / "casual_eval_b.jsonl")]

    print("\nsurface separation on the FULL deploy cell vs the never-reframed subset")
    print(f"{'contrast':<34} {'format':>9} {'bow':>9}")
    for nm, ev in (("casual_eval_a", eval_a), ("casual_eval_b", eval_b)):
        for sub_nm, sub in (("full", uids), ("never-reframed", clean)):
            dep = [rewrites[u] for u in sub]
            texts = ev + dep
            y = np.array([1] * len(ev) + [0] * len(dep))
            f = roc_auc_score(y, oof_scores(texts, y, "format", args.seed))
            b = roc_auc_score(y, oof_scores(texts, y, "bow", args.seed))
            print(f"{nm + ' vs ' + sub_nm:<34} {f:>9.3f} {b:>9.3f}")


if __name__ == "__main__":
    main()
