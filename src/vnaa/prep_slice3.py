"""Slice 3: sample real multi-turn conversations and derive matching skeletons.

The deployment side of the multi-turn arm is real WildChat traffic. The
evaluation side will be authored against *skeletons* derived from these
conversations — turn count and per-turn target lengths — so that conversation
shape is matched per item rather than per pool. Slice 2b's headline was nearly
lost to pool-level matching that hid a 13x per-item gap, so shape matching here
is item-level by construction.

Filters, and why each one is here:

- English only, and at least 2 user turns, else it is not a multi-turn probe.
- Truncated to end on a user turn, because every read point is defined relative
  to the final user turn.
- No user turn that looks like a quiz item (`looks_like_quiz`): real traffic
  contains homework, and slice 2b showed those items are indistinguishable in
  content from benchmark evals, so they are label noise on the deploy side.
- Per-turn length caps, so a single 4000-character pasted document does not
  dominate a conversation the eval side then has to match.

Outputs (both gitignored; WildChat is gated and these are real user messages):
  data/wildchat_multiturn.json   the sampled conversations
  data/slice3_skeletons.json     turn counts + per-turn target lengths only

The skeleton file carries no prompt text, so it is what the eval-side authoring
pass consumes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import re

from vnaa.prep_2b_batches import looks_like_quiz

MIN_USER_TURNS = 2
MAX_USER_TURNS = 4
MIN_TURN_CHARS = 20
MAX_TURN_CHARS = 600

# `looks_like_quiz` needs a digit AND a quiz cue, which multi-turn coursework
# evades: the first conversations sampled here were people pasting
# multiple-choice questions ("which of the following has less effect on...",
# with the assistant replying "B) Use them as a springboard"). Those are
# evaluation items in content, sitting in the deployment cell — precisely the
# label noise slice 2b warned about, in a form the single-turn filter missed.
_COURSEWORK = re.compile(
    r"(which of the following|all of the following|which of these|"
    r"select the (correct|best)|choose the (correct|best)|true or false|"
    r"multiple choice|the correct answer|answer the question|"
    r"\b[a-e]\)\s|\([a-e]\)\s|^\s*[a-e][\.\)]\s)",
    re.I | re.M,
)


def looks_like_coursework(text: str) -> bool:
    """Quiz/homework content in either speaker's turn."""
    return bool(_COURSEWORK.search(text)) or looks_like_quiz(text)


def usable(conv: list[dict]) -> list[dict] | None:
    """Normalize one WildChat conversation, or None if it fails the filters."""
    msgs = []
    for m in conv:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            return None
        if m.get("language") not in (None, "English"):
            return None
        if not (MIN_TURN_CHARS <= len(content) <= MAX_TURN_CHARS):
            return None
        msgs.append({"role": role, "content": content})

    # Must alternate user/assistant starting with user; WildChat is chat data so
    # this is the norm, but a malformed one would break the span logic.
    if not msgs or msgs[0]["role"] != "user":
        return None
    for a, b in zip(msgs, msgs[1:]):
        if a["role"] == b["role"]:
            return None

    # Truncate to end on a user turn.
    while msgs and msgs[-1]["role"] != "user":
        msgs.pop()

    n_user = sum(m["role"] == "user" for m in msgs)
    if not (MIN_USER_TURNS <= n_user <= MAX_USER_TURNS):
        return None
    # Check BOTH speakers: an assistant turn answering "B) ..." is the clearest
    # evidence the user pasted a multiple-choice item, even when the user's own
    # wording looks innocuous.
    if any(looks_like_coursework(m["content"]) for m in msgs):
        return None
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample slice-3 multi-turn deploy data")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--scan-limit", type=int, default=60000,
                    help="max WildChat records to stream before giving up")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    kept: list[list[dict]] = []
    scanned = 0
    for rec in ds:
        scanned += 1
        if scanned > args.scan_limit or len(kept) >= args.n * 3:
            break
        conv = usable(rec.get("conversation") or [])
        if conv is not None:
            kept.append(conv)
    print(f"scanned {scanned} records, {len(kept)} passed filters")
    if len(kept) < args.n:
        raise SystemExit(f"only {len(kept)} usable conversations, need {args.n}")

    rng = np.random.default_rng(args.seed)
    # Sample evenly across turn counts so the eval side is not forced to match a
    # distribution dominated by the shortest conversations.
    by_turns: dict[int, list[list[dict]]] = {}
    for c in kept:
        by_turns.setdefault(sum(m["role"] == "user" for m in c), []).append(c)
    print("available by user-turn count:",
          {k: len(v) for k, v in sorted(by_turns.items())})

    chosen: list[list[dict]] = []
    per = args.n // len(by_turns)
    for k in sorted(by_turns):
        pool = by_turns[k]
        take = min(per, len(pool))
        chosen += [pool[i] for i in rng.permutation(len(pool))[:take]]
    # Top up from the largest remaining pool if rounding left us short.
    if len(chosen) < args.n:
        rest = [c for c in kept if c not in chosen]
        chosen += [rest[i] for i in rng.permutation(len(rest))[: args.n - len(chosen)]]
    chosen = chosen[: args.n]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "wildchat_multiturn.json", "w", encoding="utf-8") as f:
        json.dump(chosen, f, ensure_ascii=False)

    skeletons = []
    for i, conv in enumerate(chosen):
        skeletons.append({
            "skeleton_id": f"sk_{i:04d}",
            "roles": [m["role"] for m in conv],
            "target_chars": [len(m["content"]) for m in conv],
        })
    with open(out_dir / "slice3_skeletons.json", "w", encoding="utf-8") as f:
        json.dump(skeletons, f, indent=1)

    turns = [sum(m["role"] == "user" for m in c) for c in chosen]
    lens = [len(m["content"]) for c in chosen for m in c]
    print(f"\nwrote {len(chosen)} conversations to {out_dir}/wildchat_multiturn.json")
    print(f"user turns per conversation: "
          f"{ {k: turns.count(k) for k in sorted(set(turns))} }")
    print(f"message length: mean {np.mean(lens):.0f}, median {np.median(lens):.0f}")
    print(f"skeletons (no text) -> {out_dir}/slice3_skeletons.json")


if __name__ == "__main__":
    main()
