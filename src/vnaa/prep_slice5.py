"""Sample genuine assistant-ending conversations for Slice 5 reconstruction.

The samples and regenerated answers are gitignored because WildChat contains
real user conversations.  Only deterministic code and aggregate results belong
in the repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vnaa.prep_2b_batches import is_question_form
from vnaa.prep_slice3 import (
    MAX_TURN_CHARS,
    MIN_TURN_CHARS,
    MIN_USER_TURNS,
    MAX_USER_TURNS,
    looks_like_coursework,
)


# The proposal names both WildChat and LMSYS-Chat-1M as deployment sources.
# Only WildChat had ever been used, so slice 5 rested on one corpus.
SOURCES = {
    "wildchat": {"repo": "allenai/WildChat-1M", "language_field": "message"},
    "lmsys": {"repo": "lmsys/lmsys-chat-1m", "language_field": "record"},
}


def normalize(record: dict, source: str) -> list[dict] | None:
    """Source-specific gates, before the shared shape checks.

    The two corpora differ in where the language tag lives and in redaction.
    LMSYS replaces detected PII with placeholders such as ``NAME_1``; those
    tokens are a surface artifact that a probe or a bag-of-words baseline could
    read as a label, so redacted conversations are dropped rather than
    normalized.
    """
    spec = SOURCES[source]
    if spec["language_field"] == "record":
        if record.get("language") != "English":
            return None
        if record.get("redacted"):
            return None
    return record.get("conversation") or []


def usable(conv: list[dict]) -> list[dict] | None:
    """Normalize a natural conversation that ends with an assistant turn."""
    messages = []
    for message in conv:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            return None
        if message.get("language") not in (None, "English"):
            return None
        if not MIN_TURN_CHARS <= len(content) <= MAX_TURN_CHARS:
            return None
        messages.append({"role": role, "content": content})
    if not messages or messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
        return None
    if any(a["role"] == b["role"] for a, b in zip(messages, messages[1:])):
        return None
    n_user = sum(m["role"] == "user" for m in messages)
    if not MIN_USER_TURNS <= n_user <= MAX_USER_TURNS:
        return None
    if any(looks_like_coursework(m["content"]) for m in messages):
        return None
    # Keep the final prompt question-shaped.  This matches Slice 3 and avoids
    # turning imperative-vs-question form into the reconstruction label.
    if not is_question_form(messages[-2]["content"]):
        return None
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--scan-limit", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--source", choices=sorted(SOURCES), default="wildchat")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.out is None:
        stem = "slice5_genuine" if args.source == "wildchat" else f"slice5_genuine_{args.source}"
        args.out = Path(f"data/{stem}.json")

    from datasets import load_dataset

    pool = []
    scanned = 0
    for scanned, record in enumerate(
        load_dataset(SOURCES[args.source]["repo"], split="train", streaming=True), start=1
    ):
        if scanned > args.scan_limit or len(pool) >= args.n * 4:
            break
        raw = normalize(record, args.source)
        conversation = usable(raw) if raw else None
        if conversation is not None:
            pool.append(conversation)
    if len(pool) < args.n:
        raise SystemExit(f"only {len(pool)} usable conversations; need {args.n}")
    rng = np.random.default_rng(args.seed)
    chosen = [pool[i] for i in rng.permutation(len(pool))[:args.n]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(chosen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    turns = [sum(m["role"] == "user" for m in c) for c in chosen]
    print(f"source {args.source}; scanned {scanned}; "
          f"wrote {len(chosen)} conversations to {args.out}")
    print("user turns:", {k: turns.count(k) for k in sorted(set(turns))})


if __name__ == "__main__":
    main()
