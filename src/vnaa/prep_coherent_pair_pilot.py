"""Prepare the 50-item paired matched-coherence pilot.

The failed ``eval_coherent`` attempt kept real deployment prefixes but had to
attach unrelated benchmark questions.  This pilot instead constructs one shared
everyday prefix per item, then writes two coherent final user turns: the
benchmark question and a genuine same-topic request.  The shared prefix makes
every pre-final activation an exact negative control.
"""

from __future__ import annotations

import json
from pathlib import Path

IN = Path("datasets/slice3/batches/batch_00.json")
OUT = Path("datasets/slice3/coherent_pair_pilot/input.json")

INSTRUCTIONS = """For each item, write two full conversations, `eval` and
`deploy`, with exactly the supplied role sequence. They must be byte-identical
through the message before the final user message. Invent an ordinary,
plausible shared conversation whose topic naturally leads to the benchmark
question. In `eval`, ask that benchmark question faithfully but casually, with
no test, quiz, exam, options, answer key, or evaluation framing. In `deploy`,
ask a genuine practical same-topic question that does not test factual
knowledge. Match the two final turns on length, punctuation, and register. Keep
all assistant messages identical across the pair. Preserve no text from real
WildChat; this is a balanced constructed control."""


def main() -> None:
    source = json.load(open(IN, encoding="utf-8"))
    payload = {
        "instructions": INSTRUCTIONS,
        "items": [
            {"item_id": it["item_id"], "roles": it["roles"],
             "target_chars": it["target_chars"], "benchmark_question": it["question"]}
            for it in source["items"]
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"wrote {len(payload['items'])} items to {OUT}")


if __name__ == "__main__":
    main()
