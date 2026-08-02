"""Loader for Devbunova's public eval-awareness-2x2 dataset.

HF: viliana-dev/eval-awareness-2x2. Built on Llama-3.1-8B-Instruct, the same
backbone as ours, n=1076 per matched cell (835 for the rewritten cells, which
are capped by how many ShareGPT queries survived rewriting).

Cells and where their text lives (each parquet has a different schema, so the
plain load_dataset() call fails on a schema merge; we read per-cell parquets):

- bench_eval:             benchmark items (MMLU/ARC/GSM8K), text in
                          `formatted_prompt`. Eval content, benchmark format.
- casual_deploy_matched:  ShareGPT first turns, length-matched to bench_eval,
                          text in `first_turn`. Deploy content, casual format.
- bench_deploy_rewritten: ShareGPT queries rewritten into benchmark templates
                          by DeepSeek-V3, text in `rewritten`. Deploy content,
                          benchmark format (the adversarial control).
- casual_eval_mutual:     MuTual dialogue first turns, text in `first_turn`.
                          Their proxy for eval content in casual format; the
                          cell our rewrite arm replaces.
- bench_deploy_formal /   genre controls: the same ShareGPT queries in formal
  casual_deploy_formal:   academic tone with and without benchmark structure.
- bench_eval_full /       unmatched full pools (3000 / 7088) behind the
  casual_eval_mutual_full: histogram matching.

This dataset is public text (benchmarks, ShareGPT, MuTual) — no canary rule,
unlike SAD.
"""

from __future__ import annotations

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "viliana-dev/eval-awareness-2x2"

# cell name -> (column holding the prompt text)
CELLS = {
    "bench_eval": "formatted_prompt",
    "casual_deploy_matched": "first_turn",
    "bench_deploy_rewritten": "rewritten",
    "casual_eval_mutual": "first_turn",
    "bench_deploy_formal": "rewritten",
    "casual_deploy_formal": "rewritten",
    "bench_eval_full": "formatted_prompt",
    "casual_eval_mutual_full": "first_turn",
}


def load_cell(cell: str) -> list[str]:
    """Return the prompt texts for one cell, in file order."""
    if cell not in CELLS:
        raise ValueError(f"unknown cell {cell!r}; known: {sorted(CELLS)}")
    path = hf_hub_download(
        REPO, f"{cell}/train-00000-of-00001.parquet", repo_type="dataset"
    )
    df = pd.read_parquet(path)
    texts = df[CELLS[cell]].tolist()
    bad = [i for i, t in enumerate(texts) if not isinstance(t, str) or not t.strip()]
    if bad:
        raise ValueError(f"cell {cell}: {len(bad)} empty/non-string prompts")
    return texts


def as_conversations(texts: list[str]) -> list[list[dict[str, str]]]:
    """Wrap each prompt as a single-turn user message for extraction."""
    return [[{"role": "user", "content": t}] for t in texts]
