"""Slice 2b dataset construction: sample, then gate.

The 2b dataset replaces Devbunova's MuTual proxy with true content-preserving
rewrites. Three files live in datasets/casual_eval_true/ (committed — all
derived from public benchmarks and the public 2x2, no canary rule):

- bench_eval_sample.jsonl   items sampled here, the rewrite inputs
- rewrites.jsonl            casual_eval_true: benchmark items as natural
                            first-turn user messages (written by Claude)
- counterparts.jsonl        casual_deploy_true: topic-matched genuine-need
                            questions, one per rewrite (kills the BoW topic
                            shortcut)

Subcommands:
  sample  draw a stratified sample of bench_eval items as rewrite inputs
  gate    surface baselines on the constructed cells. Probe-blind by design:
          acceptance is decided here, never by probe score (Xiong et al.
          selected rewrites by probe score — circular for dataset
          construction).

Gate criteria (from PLAN.md):
- rewrites vs casual_deploy_matched: format and length AUROC ~0.5 (<= 0.60)
- rewrites vs counterparts (topic-matched): format, length AND BoW <= 0.60 —
  topic matching makes the BoW bound meaningful here, unlike against
  casual_deploy_matched where topics differ by construction
- manual audit of a random sample for content preservation happens outside
  this script and is recorded in the dataset README.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_auc_score

from vnaa.baselines import (
    bag_of_words_scores,
    bootstrap_auroc_ci,
    format_scores,
    make_split,
    strength,
)
from vnaa.data_2x2 import REPO, load_cell

OUT_DIR = Path("datasets/casual_eval_true")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_sample(args: argparse.Namespace) -> None:
    p = hf_hub_download(REPO, "bench_eval/train-00000-of-00001.parquet",
                        repo_type="dataset")
    df = pd.read_parquet(p)
    df["uid"] = [f"{s}_{i}" for i, s in enumerate(df["source"])]

    rng = np.random.default_rng(args.seed)
    parts = []
    for source, group in df.groupby("source"):
        n = int(round(args.n * len(group) / len(df)))
        parts.append(group.iloc[rng.permutation(len(group))[:n]])
    sample = pd.concat(parts)

    def norm_choices(raw: str) -> list[str]:
        """The parquet stores choices as a JSON string in three shapes:
        ARC {'text': [...], 'label': [...]}, MMLU ['a', 'b', ...],
        GSM8K []. Normalize to a list of option texts."""
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = parsed["text"]
        return [str(t) for t in parsed]

    rows = [
        {
            "uid": r.uid,
            "source": r.source,
            "question": r.question,
            "choices": norm_choices(r.choices),
            "answer": r.answer,
        }
        for r in sample.itertuples()
    ]
    out = OUT_DIR / "bench_eval_sample.jsonl"
    write_jsonl(out, rows)
    counts = sample["source"].value_counts().to_dict()
    print(f"wrote {len(rows)} items to {out} | by source: {counts}")


def gate_contrast(
    name: str,
    pos_texts: list[str],
    neg_texts: list[str],
    seed: int,
    include_bow_in_gate: bool,
) -> bool:
    texts = pos_texts + neg_texts
    labels = np.array([1] * len(pos_texts) + [0] * len(neg_texts))
    train_idx, test_idx = make_split(labels, 0.3, seed)
    y = labels[test_idx]

    scored = {
        "length (char)": np.array([len(texts[i]) for i in test_idx], dtype=float),
        "length (word)": np.array([len(texts[i].split()) for i in test_idx], dtype=float),
        "format (structure)": format_scores(texts, labels, train_idx, test_idx, seed),
        "bag-of-words": bag_of_words_scores(texts, labels, train_idx, test_idx, seed=seed),
    }
    print(f"\n--- {name} (n={len(pos_texts)} vs {len(neg_texts)}) ---")
    ok = True
    for bname, scores in scored.items():
        auroc = roc_auc_score(y, scores)
        ci = bootstrap_auroc_ci(y, scores, seed=seed)
        gated = include_bow_in_gate or not bname.startswith("bag")
        verdict = ""
        if gated:
            passed = strength(auroc) <= 0.10
            verdict = "  PASS" if passed else "  FAIL (need strength <= 0.10)"
            ok = ok and passed
        print(f"{bname:<20} {auroc:>7.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
              f"  strength {strength(auroc):.4f}{verdict}")
    return ok


def cmd_gate(args: argparse.Namespace) -> None:
    rewrites = read_jsonl(OUT_DIR / "rewrites.jsonl")
    rw_texts = [r["text"] for r in rewrites]
    print(f"rewrites: n={len(rw_texts)}")

    deploy = load_cell("casual_deploy_matched")

    all_ok = gate_contrast(
        "casual_eval_true vs casual_deploy_matched",
        rw_texts, deploy, args.seed, include_bow_in_gate=False,
    )

    cp_path = OUT_DIR / "counterparts.jsonl"
    if cp_path.exists():
        counterparts = read_jsonl(cp_path)
        cp_texts = [r["text"] for r in counterparts]
        all_ok &= gate_contrast(
            "casual_eval_true vs counterparts (topic-matched)",
            rw_texts, cp_texts, args.seed, include_bow_in_gate=True,
        )
        all_ok &= gate_contrast(
            "counterparts vs casual_deploy_matched",
            cp_texts, deploy, args.seed, include_bow_in_gate=False,
        )
    else:
        print("\n(counterparts.jsonl not present yet; skipped)")
        all_ok = False

    print(f"\nGATE {'PASSED' if all_ok else 'NOT PASSED'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Slice 2b dataset build")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("sample")
    sp.add_argument("--n", type=int, default=400)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_sample)

    gp = sub.add_parser("gate")
    gp.add_argument("--seed", type=int, default=0)
    gp.set_defaults(func=cmd_gate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
