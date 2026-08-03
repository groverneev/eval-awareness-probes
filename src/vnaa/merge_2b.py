"""Slice 2b: merge rewrite batches, validate, and run the probe-blind gate.

Batch outputs live at datasets/casual_eval_true/batches/out_XX.json:

  {"batch": 0,
   "eval":   [{"uid": "arc_123", "style_a": "...", "style_b": "..."}, ...],
   "deploy": [{"uid": "dep_0000", "text": "..."}, ...]}

Merged cells (jsonl, one object per line, committed):
  casual_eval_a.jsonl    benchmark question, plain casual delivery
  casual_eval_b.jsonl    same question wrapped in a personal need
  casual_deploy_rw.jsonl real ShareGPT prompts rewritten in the same register

The gate is probe-blind: acceptance depends only on surface classifiers and
manual audit, never on probe score. The decisive contrasts are A vs deploy_rw
and B vs deploy_rw — all three cells are Claude-authored, so authorship cannot
carry the label. We also report each cell against REAL ShareGPT, which is
expected to fail; that number is the authorship-confound control, reported as
a finding rather than gated on.

Gate: format, length and bag-of-words strength <= 0.10 on the decisive
contrasts, scored out-of-fold with 5-fold CV so every item is scored by a model
that never saw it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vnaa.baselines import bootstrap_auroc_ci, format_features, strength
from vnaa.build_2b import OUT_DIR, read_jsonl, write_jsonl
from vnaa.data_2x2 import load_cell
from vnaa.prep_2b_batches import BATCH_DIR

CELLS = {
    "casual_eval_a": "casual_eval_a.jsonl",
    "casual_eval_b": "casual_eval_b.jsonl",
    "casual_deploy_wc": "casual_deploy_wc.jsonl",
    "casual_deploy_tm": "casual_deploy_tm.jsonl",
}
GATE_MAX_STRENGTH = 0.10

# Two deployment cells, because they answer different questions:
#   casual_deploy_wc  real WildChat prompts, rewritten. Carries the realism
#                     claim. Bag-of-words CANNOT reach chance here — real users
#                     do not discuss xylem cells, so topic separates the classes
#                     whatever we do (measured: BoW 0.935 after best-effort
#                     topic matching, vs 0.951 for ShareGPT). BoW is therefore
#                     reported as the bar the probe must beat, not gated.
#   casual_deploy_tm  synthetic genuine-need messages on each eval item's own
#                     subject. The only construction that can drive BoW toward
#                     chance, so this is where BoW IS gated. Synthetic, so it
#                     cannot carry the realism claim on its own.
# Length and format are gated on both.
BOW_GATED_AGAINST = {"casual_deploy_tm"}


def cmd_merge(args: argparse.Namespace) -> None:
    inputs = {}
    for p in sorted(BATCH_DIR.glob("batch_*.json")):
        with open(p, encoding="utf-8") as f:
            spec = json.load(f)
        inputs[spec["batch"]] = spec

    rows_a, rows_b, rows_d, rows_t = [], [], [], []
    missing = []
    for b, spec in sorted(inputs.items()):
        out_path = BATCH_DIR / f"out_{b:02d}.json"
        if not out_path.exists():
            missing.append(out_path.name)
            continue
        with open(out_path, encoding="utf-8") as f:
            out = json.load(f)

        for key, spec_key, out_key in (("eval", "eval_items", "eval"),
                                       ("deploy", "deploy_items", "deploy"),
                                       ("topic", "topic_items", "topic")):
            want = {it["uid"] for it in spec[spec_key]}
            got = {r["uid"] for r in out.get(out_key, [])}
            if want - got:
                raise SystemExit(
                    f"{out_path.name}: missing {key} uids {sorted(want - got)[:5]}")

        for r in out["eval"]:
            for style, key, sink in (("A", "style_a", rows_a), ("B", "style_b", rows_b)):
                text = r[key].strip()
                if not text:
                    raise SystemExit(f"{out_path.name}: empty {key} for {r['uid']}")
                sink.append({"uid": r["uid"], "batch": b, "style": style, "text": text})
        for out_key, style, sink in (("deploy", "deploy_wc", rows_d),
                                     ("topic", "deploy_tm", rows_t)):
            for r in out[out_key]:
                text = r["text"].strip()
                if not text:
                    raise SystemExit(f"{out_path.name}: empty text for {r['uid']}")
                sink.append({"uid": r["uid"], "batch": b, "style": style, "text": text})

    if missing:
        print(f"WARNING: no output yet for {missing}")
    for rows, fname in ((rows_a, CELLS["casual_eval_a"]),
                        (rows_b, CELLS["casual_eval_b"]),
                        (rows_d, CELLS["casual_deploy_wc"]),
                        (rows_t, CELLS["casual_deploy_tm"])):
        write_jsonl(OUT_DIR / fname, rows)
        lens = [len(r["text"]) for r in rows]
        print(f"{fname}: n={len(rows)} mean_chars={np.mean(lens):.0f}")


def oof_scores(texts: list[str], labels: np.ndarray, kind: str, seed: int) -> np.ndarray:
    """Out-of-fold predictions so every item is scored by an unseen model."""
    scores = np.zeros(len(texts))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    x_fmt = np.array([format_features(t) for t in texts], dtype=float)
    for tr, te in skf.split(texts, labels):
        if kind == "format":
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(x_fmt[tr], labels[tr])
            scores[te] = clf.predict_proba(x_fmt[te])[:, 1]
        else:
            vec = CountVectorizer(lowercase=True, min_df=1)
            xtr = vec.fit_transform([texts[i] for i in tr])
            clf = LogisticRegression(max_iter=2000)
            clf.fit(xtr, labels[tr])
            scores[te] = clf.predict_proba(vec.transform([texts[i] for i in te]))[:, 1]
    return scores


def contrast(
    name: str,
    pos: list[str],
    neg: list[str],
    seed: int,
    gated: bool,
    gate_bow: bool = True,
) -> bool:
    """Score one contrast. gate_bow=False reports bag-of-words without gating it,
    for contrasts where topic difference IS the label rather than an artifact."""
    texts = pos + neg
    labels = np.array([1] * len(pos) + [0] * len(neg))
    scored = {
        "length (char)": np.array([len(t) for t in texts], dtype=float),
        "length (word)": np.array([len(t.split()) for t in texts], dtype=float),
        "format (structure)": oof_scores(texts, labels, "format", seed),
        "bag-of-words": oof_scores(texts, labels, "bow", seed),
    }
    tag = "GATED" if gated else "control, not gated"
    print(f"\n--- {name}  (n={len(pos)} vs {len(neg)}, {tag}) ---")
    ok = True
    for bname, scores in scored.items():
        auroc = roc_auc_score(labels, scores)
        ci = bootstrap_auroc_ci(labels, scores, seed=seed)
        s = strength(auroc)
        verdict = ""
        this_gated = gated and (gate_bow or not bname.startswith("bag"))
        if this_gated:
            passed = s <= GATE_MAX_STRENGTH
            ok = ok and passed
            verdict = "  PASS" if passed else "  FAIL"
        elif gated:
            verdict = "  (reported, probe must beat this)"
        print(f"{bname:<20} {auroc:>7.4f} [{ci[0]:.4f}, {ci[1]:.4f}]  "
              f"strength {s:.4f}{verdict}")
    return ok


def cmd_gate(args: argparse.Namespace) -> None:
    cells = {k: [r["text"] for r in read_jsonl(OUT_DIR / v)] for k, v in CELLS.items()}
    for k, v in cells.items():
        print(f"{k}: n={len(v)}")
    real = load_cell("casual_deploy_matched")
    rng = np.random.default_rng(args.seed)
    n = len(cells["casual_deploy_wc"])
    real_s = [real[i] for i in rng.permutation(len(real))[:n]]

    ok = True
    for eval_cell in ("casual_eval_a", "casual_eval_b"):
        for dep_cell in ("casual_deploy_wc", "casual_deploy_tm"):
            ok &= contrast(
                f"{eval_cell} vs {dep_cell}", cells[eval_cell], cells[dep_cell],
                args.seed, gated=True, gate_bow=dep_cell in BOW_GATED_AGAINST,
            )
    contrast("casual_deploy_wc vs REAL prompts (authorship control)",
             cells["casual_deploy_wc"], real_s, args.seed, gated=False)
    contrast("casual_eval_a vs casual_eval_b (style separation)",
             cells["casual_eval_a"], cells["casual_eval_b"], args.seed, gated=False)
    contrast("casual_deploy_wc vs casual_deploy_tm (deploy-cell parity)",
             cells["casual_deploy_wc"], cells["casual_deploy_tm"], args.seed, gated=False)

    print(f"\nGATE {'PASSED' if ok else 'NOT PASSED'} "
          f"(threshold: strength <= {GATE_MAX_STRENGTH} on gated contrasts)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Slice 2b merge and gate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.set_defaults(func=cmd_merge)
    g = sub.add_parser("gate")
    g.add_argument("--seed", type=int, default=0)
    g.set_defaults(func=cmd_gate)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
