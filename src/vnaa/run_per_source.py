"""Per-source breakdown: is the signal eval-recognition or quiz-shape detection?

The concern, logged when the deploy pool was built: a GSM8K word problem asked
casually IS what a real user asks. Its eval-ness lives in provenance, not in
the text. ARC and MMLU items are knowledge questions with a canonical answer,
which is a more distinctive shape.

So if the probe separates ARC and MMLU items from deployment but does markedly
worse on GSM8K, what it has learned is "knowledge-quiz-shaped", not
"this is an evaluation". That is a materially weaker claim, and it is
measurable here with no new extraction.

For the topic-matched deploy cell the counterparts are keyed to their own eval
item (`tm_<eval uid>`), so both sides can be restricted to the same source —
the cleanest version of the contrast. For the WildChat cell the deploy items
carry no source, so only the eval side is restricted.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from vnaa.build_2b import OUT_DIR, read_jsonl
from vnaa.probe import fit_layer
from vnaa.run_transfer import PAIRS, load_acts, load_texts, split

SOURCES = ("arc", "mmlu", "gsm8k")
OUR_TARGETS = [
    ("casual_eval_a", "casual_deploy_wc"),
    ("casual_eval_b", "casual_deploy_wc"),
    ("casual_eval_a", "casual_deploy_tm"),
    ("casual_eval_b", "casual_deploy_tm"),
]


def uid_source(uid: str) -> str | None:
    m = re.search(r"(arc|mmlu|gsm8k)", uid)
    return m.group(1) if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-source probe breakdown")
    ap.add_argument("--results", default="results")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    results = Path(args.results)

    cells = sorted({c for pair in PAIRS.values() for c in pair})
    texts = {c: load_texts(c) for c in cells}
    acts = {c: load_acts(c, results) for c in cells}
    uids = {
        c: [r["uid"] for r in read_jsonl(OUT_DIR / f"{c}.jsonl")]
        for c in cells if (OUT_DIR / f"{c}.jsonl").exists()
    }
    srcs = {c: [uid_source(u) for u in v] for c, v in uids.items()}
    for c in ("casual_eval_a", "casual_deploy_tm"):
        counts = {s: srcs[c].count(s) for s in SOURCES}
        print(f"{c}: {counts}")

    # (train pair, source) -> list of AUROCs across seeds and target cells
    acc: dict[tuple[str, str, str], list[float]] = {}

    for seed in args.seeds:
        splits = {c: split(len(texts[c]), args.test_frac, seed) for c in cells}

        def part(cell, which):
            return splits[cell][0 if which == "train" else 1]

        for train_name, (tp, tn) in PAIRS.items():
            tr_p, tr_n = part(tp, "train"), part(tn, "train")
            xa = np.concatenate([acts[tp][tr_p], acts[tn][tr_n]])
            xt = [texts[tp][i] for i in tr_p] + [texts[tn][i] for i in tr_n]
            y = np.concatenate([np.ones(len(tr_p)), np.zeros(len(tr_n))])

            te_p, te_n = part(tp, "test"), part(tn, "test")
            xa_te = np.concatenate([acts[tp][te_p], acts[tn][te_n]])
            y_te = np.concatenate([np.ones(len(te_p)), np.zeros(len(te_n))])
            aur = [
                roc_auc_score(y_te, fit_layer(xa[:, L, :], y).project(xa_te[:, L, :]))
                for L in range(xa.shape[1])
            ]
            best = int(np.argmax(np.abs(np.array(aur) - 0.5)))
            probe = fit_layer(xa[:, best, :], y)
            sign = 1.0 if aur[best] >= 0.5 else -1.0

            vec = CountVectorizer(lowercase=True, min_df=2)
            bow = LogisticRegression(max_iter=2000).fit(vec.fit_transform(xt), y)

            for ev, dep in OUR_TARGETS:
                if ev in (tp, tn) and dep in (tp, tn):
                    continue  # skip in-distribution
                ev_te, dep_te = part(ev, "test"), part(dep, "test")
                for s in SOURCES:
                    ei = [i for i in ev_te if srcs[ev][i] == s]
                    di = ([i for i in dep_te if srcs[dep][i] == s]
                          if dep == "casual_deploy_tm" else list(dep_te))
                    if len(ei) < 5 or len(di) < 5:
                        continue
                    yy = np.concatenate([np.ones(len(ei)), np.zeros(len(di))])
                    p_sc = sign * np.concatenate([
                        probe.project(acts[ev][ei][:, best, :]),
                        probe.project(acts[dep][di][:, best, :]),
                    ])
                    b_sc = bow.predict_proba(vec.transform(
                        [texts[ev][i] for i in ei] + [texts[dep][i] for i in di]))[:, 1]
                    acc.setdefault((train_name, s, "probe"), []).append(roc_auc_score(yy, p_sc))
                    acc.setdefault((train_name, s, "bow"), []).append(roc_auc_score(yy, b_sc))

    print(f"\nTransfer AUROC by benchmark source "
          f"(mean +/- sd over {len(args.seeds)} seeds x target cells)\n")
    print(f"{'trained on':<20}" + "".join(f"{s:>18}" for s in SOURCES))
    for method in ("probe", "bow"):
        print(f"-- {method}")
        for train_name in PAIRS:
            row = ""
            for s in SOURCES:
                v = np.array(acc.get((train_name, s, method), [np.nan]))
                row += f"{np.nanmean(v):>11.3f}+-{np.nanstd(v):.2f}"
            print(f"{train_name:<20}{row}")

    print(f"\n{'source':<10} {'probe':>16} {'bag-of-words':>18}   n")
    for s in SOURCES:
        p = np.concatenate([np.array(v) for (t, ss, m), v in acc.items()
                            if ss == s and m == "probe"])
        b = np.concatenate([np.array(v) for (t, ss, m), v in acc.items()
                            if ss == s and m == "bow"])
        print(f"{s:<10} {p.mean():>10.3f}+-{p.std():.3f} {b.mean():>12.3f}+-{b.std():.3f}  {len(p)}")


if __name__ == "__main__":
    main()
