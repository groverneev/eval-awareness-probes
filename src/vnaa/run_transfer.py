"""Slice 2b: cross-cell transfer for the probe AND the surface baselines.

Why transfer rather than the in-distribution baseline gap: eval and deploy
prompts differ in many correlated ways at once (topic, speech act, vocabulary),
and slice-2b measurement showed the last two cannot be matched away — an
impersonal question with a canonical answer versus a first-person request for
help IS the eval/deploy distinction. So a high in-distribution AUROC, for the
probe or for bag-of-words, does not say WHICH difference was used.

Transfer breaks the tie. Fit on one cell pair, freeze, score a pair built by a
completely different procedure. The two pairs share only eval-vs-deploy;
everything surface about them differs. A classifier that learned MCQ shape or
science vocabulary collapses to ~0.5; one that learned something general
survives. Running the identical test on the baselines makes probe-vs-baseline
meaningful without needing baselines at chance, which is unachievable.

This is also the test Devbunova used (their probe: 0.99 in-distribution, 94.5%
false positives on format-transplanted deploy prompts; we measured 96.0%), so
our numbers are directly comparable to theirs.

Layer selection: the probe's best layer is chosen on the TRAINING pair's
held-out split and then held fixed for every transfer readout. Choosing it per
target would leak the test signal into model selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vnaa.baselines import bootstrap_auroc_ci, format_features, strength
from vnaa.build_2b import OUT_DIR, read_jsonl
from vnaa.data_2x2 import load_cell
from vnaa.probe import fit_layer

MODEL_TAG = "Llama-3.1-8B-Instruct"

# Cell pairs to train on and transfer to. Devbunova's cells and ours are built
# by entirely different procedures, which is exactly what makes the transfer
# informative.
PAIRS = {
    "devb_standard": ("bench_eval", "casual_deploy_matched"),
    "devb_fmt_matched": ("bench_eval", "bench_deploy_rewritten"),
    "ours_a_wc": ("casual_eval_a", "casual_deploy_wc"),
    "ours_a_tm": ("casual_eval_a", "casual_deploy_tm"),
    "ours_b_wc": ("casual_eval_b", "casual_deploy_wc"),
    "ours_b_tm": ("casual_eval_b", "casual_deploy_tm"),
}
OURS = {"casual_eval_a", "casual_eval_b", "casual_deploy_wc", "casual_deploy_tm"}


def load_texts(cell: str) -> list[str]:
    if cell in OURS:
        return [r["text"] for r in read_jsonl(OUT_DIR / f"{cell}.jsonl")]
    return load_cell(cell)


def load_acts(cell: str, results: Path) -> np.ndarray:
    prefix = "s2b" if cell in OURS else "s2a"
    p = results / f"{prefix}_{MODEL_TAG}_{cell}.npz"
    if not p.exists():
        raise SystemExit(f"missing activations {p}")
    return np.load(p)["acts"].astype(np.float64)


def split(n: int, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.random.default_rng(seed).permutation(n)
    n_test = int(round(n * test_frac))
    return idx[n_test:], idx[:n_test]


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-cell transfer matrix")
    ap.add_argument("--results", default="results")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    results = Path(args.results)

    cells = sorted({c for pair in PAIRS.values() for c in pair})
    texts = {c: load_texts(c) for c in cells}
    acts = {c: load_acts(c, results) for c in cells}
    splits = {c: split(len(texts[c]), args.test_frac, args.seed) for c in cells}
    n_layers = acts[cells[0]].shape[1]

    def pair_data(name: str, part: str):
        pos, neg = PAIRS[name]
        i = 0 if part == "train" else 1
        pi, ni = splits[pos][i], splits[neg][i]
        x_act = np.concatenate([acts[pos][pi], acts[neg][ni]])
        x_txt = [texts[pos][j] for j in pi] + [texts[neg][j] for j in ni]
        y = np.concatenate([np.ones(len(pi)), np.zeros(len(ni))])
        return x_act, x_txt, y

    out: dict[str, dict] = {}
    for train_name in PAIRS:
        xa_tr, xt_tr, y_tr = pair_data(train_name, "train")

        # probe: pick best layer on this pair's own held-out split
        xa_te0, _, y_te0 = pair_data(train_name, "test")
        aur = np.empty(n_layers)
        probes = []
        for L in range(n_layers):
            pr = fit_layer(xa_tr[:, L, :], y_tr)
            probes.append(pr)
            aur[L] = roc_auc_score(y_te0, pr.project(xa_te0[:, L, :]))
        best = int(np.argmax(np.abs(aur - 0.5)))
        probe, sign = probes[best], (1.0 if aur[best] >= 0.5 else -1.0)

        fmt = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        fmt.fit(np.array([format_features(t) for t in xt_tr]), y_tr)
        vec = CountVectorizer(lowercase=True, min_df=2)
        bow = LogisticRegression(max_iter=2000)
        bow.fit(vec.fit_transform(xt_tr), y_tr)

        row: dict[str, dict] = {}
        for test_name in PAIRS:
            xa_te, xt_te, y_te = pair_data(test_name, "test")
            scores = {
                "probe": sign * probe.project(xa_te[:, best, :]),
                "format": fmt.predict_proba(
                    np.array([format_features(t) for t in xt_te]))[:, 1],
                "bow": bow.predict_proba(vec.transform(xt_te))[:, 1],
                "length": -np.array([len(t) for t in xt_te], dtype=float),
            }
            row[test_name] = {
                k: {"auroc": float(roc_auc_score(y_te, v)),
                    "ci": bootstrap_auroc_ci(y_te.astype(int), v, seed=args.seed)}
                for k, v in scores.items()
            }
        out[train_name] = {"best_layer": best, "to": row}

        print(f"\n=== trained on {train_name}  (layer {best}) ===")
        print(f"{'tested on':<18} {'probe':>16} {'format':>16} {'bow':>16}")
        for test_name, r in row.items():
            mark = " *" if test_name == train_name else "  "
            cells_s = "".join(
                f"{r[k]['auroc']:>10.3f}({strength(r[k]['auroc']):.2f})"
                for k in ("probe", "format", "bow")
            )
            print(f"{test_name:<18}{cells_s}{mark}")

    # Headline: mean off-diagonal (true transfer) strength per method.
    print(f"\n{'method':<10} {'mean in-dist':>13} {'mean transfer':>14} {'drop':>8}")
    for k in ("probe", "format", "bow", "length"):
        ind = [out[a]["to"][a][k]["auroc"] for a in PAIRS]
        off = [out[a]["to"][b][k]["auroc"] for a in PAIRS for b in PAIRS if a != b]
        mi, mo = np.mean([strength(v) for v in ind]), np.mean([strength(v) for v in off])
        print(f"{k:<10} {mi:>13.3f} {mo:>14.3f} {mi - mo:>8.3f}")

    p = results / f"s2b_{MODEL_TAG}_transfer.json"
    with open(p, "w") as f:
        json.dump({"seed": args.seed, "pairs": {k: list(v) for k, v in PAIRS.items()},
                   "matrix": out}, f, indent=2)
    print(f"\nsaved {p}")


if __name__ == "__main__":
    main()
