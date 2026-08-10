"""Cross-condition transfer between the two valid Slice-3 controls."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vnaa.baselines import format_features
from vnaa.probe import fit_layer
from vnaa.run_coherent_pair_pilot import load_cells as load_coherent, path as coherent_path
from vnaa.run_slice3 import (
    MODEL_TAG,
    acts_path,
    differing_final_messages,
    load_cells as load_slice3,
    split,
)

RESULTS = Path("results")
READ_POINT = "last_user_token"


def load_pairs():
    original = load_slice3()
    coherent = load_coherent(Path("datasets/slice3/coherent_pairs/validated.json"))
    return {
        "eval_shared": (original["eval_shared"], original["deploy"],
                        np.load(acts_path("eval_shared", READ_POINT))["acts"].astype(np.float64),
                        np.load(acts_path("deploy", READ_POINT))["acts"].astype(np.float64)),
        "coherent_pairs": (coherent["eval"], coherent["deploy"],
                           np.load(coherent_path("coherent_pairs", "eval", READ_POINT))["acts"].astype(np.float64),
                           np.load(coherent_path("coherent_pairs", "deploy", READ_POINT))["acts"].astype(np.float64)),
    }


def run_direction(source, target, seed, test_frac):
    pos, neg, ap, an = source
    tp, tn, bp, bn = target
    sp_tr, sp_te = split(len(pos), test_frac, seed)
    sn_tr, sn_te = split(len(neg), test_frac, seed)
    tp_tr, tp_te = split(len(tp), test_frac, seed)
    tn_tr, tn_te = split(len(tn), test_frac, seed)

    xtr = np.r_[ap[sp_tr], an[sn_tr]]
    ytr = np.r_[np.ones(len(sp_tr)), np.zeros(len(sn_tr))]
    xsrc = np.r_[ap[sp_te], an[sn_te]]
    ysrc = np.r_[np.ones(len(sp_te)), np.zeros(len(sn_te))]
    probes, aur = [], []
    for layer in range(xtr.shape[1]):
        probe = fit_layer(xtr[:, layer], ytr)
        probes.append(probe)
        aur.append(roc_auc_score(ysrc, probe.project(xsrc[:, layer])))
    best = int(np.argmax(np.abs(np.asarray(aur) - .5)))
    sign = 1 if aur[best] >= .5 else -1

    xte = np.r_[bp[tp_te], bn[tn_te]]
    yte = np.r_[np.ones(len(tp_te)), np.zeros(len(tn_te))]
    source_pos_text, source_neg_text = differing_final_messages(pos, neg)
    target_pos_text, target_neg_text = differing_final_messages(tp, tn)
    source_text = ([source_pos_text[i] for i in sp_tr]
                   + [source_neg_text[i] for i in sn_tr])
    target_text = ([target_pos_text[i] for i in tp_te]
                   + [target_neg_text[i] for i in tn_te])
    vec = CountVectorizer(lowercase=True, min_df=2)
    bow = LogisticRegression(max_iter=2000).fit(vec.fit_transform(source_text), ytr)
    fmt = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    fmt.fit(np.array([format_features(t) for t in source_text]), ytr)
    return {
        "best_layer": best,
        "probe": float(roc_auc_score(yte, sign * probes[best].project(xte[:, best]))),
        "bow": float(roc_auc_score(yte, bow.predict_proba(vec.transform(target_text))[:, 1])),
        "format": float(roc_auc_score(yte, fmt.predict_proba(np.array([format_features(t) for t in target_text]))[:, 1])),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--test-frac", type=float, default=.3)
    args = ap.parse_args()
    pairs = load_pairs()
    rows = {}
    for train, test in (("eval_shared", "coherent_pairs"), ("coherent_pairs", "eval_shared")):
        vals = [run_direction(pairs[train], pairs[test], s, args.test_frac) for s in args.seeds]
        rows[f"{train}_to_{test}"] = vals
        print(f"{train} -> {test}")
        for key in ("probe", "bow", "format", "best_layer"):
            a = np.array([v[key] for v in vals])
            print(f"  {key:<10} {a.mean():.3f} +/- {a.std():.3f}")
        margin = np.abs([v["probe"] - .5 for v in vals]) - np.abs([v["bow"] - .5 for v in vals])
        print(f"  probe-BoW strength margin {margin.mean():+.3f} +/- {margin.std():.3f}")
    out = RESULTS / f"s3_{MODEL_TAG}_cross_control_transfer.json"
    json.dump({
        "seeds": args.seeds,
        "read_point": READ_POINT,
        "paired_surface_text_view": "differing_final_message_only",
        "directions": rows,
    }, open(out, "w"), indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
