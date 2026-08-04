"""Slice 3: extract multi-turn activations at every read point and probe.

Two phases so GPU work and analysis rerun independently:

    python -m vnaa.run_slice3 extract --dtype bfloat16 --device-map auto
    python -m vnaa.run_slice3 analyze

A multi-turn conversation has no single obvious place that represents "what
the model makes of this situation", so we read four positions and report each:
last_user_token, post_instruction_token, final_turn_mean, and the per-turn
trajectory. Which one carries the signal is itself a result — if only
final_turn_mean works, the signal is about the question just asked rather than
the situation as a whole.

Every number is reported against the surface baselines computed on the same
split, and the decisive comparison is the same as slice 2b: not the raw AUROC
but the margin over bag-of-words, because eval and deploy content genuinely
differ and no amount of matching drives a lexical baseline to chance.
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

from vnaa.baselines import format_features, strength
from vnaa.extract import READ_POINTS
from vnaa.gate_slice3 import CELLS, serialize


def first_user(conv: list[dict]) -> str:
    """Text visible at post_instruction_token: the opening user turn only."""
    for m in conv:
        if m["role"] == "user":
            return m["content"]
    raise ValueError("no user turn")
from vnaa.prep_slice3_batches import BATCH_DIR
from vnaa.probe import fit_layer

MODEL_TAG = "Llama-3.1-8B-Instruct"
RESULTS = Path("results")


def load_cells() -> dict[str, list[list[dict]]]:
    """The three written cells, plus eval_shared if it has been built.

    eval_shared is the recombination control (deploy conversation, eval final
    turn), so it lives in its own file rather than the per-batch writer output.
    """
    cells: dict[str, list[list[dict]]] = {c: [] for c in CELLS}
    for p in sorted(BATCH_DIR.glob("out_*.json")):
        for it in json.load(open(p, encoding="utf-8"))["items"]:
            for c in CELLS:
                cells[c].append(it[c])
    shared = Path("datasets/slice3/eval_shared.json")
    if shared.exists():
        cells["eval_shared"] = [
            it["eval_shared"] for it in json.load(open(shared, encoding="utf-8"))["items"]
        ]
    coherent = Path("datasets/slice3/eval_coherent.json")
    if coherent.exists():
        cells["eval_coherent"] = [
            it["eval_coherent"] for it in json.load(open(coherent, encoding="utf-8"))["items"]
        ]
    return cells


def acts_path(cell: str, read_point: str) -> Path:
    return RESULTS / f"s3_{MODEL_TAG}_{cell}_{read_point}.npz"


def cmd_extract(args: argparse.Namespace) -> None:
    import torch

    from vnaa.extract import extract_activations, extract_trajectories, load_model

    cells = load_cells()
    print({c: len(v) for c, v in cells.items()})
    RESULTS.mkdir(exist_ok=True)

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    lm = load_model(args.model, device=args.device, dtype=dtype,
                    device_map=args.device_map)
    print(f"model={args.model} n_layers={lm.n_layers}")

    for cell, convs in cells.items():
        for rp in READ_POINTS:
            p = acts_path(cell, rp)
            if p.exists():
                continue
            acts = extract_activations(lm, convs, read_point=rp, show_progress=False)
            assert np.isfinite(acts).all(), f"non-finite in {cell}/{rp}"
            np.savez_compressed(p, acts=acts)
            print(f"saved {p.name} {acts.shape}")
        tp = RESULTS / f"s3_{MODEL_TAG}_{cell}_trajectory.npz"
        if not tp.exists():
            trajs = extract_trajectories(lm, convs, show_progress=False)
            # Ragged turn counts: store flat with an index of turns per item.
            np.savez_compressed(
                tp,
                acts=np.concatenate(trajs, axis=0),
                n_turns=np.array([t.shape[0] for t in trajs]),
            )
            print(f"saved {tp.name}")


def split(n: int, test_frac: float, seed: int):
    idx = np.random.default_rng(seed).permutation(n)
    k = int(round(n * test_frac))
    return idx[k:], idx[:k]


def score_pair(acts_pos, acts_neg, txt_pos, txt_neg, seeds, test_frac):
    """Probe (best layer picked on train) and baselines, averaged over seeds."""
    out = {k: [] for k in ("probe", "format", "bow", "length", "best_layer")}
    for seed in seeds:
        tr_p, te_p = split(len(acts_pos), test_frac, seed)
        tr_n, te_n = split(len(acts_neg), test_frac, seed)
        x_tr = np.concatenate([acts_pos[tr_p], acts_neg[tr_n]])
        y_tr = np.concatenate([np.ones(len(tr_p)), np.zeros(len(tr_n))])
        x_te = np.concatenate([acts_pos[te_p], acts_neg[te_n]])
        y_te = np.concatenate([np.ones(len(te_p)), np.zeros(len(te_n))])

        aur = []
        probes = []
        for L in range(x_tr.shape[1]):
            pr = fit_layer(x_tr[:, L, :], y_tr)
            probes.append(pr)
            aur.append(roc_auc_score(y_te, pr.project(x_te[:, L, :])))
        best = int(np.argmax(np.abs(np.array(aur) - 0.5)))
        out["probe"].append(aur[best])
        out["best_layer"].append(best)

        t_tr = [txt_pos[i] for i in tr_p] + [txt_neg[i] for i in tr_n]
        t_te = [txt_pos[i] for i in te_p] + [txt_neg[i] for i in te_n]
        fmt = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        fmt.fit(np.array([format_features(t) for t in t_tr]), y_tr)
        out["format"].append(roc_auc_score(
            y_te, fmt.predict_proba(np.array([format_features(t) for t in t_te]))[:, 1]))
        vec = CountVectorizer(lowercase=True, min_df=2)
        bow = LogisticRegression(max_iter=2000).fit(vec.fit_transform(t_tr), y_tr)
        out["bow"].append(roc_auc_score(y_te, bow.predict_proba(vec.transform(t_te))[:, 1]))
        out["length"].append(roc_auc_score(
            y_te, np.array([len(t) for t in t_te], dtype=float)))
    return out


def trajectory_items(path: Path) -> list[np.ndarray]:
    """Restore the ragged per-conversation trajectory cache."""
    cached = np.load(path)
    flat, n_turns = cached["acts"], cached["n_turns"]
    ends = np.cumsum(n_turns)
    starts = np.r_[0, ends[:-1]]
    return [flat[s:e].astype(np.float64) for s, e in zip(starts, ends)]


def through_user_turn(conv: list[dict], turn: int) -> str:
    """Serialized causal text through the zero-indexed user turn."""
    seen = 0
    for i, message in enumerate(conv):
        if message["role"] == "user":
            if seen == turn:
                return serialize(conv[: i + 1])
            seen += 1
    raise IndexError(f"conversation has only {seen} user turns")


def analyze_trajectories(cells, seeds, test_frac):
    """Probe each user-turn read point, with an equally causal text baseline."""
    results: dict[str, dict] = {}
    neg = trajectory_items(RESULTS / f"s3_{MODEL_TAG}_deploy_trajectory.npz")
    for pos_cell in [c for c in ("eval_disguised", "eval_obvious", "eval_shared", "eval_coherent")
                     if c in cells]:
        pos = trajectory_items(RESULTS / f"s3_{MODEL_TAG}_{pos_cell}_trajectory.npz")
        max_turns = max(len(x) for x in pos)
        for turn in range(max_turns):
            # Cells are shape-matched per item. Retain only pairs that have this
            # user turn, rather than treating absent turns as a feature.
            keep = [i for i, (p, n) in enumerate(zip(pos, neg))
                    if len(p) > turn and len(n) > turn]
            a_pos = np.stack([pos[i][turn] for i in keep])
            a_neg = np.stack([neg[i][turn] for i in keep])
            t_pos = [through_user_turn(cells[pos_cell][i], turn) for i in keep]
            t_neg = [through_user_turn(cells["deploy"][i], turn) for i in keep]
            r = score_pair(a_pos, a_neg, t_pos, t_neg, seeds, test_frac)
            results[f"{pos_cell}|user_turn_{turn + 1}"] = {
                **{k: (float(np.mean(v)), float(np.std(v))) for k, v in r.items()},
                "n": len(keep),
            }
    # Align from the end as well. This is the causal diagnostic for eval_shared:
    # its prefixes are byte-identical to deploy, but its final user turn differs.
    # Absolute turn indices mix final turns from short conversations with earlier
    # turns from longer ones, so they cannot answer that control cleanly.
    for pos_cell in [c for c in ("eval_disguised", "eval_obvious", "eval_shared", "eval_coherent")
                     if c in cells]:
        pos = trajectory_items(RESULTS / f"s3_{MODEL_TAG}_{pos_cell}_trajectory.npz")
        max_turns = max(len(x) for x in pos)
        for offset in range(max_turns):
            keep = [i for i, (p, n) in enumerate(zip(pos, neg))
                    if len(p) > offset and len(n) > offset]
            a_pos = np.stack([pos[i][-1 - offset] for i in keep])
            a_neg = np.stack([neg[i][-1 - offset] for i in keep])
            t_pos = [through_user_turn(cells[pos_cell][i], len(pos[i]) - 1 - offset)
                     for i in keep]
            t_neg = [through_user_turn(cells["deploy"][i], len(neg[i]) - 1 - offset)
                     for i in keep]
            r = score_pair(a_pos, a_neg, t_pos, t_neg, seeds, test_frac)
            results[f"{pos_cell}|from_final_{offset}"] = {
                **{k: (float(np.mean(v)), float(np.std(v))) for k, v in r.items()},
                "n": len(keep),
            }
    return results


def cmd_analyze(args: argparse.Namespace) -> None:
    cells = load_cells()
    n = len(cells["deploy"])
    print(f"n = {n} per cell, {len(args.seeds)} seeds, test_frac {args.test_frac}\n")

    results: dict[str, dict] = {}
    for pos_cell in [c for c in ("eval_disguised", "eval_obvious", "eval_shared", "eval_coherent")
                     if c in cells]:
        for rp in READ_POINTS:
            a_pos = np.load(acts_path(pos_cell, rp))["acts"].astype(np.float64)
            a_neg = np.load(acts_path("deploy", rp))["acts"].astype(np.float64)
            # The text baseline must see exactly what the probe's read position
            # sees, or the comparison is rigged. Attention is causal, so a
            # hidden state at token i depends only on tokens <= i:
            #   post_instruction_token sits at the end of the FIRST user turn,
            #     so it has not seen the rest of the conversation;
            #   last_user_token and final_turn_mean sit in the final user turn,
            #     so every one of their tokens has seen everything before it.
            # Giving final_turn_mean a final-turn-only baseline (as a first pass
            # here did) hands the probe the whole conversation while the
            # baseline reads one turn, which inflated its margin from +0.08 to
            # +0.27.
            fn = first_user if rp == "post_instruction_token" else serialize
            t_pos = [fn(c) for c in cells[pos_cell]]
            t_neg = [fn(c) for c in cells["deploy"]]
            r = score_pair(a_pos, a_neg, t_pos, t_neg, args.seeds, args.test_frac)
            results[f"{pos_cell}|{rp}"] = {k: (float(np.mean(v)), float(np.std(v)))
                                           for k, v in r.items()}

    print(f"{'cell':<16} {'read point':<24} {'probe':>14} {'bow':>14} "
          f"{'format':>14} {'probe-bow':>11} {'layer':>6}")
    for key, r in results.items():
        cell, rp = key.split("|")
        margin = strength(r["probe"][0]) - strength(r["bow"][0])
        print(f"{cell:<16} {rp:<24} {r['probe'][0]:>8.3f}+-{r['probe'][1]:.2f} "
              f"{r['bow'][0]:>8.3f}+-{r['bow'][1]:.2f} "
              f"{r['format'][0]:>8.3f}+-{r['format'][1]:.2f} "
              f"{margin:>+11.3f} {r['best_layer'][0]:>6.0f}")

    trajectories = analyze_trajectories(cells, args.seeds, args.test_frac)
    print("\ntrajectory (causal text baseline through each user turn)")
    print(f"{'cell':<16} {'turn':<14} {'n':>4} {'probe':>14} {'bow':>14} "
          f"{'probe-bow':>11} {'layer':>6}")
    for key, r in trajectories.items():
        if "|user_turn_" not in key:
            continue
        cell, turn = key.split("|")
        margin = strength(r["probe"][0]) - strength(r["bow"][0])
        print(f"{cell:<16} {turn:<14} {r['n']:>4} "
              f"{r['probe'][0]:>8.3f}+-{r['probe'][1]:.2f} "
              f"{r['bow'][0]:>8.3f}+-{r['bow'][1]:.2f} "
              f"{margin:>+11.3f} {r['best_layer'][0]:>6.0f}")

    print("\ntrajectory aligned from final user turn (0 = final)")
    print(f"{'cell':<16} {'offset':<14} {'n':>4} {'probe':>14} {'bow':>14} "
          f"{'probe-bow':>11} {'layer':>6}")
    for key, r in trajectories.items():
        if "|from_final_" not in key:
            continue
        cell, offset = key.split("|")
        margin = strength(r["probe"][0]) - strength(r["bow"][0])
        print(f"{cell:<16} {offset:<14} {r['n']:>4} "
              f"{r['probe'][0]:>8.3f}+-{r['probe'][1]:.2f} "
              f"{r['bow'][0]:>8.3f}+-{r['bow'][1]:.2f} "
              f"{margin:>+11.3f} {r['best_layer'][0]:>6.0f}")

    out = RESULTS / f"s3_{MODEL_TAG}_analysis.json"
    with open(out, "w") as f:
        json.dump({"n_per_cell": n, "seeds": args.seeds, "results": results,
                   "trajectory": trajectories}, f, indent=2)
    print(f"\nsaved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Slice 3 multi-turn runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--model", default=f"meta-llama/{MODEL_TAG}")
    e.add_argument("--device", default=None)
    e.add_argument("--device-map", default=None)
    e.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    e.set_defaults(func=cmd_extract)
    a = sub.add_parser("analyze")
    a.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a.add_argument("--test-frac", type=float, default=0.3)
    a.set_defaults(func=cmd_analyze)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
