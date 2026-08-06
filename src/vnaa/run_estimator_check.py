"""Re-score the headline contrasts with a second estimator and null baselines.

Answers two questions the reported numbers could not:

1. Does the result depend on the difference-of-means estimator?  Every headline
   figure in the project comes from one direction-fitting rule.  Here the same
   cached activations and the same splits are re-scored with L2 logistic
   regression, layer picked by cross-validation inside the training split.

2. What does "no signal" actually look like on these activations?  The proposal
   promises a random probe near 0.5 at the bottom of the baseline hierarchy.
   This reports two: shuffled training labels (refit per draw) and random unit
   directions (nothing fit at all).

No GPU.  Everything runs off cached ``.npz`` activations.

    python -m vnaa.run_estimator_check points
    python -m vnaa.run_estimator_check transfer

Slice 1's 8B activations were produced on Kaggle and were never cached locally,
so it is absent here.  Adding it needs a short re-extraction, not new analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from vnaa.baselines import strength
from vnaa.estimators import (
    DEFAULT_C,
    fit_layer_logistic,
    logistic_probe_auroc,
    permutation_null,
    random_direction_null,
    select_layer_cv,
    train_shuffled_direction_null,
)
from vnaa.extract import ASSISTANT_READ_POINTS, READ_POINTS
from vnaa.gate_slice3 import serialize
from vnaa.probe import fit_layer
from vnaa.run_slice3 import MODEL_TAG, RESULTS, acts_path, first_user, load_cells, split

Contrast = tuple[np.ndarray, np.ndarray, list[str], list[str]]


def text_view(read_point: str):
    """The text a baseline may see, matched to what the read position can see."""
    return first_user if read_point == "post_instruction_token" else serialize


def slice3_contrast(cells, cell: str, read_point: str) -> Contrast:
    pos = np.load(acts_path(cell, read_point))["acts"].astype(np.float64)
    neg = np.load(acts_path("deploy", read_point))["acts"].astype(np.float64)
    view = text_view(read_point)
    return pos, neg, [view(c) for c in cells[cell]], [view(c) for c in cells["deploy"]]


def coherent_contrast(read_point: str, mask: np.ndarray | None = None) -> Contrast:
    from vnaa.run_coherent_pair_pilot import load_cells as load_pairs
    from vnaa.run_silence_filter import DEFAULT_DATASET, TAG, path

    cells = load_pairs(DEFAULT_DATASET)
    pos = np.load(path(TAG, "eval", read_point))["acts"].astype(np.float64)
    neg = np.load(path(TAG, "deploy", read_point))["acts"].astype(np.float64)
    view = text_view(read_point)
    txt_pos = [view(c) for c in cells["eval"]]
    txt_neg = [view(c) for c in cells["deploy"]]
    if mask is not None:
        pos, neg = pos[mask], neg[mask]
        txt_pos = [t for t, keep in zip(txt_pos, mask) if keep]
        txt_neg = [t for t, keep in zip(txt_neg, mask) if keep]
    return pos, neg, txt_pos, txt_neg


def silent_mask() -> np.ndarray:
    """Pairs that terminated naturally in both cells and verbalized nothing."""
    from vnaa.run_silence_filter import LABELS

    rows = json.loads(LABELS.read_text(encoding="utf-8"))["items"]
    return np.array([
        (row["eval_terminated"] and row["deploy_terminated"])
        and not (row["eval_explicit_awareness"] or row["deploy_explicit_awareness"])
        for row in rows
    ])


def slice5_contrast(tag: str, dataset: Path, reconstructions: Path,
                    read_point: str, mask: np.ndarray | None = None,
                    genuine_tag: str | None = None) -> Contrast:
    from vnaa.run_slice5 import acts_path as s5_acts_path
    from vnaa.run_slice5 import load_pairs

    genuine, rebuilt, keep = load_pairs(
        argparse.Namespace(dataset=dataset, reconstructions=reconstructions))
    keep = keep if mask is None else (keep & mask)
    # The genuine side is the same conversations under every condition, so only
    # one condition's genuine activations were ever extracted.  run_slice5.compare
    # resolves this the same way, by reading the left condition's copy.
    pos = np.load(s5_acts_path(tag, "resampled", read_point))["acts"].astype(np.float64)[keep]
    neg = np.load(s5_acts_path(genuine_tag or tag, "genuine", read_point))[
        "acts"].astype(np.float64)[keep]
    return (pos, neg,
            [serialize(c) for c, k in zip(rebuilt, keep) if k],
            [serialize(c) for c, k in zip(genuine, keep) if k])


def score_contrast(contrast: Contrast, seeds, test_frac, n_shuffles, n_draws,
                   n_permutations, C=DEFAULT_C) -> dict:
    """Difference-of-means, logistic regression, and all three nulls.

    Both estimators see exactly the same splits, so the comparison isolates the
    fitting rule.  Neither consults the test split when choosing a layer.

    The permutation null is run on the first seed only.  It is a property of the
    contrast rather than of a split, and rerunning the whole pipeline hundreds of
    times per seed buys nothing but wall clock.
    """
    acts_pos, acts_neg, _, _ = contrast
    out: dict[str, list] = {k: [] for k in (
        "dom", "dom_layer", "logistic", "logistic_layer",
        "null_shuffled", "null_random")}
    for seed in seeds:
        tr_p, te_p = split(len(acts_pos), test_frac, seed)
        tr_n, te_n = split(len(acts_neg), test_frac, seed)
        x_tr = np.concatenate([acts_pos[tr_p], acts_neg[tr_n]])
        y_tr = np.concatenate([np.ones(len(tr_p)), np.zeros(len(tr_n))])
        x_te = np.concatenate([acts_pos[te_p], acts_neg[te_n]])
        y_te = np.concatenate([np.ones(len(te_p)), np.zeros(len(te_n))])

        train_auroc = []
        probes = []
        for layer in range(x_tr.shape[1]):
            probe = fit_layer(x_tr[:, layer, :], y_tr)
            probes.append(probe)
            train_auroc.append(roc_auc_score(y_tr, probe.project(x_tr[:, layer, :])))
        best = int(np.argmax(np.abs(np.array(train_auroc) - 0.5)))
        out["dom"].append(roc_auc_score(y_te, probes[best].project(x_te[:, best, :])))
        out["dom_layer"].append(best)

        auroc, layer = logistic_probe_auroc(x_tr, y_tr, x_te, y_te, C=C, seed=seed)
        out["logistic"].append(auroc)
        out["logistic_layer"].append(layer)

        # The geometry diagnostics are reported at the layer the real probe
        # chose, so they answer "what does an uninformed direction get at the
        # layer we report" rather than "what is the luckiest of 33 layers".
        out["null_shuffled"].append(
            train_shuffled_direction_null(x_tr, y_tr, x_te, y_te, best, n_shuffles, seed))
        out["null_random"].append(
            random_direction_null(x_te, y_te, best, n_draws, seed))

        if seed == seeds[0]:
            pooled = np.concatenate([x_tr, x_te])
            pooled_labels = np.concatenate([y_tr, y_te])
            out["permutation"] = permutation_null(
                pooled, pooled_labels, len(y_tr),
                observed_strength=strength(out["dom"][-1]),
                n_permutations=n_permutations, seed=seed)
    return out


def summarize(raw: dict) -> dict:
    return {
        "dom": [float(np.mean(raw["dom"])), float(np.std(raw["dom"]))],
        "dom_layer": float(np.mean(raw["dom_layer"])),
        "logistic": [float(np.mean(raw["logistic"])), float(np.std(raw["logistic"]))],
        "logistic_layer": float(np.mean(raw["logistic_layer"])),
        "permutation_null": raw["permutation"],
        "geometry_shuffled_strength": float(np.mean(
            [n["mean_strength"] for n in raw["null_shuffled"]])),
        "geometry_shuffled_auroc": float(np.mean(
            [n["mean_auroc"] for n in raw["null_shuffled"]])),
        "geometry_random_strength": float(np.mean(
            [n["mean_strength"] for n in raw["null_random"]])),
    }


def cmd_points(args: argparse.Namespace) -> None:
    cells = load_cells()
    contrasts: dict[str, Contrast] = {}

    for cell in ("eval_shared", "eval_disguised", "eval_obvious"):
        for read_point in READ_POINTS:
            contrasts[f"s3_{cell}|{read_point}"] = slice3_contrast(cells, cell, read_point)

    for read_point in READ_POINTS:
        contrasts[f"s3_coherent_pairs|{read_point}"] = coherent_contrast(read_point)
    mask = silent_mask()
    contrasts["s3_coherent_silent|last_user_token"] = coherent_contrast(
        "last_user_token", mask)

    data = Path("data")
    s5 = [
        ("reconstruction_n250", data / "slice5_genuine_n250.json",
         data / "slice5_reconstructed_n250.json", None, None),
    ]
    # The decomposition is only interpretable on the shared terminated subset,
    # so both conditions are masked to the same 146 pairs as the reported run.
    swap_recon = data / "slice5_reconstructed_llama32_3b_n250.json"
    ablate_recon = data / "slice5_reconstructed_final_user_only_n250.json"
    if swap_recon.exists() and ablate_recon.exists():
        from vnaa.run_slice5 import load_pairs, shared_terminated_mask

        genuine_path = data / "slice5_genuine_n250.json"
        _, _, left = load_pairs(argparse.Namespace(
            dataset=genuine_path, reconstructions=swap_recon))
        _, _, right = load_pairs(argparse.Namespace(
            dataset=genuine_path, reconstructions=ablate_recon))
        shared = shared_terminated_mask(left, right)
        swap_tag = "model_swap_llama32_3b_n250"
        s5 += [
            (swap_tag, genuine_path, swap_recon, shared, None),
            ("context_ablation_final_user_only_n250", genuine_path, ablate_recon,
             shared, swap_tag),
        ]
    for tag, dataset, reconstructions, s5_mask, genuine_tag in s5:
        if not dataset.exists() or not reconstructions.exists():
            print(f"skipping s5 {tag}: private artifacts absent")
            continue
        for read_point in ASSISTANT_READ_POINTS:
            contrasts[f"s5_{tag}|{read_point}"] = slice5_contrast(
                tag, dataset, reconstructions, read_point, s5_mask, genuine_tag)

    results = {}
    header = (f"{'contrast':<44} {'n':>4} {'diff-of-means':>15} {'logistic':>15} "
              f"{'perm p':>7} {'perm p95':>9} {'geom':>6}")
    print(header)
    print("-" * len(header))
    for name, contrast in contrasts.items():
        raw = score_contrast(contrast, args.seeds, args.test_frac,
                             args.n_shuffles, args.n_draws,
                             args.n_permutations, args.C)
        summary = summarize(raw)
        summary["n_per_cell"] = len(contrast[0])
        results[name] = summary
        null = summary["permutation_null"]
        print(f"{name:<44} {summary['n_per_cell']:>4} "
              f"{summary['dom'][0]:>8.3f}+-{summary['dom'][1]:.3f} "
              f"{summary['logistic'][0]:>8.3f}+-{summary['logistic'][1]:.3f} "
              f"{null['p_value']:>7.3f} {null['p95_strength']:>9.3f} "
              f"{summary['geometry_shuffled_strength']:>6.3f}")

    out = RESULTS / f"estimator_check_{MODEL_TAG}_points.json"
    payload = {
        "seeds": args.seeds,
        "test_frac": args.test_frac,
        "logistic_C": args.C,
        "n_permutations": args.n_permutations,
        "n_shuffles": args.n_shuffles,
        "n_random_draws": args.n_draws,
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved {out}")


def cmd_transfer(args: argparse.Namespace) -> None:
    """Cross-cell transfer under the logistic estimator, math-free cells only.

    Slice 2b's headline is a transfer margin, not an in-distribution AUROC, so
    the estimator check has to be run on transfer to mean anything.  gsm8k items
    are excluded on both sides for the reason recorded in CLAUDE.md: their digit
    density separates the cells without any model involvement.
    """
    from vnaa.run_transfer import PAIRS, cell_sources, load_acts, load_texts

    results_dir = Path(args.results)
    texts, acts = {}, {}
    for cell in sorted({c for pair in PAIRS.values() for c in pair}):
        cell_texts = load_texts(cell)
        cell_acts = load_acts(cell, results_dir)
        sources = cell_sources(cell, len(cell_texts))
        keep = [i for i, s in enumerate(sources) if s != "gsm8k"]
        texts[cell] = [cell_texts[i] for i in keep]
        acts[cell] = cell_acts[keep]

    rows = {}
    for train_name in PAIRS:
        per_seed = {test: [] for test in PAIRS}
        layers = []
        for seed in args.seeds:
            splits = {c: split(len(texts[c]), args.test_frac, seed) for c in texts}

            def part(name: str, which: int):
                pos, neg = PAIRS[name]
                pi, ni = splits[pos][which], splits[neg][which]
                x = np.concatenate([acts[pos][pi], acts[neg][ni]])
                y = np.concatenate([np.ones(len(pi)), np.zeros(len(ni))])
                return x, y

            x_tr, y_tr = part(train_name, 0)
            best, _ = select_layer_cv(x_tr, y_tr, C=args.C, seed=seed)
            probe = fit_layer_logistic(x_tr[:, best, :], y_tr, args.C, seed)
            layers.append(best)
            # Freeze the sign on the training pair.  A direction that transfers
            # inverted is still a failure to transfer, not a hidden success.
            sign = 1.0 if roc_auc_score(
                y_tr, probe.project(x_tr[:, best, :])) >= 0.5 else -1.0
            for test_name in PAIRS:
                x_te, y_te = part(test_name, 1)
                per_seed[test_name].append(
                    roc_auc_score(y_te, sign * probe.project(x_te[:, best, :])))
        rows[train_name] = {
            "best_layer": float(np.mean(layers)),
            "to": {test: [float(np.mean(v)), float(np.std(v))]
                   for test, v in per_seed.items()},
        }

    names = list(PAIRS)
    print(f"logistic estimator, math-free transfer (rows train, cols test)\n")
    print(f"{'train':<18}" + "".join(f"{n[:16]:>18}" for n in names))
    for train_name in names:
        cells_out = "".join(
            f"{rows[train_name]['to'][t][0]:>18.3f}" for t in names)
        print(f"{train_name:<18}{cells_out}")

    off_diagonal = [strength(rows[a]["to"][b][0])
                    for a in names for b in names if a != b]
    print(f"\nmean off-diagonal transfer strength: {np.mean(off_diagonal):.3f}")

    out = RESULTS / f"estimator_check_{MODEL_TAG}_transfer_mathfree.json"
    out.write_text(json.dumps({
        "seeds": args.seeds, "test_frac": args.test_frac, "logistic_C": args.C,
        "mean_off_diagonal_strength": float(np.mean(off_diagonal)),
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("points", help="point read-outs across slices 3 and 5")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--test-frac", type=float, default=0.3)
    p.add_argument("--n-permutations", type=int, default=500)
    p.add_argument("--n-shuffles", type=int, default=200)
    p.add_argument("--n-draws", type=int, default=200)
    p.add_argument("--C", type=float, default=DEFAULT_C)
    p.set_defaults(func=cmd_points)

    t = sub.add_parser("transfer", help="slice 2b cross-cell transfer, math-free")
    t.add_argument("--results", default="results")
    t.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    t.add_argument("--test-frac", type=float, default=0.3)
    t.add_argument("--C", type=float, default=DEFAULT_C)
    t.set_defaults(func=cmd_transfer)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
