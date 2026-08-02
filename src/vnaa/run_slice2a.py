"""Slice 2a: our probe machinery on Devbunova's public 2x2 dataset.

Two phases, so extraction (GPU, slow) and analysis (CPU, instant) can rerun
independently:

  python -m vnaa.run_slice2a extract --model meta-llama/Llama-3.1-8B-Instruct \
      --dtype bfloat16 --device-map auto
  python -m vnaa.run_slice2a analyze --model meta-llama/Llama-3.1-8B-Instruct

Extract caches per-cell activations to results/s2a_{model}_{cell}.npz and skips
cells already cached.

Analyze fits difference-of-means probes under two training regimes and reports
what Devbunova reports, in our metrics:

- standard: bench_eval vs casual_deploy_matched (the confounded contrast).
  Their result: this probe calls bench_deploy_rewritten "evaluation" 94.5% of
  the time, i.e. it is a format detector.
- paired: (bench_eval + casual_eval_mutual) vs (casual_deploy_matched +
  bench_deploy_rewritten). Their result: format-invariance restored. Our
  surface baselines showed casual_eval_mutual is surface-distinguishable
  (format classifier 0.9988), so this regime's success is exactly what our
  rewrite arm stress-tests.

Layer selection: the best layer is chosen on the in-distribution held-out
split of the training contrast, then held fixed for every transfer readout.
Picking the best layer per transfer target would leak the test signal into
model selection.

The classified-as-eval fraction uses a threshold chosen by Youden's J on the
train split, mirroring their 0.5-probability readout in our projection space.
Each cell keeps a fixed 30% held-out slice (by seed), so train never sees any
prompt scored at test time, in any regime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from vnaa.baselines import bootstrap_auroc_ci, strength
from vnaa.data_2x2 import as_conversations, load_cell
from vnaa.probe import fit_layer

CELLS = [
    "bench_eval",
    "casual_deploy_matched",
    "bench_deploy_rewritten",
    "casual_eval_mutual",
]


def acts_path(out_dir: Path, model: str, cell: str) -> Path:
    tag = model.split("/")[-1]
    return out_dir / f"s2a_{tag}_{cell}.npz"


def cmd_extract(args: argparse.Namespace) -> None:
    import torch

    from vnaa.extract import extract_activations, load_model

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    todo = [c for c in CELLS if not acts_path(out_dir, args.model, c).exists()]
    done = [c for c in CELLS if c not in todo]
    if done:
        print(f"cached, skipping: {done}")
    if not todo:
        print("all cells cached; nothing to extract")
        return

    lm = load_model(args.model, device=args.device, dtype=dtype,
                    device_map=args.device_map)
    print(f"model={args.model} device={lm.device} n_layers={lm.n_layers}")

    for cell in todo:
        texts = load_cell(cell)
        if args.limit:
            texts = texts[: args.limit]
        print(f"extracting {cell}: n={len(texts)}")
        acts = extract_activations(lm, as_conversations(texts))
        assert np.isfinite(acts).all(), f"non-finite activations in {cell}"
        np.savez_compressed(acts_path(out_dir, args.model, cell), acts=acts)
        print(f"saved {acts_path(out_dir, args.model, cell)}")


def held_split(n: int, test_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic per-cell split; the same held-out slice in every regime."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(n * test_frac))
    return idx[n_test:], idx[:n_test]


def youden_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    s, y = scores[order], labels[order]
    n_pos, n_neg = y.sum(), (1 - y).sum()
    best_j, best_t = -1.0, s[0]
    tp, fp = n_pos, n_neg
    for i in range(len(s)):
        j = tp / n_pos - fp / n_neg
        if j > best_j:
            best_j, best_t = j, s[i]
        if y[i] == 1:
            tp -= 1
        else:
            fp -= 1
    return float(best_t)


def cmd_analyze(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    acts = {}
    for cell in CELLS:
        p = acts_path(out_dir, args.model, cell)
        if not p.exists():
            raise SystemExit(f"missing {p}; run the extract phase first")
        acts[cell] = np.load(p)["acts"].astype(np.float64)
    n_layers = acts[CELLS[0]].shape[1]

    splits = {c: held_split(len(acts[c]), args.test_frac, args.seed) for c in CELLS}
    tr = {c: acts[c][splits[c][0]] for c in CELLS}
    te = {c: acts[c][splits[c][1]] for c in CELLS}

    regimes = {
        "standard": (["bench_eval"], ["casual_deploy_matched"]),
        "paired": (
            ["bench_eval", "casual_eval_mutual"],
            ["casual_deploy_matched", "bench_deploy_rewritten"],
        ),
    }
    results: dict[str, dict] = {}

    for name, (pos_cells, neg_cells) in regimes.items():
        x_tr = np.concatenate([tr[c] for c in pos_cells + neg_cells])
        y_tr = np.concatenate(
            [np.ones(len(tr[c])) for c in pos_cells]
            + [np.zeros(len(tr[c])) for c in neg_cells]
        )
        x_te = np.concatenate([te[c] for c in pos_cells + neg_cells])
        y_te = np.concatenate(
            [np.ones(len(te[c])) for c in pos_cells]
            + [np.zeros(len(te[c])) for c in neg_cells]
        )

        per_layer = np.empty(n_layers)
        probes = []
        for layer in range(n_layers):
            probe = fit_layer(x_tr[:, layer, :], y_tr)
            probes.append(probe)
            per_layer[layer] = roc_auc_score(y_te, probe.project(x_te[:, layer, :]))
        best = int(np.argmax(np.abs(per_layer - 0.5)))
        probe = probes[best]
        sign = 1.0 if per_layer[best] >= 0.5 else -1.0

        print(f"\n=== regime: {name} ===")
        print(f"train: {pos_cells} (1) vs {neg_cells} (0), n={len(x_tr)}")
        ci = bootstrap_auroc_ci(
            y_te.astype(int),
            probe.project(x_te[:, best, :]),
            seed=args.seed,
        )
        print(
            f"best layer {best}/{n_layers - 1} (0=embeddings)  "
            f"in-dist AUROC {per_layer[best]:.4f} "
            f"[{ci[0]:.4f}, {ci[1]:.4f}]  strength {strength(per_layer[best]):.4f}"
        )
        print(f"layer curve (every 4th): "
              + " ".join(f"{i}:{per_layer[i]:.3f}" for i in range(0, n_layers, 4)))

        thr = youden_threshold(sign * probe.project(x_tr[:, best, :]), y_tr)
        frac_eval = {}
        for cell in CELLS:
            s = sign * probe.project(te[cell][:, best, :])
            frac_eval[cell] = float((s >= thr).mean())
        print("held-out fraction classified as eval (Devbunova-comparable):")
        for cell, frac in frac_eval.items():
            print(f"  {cell:<24} {frac:.3f}")

        transfers = {}
        for pos, neg in [
            ("bench_eval", "bench_deploy_rewritten"),
            ("casual_eval_mutual", "casual_deploy_matched"),
            ("bench_deploy_rewritten", "casual_deploy_matched"),
        ]:
            s = np.concatenate(
                [probe.project(te[pos][:, best, :]), probe.project(te[neg][:, best, :])]
            )
            y = np.concatenate([np.ones(len(te[pos])), np.zeros(len(te[neg]))])
            transfers[f"{pos} vs {neg}"] = float(roc_auc_score(y, s))
        print("transfer AUROCs at the fixed best layer:")
        for k, v in transfers.items():
            print(f"  {k:<48} {v:.4f}  (strength {strength(v):.4f})")

        results[name] = {
            "best_layer": best,
            "in_dist_auroc": float(per_layer[best]),
            "in_dist_ci": ci,
            "auroc_by_layer": per_layer.tolist(),
            "frac_classified_eval": frac_eval,
            "transfer_auroc": transfers,
        }

    tag = args.model.split("/")[-1]
    out = out_dir / f"s2a_{tag}_analysis.json"
    with open(out, "w") as f:
        json.dump({"model": args.model, "seed": args.seed, "regimes": results}, f, indent=2)
    print(f"\nsaved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Slice 2a: probe on the public 2x2")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract")
    ex.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ex.add_argument("--device", default=None)
    ex.add_argument("--device-map", default=None)
    ex.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ex.add_argument("--limit", type=int, default=None, help="cap per cell (debug)")
    ex.add_argument("--out-dir", default="results")
    ex.set_defaults(func=cmd_extract)

    an = sub.add_parser("analyze")
    an.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    an.add_argument("--test-frac", type=float, default=0.3)
    an.add_argument("--seed", type=int, default=0)
    an.add_argument("--out-dir", default="results")
    an.set_defaults(func=cmd_analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
