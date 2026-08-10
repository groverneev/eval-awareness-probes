"""Extract and analyze the paired matched-coherence pilot.

This is deliberately a 50-pair gate before authoring a 250-pair replacement
for the invalid real-prefix control.  The two cells have byte-identical prefixes
within an item, so pre-final separation is a hard implementation check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vnaa.extract import READ_POINTS
from vnaa.run_slice3 import (
    MODEL_TAG,
    RESULTS,
    differing_final_messages,
    first_user,
    score_pair,
)

PILOT = Path("datasets/slice3/coherent_pair_pilot/validated.json")


def load_cells(dataset: Path = PILOT):
    rows = json.load(open(dataset, encoding="utf-8"))["items"]
    return {"eval": [r["eval"] for r in rows], "deploy": [r["deploy"] for r in rows]}


def path(tag: str, cell: str, rp: str) -> Path:
    return RESULTS / f"s3_{MODEL_TAG}_{tag}_{cell}_{rp}.npz"


def extract(args) -> None:
    import torch
    from vnaa.extract import extract_activations, load_model

    cells = load_cells(args.dataset)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    lm = load_model(args.model, dtype=dtype, device=args.device, device_map=args.device_map)
    RESULTS.mkdir(exist_ok=True)
    for cell, convs in cells.items():
        for rp in READ_POINTS:
            out = path(args.tag, cell, rp)
            if out.exists():
                continue
            acts = extract_activations(lm, convs, read_point=rp, show_progress=False)
            np.savez_compressed(out, acts=acts)
            print(f"saved {out.name} {acts.shape}")


def analyze(args) -> None:
    cells = load_cells(args.dataset)
    results = {}
    for rp in READ_POINTS:
        pos = np.load(path(args.tag, "eval", rp))["acts"].astype(np.float64)
        neg = np.load(path(args.tag, "deploy", rp))["acts"].astype(np.float64)
        if rp == "post_instruction_token":
            text_pos = [first_user(c) for c in cells["eval"]]
            text_neg = [first_user(c) for c in cells["deploy"]]
        else:
            text_pos, text_neg = differing_final_messages(
                cells["eval"], cells["deploy"]
            )
        result = score_pair(
            pos, neg, text_pos, text_neg, args.seeds, args.test_frac
        )
        results[rp] = {k: (float(np.mean(v)), float(np.std(v))) for k, v in result.items()}
    for rp, r in results.items():
        print(f"{rp:<24} probe {r['probe'][0]:.3f}+/-{r['probe'][1]:.3f} "
              f"bow {r['bow'][0]:.3f}+/-{r['bow'][1]:.3f} "
              f"format {r['format'][0]:.3f}+/-{r['format'][1]:.3f}")
    out = RESULTS / f"s3_{MODEL_TAG}_{args.tag}_analysis.json"
    json.dump({
        "n": len(cells["eval"]),
        "paired_surface_text_view": "differing_final_message_only",
        "results": results,
    }, open(out, "w"), indent=2)
    print(f"saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="command", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--model", default=f"meta-llama/{MODEL_TAG}")
    e.add_argument("--device", default=None)
    e.add_argument("--device-map", default=None)
    e.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    e.add_argument("--dataset", type=Path, default=PILOT)
    e.add_argument("--tag", default="coherent_pilot")
    e.set_defaults(func=extract)
    a = sp.add_parser("analyze")
    a.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a.add_argument("--test-frac", type=float, default=0.3)
    a.add_argument("--dataset", type=Path, default=PILOT)
    a.add_argument("--tag", default="coherent_pilot")
    a.set_defaults(func=analyze)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
