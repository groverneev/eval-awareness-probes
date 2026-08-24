"""Post-result Priority 8 steering inference and direction controls.

The frozen audit is Amendment 2 of ``datasets/steering/prespec.md``.  It adds
prompt-clustered inference to the already generated learned-direction outputs
and evaluates five seeded, norm-matched random directions that are explicitly
orthogonal to the learned direction.  Original steering artifacts are read but
never overwritten.

Operations::

    python -m vnaa.run_steering_priority8 prepare                 # CPU
    python -m vnaa.run_steering_priority8 analyze                 # CPU, learned result now
    python -m vnaa.run_steering_priority8 readout --dtype bfloat16
    python -m vnaa.run_steering_priority8 generate --set self_report
    python -m vnaa.run_steering_priority8 generate --set safety
    python -m vnaa.run_steering_priority8 analyze                 # CPU, final audit

The readout and generation operations load the target model and require GPU
compute for the paper run.  ``prepare`` and ``analyze`` do not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

from vnaa.extract import extract_activations, load_model
from vnaa.run_silence_filter import atomic_json, greedy_reply
from vnaa.run_slice3 import MODEL_TAG, RESULTS
from vnaa.run_steering import (
    DIRECTION,
    MAX_REPEAT_FRACTION,
    MIN_LENGTH_RATIO,
    PROMPTS,
    READOUT,
    as_conversation,
    chosen_layer,
    direction_for,
    generations_path,
    is_refusal,
    load_direction,
    prompt_set,
    repeat_fraction,
    self_report_label,
)
from vnaa.steering import steer


AUDIT_ID = "postresult_steering_priority8_v1"
PRESPEC = Path("datasets/steering/prespec.md")
AUDIT_DIR = RESULTS / "steering_priority8"
CONTROL_DIRECTIONS = AUDIT_DIR / f"steer_{MODEL_TAG}_orthogonal_controls.npz"
CONTROL_READOUT = AUDIT_DIR / f"steer_{MODEL_TAG}_control_readout.json"
PROMPT_INFERENCE = AUDIT_DIR / f"steer_{MODEL_TAG}_prompt_inference.json"

CONTROL_SEEDS = tuple(range(20260823, 20260828))
CONTROL_ALPHAS = (-2.0, 2.0)
NONDEGENERATE_ALPHAS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20260823
ORTHOGONAL_TOLERANCE = 1e-12


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def control_generations_path(name: str) -> Path:
    return AUDIT_DIR / f"steer_{MODEL_TAG}_control_generations_{name}.json"


def random_orthogonal_directions(
    target: np.ndarray, seeds: tuple[int, ...] | list[int]
) -> np.ndarray:
    """Seeded isotropic directions projected orthogonal to ``target``."""
    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 1 or not np.isfinite(target).all():
        raise ValueError("target direction must be one finite vector")
    norm = float(np.linalg.norm(target))
    if norm <= 0:
        raise ValueError("target direction has zero norm")
    unit = target / norm

    controls = []
    for seed in seeds:
        vector = np.random.Generator(np.random.PCG64(int(seed))).standard_normal(unit.size)
        vector -= float(vector @ unit) * unit
        vector_norm = float(np.linalg.norm(vector))
        if vector_norm <= 0:
            raise RuntimeError(f"seed {seed} produced a zero orthogonal vector")
        controls.append(vector / vector_norm)
    return np.stack(controls)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def prepare_control_artifact() -> dict:
    frozen = load_direction()
    layer = chosen_layer(frozen, None)
    target, scale = direction_for(frozen, layer)
    target = np.asarray(target, dtype=np.float64)
    target /= np.linalg.norm(target)
    controls = random_orthogonal_directions(target, CONTROL_SEEDS)
    cosines = controls @ target

    expected = {
        "directions": controls,
        "seeds": np.asarray(CONTROL_SEEDS, dtype=np.int64),
        "target_direction": target,
        "target_cosines": cosines,
        "layer": np.asarray(layer, dtype=np.int64),
        "scale": np.asarray(scale, dtype=np.float64),
        "source_direction_sha256": np.asarray(file_sha256(DIRECTION)),
        "prespec_sha256": np.asarray(file_sha256(PRESPEC)),
        "audit_id": np.asarray(AUDIT_ID),
    }
    if CONTROL_DIRECTIONS.exists():
        saved = np.load(CONTROL_DIRECTIONS)
        for key, value in expected.items():
            if key not in saved.files or not np.array_equal(saved[key], value):
                raise ValueError(
                    f"{CONTROL_DIRECTIONS} conflicts with the frozen {key}; "
                    "do not overwrite an existing control audit"
                )
    else:
        _atomic_npz(CONTROL_DIRECTIONS, **expected)
    return load_control_artifact()


def load_control_artifact() -> dict:
    if not CONTROL_DIRECTIONS.exists():
        raise SystemExit(
            f"{CONTROL_DIRECTIONS} missing; run `run_steering_priority8 prepare` first"
        )
    saved = np.load(CONTROL_DIRECTIONS)
    result = {key: saved[key] for key in saved.files}
    directions = np.asarray(result["directions"], dtype=np.float64)
    target = np.asarray(result["target_direction"], dtype=np.float64)
    if not np.allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-12):
        raise ValueError("control directions are not unit norm")
    if np.max(np.abs(directions @ target)) > ORTHOGONAL_TOLERANCE:
        raise ValueError("control directions are not orthogonal to the learned direction")
    if tuple(int(seed) for seed in result["seeds"]) != CONTROL_SEEDS:
        raise ValueError("control seed set differs from the frozen audit")
    if str(result["source_direction_sha256"]) != file_sha256(DIRECTION):
        raise ValueError("learned direction artifact changed after controls were prepared")
    if str(result["prespec_sha256"]) != file_sha256(PRESPEC):
        raise ValueError("Priority 8 prespec changed after controls were prepared")
    if str(result["audit_id"]) != AUDIT_ID:
        raise ValueError("control artifact belongs to a different audit")
    return result


def cmd_prepare(_args) -> None:
    artifact = prepare_control_artifact()
    print(f"saved {CONTROL_DIRECTIONS}")
    print(
        f"layer {int(artifact['layer'])}; scale {float(artifact['scale']):.6f}; "
        f"{len(artifact['seeds'])} controls"
    )
    for seed, cosine in zip(artifact["seeds"], artifact["target_cosines"]):
        print(f"  seed {int(seed)} cosine(target) {float(cosine):+.3e}")


def _binary_outcome(name: str, reply: str) -> int:
    if name == "self_report":
        return int(self_report_label(reply) == "evaluation")
    if name == "safety":
        return int(is_refusal(reply))
    raise ValueError(f"unknown steering outcome {name!r}")


def outcome_table(
    name: str, items: list[dict], alphas: tuple[float, ...] | list[float]
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Return prompt-aligned binary outcomes, rejecting missing or mixed rows."""
    requested = tuple(float(alpha) for alpha in alphas)
    collected: dict[float, dict[int, int]] = {alpha: {} for alpha in requested}
    for row in items:
        alpha = float(row["alpha"])
        if alpha not in collected:
            continue
        index = int(row["index"])
        if index in collected[alpha]:
            raise ValueError(f"duplicate row for alpha {alpha} prompt {index}")
        collected[alpha][index] = _binary_outcome(name, row["reply"])

    first = set(collected[requested[0]])
    if not first:
        raise ValueError(f"no rows for alpha {requested[0]}")
    for alpha in requested[1:]:
        if set(collected[alpha]) != first:
            raise ValueError("alphas do not contain the same prompt indices")
    indices = np.asarray(sorted(first), dtype=np.int64)
    outcomes = {
        alpha: np.asarray([collected[alpha][int(index)] for index in indices], dtype=np.int8)
        for alpha in requested
    }
    return indices, outcomes


def paired_binary_contrast(
    high: np.ndarray,
    low: np.ndarray,
    *,
    high_alpha: float,
    low_alpha: float,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Prompt-paired risk difference, cluster bootstrap, and exact McNemar test."""
    high = np.asarray(high, dtype=np.int8)
    low = np.asarray(low, dtype=np.int8)
    if high.ndim != 1 or high.shape != low.shape or len(high) == 0:
        raise ValueError("paired outcomes must be nonempty vectors of equal length")
    if not np.isin(high, [0, 1]).all() or not np.isin(low, [0, 1]).all():
        raise ValueError("paired outcomes must be binary")

    differences = high.astype(np.float64) - low.astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(differences), size=(n_bootstrap, len(differences)))
    boot = differences[draws].mean(axis=1)
    positive = int(np.sum((high == 1) & (low == 0)))
    negative = int(np.sum((high == 0) & (low == 1)))
    discordant = positive + negative
    p_value = (
        float(binomtest(positive, discordant, p=0.5, alternative="two-sided").pvalue)
        if discordant
        else 1.0
    )
    return {
        "high_alpha": float(high_alpha),
        "low_alpha": float(low_alpha),
        "n_prompts": int(len(high)),
        "high_rate": float(high.mean()),
        "low_rate": float(low.mean()),
        "risk_difference": float(differences.mean()),
        "bootstrap_95_interval": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "discordant_high_only": positive,
        "discordant_low_only": negative,
        "mcnemar_exact_two_sided_p": p_value,
    }


def prompt_slope_inference(
    outcomes: dict[float, np.ndarray],
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Mean within-prompt OLS slope with a prompt-cluster bootstrap interval."""
    alphas = np.asarray(sorted(outcomes), dtype=np.float64)
    matrix = np.stack([outcomes[float(alpha)] for alpha in alphas], axis=1).astype(float)
    centered = alphas - alphas.mean()
    slopes = (matrix @ centered) / float(centered @ centered)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(slopes), size=(n_bootstrap, len(slopes)))
    boot = slopes[draws].mean(axis=1)
    return {
        "alphas": alphas.tolist(),
        "n_prompts": int(len(slopes)),
        "mean_within_prompt_slope": float(slopes.mean()),
        "bootstrap_95_interval": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
    }


def learned_prompt_inference(
    name: str,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    path = generations_path(name)
    saved = json.loads(path.read_text(encoding="utf-8"))
    indices, outcomes = outcome_table(name, saved["items"], NONDEGENERATE_ALPHAS)
    contrasts = {
        "plus2_minus_minus2": paired_binary_contrast(
            outcomes[2.0], outcomes[-2.0], high_alpha=2.0, low_alpha=-2.0,
            n_bootstrap=n_bootstrap, seed=seed,
        ),
        "plus2_minus_zero": paired_binary_contrast(
            outcomes[2.0], outcomes[0.0], high_alpha=2.0, low_alpha=0.0,
            n_bootstrap=n_bootstrap, seed=seed,
        ),
        "minus2_minus_zero": paired_binary_contrast(
            outcomes[-2.0], outcomes[0.0], high_alpha=-2.0, low_alpha=0.0,
            n_bootstrap=n_bootstrap, seed=seed,
        ),
    }
    return {
        "source": str(path),
        "source_sha256": file_sha256(path),
        "prompt_indices": indices.tolist(),
        "outcome": "evaluation" if name == "self_report" else "refusal",
        "primary_contrast": "plus2_minus_minus2",
        "contrasts": contrasts,
        "whole_curve": prompt_slope_inference(
            outcomes, n_bootstrap=n_bootstrap, seed=seed
        ),
    }


def _artifact_header(model: str, controls: dict, operation: str) -> dict:
    return {
        "audit_id": AUDIT_ID,
        "operation": operation,
        "model": model,
        "prompts_sha256": file_sha256(PROMPTS),
        "prespec_sha256": file_sha256(PRESPEC),
        "control_directions_sha256": file_sha256(CONTROL_DIRECTIONS),
        "steering_layer": int(controls["layer"]),
        "scale": float(controls["scale"]),
        "control_seeds": [int(seed) for seed in controls["seeds"]],
        "alphas": list(CONTROL_ALPHAS),
    }


def _load_or_initialize_json(path: Path, expected: dict) -> dict:
    if not path.exists():
        return {**expected, "items": []}
    saved = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(
                f"{path} was made with a different {key}; do not mix control runs"
            )
    return saved


def cmd_readout(args) -> None:
    controls = load_control_artifact()
    frozen = load_direction()
    conversations = [as_conversation(text) for text in prompt_set("self_report")]
    expected = _artifact_header(args.model, controls, "independent_readout")
    saved = _load_or_initialize_json(CONTROL_READOUT, expected)
    done = {(int(row["control_seed"]), float(row["alpha"])) for row in saved["items"]}

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    lm = load_model(args.model, device=args.device, dtype=dtype, device_map=args.device_map)
    readout_layer = int(frozen["readout_layer"])
    readout_direction = np.asarray(frozen["readout_direction"], dtype=np.float64)
    deploy_reference = float(frozen["reference_deploy"].mean())
    eval_reference = float(frozen["reference_eval"].mean())
    span = eval_reference - deploy_reference

    for seed, direction in zip(controls["seeds"], controls["directions"]):
        for alpha in CONTROL_ALPHAS:
            key = (int(seed), float(alpha))
            if key in done:
                continue
            with steer(
                lm.model, int(controls["layer"]), direction,
                alpha * float(controls["scale"]),
            ):
                acts = extract_activations(
                    lm, conversations, read_point="last_user_token", show_progress=False
                )
            scores = acts[:, readout_layer, :].astype(np.float64) @ readout_direction
            saved["items"].append({
                "control_seed": int(seed),
                "alpha": float(alpha),
                "mean": float(scores.mean()),
                "sd": float(scores.std()),
                "gap_fraction": float((scores.mean() - deploy_reference) / span),
            })
            atomic_json(CONTROL_READOUT, saved)
            print(
                f"seed {int(seed)} alpha {alpha:+.1f}: readout {scores.mean():+.3f} "
                f"({saved['items'][-1]['gap_fraction']:+.3f} of gap)", flush=True,
            )
    print(f"saved {CONTROL_READOUT}")


def cmd_generate(args) -> None:
    controls = load_control_artifact()
    prompts = prompt_set(args.set)
    out = control_generations_path(args.set)
    expected = {
        **_artifact_header(args.model, controls, f"generation_{args.set}"),
        "set": args.set,
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
    }
    saved = _load_or_initialize_json(out, expected)
    done = {
        (int(row["control_seed"]), float(row["alpha"]), int(row["index"]))
        for row in saved["items"]
    }

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    lm = load_model(args.model, device=args.device, dtype=dtype, device_map=args.device_map)
    for seed, direction in zip(controls["seeds"], controls["directions"]):
        for alpha in CONTROL_ALPHAS:
            with steer(
                lm.model, int(controls["layer"]), direction,
                alpha * float(controls["scale"]),
            ):
                for index, text in enumerate(prompts):
                    key = (int(seed), float(alpha), index)
                    if key in done:
                        continue
                    reply, tokens, terminated = greedy_reply(
                        lm, as_conversation(text), args.max_new_tokens
                    )
                    saved["items"].append({
                        "control_seed": int(seed),
                        "alpha": float(alpha),
                        "index": index,
                        "reply": reply,
                        "generated_tokens": tokens,
                        "terminated": terminated,
                    })
                    atomic_json(out, saved)
            print(
                f"seed {int(seed)} alpha {alpha:+.1f}: {len(prompts)} generations done",
                flush=True,
            )
    print(f"saved {out}")


def _baseline_mean_tokens(name: str) -> float:
    saved = json.loads(generations_path(name).read_text(encoding="utf-8"))
    rows = [row for row in saved["items"] if float(row["alpha"]) == 0.0]
    if not rows:
        raise ValueError(f"learned {name} artifact has no zero baseline")
    return float(np.mean([row["generated_tokens"] for row in rows]))


def control_prompt_inference(
    name: str,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict] | None:
    path = control_generations_path(name)
    if not path.exists():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    baseline_tokens = _baseline_mean_tokens(name)
    results = []
    for control_seed in CONTROL_SEEDS:
        rows = [
            row for row in saved["items"] if int(row["control_seed"]) == control_seed
        ]
        indices, outcomes = outcome_table(name, rows, CONTROL_ALPHAS)
        endpoint_quality = []
        for alpha in CONTROL_ALPHAS:
            endpoint = [row for row in rows if float(row["alpha"]) == alpha]
            mean_tokens = float(np.mean([row["generated_tokens"] for row in endpoint]))
            repeats = float(np.mean([repeat_fraction(row["reply"]) for row in endpoint]))
            degenerate = repeats > MAX_REPEAT_FRACTION or (
                mean_tokens < MIN_LENGTH_RATIO * baseline_tokens
            )
            endpoint_quality.append({
                "alpha": alpha,
                "n": len(endpoint),
                "mean_tokens": mean_tokens,
                "repeat_fraction": repeats,
                "degenerate": bool(degenerate),
            })
        contrast = paired_binary_contrast(
            outcomes[2.0], outcomes[-2.0], high_alpha=2.0, low_alpha=-2.0,
            n_bootstrap=n_bootstrap, seed=seed,
        )
        results.append({
            "control_seed": control_seed,
            "prompt_indices": indices.tolist(),
            "endpoint_quality": endpoint_quality,
            "eligible": not any(row["degenerate"] for row in endpoint_quality),
            "plus2_minus_minus2": contrast,
        })
    return results


def _control_readout_summary() -> list[dict] | None:
    if not CONTROL_READOUT.exists():
        return None
    saved = json.loads(CONTROL_READOUT.read_text(encoding="utf-8"))
    output = []
    for seed in CONTROL_SEEDS:
        rows = sorted(
            [row for row in saved["items"] if int(row["control_seed"]) == seed],
            key=lambda row: row["alpha"],
        )
        if len(rows) != len(CONTROL_ALPHAS):
            raise ValueError(f"control readout seed {seed} is incomplete")
        output.append({
            "control_seed": seed,
            "rows": rows,
            "plus2_minus_minus2_mean": float(rows[1]["mean"] - rows[0]["mean"]),
            "plus2_minus_minus2_gap_fraction": float(
                rows[1]["gap_fraction"] - rows[0]["gap_fraction"]
            ),
        })
    return output


def _specificity(learned: dict, controls: list[dict] | None) -> dict:
    primary = learned["contrasts"]["plus2_minus_minus2"]
    interval = primary["bootstrap_95_interval"]
    if controls is None:
        return {"status": "control generations not complete"}
    eligible = [
        row["plus2_minus_minus2"]["risk_difference"]
        for row in controls if row["eligible"]
    ]
    if not eligible:
        return {"status": "no nondegenerate control direction"}
    learned_effect = float(primary["risk_difference"])
    return {
        "status": "complete",
        "criterion": (
            "predicted positive learned effect; 95% prompt-bootstrap interval above zero; "
            "magnitude exceeds every nondegenerate control point estimate"
        ),
        "learned_effect": learned_effect,
        "learned_interval": interval,
        "eligible_control_effects": eligible,
        "max_absolute_control_effect": float(max(abs(value) for value in eligible)),
        "direction_specific": bool(
            learned_effect > 0
            and interval[0] > 0
            and abs(learned_effect) > max(abs(value) for value in eligible)
        ),
    }


def cmd_analyze(args) -> None:
    output = {
        "audit_id": AUDIT_ID,
        "status": "post-result mentor-requested robustness audit",
        "prespec": str(PRESPEC),
        "prespec_sha256": file_sha256(PRESPEC),
        "bootstrap": {"draws": args.bootstrap, "seed": args.seed, "unit": "prompt"},
        "learned_direction": {},
        "controls": {},
    }
    for name in ("self_report", "safety"):
        learned = learned_prompt_inference(
            name, n_bootstrap=args.bootstrap, seed=args.seed
        )
        controls = control_prompt_inference(
            name, n_bootstrap=args.bootstrap, seed=args.seed
        )
        output["learned_direction"][name] = learned
        output["controls"][name] = controls
        output.setdefault("specificity", {})[name] = _specificity(learned, controls)

        primary = learned["contrasts"]["plus2_minus_minus2"]
        print(
            f"{name}: +2 minus -2 = {primary['risk_difference']:+.4f} "
            f"[{primary['bootstrap_95_interval'][0]:+.4f}, "
            f"{primary['bootstrap_95_interval'][1]:+.4f}], "
            f"exact p={primary['mcnemar_exact_two_sided_p']:.6g}"
        )
        slope = learned["whole_curve"]
        print(
            f"  mean within-prompt slope = {slope['mean_within_prompt_slope']:+.4f} "
            f"[{slope['bootstrap_95_interval'][0]:+.4f}, "
            f"{slope['bootstrap_95_interval'][1]:+.4f}]"
        )
        if controls is None:
            print("  controls: not generated yet")

    output["controls"]["readout"] = _control_readout_summary()
    if CONTROL_DIRECTIONS.exists():
        artifact = load_control_artifact()
        output["controls"]["direction_metadata"] = {
            "artifact": str(CONTROL_DIRECTIONS),
            "artifact_sha256": file_sha256(CONTROL_DIRECTIONS),
            "layer": int(artifact["layer"]),
            "scale": float(artifact["scale"]),
            "seeds": [int(value) for value in artifact["seeds"]],
            "target_cosines": [float(value) for value in artifact["target_cosines"]],
        }
    else:
        output["controls"]["direction_metadata"] = None

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(PROMPT_INFERENCE, output)
    print(f"saved {PROMPT_INFERENCE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    prepare = sub.add_parser("prepare")
    prepare.set_defaults(func=cmd_prepare)

    readout = sub.add_parser("readout")
    readout.add_argument("--model", default=f"meta-llama/{MODEL_TAG}")
    readout.add_argument("--device", default=None)
    readout.add_argument("--device-map", default=None)
    readout.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    readout.set_defaults(func=cmd_readout)

    generate = sub.add_parser("generate")
    generate.add_argument("--set", required=True, choices=["self_report", "safety"])
    generate.add_argument("--model", default=f"meta-llama/{MODEL_TAG}")
    generate.add_argument("--device", default=None)
    generate.add_argument("--device-map", default=None)
    generate.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    generate.add_argument("--max-new-tokens", type=int, default=256)
    generate.set_defaults(func=cmd_generate)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    analyze.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
