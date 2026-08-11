# Cross-model confirmatory suite v2

This runbook executes the frozen manifest at
`datasets/cross_model/confirmatory_suite_v2.json`. V2 was frozen before any
Qwen3 activations were extracted and preserves v1 as the pre-amendment record.
It intentionally reruns only
the valid paired Slice-3 controls. It does not include failed Slice-2b cells,
the invalid Slice-3 realism construction, Slice 5, or the failed authorship-v2
pilot.

## Frozen panel

- Reference: `meta-llama/Llama-3.1-8B-Instruct` (already completed).
- Primary cross-family replication 1: `Qwen/Qwen3-8B`.
- Primary cross-family replication 2: `google/gemma-3-12b-it`.
- Primary cross-family replication 3:
  `mistralai/Ministral-3-8B-Instruct-2512-BF16`.

All three primary models must be run regardless of earlier results. Every
repository is pinned to the immutable revision in the manifest. Use
unquantized bfloat16 weights. Gemma requires accepting Google's Hugging Face
usage terms before the pod can download it. Gemma 3 and Ministral 3 require
separate validated text-only adapters for their multimodal wrappers; do not run
them through the causal-LM loader used for Qwen.

## What the core run performs

For each primary model, the runner makes exactly 2,824 short forward passes:
one independent pass through every user-turn boundary in the positive and
negative histories for 250 `eval_shared` pairs and 250 matched-coherence pairs.
Prefixwise extraction is intentional. It prevents different final-turn lengths
from selecting different BF16 attention kernels and introducing numerical
differences at an otherwise byte-identical earlier prefix.

The CPU analysis then runs:

- a numerical identical-prefix check before the differing final user turn;
- grouped five-fold, train-only layer selection;
- word-count, word TF-IDF, character TF-IDF, format, and length baselines;
- paired bootstrap intervals and paired prediction-randomization tests; and
- transfer in both directions between the two controls.

Raw probe AUROC is not enough for a probe-specific claim. The frozen reporting
rule requires reporting the paired interval against the strongest text
baseline, whether favorable or unfavorable.

## Before renting a GPU

```powershell
$env:PYTHONPATH='.'
uv run python -m vnaa.cross_model_suite validate
uv run python -m vnaa.cross_model_suite plan
uv run pytest -q
```

Run an eight-pair Qwen smoke test on the local RTX 5070 Ti. `device-map auto`
allows CPU offload if the unquantized weights and inference overhead exceed the
16 GB VRAM limit. The smoke artifact has its own directory and cannot be reused
as a full result.

```powershell
$env:HF_HUB_DISABLE_XET='1'
uv run python -m vnaa.cross_model_suite extract `
  --model Qwen/Qwen3-8B `
  --device-map auto --dtype bfloat16 --limit 8

uv run python -m vnaa.cross_model_suite analyze `
  --model Qwen/Qwen3-8B `
  --limit 8 --folds 2 --n-boot 50 --n-permutations 50
```

On this Windows desktop the Hugging Face Xet transfer backend was much slower
than standard HTTP for Qwen's five weight shards. Keep
`HF_HUB_DISABLE_XET=1` for the initial download. This changes only transport,
not the pinned model bytes.

The smoke analysis is an implementation check only and must never be reported
as evidence.

## Full GPU run

Use a RunPod Pod rather than Serverless because the job needs a repository,
persistent activation artifacts, and several analysis commands. A single 48 GB
A40 is the cost-first choice for the 7.6B and 9B models; an RTX A6000 is the
fallback, and an L40S is the faster but more expensive option. Provision 100 GB
of disk, run one model at a time, and delete the pod only after checksumming and
copying all artifacts locally.

```bash
export PYTHONPATH=.
uv run python -m vnaa.cross_model_suite validate

uv run python -m vnaa.cross_model_suite extract \
  --model Qwen/Qwen3-8B \
  --device-map auto --dtype bfloat16
uv run python -m vnaa.cross_model_suite analyze \
  --model Qwen/Qwen3-8B
```

Add Gemma 3 and Ministral 3 commands only after their text-only adapters pass
the same local smoke and hidden-state shape tests.

As checked on 2026-08-11, RunPod's public Pod pricing listed the 48 GB A40 at
$0.44/hour, RTX A6000 at $0.53/hour, and L40S at $0.99/hour:
https://www.runpod.io/pricing. Runtime has not yet been benchmarked on the new
runner. A conservative planning range is 2--4 GPU-hours per model for the core
extraction, or roughly $0.88--$1.76 for the first Qwen run on an A40, excluding
setup time and storage. Treat that as a budget envelope, not a measured cost.

Do not use 4-bit or 8-bit quantization. Models outside 5--12B are not part of
the v2 primary panel.

## Completion checks

For each model, retain:

- four trajectory `.npz` files under its collision-resistant model directory;
- `analysis.json` with the model repository, manifest hash, dataset hashes, and
  inference settings;
- terminal logs and wall-clock duration; and
- SHA-256 hashes of every artifact copied off the pod.

Do not run the silence or steering extensions until all core analyses are
complete. Those generation-heavy arms are secondary in the frozen manifest and
will receive separate model-aware adapters and cost checks before execution.

## Completed Qwen3-8B core

The full local Qwen3-8B run completed on 2026-08-11 at the frozen revision.
All four 250-pair caches passed provenance checks and both temporal controls had
exactly zero pre-final difference. The primary grouped results are:

| Test | Probe AUROC | Strongest text AUROC | Difference (95% CI) | p |
|---|---:|---:|---:|---:|
| `eval_shared` within | 0.99357 | 0.95293 | +0.04064 [+0.02364, +0.05982] | 0.0050 |
| `coherent_pairs` within | 0.94653 | 0.94474 | +0.00179 [-0.01495, +0.01798] | 0.4448 |
| eval-shared -> coherent | 0.88195 | 0.86614 | +0.01581 [-0.01079, +0.04309] | 0.0795 |
| coherent -> eval-shared | 0.82446 | 0.84854 | -0.02408 [-0.06007, +0.01192] | 0.8486 |

These results do not change the stopping rule: Gemma 3 and Ministral 3 must run
regardless of the Qwen outcome.
