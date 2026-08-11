# Cross-model confirmatory suite v1

This runbook executes the frozen manifest at
`datasets/cross_model/confirmatory_suite_v1.json`. It intentionally reruns only
the valid paired Slice-3 controls. It does not include failed Slice-2b cells,
the invalid Slice-3 realism construction, Slice 5, or the failed authorship-v2
pilot.

## Frozen panel

- Reference: `meta-llama/Llama-3.1-8B-Instruct` (already completed).
- Primary cross-family replication 1: `Qwen/Qwen2.5-7B-Instruct`.
- Primary cross-family replication 2: `google/gemma-2-9b-it`.
- Deferred scale test: `meta-llama/Llama-3.1-70B-Instruct` only after both
  primary models, budget permitting.

Both primary models must be run regardless of the first model's result. Use
unquantized bfloat16 weights. Gemma requires accepting Google's Hugging Face
usage terms before the pod can download it.

## What the core run performs

For each primary model, the runner makes exactly 1,000 forward passes: positive
and negative histories for 250 `eval_shared` pairs and 250 matched-coherence
pairs. It saves every user-turn boundary in one pass, so the last-user readout
and the temporal control do not require duplicate inference.

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
uv run python -m vnaa.cross_model_suite extract `
  --model Qwen/Qwen2.5-7B-Instruct `
  --device-map auto --dtype bfloat16 --limit 8

uv run python -m vnaa.cross_model_suite analyze `
  --model Qwen/Qwen2.5-7B-Instruct `
  --limit 8 --folds 2 --n-boot 50 --n-permutations 50
```

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
  --model Qwen/Qwen2.5-7B-Instruct \
  --device-map auto --dtype bfloat16
uv run python -m vnaa.cross_model_suite analyze \
  --model Qwen/Qwen2.5-7B-Instruct

uv run python -m vnaa.cross_model_suite extract \
  --model google/gemma-2-9b-it \
  --device-map auto --dtype bfloat16
uv run python -m vnaa.cross_model_suite analyze \
  --model google/gemma-2-9b-it
```

As checked on 2026-08-11, RunPod's public Pod pricing listed the 48 GB A40 at
$0.44/hour, RTX A6000 at $0.53/hour, and L40S at $0.99/hour:
https://www.runpod.io/pricing. Runtime has not yet been benchmarked on the new
runner. A conservative planning range is 2--4 GPU-hours per model for the core
extraction, or roughly $1.76--$3.52 total for both models on an A40, excluding
setup time and storage. Treat that as a budget envelope, not a measured cost.

Do not use 4-bit or 8-bit quantization. The optional 70B model has roughly 141
GB of bfloat16 weights before inference overhead and therefore needs at least a
multi-GPU 80 GB setup; it is not part of the initial run.

## Completion checks

For each model, retain:

- four trajectory `.npz` files under its collision-resistant model directory;
- `analysis.json` with the model repository, manifest hash, dataset hashes, and
  inference settings;
- terminal logs and wall-clock duration; and
- SHA-256 hashes of every artifact copied off the pod.

Do not run the silence or steering extensions until both core analyses are
complete. Those generation-heavy arms are secondary in the frozen manifest and
will receive separate model-aware adapters and cost checks before execution.
