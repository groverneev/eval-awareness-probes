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
- grouped five-fold predictions with pair-grouped layer selection nested inside
  every outer training fold;
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
uv run python -m pytest -q
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

uv run python -m vnaa.cross_model_suite extract \
  --model google/gemma-3-12b-it \
  --device-map auto --dtype bfloat16
uv run python -m vnaa.cross_model_suite analyze \
  --model google/gemma-3-12b-it

uv run python -m vnaa.cross_model_suite extract \
  --model mistralai/Ministral-3-8B-Instruct-2512-BF16 \
  --device-map auto --dtype bfloat16
uv run python -m vnaa.cross_model_suite analyze \
  --model mistralai/Ministral-3-8B-Instruct-2512-BF16
```

Gemma 3 and Ministral 3 use explicit text-only adapters for their official
multimodal wrappers. Their real-tokenizer and eight-pair smoke gates passed
before these full commands were run.

As checked on 2026-08-11, RunPod's public Pod pricing listed the 48 GB A40 at
$0.44/hour, RTX A6000 at $0.53/hour, and L40S at $0.99/hour:
https://www.runpod.io/pricing. Runtime has not yet been benchmarked on the new
runner. The measured full extraction times on separate A40s were 11m09s for
Gemma and 12m15s for Ministral; full analyses took about 4.5 and 5.3 minutes.
Including setup, downloads, smokes, analysis, and artifact transfer, both pods
together cost approximately $0.86 at $0.44/hour each before termination.

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

## Completed primary panel

The full local Qwen3-8B run completed on 2026-08-11 at the frozen revision.
All four 250-pair caches passed provenance checks and both temporal controls had
exactly zero pre-final difference. The table reports the uniform nested
layer-selection analysis completed after the audit described below:

| Test | Probe AUROC | Strongest text AUROC | Difference (95% CI) | p |
|---|---:|---:|---:|---:|
| `eval_shared` within | 0.99192 | 0.95293 | +0.03899 [+0.02182, +0.05823] | 0.0025 |
| `coherent_pairs` within | 0.94504 | 0.94474 | +0.00030 [-0.01628, +0.01643] | 0.4918 |
| eval-shared -> coherent | 0.88195 | 0.86614 | +0.01581 [-0.01079, +0.04309] | 0.0795 |
| coherent -> eval-shared | 0.82446 | 0.84854 | -0.02408 [-0.06007, +0.01192] | 0.8486 |

Gemma 3 and Ministral 3 were subsequently completed on the same 250-pair
endpoints. All eight additional caches passed provenance and temporal checks.

| Model | Test | Probe | Strongest text | Difference (95% CI) | p |
|---|---|---:|---:|---:|---:|
| Gemma | `eval_shared` within | 0.97296 | 0.95293 | +0.02003 [-0.00021, +0.04043] | 0.0885 |
| Gemma | `coherent_pairs` within | 0.85902 | 0.94474 | -0.08571 [-0.11331, -0.05891] | 1.0000 |
| Gemma | eval-shared -> coherent | 0.72642 | 0.86614 | -0.13973 [-0.17575, -0.10417] | 1.0000 |
| Gemma | coherent -> eval-shared | 0.65408 | 0.84854 | -0.19446 [-0.23496, -0.15188] | 1.0000 |
| Ministral | `eval_shared` within | 0.98309 | 0.95293 | +0.03016 [+0.01281, +0.04904] | 0.0060 |
| Ministral | `coherent_pairs` within | 0.92790 | 0.94474 | -0.01683 [-0.03563, +0.00282] | 0.8531 |
| Ministral | eval-shared -> coherent | 0.84232 | 0.86614 | -0.02382 [-0.05075, +0.00519] | 0.9030 |
| Ministral | coherent -> eval-shared | 0.87557 | 0.84854 | +0.02702 [-0.00619, +0.06111] | 0.0660 |

The stopping rule is satisfied. The panel supports cross-family temporal
decodability but not architecture-invariant effect size. Llama, Qwen, and
Ministral establish positive paired advantages on `eval_shared`; Gemma's
positive margin is inconclusive. No replication model beats character TF-IDF
on coherent or transfer tests.

### Post-result layer-selection audit

The first completed analysis selected the direction-of-means layer by
in-sample outer-training AUROC. This respected the outer holdout but proved
unstable when Gemma's high-dimensional training curves saturated: its reported
within-control AUROCs were initially 0.72667 and 0.64744. After the anomaly was
observed, the runner was corrected to select layers by pair-grouped CV nested
inside each outer training fold and was rerun uniformly for Llama, Qwen, Gemma,
and Ministral. Gemma then selected layer 48 in all five `eval_shared` folds and
layer 9 in all five coherent-pair folds, producing the corrected table above.
Transfer already used grouped source-only CV and did not change. This is a
transparent post-result method correction, not part of the original frozen
analysis.

### Post-result readout audit

The Gemma matched-coherence result prompted a second, uniformly applied audit
of final-user-turn mean pooling. `src/vnaa/run_readout_audit.py` extracts or
validates the mean caches, reuses the paired outer folds and nested grouped
selector, and can exclude embedding layer zero from candidate layers.

| Model | Shared token / mean | Coherent token / mean all / mean no embedding |
|---|---:|---:|
| Llama | 0.98779 / 0.99158 | 0.94682 / 0.93218 / 0.93251 |
| Qwen | 0.99192 / 0.97850 | 0.94504 / 0.94392 / 0.93323 |
| Gemma | 0.97296 / 0.94883 | 0.85902 / 0.93877 / 0.63800 |
| Ministral | 0.98309 / 0.98690 | 0.92790 / 0.87750 / 0.93803 |

Unrestricted coherent mean pooling selects embedding layer zero in every Gemma
and Qwen outer fold, three Llama folds, and one Ministral fold. Excluding that
lexical embedding average leaves Llama, Qwen, and Ministral at 0.93251--0.93803
but reduces Gemma to 0.63800. The audit therefore supports architecture-by-
readout sensitivity, not a uniform mean-pooling improvement or a simple Gemma
extraction fix. Outputs are under `results/readout_audit/` and are intentionally
ignored because the mean activation caches are large.
