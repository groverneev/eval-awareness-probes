# Qwen3 dense scale ladder v1

This runbook executes the frozen manifest at
`datasets/qwen_scale/scale_ladder_v1.json`. The manifest was frozen before any
new scale-study activations were extracted. It reuses the two valid 250-pair
temporal controls and the completed cross-model runner without introducing new
prompts.

## Frozen checkpoints

Run all six original dense Qwen3 checkpoints, regardless of intermediate
results:

- `Qwen/Qwen3-0.6B`
- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-14B`
- `Qwen/Qwen3-32B`

The 8B checkpoint is re-extracted rather than borrowing its local cross-family
cache, so every scale point shares one GPU and software environment. Qwen3.8,
sparse MoE checkpoints, quantized weights, and later Qwen generations are not
part of the ladder.

## Validate before GPU extraction

```powershell
$env:PYTHONPATH='.'
.venv\Scripts\python -m vnaa.cross_model_suite `
  --manifest datasets/qwen_scale/scale_ladder_v1.json `
  --results results/qwen_scale/v1 validate
.venv\Scripts\python -m vnaa.cross_model_suite `
  --manifest datasets/qwen_scale/scale_ladder_v1.json `
  --results results/qwen_scale/v1 plan
.venv\Scripts\python -m pytest -q
```

## Uniform full extraction and primary analysis

Use one 80GB GPU pod, unquantized bfloat16 weights, and the same environment
for every checkpoint. On Linux:

```bash
export PYTHONPATH=.
export HF_HUB_DISABLE_XET=1
MANIFEST=datasets/qwen_scale/scale_ladder_v1.json
RESULTS=results/qwen_scale/v1

for MODEL in \
  Qwen/Qwen3-0.6B \
  Qwen/Qwen3-1.7B \
  Qwen/Qwen3-4B \
  Qwen/Qwen3-8B \
  Qwen/Qwen3-14B \
  Qwen/Qwen3-32B
do
  uv run python -m vnaa.cross_model_suite \
    --manifest "$MANIFEST" --results "$RESULTS" extract \
    --model "$MODEL" --device-map auto --dtype bfloat16
  uv run python -m vnaa.cross_model_suite \
    --manifest "$MANIFEST" --results "$RESULTS" analyze \
    --model "$MODEL"
done

# Reconstruct and validate paired predictions needed for the scale bootstrap.
uv run python -m vnaa.run_qwen_scale_analysis \
  --manifest "$MANIFEST" --results "$RESULTS" export

# After the historical cross-family 8B analysis is available at its default
# local path, run the prespecified paired-bootstrap slope analysis.
uv run python -m vnaa.run_qwen_scale_analysis \
  --manifest "$MANIFEST" --results "$RESULTS" aggregate
```

Do not delete a checkpoint cache until its four trajectory files and
`analysis.json` pass provenance and temporal checks. Copy every artifact and a
SHA-256 inventory off the pod before termination.

## Reporting boundary

The primary outcomes are activation-minus-strongest-text AUROC on shared prefix
and matched coherence. A consistent within-family trend requires the paired
bootstrap slope interval against log10 exact parameter count to exclude zero
in the same direction on both controls. Transfer, normalized selected depth,
and nonlinear RBF results are secondary. Six checkpoints can establish a
within-family pattern for this pipeline, not a universal scaling law.
