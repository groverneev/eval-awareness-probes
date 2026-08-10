# Probing Evaluation Awareness in Realistic Multi-Turn Conversations

Do language models internally recognize when they are being evaluated, and does
that recognition survive when the evaluation is disguised as a realistic
conversation? We train linear probes on a model's residual-stream activations to
separate evaluation contexts from deployment contexts, and test whether the
signal holds up once we remove the surface cues (formatting, length, single-turn
structure) that a probe could read instead of genuine recognition.

Full proposal, methodology, and citations: `VNAA-Proposal.pdf`. Detailed
results, gotchas, and the agent-facing project state: `CLAUDE.md`.

Author information is withheld for double-blind review.

## Status

All experimental work is complete on Llama-3.1-8B: the single-turn replication
and its surface-baseline audit, the format-matched cross-cell transfer test,
the multi-turn `eval_shared` and matched-coherence controls (with a
silence-verified subset), a deployment-simulation reconstruction with a
model-swap/context-ablation decomposition (replicated on a second corpus,
LMSYS-Chat-1M), and a causal steering arm. A second estimator (logistic
regression) and full-pipeline permutation nulls confirm no conclusion depends
on the fitting method. What remains is paper assembly.

## Headline results

**Slice 1 — single-turn replication.** Probe reaches AUROC 0.994 at layer 4/32,
but a structure-only classifier (no words) reaches 0.9675 and bag-of-words
0.9517 on the same split — the probe's margin over the best surface baseline
is only +0.027, within noise. This cell is fully confounded (SAD-eval prompts
look like benchmark items; WildChat-deploy prompts don't), which motivates
everything after it.

**Slice 2 — format-matched transfer (ARC+MMLU, 5 seeds).** Probe beats
bag-of-words under cross-cell transfer by **+0.071 ± 0.013**, positive on
every seed and every format-matched training cell. Training on the standard
confounded contrast is the sole exception (−0.087 ± 0.032): on the setup the
field actually uses, activations transfer *worse* than word counts.

**Slice 3 — multi-turn.** `eval_shared` (deploy conversation, final user turn
swapped for a benchmark question, byte-identical prefix) is exactly chance
before the swap and **0.995** at the final turn, a **+0.226** margin over
bag-of-words. The matched-coherence control repairs its topic-shift caveat
and reaches **0.951 ± 0.011** (+0.099 over BoW) with a naturally-continuing
final question. Both controls' silence-filtered subsets (responses that never
verbalize awareness, confirmed by a blinded judge with planted controls) hold:
0.937 and 0.998 respectively.

**Slice 5 — deployment-simulation reconstruction.** Real WildChat/LMSYS
conversations with only the final assistant reply regenerated give exact
provenance labels. Probe reaches 0.792–0.822 vs a format baseline of ~0.67–0.68.
A model-swap vs context-ablation decomposition shows the effect isn't
attributable to the probed model's own authorship.

**Steering (pre-specified before any steered output existed).** Manipulation
check passes; self-report shifts strongly (0.139 → 0.889 across the
coefficient grid); safety refusal rate shows no large effect (underpowered at
n=48, not a clean null).

## Repository layout

```
src/vnaa/       probing, extraction, dataset, and analysis code (see CLAUDE.md
                for the full module map)
scripts/        figure generation, GPU runbooks
datasets/       committed batch specs, judge rubric, steering pre-spec
                (never real prompt text from gated corpora)
tests/          unit tests for probe math, read points, estimators, etc.
```

`data/` and `results/` are gitignored — datasets and cached activations are
large, reproducible, and some derive from gated/licensed corpora (see
CLAUDE.md's SAD canary discipline note).

## Setup

Requires Python 3.11–3.13 (not 3.14; PyTorch doesn't support it yet).

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Model access: Llama models are gated on Hugging Face. Request access to
`meta-llama/Llama-3.2-1B-Instruct` and `meta-llama/Llama-3.1-8B-Instruct`, then
log in with a token that has "read access to public gated repos":

```bash
.venv/bin/hf auth login
```

Data: slice 1 uses the SAD `stages_oversight` task. Download the encrypted
oversight structs once:

```bash
mkdir -p data/sad
curl -sL -o data/sad/oversight_structs.zip \
  https://github.com/LRudL/sad/raw/main/sad/stages/oversight/structs.zip
```

SAD question text must never be committed anywhere scrapable (leaking it
invalidates the benchmark). The loader reads it from the encrypted zip in
memory and never writes prompt text to disk.

## Running slice 1

```bash
# local sanity check on the small model
.venv/bin/python -m vnaa.run_slice1 \
  --sad-repo data/sad/oversight_structs.zip \
  --model meta-llama/Llama-3.2-1B-Instruct

# the real replication (needs a ~24 GB GPU)
.venv/bin/python -m vnaa.run_slice1 \
  --sad-repo data/sad/oversight_structs.zip \
  --model meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16
```

Output: a per-layer AUROC table (evaluation vs deployment), plus cached
activations and the AUROC curve saved under `results/`.

## Tests

```bash
.venv/bin/python -m pytest
```

## Compute

White-box probing needs the model's internal activations, which no hosted API
returns, so this project runs on self-hosted open-weight models on a GPU, not
API credits. Forward-pass-only work fits comfortably on a free GPU tier
(Kaggle, Lightning AI); the generation-heavy silence-filter and steering runs
used a rented GPU. See `CLAUDE.md` for the full compute policy and environment
details.
