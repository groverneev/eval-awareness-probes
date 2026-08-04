# Probing Evaluation Awareness in Realistic Multi-Turn Conversations

Do language models internally recognize when they are being evaluated, and does
that recognition survive when the evaluation is disguised as a realistic
conversation? We train linear probes on a model's residual-stream activations to
separate evaluation contexts from deployment contexts, and test whether the
signal holds up once we remove the surface cues (formatting, length, single-turn
structure) that a probe could read instead of genuine recognition.

Authors: Neev Grover, Apurva Reddy, Vedaswaroop Kunamneni, Arya Kumar, Maheep
Chaudhary.

## Status

Slices 1 and 2 are done on Llama-3.1-8B. Slice 3 (multi-turn) is next.

**Slice 1 — the replication, and why it is not enough.** The probe reaches
AUROC 0.994 at layer 4 of 32, reproducing the prior single-turn result, and
length is ruled out (0.37–0.45). But a classifier reading only structure — no
words at all — reaches 0.9675 and bag-of-words 0.9517 on the same split. The
probe's margin over the best surface baseline is +0.027, within noise at
n=120. The obvious-eval cell is confounded: it cannot separate "the model
recognizes evaluation" from "these prompts are shaped like tests."

**Slice 2 — format-matched cells and cross-cell transfer.** We replicated
Devbunova's 2×2 on our machinery (their standard probe calls format-transplanted
deployment prompts "evaluation" 96.0% of the time; they report 94.5%), then
built four new cells where every prompt is model-authored so authorship cannot
carry the label, lengths match by construction, and one deploy cell is
topic-matched to each eval item.

Surface baselines cannot be driven to chance, and we show why: the residual
signal is *speech act* — an impersonal question with a canonical answer versus
a first-person request for help is what the eval/deploy distinction is. So the
decisive test is cross-cell **transfer**: fit on one cell pair, freeze, score a
differently-constructed pair, and run the identical test on every baseline.

Headline (ARC+MMLU, 5 seeds; GSM8K excluded because digit density alone scores
0.949 there, so those items measure numeral counting):

- The probe beats bag-of-words under transfer by **+0.067 ± 0.016**, positive
  on every seed. Every format-matched training cell gives a positive margin.
- **Training on the standard confounded contrast is the sole exception**
  (−0.086 ± 0.034, win rate 12%): on the setup the field actually uses,
  activations generalize *worse* than word counts.
- The direction travels one way. Trained on our realistic cells it scores
  0.949 on the standard benchmark cell; trained on the standard cell it scores
  0.578 on ours, barely above chance.
- Honest limit: benchmark format on *both* sides with only content differing
  (`bench_eval` vs `bench_deploy_rewritten`) defeats every direction we can
  learn — 0.31–0.55, at or below chance. That is where content-only eval
  recognition would have to appear, and it does not.

Figures in `figures/`; full detail and caveats in `CLAUDE.md`.

## Approach in one paragraph

We build the thinnest end-to-end pipeline first (one model, one probe, obvious
evaluation vs deployment prompts) and verify it reproduces prior results, then
widen it one step at a time toward the hard case: multi-turn, format-matched,
silence-filtered conversations. Each step is mostly a dataset swap on the same
probe code.

## Repository layout

```
src/vnaa/
  probe.py        per-layer difference-of-means probe + AUROC readout
  extract.py      residual-stream activation extraction
  data.py         SAD stages_oversight loader (evaluation vs deployment)
  run_slice1.py   slice-1 runner: data -> extract -> probe
tests/
  test_probe.py   probe math checked against known-answer synthetic data
  ...             model-dependent smoke/visualization scripts
```

`data/` and `results/` are gitignored (datasets and cached activations are
large and reproducible).

## Setup

Requires Python 3.11–3.13 (not 3.14; PyTorch does not support it yet).

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Model access: Llama models are gated on Hugging Face. Request access to
`meta-llama/Llama-3.2-1B-Instruct` and `meta-llama/Llama-3.1-8B-Instruct`, then
log in with a token that has the "read access to public gated repos" permission:

```bash
.venv/bin/hf auth login
```

Data: slice 1 uses the SAD `stages_oversight` task. Download the (encrypted)
oversight structs once:

```bash
mkdir -p data/sad
curl -sL -o data/sad/oversight_structs.zip \
  https://github.com/LRudL/sad/raw/main/sad/stages/oversight/structs.zip
```

Note: SAD question text must never be committed anywhere scrapable (it is a
benchmark; leaking it invalidates it). The loader reads it from the encrypted
zip in memory and never writes prompt text to disk. Do not commit SAD prompts.

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

## Running on a cloud GPU (the 8B replication)

The 8B does not fit on a 16 GB laptop, so the real run uses a ~24 GB GPU. On a
Lightning AI Studio (single L4), from a terminal:

```bash
git clone https://github.com/groverneev/Algoverse-VNAA.git
cd Algoverse-VNAA
bash scripts/setup_gpu.sh      # installs deps, HF login, downloads SAD data
# then switch the Studio to the L4 GPU and run:
python -m vnaa.run_slice1 \
  --sad-repo data/sad/oversight_structs.zip \
  --model meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16
```

`setup_gpu.sh` installs on top of the Studio's existing CUDA PyTorch (it does not
reinstall torch, which could pull a CPU-only build). Do setup on the free CPU
machine; only switch to the GPU for the model run.

## Tests

```bash
.venv/bin/python -m pytest
```

## Compute

White-box probing needs the model's internal activations, which no hosted API
returns — so this work runs on self-hosted open-weight models on a GPU, not
through API credits. Slice 1 is a few hundred forward passes with no generation,
so it fits comfortably on a free GPU tier (Kaggle, Lightning AI). See `CLAUDE.md`
for details and the program compute policy.
