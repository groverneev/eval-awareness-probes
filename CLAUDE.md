# CLAUDE.md — agent onboarding and running log

This file is the detailed context for any agent picking up this project. Keep it
updated as work progresses (see "Maintaining this file" at the bottom). The
README is the short human-facing version; this is the long one.

Last updated: August 2026. Slice 1 done: replicated on Llama-3.1-8B (Kaggle,
2x T4). Forward plan written (`PLAN.md`). New primary machine: Windows desktop
with RTX 5070 Ti (see Environment).

GPU host decision: tried Lightning AI first, abandoned it — the site 404'd on the
Studio dashboard and the account showed 0.00 free credits (a `.org` school email,
not `.edu`, likely got no free grant). Pivoted to **Kaggle** (2x T4, genuinely
free, 30 GPU-hr/week). The 8B in bf16 (~16 GB) does not fit on one 16 GB T4, so it
shards across both T4s with `--device-map auto` (needs `accelerate`). See
`scripts/kaggle_slice1.md` for the exact notebook cells. `scripts/setup_gpu.sh`
is the single-GPU path (kept for a future single-24GB-card host).

---

## The idea

AI safety evaluations are only trustworthy if models behave the same under test
as in real use. Prior work shows models can tell evaluation from deployment,
both behaviorally and from internal activations (Needham et al. 2025, Nguyen et
al. 2025, Chaudhary et al. 2025). Nguyen argues a linear probe on the residual
stream captures genuine eval-awareness; Devbunova (2026) counters that the probe
just tracks surface format (benchmark-canonical structure) and fails to
generalize to free-form prompts. Both sides are single-turn, so they cannot
separate genuine recognition from surface-format detection.

Our contribution: test white-box probing in the realistic, multi-turn,
format-matched, silence-filtered regime — the axis deployment-simulation sets
aside. If the probe stays well above a hierarchy of surface baselines when the
evaluation is disguised as a real conversation, that is genuine recognition. If
it collapses to baseline, realism is an effective defense and prior "awareness"
was largely format detection. The gap between the probe and the baselines is the
result, not the raw AUROC.

Full proposal: `Probing Evaluation Awareness in Realistic Multi-Turn
Conversations` (the source PDF, not in the repo). Authors: Neev Grover, Apurva
Reddy, Vedaswaroop Kunamneni, Arya Kumar, Maheep Chaudhary.

## Build strategy: thinnest slice first, then widen

Build the minimal end-to-end path that produces one honest AUROC number, verify
it, then expand by swapping datasets on the same machinery. Do NOT build the
whole apparatus from the proposal at once.

Expansion order (each is mostly a dataset swap on the same probe code):
1. **Slice 1 (current):** single-turn, obvious eval (SAD) vs deployment
   (WildChat). Pipeline validation / replication, not a hypothesis test.
2. Format-matched single-turn pairs.
3. Multi-turn (adds four token read-points: last user token, post-instruction
   token, mean-pool over final turn, per-turn trajectory).
4. The decisive cell: multi-turn + format-matched + silence-filtered.
5. Deployment-simulation reconstruction (resample final assistant turn; model-
   swap vs eval-context ablation).
6. Silence-filtering internality test (awareness judge; unaware subset; probe-
   vs-verbalization detectability).
7. Steering arm (last).

Success criterion for slice 1: high `|AUROC - 0.5|` concentrated in early-to-
middle layers on the obvious-eval cell.

## How the user (Neev) likes to work

- One slice at a time. Finish and verify before moving on. Do not jump ahead.
- Ground claims in a concrete run before generalizing. A small real number beats
  a big abstract framework.
- Correct him directly when he is wrong; he prefers that to hedging.
- Prose: brief, one job per sentence, minimal em dashes, minimal fancy language.
- State explicitly which slice we are on and what the success criterion is.
- **Do not add a `Co-Authored-By: Claude` trailer to commits** (or any self-
  attribution). Plain commit messages only.

## Environment

Two machines. As of August 2026 the Windows desktop is primary.

**Windows desktop (primary):** Windows 11, RTX 5070 Ti (16 GB VRAM, Blackwell
sm_120), 64 GB DDR5, 2 TB NVMe.
- System Python is 3.14 (too new for PyTorch). Use the `.venv` (Python 3.12)
  created with `uv sync`. Run everything as `.venv\Scripts\python`.
- Blackwell needs the cu128 PyTorch wheels; pyproject pins the
  `pytorch-cu128` index for torch on win32 only, so `uv sync` just works.
- **The 8B almost fits.** bf16 weights are ~16.06 GB vs 16.3 GB VRAM; with
  CUDA overhead it does not fit fully. Use `device_map="auto"` (accelerate
  offloads the last few layers to the 64 GB of system RAM). Extraction at
  batch 1 is fine this way and beats Kaggle round-trips for iteration. Never
  4-bit quantize (perturbs the activations we measure).
- The 1B fits entirely in VRAM for fast debugging.
- Kaggle (2x T4) remains the fallback for fully-on-GPU runs, e.g. long
  generation sweeps in slices 4-5 if local offload is too slow.

**Mac (secondary):** Apple M3, 16 GB unified memory. Same 3.14/3.12 story;
`.venv/bin/python`. torch has MPS, no CUDA. The 8B does not fit at all here;
debug on the 1B.

Both: transformers 5.x (API differs from the 4.x most probing code targets —
first suspect for weird API errors; `output_hidden_states` still works;
loading uses `dtype=` not the deprecated `torch_dtype=`).

## Model access (Hugging Face)

- Llama is gated. Account `LightningPro1` is approved for both Llama-3.2-1B
  and Llama-3.1-8B (verified by file download on the Windows desktop, Aug
  2026). The desktop is logged in via `.venv\Scripts\hf.exe auth login`.
- Gotcha that cost time: a fine-grained token without the "read access to public
  gated repos" scope gives 403 even after approval. Use a token with gated-repo
  read access (`hf auth login`). The metadata endpoint returning 200 is not
  proof of access; file access is.

## Compute policy (Algoverse program) — the important constraints

- **APIs cannot serve this project.** White-box probing needs hidden states; no
  hosted endpoint (Azure, Cerebras, NVIDIA Build, OpenRouter) returns them. We
  are locked to self-hosted open weights on a GPU. Tell teammates this — "just
  use the Azure credits for the 8B" does not work.
- **Slice 1 is free.** ~200 forward passes, no generation = minutes of GPU. The
  policy says validate on free T4 first; only pay when you have measured you need
  more. Do not rent AWS or file the reimbursement form for slice 1.
- The policy explicitly blesses this: scenario "train a linear probe on hidden
  states → T4 is usually enough. Reimbursable."
- **Hardware wrinkle for the 8B:** 8B bf16 is ~16 GB, and a free T4 is exactly
  16 GB (too tight with activations). Use a free 24 GB-class card: Kaggle
  (2×T4 = 32 GB, 30 hr/week) or Lightning AI (L4 24 GB). Avoid 4-bit here — it
  perturbs the activations.
- Paid GPU is for training/fine-tuning, NOT inference. If we ever self-host a
  large model on paid GPU (e.g. the 70B scaling arm), pre-approve on the form.

## Data

Slice 1 uses SAD `stages_oversight`, which already pairs both classes:
- `test_oversight`: evaluation prompts (label 1). Sources: `test_anthropic`,
  `test_bigbench`, `test_wmdp` (~67 each, 200 total).
- `deploy_oversight`: deployment prompts (label 0), all `wildchat_sampled`
  (200). So SAD's deployment side is already WildChat — slice 1 is literally
  "SAD-eval vs WildChat-deploy," pre-assembled, matching prior work.

We use the raw `body` of each sample as a single-turn user message. We do NOT
apply SAD's own "is this test or deployment?" MCQ wrapper — for probing we feed
the underlying prompt and read activations, using the stage as ground truth.

**SAD canary discipline (required by SAD's license):** SAD question text must
never appear in plain text anywhere scrapable, including this public repo. The
loader (`data.py`) reads prompts from the encrypted `structs.zip` in memory
(public password `sadtimesforthesetimes`) and never writes prompt text to disk.
Cached *activations* are fine (not the question text). Never commit SAD prompts,
and never paste them into files that get committed.

The encrypted zip lives at `data/sad/oversight_structs.zip` (gitignored, ~1.3
MB). Re-download with the curl command in the README if missing. It survives
teardowns; the full 1.1 GB SAD clone in scratchpad does not.

## Code and key technical decisions

- `probe.py` — per-layer difference-of-means. Direction = normalized
  mean(eval) − mean(deploy), fit on train only. Score = projection onto
  direction; report AUROC per layer and the sign-invariant `|AUROC − 0.5|`
  (raw AUROC can drop below 0.5 under distribution shift). Verified against
  known-answer synthetic data in `test_probe.py` (6 tests). Two lessons baked
  in: a shuffled-label null must be averaged over many shuffles (one draw leaks
  residual signal against strong separation), and separation must be spread
  across dims to be realistic.
- `extract.py` — loads model, runs each conversation with
  `output_hidden_states=True`, reads one token position across every layer.
  Decisions: (a) the chat template already adds exactly one BOS — we feed its
  ids straight in, never re-tokenize; (b) `hidden_states` has n_layers + 1
  entries — index 0 is the embedding layer, 1..n are transformer blocks; we keep
  all and record that index 0 is embeddings; (c) read point `last_user_token`
  templates with `add_generation_prompt=False` and reads the final token (the
  `<|eot_id|>` closing the user turn) — this is a swappable parameter because the
  multi-turn arm adds more read points; (d) batch size 1 so the read index is
  always the true last token with no padding.
- `data.py` — SAD loader (above). Accepts either a SAD repo root or a direct
  path to `structs.zip`.
- `run_slice1.py` — ties it together: load data, extract, stratified train/test
  split, `fit_score`, print per-layer AUROC, save activations + AUROC curve to
  `results/` (never prompt text).

Note on the read point and layer 0: because the read token is the same
`<|eot_id|>` for every prompt, the embedding layer (index 0) carries no per-
example info, so its AUROC is exactly 0.5. Separation appears only after block 1
once attention has aggregated the turn. This is expected and a good sanity check;
it does NOT rule out surface confounds (length/format still get aggregated into
that position).

## Results so far

**Slice 1 on Llama-3.2-1B (local, MPS), 200 eval / 200 deploy, stratified 280
train / 120 test:**
- Layer 0 (embeddings): AUROC 0.500 (as expected).
- Layers 1–16: 0.91 → 0.99. Best layer 5 of 16, AUROC 0.992.
- Shape matches the replication prediction: early-to-middle peak, high and stable
  after.
- **Surface-baseline check (length):** on the same test split, token-length
  AUROC = 0.37, char-length = 0.39 (eval prompts are shorter: median 57 vs 101
  tokens). The probe's 0.99 is NOT a length artifact — probe strength 0.49 vs
  length strength ~0.13.

**Slice 1 on Llama-3.1-8B (Kaggle, 2x T4 sharded, bf16), 200 eval / 200 deploy,
same stratified 280/120 split:**
- Layer 0 (embeddings): AUROC 0.500.
- Layers 1–3: 0.71 → 0.97 → 0.99. Best layer 4 of 32, AUROC 0.994.
- Layers 5–32: 0.97–0.99, high and stable.
- This is the headline replication: early-layer peak on the target model, a more
  meaningful "early-to-middle" claim on 32 layers than on the 1B's 16.
- Length baseline is model-independent (same prompts/split): token-length AUROC
  0.37. The probe clears it by the same wide margin on the 8B.

**Surface baselines on the same 120-prompt held-out split (`baselines.py`):**

| reads | AUROC | strength |
|---|---|---|
| activations (probe, 8B) | 0.9944 | 0.494 |
| format / structure only (no words) | 0.9675 | 0.468 |
| bag-of-words (word counts only) | 0.9517 | 0.452 |
| char length | 0.3929 | 0.107 |
| word-count length | 0.4506 | 0.049 |

**This is the key result so far, and it is a negative one.** A classifier that
reads no words at all — only `A)`/`(B)` option markers, digit density, colons,
newlines, punctuation ratios — reaches 0.9675. Bag-of-words on word counts
reaches 0.9517. The probe's margin over the best surface baseline is only
**+0.027 in strength**, which at n=120 (60/60) is within sampling noise. We must
NOT claim the probe reliably beats surface features on this data.

Internal check: length alone is near chance (strength 0.05–0.11), so the format
classifier's 0.97 is not length leaking in — it is genuinely the
benchmark-canonical MCQ shape.

Honest scope after baselines: the pipeline is correct and the replication
reproduced, but **the obvious-eval cell is fully confounded.** SAD-eval prompts
look and read like benchmark items; WildChat-deploy prompts do not. Surface
structure and eval-ness are perfectly entangled, so this cell cannot distinguish
"the model internally recognizes evaluation" from "these prompts are shaped like
tests." This does not show the model lacks eval-awareness, and does not make the
probe wrong — it shows this dataset cannot answer the question. It is
Devbunova's critique measured directly on our own data, and it is the empirical
mandate for the format-matched arm: we need pairs where surface classifiers are
forced to ~0.5 by construction, so whatever the probe still finds cannot be
surface.

## What's next

The full forward plan lives in `PLAN.md` (written Aug 2026 after a research
pass). Summary: slice 2 is the format-matched 2x2. Key research findings that
shaped it: Devbunova's 2x2 dataset is public (HF
`viliana-dev/eval-awareness-2x2`, GitHub `viliana-dev/eval-awareness-format`)
and was built on Llama-3.1-8B-Instruct, our exact model, n=1076/cell; their
Casual-Eval cell is a MuTual proxy (admitted), which is the weakness our
rewrite arm fixes; their paired-training result (format-invariance restored)
is the claim our Casual-Eval-True cell stress-tests. Rewrite acceptance gates
are probe-blind (surface baselines to ~0.5, manual audit), never selected by
probe score (Xiong et al. selected by probe score — circular for dataset
construction). Statistics upgrade from slice 2 on: bootstrap CIs, >=3 seeds.

1. **[DONE] Slice 1 on Llama-3.1-8B** — replicated on Kaggle (2x T4).
2. **[DONE] Surface-baseline hierarchy** — obvious cell fully confounded (see
   Results).
3. Slice 2 (per PLAN.md): replicate Devbunova on our machinery, then replace
   their proxy Casual-Eval with true content-preserving rewrites.

## Gotchas / hard-won notes

- **uv on the Windows desktop cannot install Pythons to its default dir**
  (`AppData\Roaming\uv\python`): extraction silently produces an empty
  directory and `uv python install` fails with "Missing expected target
  directory". Workaround: `UV_PYTHON_INSTALL_DIR=C:/Users/grove/uv-pythons`.
  The 3.12 interpreter lives there; the repo `.venv` already points at it, so
  day-to-day `uv sync` works without the env var. Only set it again if
  installing another Python version.

- transformers 5.x: use `dtype=`, not `torch_dtype=`. `apply_chat_template(...,
  return_dict=True)` returns a `BatchEncoding`, not a bare tensor.
- The scratchpad directory and full SAD clone get wiped on session teardown.
  Durable state lives in the repo's gitignored `data/` (the encrypted SAD zip)
  and `results/` (activations).
- Do not trust HF metadata 200s for gated access; test file access.
- Model load of the 1B takes ~1–2.5 min the first time (download + load).

## Maintaining this file

Update CLAUDE.md and README.md as part of finishing each unit of work — new
results, new decisions, changed status, new gotchas. CLAUDE.md gets the details
and reasoning; README.md gets the short human summary. Never put SAD prompt text
in either (canary rule). Keep the "Last updated" line and "Results so far"
current.
