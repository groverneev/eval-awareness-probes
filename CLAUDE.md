# CLAUDE.md — agent onboarding and running log

This file is the detailed context for any agent picking up this project. Keep it
updated as work progresses (see "Maintaining this file" at the bottom). The
README is the short human-facing version; this is the long one.

Last updated: August 2026. Slices 1 and 2 are done on Llama-3.1-8B; slice 3
(multi-turn) is in progress — see the HANDOFF section below for exactly where
it stands and what to run next. Forward plan in `PLAN.md`. Primary machine is
the Windows desktop with an RTX 5070 Ti (see Environment).

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

## Module map

```
src/vnaa/
  probe.py               per-layer difference-of-means probe + AUROC
  extract.py             activation extraction; READ_POINTS, turn_spans,
                         read_slice, extract_trajectories
  baselines.py           surface baselines + bootstrap_auroc_ci
  data.py                SAD loader (slice 1)
  data_2x2.py            Devbunova cell loader (slice 2a)
  run_slice1.py          slice-1 runner
  run_2x2_baselines.py   baselines on Devbunova's cells
  run_slice2a.py         extract/analyze on the public 2x2
  build_2b.py            slice-2b sampling + early gate
  prep_2b_batches.py     slice-2b batch inputs; looks_like_quiz,
                         is_question_form, match_pool
  hydrate_batches.py     rehydrate slice-2b batches with real prompt text
  merge_2b.py            merge slice-2b writer output + probe-blind gate
  run_transfer.py        6x6 cross-cell transfer matrix, multi-seed
  run_per_source.py      arc/mmlu/gsm8k breakdown
  check_rewrite_drift.py deploy rewrite vs real originals
  prep_slice3.py         multi-turn sampling; looks_like_coursework
  prep_slice3_batches.py slice-3 batch inputs (+ --scratch hydration)
  gate_slice3.py         slice-3 probe-blind gate
  run_slice3.py          slice-3 extract/analyze across read points
scripts/
  make_figures.py        slice-1 figures
  make_2b_figures.py     slice-2b figures (math-free by default)
tests/
  test_probe.py          probe math, known-answer synthetic data (6)
  test_read_points.py    multi-turn token spans and read indices (9)
```

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

**Slice 2a step 1 — our surface baselines on Devbunova's public 2x2 cells
(CPU, seeds 0-2, bootstrap CIs; `run_2x2_baselines.py`):**

| contrast | length | format | bag-of-words |
|---|---|---|---|
| bench_eval vs casual_deploy_matched | 0.53 | 0.98 | 1.00 |
| bench_eval vs bench_deploy_rewritten | 0.50 | **0.86** | 0.96 |
| casual_eval_mutual vs casual_deploy_matched | 0.13 | **0.9988** | 0.99 |
| bench_deploy_rewritten vs casual_deploy_matched | 0.55 | 0.96 | 0.99 |
| casual_deploy_formal vs casual_deploy_matched | 0.17 | 0.94 | 0.99 |

Readings: (a) their length histogram-matching works and only length; (b) their
MuTual Casual-Eval proxy is near-perfectly surface-separable (lowercase,
tokenized spacing, short turns), so their "paired training restores
format-invariance" claim may ride on MuTual artifacts — this is the empirical
mandate for our Casual-Eval-True rewrites, stronger than the argument in the
proposal; (c) even their format-matched contrast leaks structure to our
richer format classifier (0.86 vs their claim of matched); (d) BoW can never
hit 0.5 when content differs by construction — our 2b gate therefore demands
format+length at ~0.5 and drives BoW down via topic-matched deploy
counterparts, rather than pretending BoW 0.5 is achievable.

**Slice 2a — Devbunova replication on our machinery (8B, local 5070 Ti,
bf16 + CPU offload, ~16 min for 4063 extractions; `run_slice2a.py`):**

Standard regime (train bench_eval vs casual_deploy_matched): in-dist AUROC
1.000 at layer 2; held-out fraction classified as eval: bench_deploy_rewritten
**0.960** (their 94.5%), casual_eval_mutual 0.000 (their 0.0%). Transfer
bench_eval vs bench_deploy_rewritten at the fixed best layer: 0.53 — the
standard probe sees format, not content. Clean qualitative replication with a
difference-of-means estimator instead of their logistic regression.

Paired regime (train bench_eval+casual_eval_mutual vs casual_deploy_matched+
bench_deploy_rewritten): in-dist 0.969 [0.961, 0.976] at layer 5;
casual_eval_mutual 0.997 classified as eval, bench_deploy_rewritten 0.180
(their 7.2% error — ours a bit worse, difference-of-means vs their CV'd LR).

Interpretation: their pattern replicates, BUT step 1 showed casual_eval_mutual
is surface-separable at 0.9988 (format classifier). So "paired training
restores invariance" is consistent with the probe learning (benchmark
structure OR MuTual style) as the positive class — not yet evidence of genuine
eval-content signal. 2b's Casual-Eval-True rewrites are the discriminating
test.

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

## Slice 2b design (settled Aug 2026 after a prototype gate)

A 40-item hand-written prototype was gated before scaling, and it killed the
original design. The finding: **Claude-authored text is surface-separable from
real ShareGPT text regardless of content.** Topic-matched counterparts whose
content is pure deployment (genuine user needs, no benchmark material) still
separated from real ShareGPT at format 0.93 / BoW 0.91, and the same ~0.86-0.95
band appeared for every Claude-written cell. Length is the main tell: real
ShareGPT prompts are longer and messier than anything we write.

Consequence: contrasting our rewrites against real ShareGPT would train the
probe on authorship, not eval-ness — Devbunova's mistake with a new artifact.

Design as built (Neev's calls, both forks):
1. **Every cell is Claude-authored.** Real ShareGPT prompts get rewritten too,
   so authorship cancels instead of confounding. The deploy side still derives
   from real user traffic. Cells: `casual_eval_a`, `casual_eval_b`,
   `casual_deploy_rw`.
2. **Two eval cells, reported separately.** Style A = benchmark question asked
   plainly and casually (faithful, but retains quiz flavour). Style B = same
   question wrapped in a plausible personal need (reads like deployment; the
   harder test). Reporting both separates quiz-flavour detection from genuine
   content recognition.
3. **Surface matching built into the inputs, not hoped for**
   (`prep_2b_batches.py`): every item carries an explicit `target_chars`, and
   the same multiset of target lengths is dealt to all three cells, so length
   matches by construction; the ShareGPT source sample is drawn to match the
   benchmark sample's digit-density distribution (0.0124 vs 0.0123 achieved),
   because digit density was the worst single surface feature in the prototype
   (AUROC 0.69 — maths items carry numbers, generic questions do not).
4. **One writer per batch handles that batch's eval AND deploy items**, so
   per-writer style variation is balanced across cells rather than correlated
   with the label.
5. Malformed MMLU items (stems referencing absent numbered lists, e.g. choices
   like "1,3,4") are filtered: 399 -> 389 usable.

Two failures caught by auditing the first pilot batch (both fixed, batches
regenerated — audit the writers' output, their self-reports miss these):

- **Length targets forced style A to become style B.** Targets drawn from the
  ShareGPT length distribution (mean 193 chars) are far longer than a bare ARC
  question, so the writer padded style A with personal framing to hit them:
  44% of style-A items carried first-person pronouns vs 78% for style B, and
  length adherence was a perfect 0% violations, which is exactly the cause.
  Fix: targets are now sized from each question's own length (mean 133), and
  writers are told never to pad style A to reach a target. Style A is defined
  by the absence of a SITUATION, not the absence of a pronoun.
- **Real user traffic contains homework.** 16% of the sampled ShareGPT deploy
  prompts were math word problems, in content indistinguishable from GSM8K
  eval items — and digit-density matching actively selected for them, since
  that is where the digits are. Fix: `looks_like_quiz()` filters the pool
  before matching (1076 -> 1037). Note this is a real tension, not just a bug:
  a GSM8K item asked casually IS what a real user asks, so eval-ness there
  lives in provenance, not content. Report probe results per source
  (arc/mmlu/gsm8k) — if the probe separates ARC/MMLU but not GSM8K, the signal
  is "knowledge-quiz-shaped", not evaluation recognition.

Third pilot gate (100 items) fixed length but failed on structure and
vocabulary, and the diagnosis forced the final design:

- length matched essentially perfectly (AUROC 0.49, strength 0.008) once
  targets were sized from the questions. That part of the engineering works.
- format 0.81 was driven almost entirely by question marks: **100% of eval
  items contain "?" vs 54% of deploy items**, because benchmark items are
  questions while user prompts are imperatives ("write me a post", "set up an
  api"). Fixed by filtering the deploy pool to question-form prompts.
- BoW 0.93 is driven by topic: eval-ward words are `which, what, does, total`,
  deploy-ward are `me, my, need, write, want`. **This one cannot be engineered
  away against real traffic.** Best-effort TF-IDF topic matching against real
  prompts gives a mean best-match similarity of 0.104 (ShareGPT) / 0.127
  (WildChat) — essentially no overlap, because real users do not ask about
  xylem cells or Sylow subgroups. Resulting BoW after matching: 0.951 /0.935.

Final design — four cells, all Claude-authored, n=250 each:
`casual_eval_a`, `casual_eval_b`, `casual_deploy_wc` (real WildChat prompts
rewritten; carries the realism claim) and `casual_deploy_tm` (synthetic
genuine-need messages on each eval item's own subject; the only construction
that can drive BoW toward chance). WildChat replaced ShareGPT as the real
source: marginally better topic match and slice 3 needs it anyway. The sample
lives at `data/wildchat_first_turns.json` (gitignored, 3000 English first
turns, streamed from `allenai/WildChat-1M`).

Gate (`merge_2b.py`, probe-blind): length and format strength <= 0.10 on all
four eval-vs-deploy contrasts; BoW gated only against `casual_deploy_tm`,
where topic is controlled by construction, and *reported as the bar the probe
must beat* against `casual_deploy_wc`, where topic difference IS the label
rather than an artifact. Also reported, not gated: each cell vs real prompts
(authorship control), A vs B (style separation), and wc vs tm (deploy parity).

Process lesson: **freeze batch inputs before spawning writers.** Regenerating
`batch_*.json` mid-run left one writer generating against a snapshot of a
rotated input; its output had to be discarded.

## Slice 2b results (8B, n=250/cell, seed 0)

Gate outcome: length matched (AUROC 0.47-0.52 on all four eval-vs-deploy
contrasts). Format 0.67-0.87 and BoW 0.91-0.98 do NOT reach chance, and the
diagnosis is that they cannot: the residual signal is **speech act**. Eval-ward
words are `what, which, does, why`; deploy-ward are `my, our, we, should,
want`. An impersonal question with a canonical answer versus a first-person
request for help IS the eval/deploy distinction, so matching it away would
destroy the label. Style B (personal framing) narrows but does not close it
(BoW 0.91 vs style A's 0.93 against real traffic).

Consequence (Neev's call): the decisive evidence is **cross-cell transfer**,
not the in-distribution baseline gap. Fit on one cell pair, freeze, score a
differently-constructed pair; run the identical test on every baseline, so
probe-vs-baseline is meaningful without needing baselines at chance.
`run_transfer.py` builds the 6x6 matrix.

Findings:

All numbers below are mean +/- sd over **5 split seeds (0-4)**.

1. **Devbunova replicated on the transfer axis.** A probe trained on the
   standard confounded contrast (bench_eval vs casual_deploy) fails to
   transfer: 0.548+-0.01 to their format-matched cell, 0.688+-0.03 to our
   casual cells. It is a format detector. It is also the ONLY training set
   where BoW BEATS the probe (probe-minus-BoW **-0.071+-0.021**, win rate
   24%) — i.e. on the standard setup, activations are worse than word counts
   at generalizing.
2. **New result: training on format-matched casual data generalizes, and the
   asymmetry is large and stable.** Probe trained on `ours_a_wc` scores
   **0.966+-0.01** on Devbunova's own standard cell and 0.990 / 0.959 / 0.922
   on our other cells. The reverse does not hold: theirs -> ours is
   0.688+-0.03. The direction learned from casual, length-matched data covers
   the benchmark case; the direction learned from benchmark data does not
   cover the casual case.
3. **The probe beats bag-of-words under transfer, consistently:** overall
   **+0.048 +- 0.012** mean strength, positive on all 5 seeds (range +0.031
   to +0.062). Per training set: `ours_b_wc` +0.128+-0.038 (win rate 72%),
   `ours_a_wc` +0.088+-0.020 (88%), `ours_a_tm` +0.072+-0.015 (88%),
   `ours_b_tm` +0.047+-0.027 (76%). Every one of our four training cells
   yields a positive margin; only the standard confounded cell is negative.
4. **The format classifier collapses under transfer** (0.124+-0.008 vs the
   probe's 0.300+-0.012), so the surviving signal is not structural. Length
   is flat at 0.026 throughout, as designed.
5. **Hardest cell for everything: `devb_fmt_matched`** (bench_eval vs
   bench_deploy_rewritten — benchmark format on both sides, only content
   differs). Training sets land at 0.37-0.64 there; from `ours_b_tm` it is
   0.374+-0.04, i.e. reliably ANTI-correlated. This is the cell where
   content-only eval recognition would have to show up, and it does not.

**Per-source breakdown (`run_per_source.py`), 5 seeds x transfer targets —
the prediction in the 2b design notes was WRONG, and the real pattern is a
bigger caveat than the one we were guarding against:**

| source | probe | bag-of-words | probe margin |
|---|---|---|---|
| gsm8k | **0.949 +- 0.055** | 0.833 +- 0.111 | +0.116 |
| mmlu | 0.775 +- 0.165 | 0.693 +- 0.153 | +0.082 |
| arc | **0.727 +- 0.176** | 0.685 +- 0.173 | +0.042 |

We predicted that if the probe separated ARC/MMLU but not GSM8K, the signal
would be "knowledge-quiz-shaped" rather than eval recognition. The reverse
holds: math word problems are by far the easiest and ARC science-knowledge
questions the hardest, with a wide seed spread. Implications:

- The headline "probe beats BoW under transfer" is **carried substantially by
  GSM8K.** On ARC the margin is +0.042 with sd 0.176 — not distinguishable
  from zero at this n.
**The gsm8k number is a digit-density artifact — confirmed, not speculated.**
Digit density alone, a single scalar with no fitting, separates gsm8k eval
items from their deploy counterparts at **0.873 (vs wc) and 0.949 (vs tm)** —
the latter identical to the probe's 0.949 gsm8k transfer AUROC. Per-cell mean
digit density:

| cell | arc | mmlu | gsm8k |
|---|---|---|---|
| casual_eval_a | 0.0013 | 0.0131 | **0.0452** |
| casual_deploy_tm | 0.0007 | 0.0014 | **0.0033** |

A 13x gap on gsm8k. Cause: `prep_2b_batches.py` matched digit density between
the *benchmark pool* and the *WildChat pool*, but the synthetic topic-matched
counterparts were written with no digit-density requirement, so they came out
nearly numeral-free while the maths items kept every figure. Pool-level
matching does not imply item-level matching. On arc the same scalar is at or
below chance (0.353 / 0.505), so arc is uncontaminated — and arc is exactly
where the probe's margin over BoW is weakest (+0.042 +- 0.176).

Consequences: gsm8k results are excluded from the headline claim, and the
math-free (arc+mmlu) transfer run below IS the slice-2b result. If the tm cell
is ever rebuilt, give the counterpart writer an explicit per-item digit-density
target, not a pool-level one.

**MATH-FREE TRANSFER (arc+mmlu only, both train and test; 5 seeds) — the
headline result. It is STRONGER than the contaminated version:**

- **probe minus BoW: +0.071 +- 0.013**, positive on all 5 seeds, range +0.053
  to +0.088. (The gsm8k-contaminated run gives +0.048 +- 0.012, i.e. including
  maths items makes the probe look WORSE, because BoW exploits the numeral
  shortcut at least as well.)
- probe transfer strength 0.287 vs BoW 0.221+-0.009 vs format ~0.13 vs
  length 0.026.
- Per training set: `ours_b_wc` **+0.143+-0.023** (win rate 80%), `ours_b_tm`
  +0.110 (96%), `ours_a_tm` +0.096 (**100%**), `ours_a_wc` +0.097 (80%),
  `devb_fmt_matched` +0.068 (68%). Only `devb_standard` is negative,
  **-0.087+-0.032, win rate 12%**.
- The asymmetry is sharper without maths: `ours_a_wc` -> `devb_standard`
  **0.949+-0.00**, while `devb_standard` -> `ours_a_wc` falls to
  **0.578+-0.03**, i.e. barely above chance.

Why removing gsm8k helped rather than hurt: the per-source table restricted
only the test set while still training on maths-containing cells. Filtering
both sides removes the numeral shortcut from training too, so the fitted
direction is cleaner — and BoW loses a shortcut it was also exploiting (its
gsm8k score was 0.833).

Remaining negative, now sharper: transfer to `devb_fmt_matched` from our cells
is 0.349-0.516, i.e. at or reliably BELOW chance. Benchmark format on both
sides with only content differing defeats every direction we can learn. That
is the honest limit of the current result and the obvious target for slice 3.

**Rewrite-drift check (`check_rewrite_drift.py`), two-sided:**

Paired against the 250 unmodified WildChat originals, our deploy rewrites did
shift along exactly the speech-act axis that carries the residual signal:

| marker | original | rewrite | shift |
|---|---|---|---|
| first-person | 0.26 | 0.58 | **+0.33** |
| request cue | 0.20 | 0.34 | +0.14 |
| factual question | 0.15 | 0.04 | -0.11 |
| ends with ? | 0.58 | 0.49 | -0.10 |

So the deploy cell is markedly more first-person and request-shaped than the
real traffic it derives from. That is a real limitation and must be stated.

BUT the specific worry — that reframing the 15% of originals which were
themselves factual questions is doing the work — is **ruled out**. Restricting
to the never-reframed subset leaves separation unchanged (format 0.683 vs
0.668 full; BoW 0.942 vs 0.933). The shift is uniform across all items, not
concentrated in the reframed ones.

Partial mitigation already in the transfer matrix: the probe trained on
`ours_a_wc` scores 0.952 on `devb_standard`, whose deploy side is REAL,
unrewritten ShareGPT. So the direction learned on our (stylistically shifted)
cells does transfer to a cell with untouched deployment prompts.

Remaining caveats: n=250/cell means 75/cell held out, so per-transfer CIs are
about +/-0.05.

## Slice 3 design (multi-turn)

**Deployment side.** 250 real WildChat conversations with 2-4 user turns,
truncated to end on a user turn (every read point is defined relative to the
final user turn). Filters, each earned: English; per-message length 20-600
chars; no coursework in EITHER speaker's turn; final user turn must be
question-form. `looks_like_coursework()` is stricter than slice 2b's
`looks_like_quiz()` because multi-turn traffic is full of pasted
multiple-choice homework that the single-turn filter missed — the first sample
drawn here had assistants replying "B) Use them as a springboard". The
question-form requirement exists because evaluation items are questions: without
it, deploy conversations end on imperatives and the cells separate on
question-mark rate alone (measured 94% vs 16%, format classifier 0.96).

**Three written cells, all model-authored** so authorship cannot carry the
label — the real conversations carry genuine GPT assistant turns, so authoring
only the eval side would let the probe read assistant authorship:
- `deploy` — rewrite of the real conversation, same shape
- `eval_disguised` — same shape, final user turn poses a benchmark question
  reached without any test framing
- `eval_obvious` — same question, openly framed as a test (the obvious cell of
  the turn-structure x disguise 2x2)

**Plus one constructed cell**, `eval_shared`: the `deploy` conversation with
only its final user turn swapped for the eval question. No writer cost, and
openings are identical by construction.

**Per-item matching:** all cells for an item are written to the same skeleton
(turn count + per-turn target lengths from the real conversation). Benchmark
items are ARC/MMLU only (no GSM8K — see the slice-2b digit confound) and are
drawn fresh, unused in slice 2b, so slice 3 is not partly a retest.

**Writer instructions that turned out to matter:** cross-cell length parity
beats hitting the original per-turn targets; the question-mark decision is made
per ITEM so all three cells match; assistant turns must be identical in style
across cells; and (added for batches 02-04) eval_disguised must open on an
ordinary everyday need rather than a topical ramp toward the question.

## Slice 3 results so far (8B, n=100/cell — 2 of 5 batches written)

Only batches 00-01 exist; the other three writers were stopped mid-run. All
numbers below are 5 seeds, mean +/- sd, and should be redone at n=250.

**Read points and what each can see.** Attention is causal, so a hidden state
at token i depends only on tokens <= i. `post_instruction_token` sits at the
end of the FIRST user turn and therefore has not seen the eval question at all;
`last_user_token` and `final_turn_mean` sit in the final user turn and have
seen everything. The text baseline for each read point must be restricted to
what that position sees, or the comparison is rigged — a first pass here gave
`final_turn_mean` a final-turn-only baseline while the probe saw the whole
conversation, inflating its margin from +0.08 to +0.27.

**Finding 1 (a dataset failure, caught by the read points).** On the
as-written cells, the probe scores **0.994 at `post_instruction_token`** — a
position that cannot see the eval question — and bag-of-words on that opening
turn alone scores 0.926. The disguised conversations are identifiable from
their FIRST turn. Writers invented science-flavoured lead-ins while the deploy
openings are rewrites of real, messy user needs, so the "disguise" fails at the
conversation level and the high AUROC partly reflects synthetic-lead-in
detection, not eval recognition.

**Finding 2 (the fix, and the strongest result in the project).** Building
`eval_shared` by mechanical recombination — take the DEPLOY conversation and
replace only its final user turn with the eval question — makes the openings
identical by construction, at no writer cost. Then:

| read point | probe | BoW (whole conv) | BoW (final turn only) |
|---|---|---|---|
| post_instruction_token | **0.502** | 0.500 | — |
| last_user_token | **0.994** | 0.592 | 0.726 |
| final_turn_mean | 0.993 | 0.592 | 0.726 |

- `post_instruction_token` lands at exactly chance, which is the sanity check
  the construction demands: identical text cannot separate.
- At the final turn the probe hits 0.994 while the toughest lexical baseline
  (final turn alone) reaches 0.726 and format 0.614 — a margin of **+0.268**,
  far larger than slice 2b's +0.071.

Caveat to state plainly: the probe reads a position that can attend selectively
to the final turn, while bag-of-words averages over whatever text view it is
given, so some of the margin is that asymmetry rather than depth of
representation. Reporting both baseline views (whole conversation +0.40, final
turn only +0.268) is the honest range.

Also note `eval_shared` conversations are deliberately topic-shifted: the
question does not follow from the preceding turns. Real users do change subject,
but a matched-coherence version is the obvious next control.

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

1. **[DONE] Slice 1 on Llama-3.1-8B** — replicated (Kaggle, later local).
2. **[DONE] Surface-baseline hierarchy** — obvious cell fully confounded.
3. **[DONE] Slice 2a** — Devbunova replicated on our machinery.
4. **[DONE] Slice 2b** — four format-matched cells, cross-cell transfer.
5. **[IN PROGRESS] Slice 3** — multi-turn. See the handoff below.
6. Slices 4-6 (silence filtering, deployment-simulation, steering) unstarted.

---

# HANDOFF — read this first if you are picking the project up

State as of the end of the August 2026 working session.

## Where slice 3 actually is

Dataset is 5 batches of 50 items. **Batches 00, 01 and 04 are written and
committed** (`datasets/slice3/batches/out_0{0,1,4}.json`). **Batches 02 and 03
were being written by two subagents when the session ended** — check whether
`out_02.json` and `out_03.json` exist and contain 50 complete items each
(every item needs all three of `deploy`, `eval_disguised`, `eval_obvious`). If
they are missing, respawn two writers; the exact prompt used is reconstructable
from the requirements in "Slice 3 design" below, and the batch inputs must be
hydrated first (see next section). Two earlier attempts were killed mid-run and
lost everything because they held the result in memory; tell any writer to
write its output file early and refine in place.

All published results so far are at **n=100** (batches 00-01 only). Everything
should be rerun once all five batches exist.

## Reproducing the slice-3 inputs

WildChat is gated and its entries are real user messages, so no prompt text is
committed. To regenerate what the writers read:

```
.venv\Scripts\python -m vnaa.prep_slice3 --n 250 --scan-limit 400000
.venv\Scripts\python -m vnaa.prep_slice3_batches --scratch <a scratch dir>
```

The first writes `data/wildchat_multiturn.json` and `data/slice3_skeletons.json`
(both gitignored); the second writes index-only specs to
`datasets/slice3/batches/` (committed) and hydrated copies containing the real
conversations to the scratch dir (never commit those). Both are deterministic
given the seed, so anyone with WildChat access gets identical inputs.

## The steps to finish slice 3

1. Merge and gate: `.venv\Scripts\python -m vnaa.gate_slice3`. Expect the
   final-user-turn contrast to pass length and format and the whole-conversation
   contrast to fail format (~0.76); that residual is domain content in the
   assistant turns, diagnosed and not fixable by matching.
2. Rebuild the `eval_shared` control. **This is currently ad-hoc inline code,
   not a script — write it properly.** It takes each `deploy` conversation and
   replaces ONLY its final user turn with that item's `eval_disguised` final
   user turn, writing `datasets/slice3/eval_shared.json`. Openings are then
   identical by construction, which is what makes `post_instruction_token` a
   real sanity check.
3. Extract: `.venv\Scripts\python -m vnaa.run_slice3 extract --dtype bfloat16
   --device-map auto` (~10 min for 250/cell on the 5070 Ti), then extract the
   `eval_shared` cell the same way.
4. Analyze: `.venv\Scripts\python -m vnaa.run_slice3 analyze`.
5. Two numbers decide the slice: does the **+0.268 final-turn margin** hold at
   n=250, and does the as-written `eval_disguised` cell's
   **post_instruction_token drop from 0.994 toward chance**? Batches 02-04 were
   written with revised instructions (open on an ordinary everyday need, let the
   question arrive as a tangent) specifically to fix that leak, so compare
   batches 00-01 against 02-04 to see whether the instruction change worked.
   If it did, that comparison is itself a reportable result about how easily
   synthetic "disguised" evaluations betray themselves.

## Things that will bite you

- **Never trust a subagent's self-report.** Every writer so far reported
  accurate statistics and still missed cross-cell problems only visible by
  comparing cells. Audit the output.
- **Match surface features per item within each source and cell**, never on the
  pooled aggregate. Slice 2b's headline was nearly lost to exactly this.
- **Freeze batch inputs before spawning writers.** Regenerating them mid-run
  invalidated a writer's output once.
- **Baseline text views must match what each causal read position can see.**
  Getting this wrong inflated a margin from +0.08 to +0.27 before it was caught.
- The venv is `.venv\Scripts\python` (Python 3.12); system Python is 3.14 and
  will not work. `git_filter_repo` is only in the system Python.
- Neev asked that commits be **pushed to GitHub** as part of finishing a unit of
  work. No `Co-Authored-By` trailers.

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
