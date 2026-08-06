# CLAUDE.md — agent onboarding and running log

This file is the detailed context for any agent picking up this project. Keep it
updated as work progresses (see "Maintaining this file" at the bottom). The
README is the short human-facing version; this is the long one.

Last updated: August 2026. Slices 1, 2, 3, and 5 are complete on
Llama-3.1-8B; slice 5 includes the model-swap/context-ablation decomposition
at n=250 and now an LMSYS-Chat-1M corpus replication. The secondary logistic
estimator, the permutation and random-probe nulls, and the calibrated model
judge for the silence filter are done (all CPU-only, off cached activations).
**Tier 2 is DONE**: steering (manipulation check passed at layer 14; self-report
shifts strongly; safety shows no large effect but is underpowered), the silence
filter on `eval_shared`, and the LMSYS replication all ran on a Runpod 4090.
The `eval_shared` silence filter also has a 512-token censoring sensitivity
rerun, which raises retention from 124/250 to 227/250 and leaves the conclusion
intact. Slice-1 8B activations have been re-extracted locally and
reproduce the Kaggle run exactly. Forward planning is maintained in this file.
Primary machine is the Windows desktop with an RTX 5070 Ti (see Environment).

**The remaining work is paper assembly, plus one scope decision.** The proposal's
Experimental Setup promises a model scaling sweep (70B, Qwen2.5-7B/32B, Gemma-2/3)
and per-class agentic/tool-use results. Neither is started. Both cannot fit before
the deadline; the recommendation is to cut the agentic/tool-use arm to future work
(it needs a dataset that does not exist) and, if any of it runs, do Qwen2.5-7B on
`eval_shared` as a one-hour check of whether the effect is Llama-specific.

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
  estimators.py          logistic probe, permutation/geometry nulls
  run_estimator_check.py re-scores headline contrasts, CPU only
  judge_batches.py       blinded judge batches + calibrated merge
  run_silence_filter.py  per-cell generate/label/analyze (--cell)
  steering.py            activation-addition hook; layer indexing
  run_steering.py        steering direction, manipulation check, behaviour
scripts/
  make_figures.py        slice-1 figures
  make_2b_figures.py     slice-2b figures (math-free by default)
  gpu_session_tier2.md   the frozen Tier 2 runbook
datasets/judge/
  awareness_judge_v1.md  the committed, hashed judge rubric
datasets/steering/
  prespec.md             steering protocol, frozen before any output
  behavior_prompts.json  the held-fixed self-report and safety sets
tests/
  test_probe.py          probe math, known-answer synthetic data (6)
  test_read_points.py    multi-turn token spans and read indices (9)
  test_estimators.py     second estimator + the three nulls (11)
  test_judge_batches.py  judge harness build/merge round trip (7)
  test_steering.py       hook arithmetic + scoring rules (27)
  test_prep_slice5.py    WildChat vs LMSYS source normalization (7)
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

## Slice 3 preliminary results (historical n=100/cell, superseded)

At this point only batches 00-01 existed. These five-seed figures are retained
for provenance only; the full n=250 results below supersede them.

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

## Slice 3 final results (8B, n=250/cell)

Full extraction completed on the local 5070 Ti (four cells, three read points,
and trajectories). All figures below are mean AUROC +/- sd across five
stratified 70/30 splits.

| cell | read point | probe | BoW | format | probe-BoW |
|---|---|---:|---:|---:|---:|
| eval_disguised | post-instruction | 0.951 +/- 0.009 | 0.856 +/- 0.017 | 0.603 | +0.095 |
| eval_disguised | final-turn mean | 0.994 +/- 0.004 | 0.912 +/- 0.011 | 0.647 | +0.082 |
| eval_shared | post-instruction | **0.497 +/- 0.001** | 0.500 | 0.500 | +0.003 |
| eval_shared | final-turn mean | **0.995 +/- 0.005** | 0.769 +/- 0.011 | 0.576 | **+0.226** |

`eval_shared` is the decisive sanity control: it is mechanically identical to
the matched deploy conversation before the final user turn. The end-aligned
trajectory is at chance for every preceding user turn (0.498, 0.498, and 0.501
at one, two, and three turns before the end) then becomes 0.995 at the final
turn. This verifies the read indices and localizes the result to the swapped
question.

The as-written `eval_disguised` cell remains invalid as a realistic-disguise
test. It is separable at the first user turn, before the model has seen the
benchmark question (probe 0.951; BoW 0.856). The batches written with revised
opening instructions lowered that probe only from 0.994 to 0.962, so they did
not eliminate the synthetic-lead-in artifact.

`eval_shared` is stronger but has a clear limit: the final question is a topic
shift. The next control is a matched-coherence version that preserves the
identical-prefix construction while making the final benchmark question a
natural continuation of the conversation.

## Estimator and null-baseline checks (August 2026, CPU only)

Two things the proposal promised and the code never delivered. Both run off
cached `.npz` activations with no GPU:

```
.venv\Scripts\python -m vnaa.run_estimator_check points
.venv\Scripts\python -m vnaa.run_estimator_check transfer
```

**Secondary estimator (proposal 3.1).** L2 logistic regression, `C=0.01`,
standardized inputs. Layer selection had to differ from the difference-of-means
path: LR on 4096 dims with a few hundred training rows separates the training
split perfectly at essentially every layer, so train-AUROC argmax selects noise.
`select_layer_cv` picks the layer by 5-fold CV *inside the training split*. The
test split is never consulted by either estimator.

Diff-of-means reproduces the published figures exactly, and LR is uniformly
equal or stronger:

| contrast (read point) | diff-of-means | logistic |
|---|---:|---:|
| s1 SAD oversight, n=200 (last user) | 0.992 | 0.993 |
| eval_shared (last user) | 0.992 | 0.998 |
| eval_shared (post-instruction) | 0.498 | 0.500 |
| coherent pairs (last user) | 0.951 | 0.989 |
| coherent pairs (post-instruction) | 0.499 | 0.499 |
| coherent silent, n=192 (last user) | 0.937 | 0.982 |
| s5 reconstruction n250 (assistant token) | 0.792 | 0.896 |
| s5 reconstruction n250 (assistant mean) | 0.822 | 0.958 |
| s5 model swap, shared 146 (assistant token) | 0.811 | 0.887 |
| s5 context ablation, shared 146 (assistant token) | 0.868 | 0.947 |

Slice 1 is the one row where LR buys essentially nothing (0.992 -> 0.993, versus
gains of 0.03-0.14 everywhere else). That is consistent with the rest of the
slice-1 story rather than a surprise: the obvious cell is fully confounded, so
the separation is already trivially linear and a better-fitted direction has
nothing left to recover. Its uninformed-direction geometry null is also lower
(0.160) than the multi-turn cells' 0.24-0.29.

Under math-free slice-2b transfer, probe-minus-BoW is **+0.056** for LR against
+0.071 for difference-of-means, with the same sign pattern per training cell and
the same sole negative (`devb_standard`, -0.052 LR / -0.087 DoM). The
asymmetry survives: `ours_a_wc` -> `devb_standard` 0.958, reverse 0.642. **No
conclusion in this project depends on the estimator.**

**Nulls (proposal 4, "a random probe near 0.5").** Three, and the difference
between them is the finding:

1. `permutation_null` is the significance test. It permutes the *entire* label
   vector and reruns the whole pipeline including layer selection. Every real
   contrast lands at **p = 0.002** (the floor at 500 permutations); null p95 is
   0.09-0.13 strength. The two identical-prefix controls land at p = 0.99
   (`eval_shared` post-instruction) and p = 0.94 (coherent pairs
   post-instruction), which is the sanity check the construction demands.
2. `train_shuffled_direction_null` permutes only the training labels and scores
   true test labels. **This is not a significance test and must never be
   reported as one.** It reaches strength **0.24-0.29** on the strongest cells.
3. `random_direction_null` fits nothing at all and reaches **~0.11**.

Why 2 and 3 are large: the eval/deploy axis carries a big share of total
activation variance (at `eval_shared` last-user token, ||mu1-mu0|| = 2.99
against a total within-class spread of 8.39 across all 4096 dims), so any
direction drawn from this data partially recovers it, with random sign. The
first null I wrote was this one, and on a single seed it made the
silence-filtered result look like p = 0.065. That reading was wrong: it is a
null contaminated by the very signal under test. **The correct null permutes
train and test labels together.**

Consequences for the paper: the proposal's assumption that a random probe sits
near 0.5 is false for these activations at the reported layers, so raw
|AUROC - 0.5| overstates how much work the probe is doing. Significance is
unaffected. Report the permutation p-value as the test and the uninformed
direction as a limitation.

Slice 1 was originally absent here because its 8B activations only ever existed
on Kaggle. It has since been re-extracted locally on the 5070 Ti (about three
minutes of forward passes, no generation) and is included above. Every other row
is unchanged to the printed precision, which also confirms that the
`run_silence_filter` and `prep_slice5` refactors did not disturb any cached
result.

## Verbalized-awareness judge (August 2026)

`judge_batches.py` plus the committed rubric `datasets/judge/awareness_judge_v1.md`.
The regex in `run_silence_filter` stays because it is deterministic and anyone
can rerun it; the judge exists to catch paraphrase the regex cannot.

The design problem: the expected answer is "nothing here", and a judge that
answers `none` to everything scores perfectly on a set with no positives. So:

- **Planted controls.** 14 hand-written replies that verbalize recognition (8
  overt, 6 paraphrase-only) and 10 hard negatives that use test/exam/benchmark
  vocabulary as subject matter. Both numbers are needed: flagging every
  occurrence of "test" would also give perfect recall.
- **Blinding.** Rows shuffled, opaque ids, one reply per row. The judge never
  sees the cell, the pair, or which rows are controls.
- **Overlap.** 52 duplicated rows under fresh ids for an agreement estimate.

Result over all 500 real replies: planted positives **14/14**, planted-negative
false alarms **0/10**, between-judge agreement **47/47** exact (5 duplicates
landed in one batch and are reported separately as within-judge consistency).
Real pairs flagged: **0/250**, agreeing with the regex on every response. Rerunning
`run_silence_filter analyze --labels results/judge/judge_labels.json` reproduces
0.937 +/- 0.033 exactly, since the mask is unchanged.

Manual audit on top of the judge, per the never-trust-a-subagent rule: all 16
non-`none` labels were planted controls with verbatim quotes, and every one of
the 32 evaluation-vocabulary contexts in the real replies is subject-matter use
(water tests, blood tests, hypothesis testing, lab glassware, NMR simulation).

**Limitation to state in the paper:** the judges ran as Claude Code subagents,
which receive this repository's `CLAUDE.md` as project context, so blinding to
the project's expectations cannot be fully guaranteed by construction. The
planted controls are the defence — a judge biased toward `none` would have
missed them, and it missed none. Batch inputs and the rubric are on disk and
hashed, so a scripted API call reproduces the run exactly if a reviewer wants
authorship-independent labels.

Process note, learned again the hard way: 5 of 52 duplicate rows landed in the
same batch as their original. Caught after the judges were already running, so
the batches were left frozen and the merge reports cross-batch and within-batch
agreement separately. If the harness is rebuilt, assign duplicates to a
different batch at build time.

## What's next

## Matched-coherence pilot (50 pairs, August 2026)

The first attempted `eval_coherent` control was rejected after manual audit:
rewriting an arbitrary benchmark question against its original real deployment
prefix softened topic shifts but did not make them coherent. Do not use those
uncommitted outputs or their high probe score.

The repaired design constructs one shared everyday-topic prefix per item, then
attaches either a faithful casual benchmark question or a same-topic genuine
deployment request. Prefixes are byte-identical within each pair and authorship
is balanced. The 50-pair pilot passed its construction checks, then the full
**n=250** paired control replicated the result: whole-conversation surface
baselines are length 0.500, format 0.520, and BoW 0.646. The pre-final read is
exactly chance (probe 0.500, BoW/format 0.500); at the final token the probe is
**0.951 +/- 0.011** versus BoW **0.852 +/- 0.012** and format 0.579, a +0.099
probe-BoW margin. Final-turn mean is 0.941 +/- 0.011. This removes the prior
topic-shift failure while retaining the causal identical-prefix sanity check.

**Cross-control transfer (last-user token, five seeds).** Freeze the direction
on one valid multi-turn control and score the other: `eval_shared` -> paired
coherent is **0.858 +/- 0.032** versus BoW 0.597 +/- 0.014 (margin +0.262);
paired coherent -> `eval_shared` is **0.788 +/- 0.099** versus BoW 0.564 +/-
0.019 (margin +0.224). This is the strongest robustness result so far: the
signal transfers both ways between independently constructed controls while
the lexical and format baselines largely collapse.

**Silence-filtered matched coherence (n=250 generated pairs, August 2026).**
We greedily generated one 8B response for every eval and deploy conversation,
with `<|eot_id|>` recognized as an end-of-turn token. The response cache records
token counts and termination, so a reply that reached the 256-token cap is
indeterminate rather than silently counted as unaware. Of 250 pairs, 192 had
both replies terminate naturally; 58 were excluded for a capped reply.

The explicit-awareness rule is narrow by design: it flags self-referential
statements about a test, quiz, benchmark, evaluation, or assessment, but not an
ordinary use of those words in the answer's subject matter. It flagged 0/250
pairs. A separate text-only audit of a random 30 pairs (60 replies) found 0
explicit or paraphrased recognition statements, and no apparent regex misses.
On the 192 terminated, non-flagged pairs at the final user token, the probe is
**0.937 +/- 0.033**, versus BoW **0.823 +/- 0.026** and format **0.567 +/-
0.016**; probe-BoW strength margin **+0.114**. The held-out layer is now chosen
using the training split only (the prior helper incorrectly selected it on the
test split); the full matched-coherence final-token result remains 0.951 +/-
0.011 after that correction.

This is an observed-silence result, not a claim that every response is known
unaware: the 58 capped pairs are outside the estimate, and zero positive
verbalization labels mean a verbalization-detectability classifier cannot be
fit for this run. The next slice is deployment-simulation reconstruction on
genuine conversations.

**Slice 5 deployment-simulation reconstruction pilot (n=50, August 2026).**
We sampled 50 genuine assistant-ending WildChat conversations (2--4 user turns)
and greedily regenerated only their final assistant reply with the same 8B.
The prefix is therefore byte-identical within every original/resampled pair and
the labels are exact response provenance, not a hand-constructed proxy. We had
to add assistant-side read points: a final-user read happens before either
candidate answer exists and cannot distinguish the conditions. Seven replies
reached the 256-token cap and are indeterminate; analysis uses the other 43.

At `final_assistant_token`, the held-out probe is **0.761 +/- 0.086**, versus
BoW **0.575 +/- 0.041** and format **0.667 +/- 0.048**. The conservative margin
over the strongest surface baseline (format) is +0.094 strength. At
`final_assistant_mean`, the figures are 0.737 +/- 0.078, 0.575 +/- 0.041, and
0.667 +/- 0.048, respectively (probe-format margin +0.070). This establishes a
small exact-label response-provenance effect, not evaluation awareness: final
answer text itself is necessarily different, and format remains a substantial
baseline. The next valid expansion is n=250 under the same frozen protocol,
followed by model-swap versus eval-context ablation to separate model authorship
from context.

**Slice 5 scale-up (n=250, August 2026).** The pilot protocol was frozen and
repeated on 250 newly sampled genuine assistant-ending WildChat conversations.
Greedy 8B replacement produced 196 naturally terminated pairs; 54 cap-hit
replies are indeterminate and excluded. At `final_assistant_token`, the held-out
probe is **0.792 +/- 0.024**, versus BoW **0.633 +/- 0.011** and format **0.680
+/- 0.021**: +0.113 strength over the strongest surface baseline. At
`final_assistant_mean`, the probe is **0.822 +/- 0.018**, with +0.142 strength
over format. The n=50 pilot (token 0.761, mean 0.737) therefore replicated and
sharpened at scale. This is still response-provenance detection, not a direct
test of eval awareness. The next required decomposition is model-swap versus
context ablation; steering remains last.

**Slice 5 model-swap/context-ablation decomposition (n=250 generated per
condition, August 2026).** Llama-3.2-3B regenerated every final reply with the
full original prefix, while the probed 8B regenerated each reply from the final
user message alone; the 8B always read the complete original conversation with
the replacement inserted. We re-scored both on the exact intersection of their
naturally terminated pairs (146/250), rather than comparing their differently
censored full samples. At `final_assistant_token`, model-swap is **0.811 +/-
0.046** versus format 0.651 (margin +0.160), and context ablation is **0.868
+/- 0.012** versus format 0.702 (margin +0.166). At `final_assistant_mean`, the
respective probe margins are +0.138 and +0.101. Context removal therefore
produces a comparably strong, token-wise stronger effect than model substitution:
the reconstruction signal is not attributable to 8B-specific authorship and
remains a provenance/context result, not evidence of evaluation awareness.

Slice 2 is the format-matched 2x2. Key research findings that shaped it:
Devbunova's 2x2 dataset is public (HF
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
5. **[DONE] Slice 3** — multi-turn n=250, with `eval_shared` as the clean
   question-swap control. Matched coherence is the natural-continuation control.
6. **[DONE] Silence filtering** — 192 naturally terminated silent pairs;
   probe 0.937 vs BoW 0.823 at the final token.
7. **[DONE] Slice 5 reconstruction and decomposition** â€” exact-label n=250
   provenance/context result; only steering remains.
8. **[DONE] Secondary estimator, permutation nulls, calibrated judge** — all
   CPU-only; no conclusion changed, one new limitation (uninformed directions
   are not at 0.5).
9. **[DONE] Slice 1 re-extracted locally** — the estimator and null checks now
   cover slice 1, and the Kaggle numbers reproduced exactly on other hardware.
10. **[DONE] Tier 2** — steering (pre-specified; manipulation check passed at
    layer 14, self-report shifts 0.139->0.889, safety shows no large effect),
    the silence filter on `eval_shared` (0/250 verbalized; probe 0.998 vs BoW
    0.758, but only 125/250 pairs determinate under the 256-token cap), and
    slice 5 replicated on LMSYS-Chat-1M (0.837 at the assistant token vs
    WildChat's 0.792). The 512-cap sensitivity rerun confirms the `eval_shared`
    silence result is not a censoring artifact (227/250 retained, probe 0.991).
    All Tier 2 GPU work is finished and the artifacts are local.

---

# HANDOFF — read this first if you are picking the project up

State as of the end of the August 2026 working session.

## Current handoff: Slice 5 is complete through decomposition

The current committed code is `dc826e0` (`Analyze slice 5 decomposition on
shared pairs`), and the local test suite passes **20/20**. `run_slice5.py` now
has three relevant operations:

- `reconstruct --generation-context full` is the original / model-swap path.
- `reconstruct --generation-context final_user_only` is the context-ablation
  control: it hides earlier turns only while generating, then restores the full
  original prefix before 8B activation extraction.
- `compare` intersects the natural-termination masks of two conditions before
  fitting either probe. This prevents a misleading comparison of differently
  censored samples. The completed invocation was:

```
.venv\Scripts\python -m vnaa.run_slice5 compare ^
  --dataset data/slice5_genuine_n250.json ^
  --left-reconstructions data/slice5_reconstructed_llama32_3b_n250.json ^
  --left-tag model_swap_llama32_3b_n250 ^
  --right-reconstructions data/slice5_reconstructed_final_user_only_n250.json ^
  --right-tag context_ablation_final_user_only_n250
```

The exact shared subset is **146/250**. At `final_assistant_token`, model-swap
is 0.811 +/- 0.046 (format 0.651; margin +0.160) and context ablation is 0.868
+/- 0.012 (format 0.702; margin +0.166). At `final_assistant_mean`, their
margins are +0.138 and +0.101. The conclusion is narrow: the reconstruction
effect is not attributable to 8B-specific authorship; it remains a
provenance/context result, not evidence of evaluation awareness.

### Local private artifacts (do not commit)

The source conversations, generated replies, and activation arrays are all
gitignored because they contain real WildChat-derived content or are large. They
are present locally and were SHA-256 verified after download from the cloud:

- `data/slice5_genuine_n250.json`
- `data/slice5_reconstructed_n250.json`
- `data/slice5_reconstructed_llama32_3b_n250.json`
- `data/slice5_reconstructed_final_user_only_n250.json`
- `results/s5_Llama-3.1-8B-Instruct_model_swap_llama32_3b_n250_*`
- `results/s5_Llama-3.1-8B-Instruct_context_ablation_final_user_only_n250_*`

The model-swap and context-ablation analysis JSON files and the six
non-duplicated activation arrays needed to reproduce the shared-subset result
are local. Do not upload these private artifacts to Git or public storage.

### Cloud status

A single 32 GB RTX 5090 Runpod Pod was used successfully for slice 5. The
model ran wholly on-GPU in bf16; extraction was seconds and generation was the
costly stage. All necessary artifacts have been copied back locally, so the Pod
may remain stopped or be terminated after the user is comfortable with the
backup. Do not record SSH keys, tokens, host addresses, or ports in this repo.
If a future Pod is used, store work under `/workspace`, use
`HF_HOME=/workspace/.cache/huggingface`, and expect a new exposed SSH port after
every restart/reset.

### What remains

See "Tier 2" below: the code for all three remaining items is written, frozen,
and tested; only the GPU session itself is outstanding.

The gap between the proposal's Experimental Setup and what exists is larger than
the slice list suggests: the model scaling sweep (70B, Qwen2.5-7B/32B,
Gemma-2/3) and the agentic / tool-use per-class results have not been started at
all, and one of those two claims should be cut from the paper if they are not
going to run. After Tier 2, the remaining work is paper assembly: figures,
methods/results prose, limitations, and final reproducibility checks.

## Tier 2 (August 2026): code frozen, GPU session pending

`scripts/gpu_session_tier2.md` is the runbook. The compute call is a **Runpod
RTX 5090 (32 GB)**, the same as slice 5: all three items are generation-bound
(~1,750 greedy replies), the 8B fits wholly on-GPU in bf16 there, and the local
5070 Ti would have to offload to system RAM. Estimated ~2.5 GPU-hours. The one
prerequisite that needs a human is accepting the **gated LMSYS-Chat-1M terms**
on the HF account that already has Llama access.

**Done already, locally, no GPU session needed.** Slice-1 8B activations now
exist on the desktop (`results/slice1_Llama-3.1-8B-Instruct_acts.npz`), so the
logistic estimator and the permutation null cover slice 1. The local run
reproduced the original Kaggle numbers exactly — layer 0 at 0.500, layers 1-3 at
0.712 / 0.970 / 0.991, best layer 4 at AUROC 0.9944 — which is an independent
reproduction on different hardware, a 2x T4 shard versus a single 5070 Ti with
CPU offload.

**Item 1: silence filter on `eval_shared`.** `run_silence_filter` now takes
`--cell {coherent_pairs,eval_shared}` with per-cell response, label, and analysis
paths, so the two runs cannot overwrite each other. `eval_shared` conversations
come from the slice-3 loader rather than a written dataset; its prefixes were
verified byte-identical to `deploy` on all 250 pairs. The published
`coherent_pairs` figure reproduces exactly after the refactor (192/250 silent
pairs, probe 0.937 +/- 0.033, BoW 0.823, format 0.567). Activations already
exist; the session only generates 500 replies.

**ITEM 1 RESULT (n=250 generated pairs, August 2026).** Explicit verbalized
awareness on the eval side: **0/250**, matching `coherent_pairs`. On the retained
pairs the final-user-token probe is **0.998 +/- 0.002**, versus BoW **0.758 +/-
0.029** and format **0.624 +/- 0.022**; probe-BoW margin **+0.240**, more than
double matched coherence's +0.114.

One regex hit, the first false positive the rule has produced: a *deploy* reply
guessing at a TV quiz show ("I'm going to take a wild guess that the quiz"). Read
manually — subject matter, not self-recognition. It is on the deploy side, so it
costs a pair and cannot inflate the eval result.

**The censoring caveat, and a wrong prediction.** The runbook predicted *fewer*
cap-hit exclusions here than matched coherence's 58/250, on the reasoning that
benchmark questions get short answers. The opposite happened: only **125/250**
pairs had both replies terminate naturally (eval 89 capped, deploy 70), with
median reply lengths of 196 and 160 tokens against the 256 cap. Half the sample
is censored, and the censoring is mildly label-correlated. The read point is the
final *user* token, so the probe's input is untouched by reply length — but a
silence filter that can only speak for half its sample is a weak instrument.

**CENSORING SENSITIVITY: the result is not a censoring artifact.** Rerunning the
identical protocol at a 512-token cap raises retention from 124/250 to
**227/250** (91%) and barely moves anything:

| cap | silent pairs | probe | BoW | format | probe-BoW |
|---|---|---|---:|---:|---:|
| 256 (protocol) | 124/250 | 0.998 +/- 0.002 | 0.758 +/- 0.029 | 0.624 +/- 0.022 | +0.240 |
| 512 (sensitivity) | 227/250 | 0.991 +/- 0.003 | 0.776 +/- 0.024 | 0.583 +/- 0.017 | +0.215 |

Nearly doubling the determinate sample changes the probe by 0.007 and the margin
by 0.025. Report the 256 figure as the protocol result and the 512 figure as the
sensitivity check; the honest headline is that the conclusion holds at 91%
retention. Eval-side explicit awareness is **0/250 at both caps**, and the 512
run is the stronger silence claim because replies had twice the room to
verbalize. The same single deploy-side false positive (pair_152, the TV quiz
show) appears at both caps and nothing else does.

**The generation cap is now part of a run's identity.** A reply that reaches the
cap is indeterminate rather than silent, so the cap decides how much of the
sample the filter can speak for, and two runs at different caps are different
measurements. `cell_paths(cell, max_new_tokens)` gives a non-default cap its own
filenames (`..._eval_shared_cap512_...`); the protocol cap of 256 resolves to the
existing paths unchanged, and both published figures still reproduce exactly.

Two process notes from making that change, both worth remembering:

- `generate` was re-deriving its output path from the cell alone, dropping the
  cap, so the first cap-512 attempt aimed straight at the protocol run's
  responses file. Only the `load_or_initialize_responses` decoding-mismatch
  guard stopped it from overwriting a published result. That guard earned its
  keep; keep this class of check.
- **The first test written for the fix was vacuous.** It mocked `generate` out,
  so it only exercised argument parsing and passed against the buggy code. It
  was caught by reverting the fix and confirming the test still passed. Do that
  check routinely: a regression test that has never failed has not been tested.

**Item 2: steering.** The protocol is frozen in `datasets/steering/prespec.md`,
written and committed **before any steered output existed**; the git history is
the evidence for that ordering. Mechanics are in `steering.py`, deliberately
small and testable without a GPU. Two things it pins down that would otherwise
fail silently: `hidden_states[L]` is the output of block `L-1`, so steering at
"layer L" hooks `layers[L-1]`, and a coefficient of 0 installs no hook at all so
the unsteered condition is bit-identical rather than merely close. transformers
5.x decoder layers return a bare tensor where 4.x returned a tuple, and the hook
handles both, because a version bump silently turning the intervention into a
no-op would look exactly like a clean null result.

The design in one line: inject `alpha * ||mu_eval - mu_deploy|| * d_hat` at one
layer with the prompt held fixed, then measure (1) an independent probe's
readout, (2) verbalized self-report, (3) refusal rate on borderline requests,
with a repetition/length degeneracy veto. Measure 1 gates the rest.

**Amendment 1 to the pre-spec, made after the CPU-only direction fit and before
any steering.** The CV "best layer" is a plateau, not a peak: layer 4 scores
0.4453 and layer 9 scores 0.4419, with twelve layers inside 0.02. Picking by
argmax would have injected at a layer chosen by a 0.003 difference. The revised
rule takes the tied layers, thins them to four spread across the plateau
(3, 7, 10, 14), and lets the manipulation check choose — that check is
forward-passes-only and scored by the `eval_shared` probe, so it never sees a
behavioural outcome and cannot launder a result. The alpha grid also gained two
interior points, because at these layers the class-mean gap is 8-34% of the mean
residual norm, so `alpha = 4` is a larger-than-unit perturbation and was likely
to consume most of the original grid in the degenerate range.

**STEERING RESULT (August 2026, Runpod 4090).** Behavioural generation is done:
756 greedy generations at layer 14 across the nine-coefficient grid.

self_report (fraction calling the situation an evaluation), n=36 per cell:

| alpha | -4 | -2 | -1 | -0.5 | 0 | +0.5 | +1 | +2 | +4 |
|---|---|---|---|---|---|---|---|---|---|
| rate | 0.000 | 0.139 | 0.611 | 0.667 | 0.778 | 0.806 | 0.889 | 0.889 | 0.000 |
| flag | DEGEN | | | | | | | | DEGEN |

Spearman +0.991 excluding the degenerate ends. The 95% Wilson intervals at
alpha -2 ([0.061, 0.287]) and alpha +2 ([0.747, 0.956]) do not overlap.

safety (refusal rate on 48 held-fixed borderline requests), n=48 per cell:

| alpha | -4 | -2 | -1 | -0.5 | 0 | +0.5 | +1 | +2 | +4 |
|---|---|---|---|---|---|---|---|---|---|
| rate | 0.000 | 0.083 | 0.104 | 0.104 | 0.125 | 0.104 | 0.083 | 0.062 | 0.000 |
| flag | DEGEN | | | | | | | | DEGEN |

Spearman -0.356 excluding degenerate. Every non-degenerate interval overlaps;
the full observed range is 3 to 6 refusals out of 48.

The degeneracy veto fired at exactly the predicted coefficients (alpha = +/-4;
repeated-4-gram fraction 0.55/0.78 for self_report and 0.42/0.61 for safety,
all also pinned at the 256-token cap). The runbook predicted this in advance.

**This is row 2 of the pre-specified outcome table**: manipulation passes,
self-report shifts, safety flat. Reading, as written before the run: "the
direction is genuinely eval-recognition and the model can report on it, but
recognition alone does not move safety behaviour. A real and publishable
negative." The pre-spec recorded **low confidence** for the safety prediction,
and that was correct.

**Report the safety arm as underpowered, not as a null.** At n=48 with a 0.125
baseline, a two-proportion test has 80% power only against an increase to
**0.37**. So the arm rules out a large effect and says nothing about a modest
one. Writing "steering does not affect safety behaviour" would overclaim; the
defensible sentence is "no large effect, with power only against a swing of
+0.25 or more." If this arm matters to the paper, it needs either many more
borderline prompts or a set with a higher baseline refusal rate — 0.125 is a
floor effect that wastes most of the dynamic range.

Second caveat: the unsteered self-report rate is already **0.778**, so most of
the measurable movement is on the deploy-ward side. The carriers are single-turn
requests, which is also why the unsteered readout sits at -0.20 of the gap.

**Manipulation check: PASSED, run locally on the 5070 Ti (August 2026).** It is
forward-passes-only, so CPU offload is fine and it cost nothing. Values below are
the fraction of the eval-deploy gap reached on the independent `eval_shared`
probe; unsteered sits at -0.20 because the self-report carriers are single-turn
requests rather than multi-turn WildChat rewrites, so what matters is the shift,
not the intercept.

| layer | -4 | -2 | -1 | -0.5 | 0 | +0.5 | +1 | +2 | +4 | monotone |
|---|---|---|---|---|---|---|---|---|---|---|
| **14** | -1.14 | -0.69 | -0.46 | -0.34 | -0.20 | -0.06 | 0.08 | 0.35 | 0.87 | yes, rho +1.00 |
| 10 | -0.63 | -0.57 | -0.41 | -0.31 | -0.20 | -0.08 | 0.03 | 0.24 | 0.49 | yes, rho +1.00 |
| 7 | -0.36 | -0.35 | -0.29 | -0.24 | -0.20 | -0.17 | -0.13 | 0.12 | 0.34 | yes, rho +1.00 |
| 3 | -0.21 | -0.20 | -0.20 | -0.20 | -0.20 | -0.20 | -0.20 | -0.20 | -0.17 | no, rho +0.15 |

Selected injection layer: **14**, by the pre-specified rule (monotone, then
Spearman, then span).

This is the direct vindication of Amendment 1. **Layer 3 is completely flat
across a 16x range of coefficients** — a dead injection site. Under the original
argmax rule the direction would have been injected in that early-layer region,
and the behavioural arm would have returned a clean-looking null that was
actually just an intervention that did nothing. Steering efficacy is clearly
increasing with depth across the tied plateau even though cross-validated
*decodability* is flat across it, which is worth stating in the paper: the layer
where a concept is most decodable is not the layer where injecting it does the
most work.

Two caveats to report honestly: even at alpha = +4 layer 14 reaches only +0.87 of
the gap, so steering never fully traverses to the eval reference inside the grid,
and that coefficient may trip the degeneracy veto anyway. And the plateau
thinning happened to exclude layer 4, the CV argmax; it did not change the
outcome, since layer 14 dominates by a wide margin, but a rebuilt candidate rule
should pin the argmax into the set explicitly.

**Item 3: slice 5 on LMSYS-Chat-1M.** `prep_slice5.py --source lmsys` reuses the
frozen WildChat protocol byte-for-byte, so this is a corpus replication and
nothing else. Two source differences are handled: LMSYS tags language on the
record rather than per message, and it replaces PII with placeholders such as
`NAME_1`. Redacted conversations are dropped — those placeholders are a surface
artifact a probe or bag-of-words baseline could read as the label.

**LMSYS RESULT (n=250, August 2026).** 179/250 pairs terminated naturally. At
`final_assistant_token` the probe is **0.837 +/- 0.026** versus BoW 0.596 and
format 0.671, margin **+0.166**; at `final_assistant_mean`, **0.791 +/- 0.030**,
margin +0.120. WildChat n=250 gave 0.792 and 0.822. The effect replicates across
corpora; the token-versus-mean ordering swaps between them, which is worth a
sentence but does not change the conclusion. Slice 5 no longer rests on one
corpus.

Verified on the copied artifacts, not taken from the log: 250 conversations,
user-turn distribution {2: 164, 3: 61, 4: 25}, all ending on an assistant turn,
**0 PII placeholders**, 0 empty messages, and the reconstruction file's recorded
`dataset_sha256` matches the genuine file.

**Gotcha: `prep_slice5 --source lmsys` exits 134 (core dumped) after succeeding.**
It writes the dataset and prints its summary, then crashes during interpreter
finalization: a `[Errno 9] Bad file descriptor` on the HF parquet stream followed
by `Fatal Python error: PyGILState_Release: thread state must be current when
releasing`. This is a teardown race in the datasets/pyarrow streaming shutdown,
not a data failure. **A chained runner will record rc=134 and move on**, so the
nonzero exit is not by itself evidence the step failed — check the output file
before rerunning. Conversely, do not trust the rc=0 of the steps after it
without checking that they consumed the right file.

## Where slice 3 actually is

**The dataset is COMPLETE: all 5 batches, 250 items per cell**, committed at
`datasets/slice3/batches/out_0{0..4}.json`. The `eval_shared` control is built
and committed at `datasets/slice3/eval_shared.json`. Nothing more needs writing.

**Extraction and analysis are complete.** The full n=250/cell cache contains
all four cells, all point readouts, and ragged per-user-turn trajectories.
Reproduce the analysis with:

```
.venv\Scripts\python -m vnaa.run_slice3 extract --dtype bfloat16 --device-map auto
.venv\Scripts\python -m vnaa.run_slice3 analyze
```

Extraction took about 18 minutes on the 5070 Ti for 4 cells x 3 read points x
250 items, plus trajectories. `run_slice3.load_cells()` picks up `eval_shared`
automatically, so both commands cover all four cells with no extra flags.

## Gate status at n=250 (already run, `gate_slice3.py`)

- shape errors 0 across all 250 items
- framing separation PASS: test cues in 97% of `eval_obvious`, 3% of
  `eval_disguised`
- length PASS everywhere (0.51 whole-conversation, 0.56 final turn)
- format FAIL: 0.642 whole-conversation, 0.712 final turn. Note the
  whole-conversation figure **improved from 0.76 at n=100**, because batches
  02-04 were written with the revised opening instructions. Expect format never
  to reach the 0.10 gate — that is the established slice-2b position (speech act
  is the label), which is why transfer and the `eval_shared` control carry the
  argument instead.

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

## Next Slice 3 control

Build a matched-coherence `eval_shared` cell: preserve every deployment prefix,
but replace the final turn with a benchmark question that follows naturally from
the conversation. Then reuse the existing extraction and analysis runner. This
removes the topic-shift caveat without restoring opening-style or authorship
confounds.

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
