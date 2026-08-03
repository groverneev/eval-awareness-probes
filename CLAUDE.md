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

1. **Devbunova replicated on the transfer axis.** A probe trained on the
   standard confounded contrast (bench_eval vs casual_deploy) fails to
   transfer: 0.53 to their format-matched cell, 0.67 to our casual cells. It
   is a format detector. It is also the ONLY training set where BoW beats the
   probe (probe-minus-BoW -0.091, 1/5 wins).
2. **New result: training on format-matched casual data generalizes, and the
   asymmetry is large.** Probe trained on `ours_a_wc` scores 0.952 on
   Devbunova's own standard cell, 0.990 / 0.972 / 0.953 on our other cells.
   The reverse direction does not hold (theirs -> ours 0.67). So the direction
   learned from casual, length-matched data covers the benchmark case, while
   the direction learned from benchmark data does not cover the casual case.
3. **The probe beats bag-of-words under transfer, modestly but consistently:**
   overall +0.031 mean strength, winning 21/30 off-diagonal transfers. Best
   when trained on our cells (+0.079 from `ours_a_wc`, 4/5 wins; +0.064 from
   `ours_a_tm`, 5/5). The clearest margins are the hard targets: transferring
   to `ours_b_wc` gives probe 0.953 [0.92, 0.98] vs BoW 0.814 [0.74, 0.88],
   and to `ours_b_tm` 0.972 [0.94, 0.99] vs BoW 0.797 [0.72, 0.87] — CIs
   disjoint in both.
4. **The format classifier collapses under transfer** (mean strength 0.132 vs
   probe 0.285), so the surviving signal is not structural.
5. **Hardest cell for everything: `devb_fmt_matched`** (bench_eval vs
   bench_deploy_rewritten — benchmark format on both sides, only content
   differs). Every training set lands at 0.43-0.68 there. This is the cell
   where content-only eval recognition would have to show up, and it does not
   yet.

Caveats to carry: n=250/cell means 75/cell held out, so transfer CIs are
+/-0.05; seed 0 only so far (>=3 seeds owed). Writers reframed some real
WildChat prompts that were themselves single-answer factual questions into
help-requests, which inflates the speech-act difference we diagnosed — needs
quantifying against the unmodified originals.

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
