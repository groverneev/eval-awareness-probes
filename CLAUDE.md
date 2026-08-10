# CLAUDE.md — agent onboarding

Readng `VNAA-Proposal.pdf` (repo root) is fairly useful. It has the
full methodology and motivation; this file only covers what changed since, what
to run, and what will bite you. However, keep in mind that the proposal is not
the final version of what we have actually ran. Only use it to get a general idea
of this paper, but not to assume what has been done. Reading 'EXPERIMENT_REPORT.md`
is highly recomended to see what we have done.

Last updated: August 2026.

## Status: all experiments done, paper assembly remains

Slices 1, 2, 3, and 5 are complete on Llama-3.1-8B (slice 5 includes the
model-swap/context-ablation decomposition and an LMSYS-Chat-1M replication).
Tier 2 is done: steering, the `eval_shared` silence filter (+ 512-token cap
sensitivity check), and the LMSYS replication, all run on a rented Runpod 4090.
Local test suite: 90/90 passing after the matched-prefix baseline correction.

**What's left is paper assembly** (figures, methods/results prose,
limitations) plus one open scope decision below. No GPU pod is currently
running.

**Open scope decision.** The proposal promises a model scaling sweep (70B,
Qwen2.5-7B/32B, Gemma-2/3) and per-class agentic/tool-use results. Neither is
started and both cannot fit before the deadline. Recommendation: cut
agentic/tool-use to future work (needs a dataset that doesn't exist); if any
scaling sweep runs, do Qwen2.5-7B on `eval_shared` only (~1 pod-hour) as a
check of whether the effect is Llama-specific.

**Single biggest risk right now (not scientific):** `data/` and `results/`
are gitignored and exist only on this desktop now that the rented pod is
torn down. Back them up off-machine.

## The idea

Do models internally recognize when they're being evaluated, and does that
survive when the evaluation is disguised as a realistic conversation? We train
linear probes on residual-stream activations to separate eval from deployment
contexts, then test whether the signal survives once surface cues (format,
length, single-turn structure) are removed. The gap between the probe and a
hierarchy of surface baselines (length, format, bag-of-words) is the result —
not the raw AUROC, which can be near-perfect for boring reasons (see Slice 1).

Full proposal, authors, and citations: `VNAA-Proposal.pdf`.

## Project working conventions

- One slice at a time; finish and verify before moving on.
- Ground claims in a concrete run before generalizing.
- Correct assumptions directly when they are wrong.
- Prose: brief, one job per sentence, minimal em dashes.
- **Double-blind review:** Do not add contributor names, usernames, personal
  repository handles, or other identifying information anywhere in repository
  files.
- **No `Co-Authored-By: Claude` trailer or self-attribution in commits.**
- Push commits to GitHub as part of finishing a unit of work.

## Environment

Windows desktop is primary

**Windows desktop:** RTX 5070 Ti (16 GB VRAM, Blackwell sm_120), 64 GB RAM.
System Python is 3.14 (too new for PyTorch) — always use `.venv\Scripts\python`
(3.12, via `uv sync`; pyproject pins the cu128 PyTorch index on win32). The 8B
in bf16 (~16 GB) doesn't quite fit in 16.3 GB VRAM with overhead — use
`device_map="auto"` so accelerate offloads the last few layers to system RAM;
extraction at batch 1 is fine this way. Never 4-bit quantize (perturbs the
measured activations). The 1B fits entirely in VRAM.

**Mac (secondary):** M3, 16 GB unified memory, MPS not CUDA, same 3.14/3.12
split. 8B doesn't fit

Both: transformers 5.x — `output_hidden_states` still works, use `dtype=` not
`torch_dtype=`, decoder layers return a bare tensor not a tuple (matters for
steering hooks).

**Kaggle (2x T4, free, 30 GPU-hr/week)** is the fallback for fully-on-GPU runs
if local offload is too slow. See `scripts/kaggle_slice1.md`.

## Model access (Hugging Face)

Llama is gated. Use an approved account for Llama-3.2-1B and Llama-3.1-8B;
the desktop is logged in via `.venv\Scripts\hf.exe auth login`. A
fine-grained token without "read access to public gated repos" gives 403 even
after approval — the metadata endpoint returning 200 is not proof of file
access.

## Compute

- No project API access is available.
- Free-tier GPUs (Kaggle, Lightning) cover everything that's just forward
  passes. Paid GPU is for the generation-heavy Tier 2 work (silence filter,
  steering) and would be for a scaling sweep, not routine inference.

## Data

Slice 1 uses SAD `stages_oversight` (`test_oversight` = eval, label 1;
`deploy_oversight` = WildChat, label 0). We feed the raw `body` as a
single-turn message and use the stage as ground truth — no MCQ wrapper.

**SAD canary discipline (required by SAD's license): SAD question text must
never appear in plain text anywhere scrapable, including this repo.**
`data.py` reads prompts from the encrypted `structs.zip` in memory (password
`sadtimesforthesetimes`) and never writes prompt text to disk. Cached
*activations* are fine; prompt text is never committed or pasted into a
committed file. Zip lives at `data/sad/oversight_structs.zip` (gitignored);
re-download with the curl command in the README if missing.

## Module map

```
src/vnaa/
  probe.py               per-layer difference-of-means probe + AUROC
  extract.py              activation extraction; READ_POINTS, turn_spans,
                          read_slice, extract_trajectories
  baselines.py            surface baselines + bootstrap_auroc_ci
  data.py                 SAD loader (slice 1)
  data_2x2.py             Devbunova cell loader (slice 2a)
  run_slice1.py            slice-1 runner
  run_2x2_baselines.py     baselines on Devbunova's cells
  run_slice2a.py           extract/analyze on the public 2x2
  build_2b.py              slice-2b sampling + early gate
  prep_2b_batches.py       slice-2b batch inputs
  hydrate_batches.py       rehydrate slice-2b batches with real prompt text
  merge_2b.py              merge slice-2b writer output + probe-blind gate
  run_transfer.py          6x6 cross-cell transfer matrix, multi-seed
  run_per_source.py        arc/mmlu/gsm8k breakdown
  check_rewrite_drift.py   deploy rewrite vs real originals
  prep_slice3.py           multi-turn sampling
  prep_slice3_batches.py   slice-3 batch inputs (+ --scratch hydration)
  gate_slice3.py           slice-3 probe-blind gate
  run_slice3.py            slice-3 extract/analyze across read points
  estimators.py            logistic probe, permutation/geometry nulls
  run_estimator_check.py   re-scores headline contrasts, CPU only
  judge_batches.py         blinded judge batches + calibrated merge
  run_silence_filter.py    per-cell generate/label/analyze (--cell)
  steering.py              activation-addition hook; layer indexing
  run_steering.py          steering direction, manipulation check, behaviour
  run_slice5.py            reconstruction / model-swap / context-ablation / lmsys
  prep_slice5.py           genuine-conversation sampling (WildChat or LMSYS)
scripts/
  make_figures.py, make_2b_figures.py     slice figures
  gpu_session_tier2.md                    the frozen Tier 2 runbook
datasets/judge/awareness_judge_v1.md      committed, hashed judge rubric
datasets/steering/prespec.md              steering protocol, frozen pre-run
tests/  test_probe.py, test_read_points.py, test_estimators.py,
        test_judge_batches.py, test_steering.py, test_prep_slice5.py
```

## Key technical decisions (extract.py / probe.py)

- Probe = per-layer difference-of-means. Direction = normalized mean(eval) −
  mean(deploy), fit on train only. Report `|AUROC − 0.5|` (sign-invariant; raw
  AUROC can drop below 0.5 under distribution shift).
- `hidden_states` has n_layers + 1 entries; index 0 is the embedding layer
  (always AUROC 0.5 at a token-identical read point — a sanity check, not
  evidence against surface confounds).
- Chat template already adds exactly one BOS — feed its ids straight in, never
  re-tokenize. Batch size 1 so the read index is always the true last token,
  no padding.
- `last_user_token` templates with `add_generation_prompt=False` and reads the
  final token (`<|eot_id|>`) closing the user turn.
- A shuffled-label null must be averaged over many shuffles — one draw leaks
  residual signal against strong separation.

## Results (headline numbers — see git history / results/ for full detail)

All figures below are mean AUROC over 5 stratified 70/30 splits unless noted.
"Strength" = `|AUROC − 0.5|`.

**Slice 1 (obvious eval, 8B, n=200/class).** Probe 0.994 at layer 4/32. But a
structure-only classifier (no words) reaches 0.9675 and bag-of-words 0.9517 on
the same split — probe margin over the best surface baseline is only +0.027,
within noise at n=120. **This cell is fully confounded**: SAD-eval prompts are
shaped like benchmark items, WildChat-deploy prompts aren't. It cannot
distinguish genuine recognition from surface-format detection — that's the
mandate for slice 2.

**Slice 2 (format-matched, cross-cell transfer, ARC+MMLU only — GSM8K
excluded, digit density alone scores 0.949 there).** Probe beats bag-of-words
under transfer by **+0.071 ± 0.013** (positive every seed) on every
format-matched training cell. The sole exception is training on the standard
confounded contrast: **−0.087 ± 0.032** — on the setup the field actually
uses, activations transfer *worse* than word counts. Direction is asymmetric:
trained on our realistic cells → 0.949 on the standard benchmark cell;
reverse → 0.578, barely above chance. Hardest cell for everyone:
`bench_eval` vs `bench_deploy_rewritten` (benchmark format both sides, content
only differs) — every direction lands at or below chance there (0.31–0.55).
That's the untouched limit; slice 3 targets it with multi-turn structure
instead of single-turn rewrites.

**Slice 3 (multi-turn, n=250/cell).** `eval_shared` (deploy conversation with
only the final user turn swapped for a benchmark question, prefix
byte-identical by construction) is the clean control: exactly chance
pre-swap (probe 0.497–0.501 across the whole trajectory), then **0.992** at
the final turn, versus final-turn-only bag-of-words 0.935 and format 0.728
(**+0.057** over BoW). The as-written `eval_disguised` cell is invalid as a realism test —
it's separable (probe 0.951) before the benchmark question even appears, from
synthetic lead-in style alone.

**Matched-coherence control (n=250)** fixes `eval_shared`'s remaining
topic-shift caveat: shared everyday-topic prefix, then either a benchmark
question or a genuine same-topic help request. Pre-final read is chance
(0.500); final token probe **0.951 ± 0.011** vs final-turn-only BoW 0.933
(+0.018 margin). Cross-control transfer remains positive (`eval_shared` ↔
matched-coherence: probe 0.858 / 0.788 vs BoW 0.792 / 0.678).

**Silence filtering (192/250 naturally-terminated matched-coherence pairs;
227/250 for `eval_shared` at a 512-token cap).** Explicit verbalized
awareness: 0/250 on every cell tested, confirmed by a blinded, planted-control
model judge (14/14 planted positives caught, 0/10 false alarms). On the
retained pairs: matched-coherence probe 0.937 ± 0.033 vs BoW 0.935 (tied);
`eval_shared` probe 0.998 ± 0.002 vs BoW 0.950 (+0.047 margin), robust to the
cap-512 sensitivity check (probe 0.991 vs BoW 0.946, +0.045).

**Slice 5 (deployment-simulation reconstruction, n=250, exact provenance
labels — real conversation, only the final assistant reply is regenerated).**
Probe 0.792 ± 0.024 (token) / 0.822 ± 0.018 (mean) vs reply-only format
0.774 (+0.019/+0.048 margin). Model-swap (Llama-3.2-3B replaces the 8B) vs
context-ablation (8B sees only the final user turn) decomposition on the
146-pair shared-termination subset: both token effects are comparable in size
(0.811 / 0.868 vs format 0.786/0.832), while mean readouts tie or lose to
format. The signal isn't attributable to 8B-specific authorship and much of it
is reply-surface information. It replicates on LMSYS-Chat-1M (0.837 token /
0.791 mean vs format 0.736) — not corpus-specific.

**Steering (pre-specified in `datasets/steering/prespec.md`, layer 14 chosen
by a monotonicity+Spearman rule over a plateau, not CV argmax — layer 3 was
completely flat, a dead injection site the argmax rule would have picked).**
Manipulation check passed (Spearman +1.00). Self-report shifts strongly:
0.139 → 0.889 across the coefficient grid (Spearman +0.991). Safety refusal
rate stays flat, 0.062–0.125 — but this arm is underpowered (80% power only
against a swing to ~0.37 at n=48), so report "no large effect", not "no
effect".

**Estimator/null robustness (`run_estimator_check.py`, CPU-only, off cached
activations).** A second estimator (L2 logistic regression, CV-selected layer)
reproduces every result and is equal or stronger everywhere except slice 1
(already trivially linear). A full-pipeline permutation null (label vector
shuffled, layer selection rerun) puts every real contrast at p = 0.002 (floor
at 500 permutations); the two identical-prefix controls correctly land at
p = 0.94–0.99. **Caveat for the paper:** an uninformed direction (fit on
shuffled *training* labels only, scored on real test labels — not a
significance test) still reaches strength 0.24–0.29 on the strongest cells,
because the eval/deploy axis carries a large share of total activation
variance. This doesn't affect significance but means raw `|AUROC-0.5|`
overstates how much work the probe does; state it as a limitation.

## Reproducing the analyses

```bash
.venv\Scripts\python -m vnaa.run_slice3 extract --dtype bfloat16 --device-map auto
.venv\Scripts\python -m vnaa.run_slice3 analyze
.venv\Scripts\python -m vnaa.run_estimator_check points
.venv\Scripts\python -m vnaa.run_estimator_check transfer
.venv\Scripts\python -m pytest
```

`data/` and `results/` are gitignored (large, and some derive from
gated/licensed corpora). Regenerating slice-3 inputs needs WildChat access —
see `prep_slice3.py` / `prep_slice3_batches.py` docstrings; hydrated outputs
containing real conversation text go to a scratch dir, never committed.

## Things that will bite you

- **Never trust a subagent's self-report.** Writers have repeatedly reported
  accurate per-item stats while missing cross-cell confounds only visible by
  comparing cells side by side (e.g. digit density, question-mark rate). Audit
  the output, not the report.
- **An exit code is not a result.** `prep_slice5 --source lmsys` exits 134
  (interpreter-teardown crash in the datasets/pyarrow shutdown path) *after*
  successfully writing its output. A chained runner records rc=134 and moves
  on regardless. Check the artifact (item counts, schema, recorded input
  hash), not the exit code, in either direction.
- **A regression test that has never failed has not been tested.** Revert the
  fix, confirm the test goes red, then restore. One case already shipped a
  test that mocked out the very function containing the bug.
- **Match surface features per item within each source/cell, never on the
  pooled aggregate.** Pool-level digit-density matching in slice 2b let GSM8K
  items keep 13x the digit density of their "matched" deploy counterparts —
  exclude GSM8K from any headline transfer number as a result.
- **Baseline text views must match what each causal read position can see.**
  A first pass gave a final-turn-only baseline to a read point that had seen
  the whole conversation, inflating a margin from +0.08 to +0.27.
- **Freeze batch inputs before spawning writer subagents.** Regenerating them
  mid-run invalidated a writer's output once.
- venv is `.venv\Scripts\python` (3.12); system Python (3.14) will not work.
  `git_filter_repo` is only in system Python if ever needed.
- `pgrep -f <pattern>` on a remote box can match the very ssh command
  containing the pattern — wait on a marker line in a log file instead.
- Verify before deleting rented GPU infrastructure: SHA-256 every artifact
  against the local copy first, and remember network volumes bill separately
  from pods and survive pod deletion.
- uv on this desktop can't install Pythons to its default dir — if you ever
  need to reinstall, set `UV_PYTHON_INSTALL_DIR=C:/Users/grove/uv-pythons`
  first (the repo `.venv` already points there, so routine `uv sync` doesn't
  need this).

## Maintaining this file

Update `EXPERIMENT_REPORT.md` as part of any completed work that changes the
experiment record, reported results, or conclusions.

Update as part of finishing each unit of work: new results, new decisions,
changed status, new gotchas. Keep it to what a fresh agent actually needs —
remove any outdated/useless info. Never put SAD prompt text in this file (or
anywhere else committed) — canary rule above.
