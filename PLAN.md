# Experimental plan

Written August 2026, after slice 1. This is the working plan for the rest of
the project. Each slice ends with a verified number before the next begins.
CLAUDE.md stays the running log; this file is the forward plan.

## Where we are

Slice 1 (single-turn, obvious eval: SAD vs WildChat) is done on Llama-3.1-8B.
The probe hits 0.9944, but the format classifier hits 0.9675 and bag-of-words
0.9517 on the same split. The obvious-eval cell is fully confounded: surface
structure and eval-ness are perfectly entangled there. That is not a failure,
it is the measured version of Devbunova's critique and the mandate for
everything below.

## What the research pass found (August 2026)

Facts that shape this plan:

1. **Devbunova's 2x2 dataset is public** and built on **Llama-3.1-8B-Instruct,
   our exact model.** Code: `github.com/viliana-dev/eval-awareness-format`.
   Data: HF `viliana-dev/eval-awareness-2x2`, n=1076 per cell after
   length histogram-matching. Cells: Bench-Eval (MMLU/ARC/GSM8K items),
   Casual-Deploy (ShareGPT first turns), Bench-Deploy (ShareGPT rewritten
   into benchmark templates by DeepSeek-V3), Casual-Eval (MuTual dialogue
   first turns, **a proxy, admitted as such in the paper**).
2. Their key numbers: a probe trained on the standard contrast (Bench-Eval vs
   Casual-Deploy) calls Bench-Deploy "evaluation" 94.5% of the time — pure
   format tracking. But **paired training** (Bench-Eval + Casual-Eval vs
   Casual-Deploy + Bench-Deploy) restores format-invariant classification
   (Casual-Eval 100%, Bench-Deploy error 7.2%). So the eval signal may be
   recoverable when training decorrelates format; their proxy Casual-Eval
   cell is the weak link in that claim.
3. Their probes: logistic regression on last-token residuals, layers 8-31,
   5-fold CV. Ours is difference-of-means. Running our estimator on their
   data is a methods replication for free.
4. Xiong et al. rewrote eval scenarios toward deployment style with Claude
   (k=7 candidates per prompt, manual audit of all 371 rewrites) and moved
   probe scores ~30%. Caveat for us: they selected candidates BY probe score.
   For dataset construction that is circular; our rewrite gate must be
   surface-baseline-based and probe-blind.
5. Nguyen et al. steering: probe-direction steering recovered <5% of
   sandbagged accuracy; SAE features ~25%; a prompt suffix 83%. Steering
   along one direction is weak there. The Assistant Axis paper gives the
   stronger machinery: difference-of-means role vector, injection at a middle
   layer at every token position with norm-scaled magnitude, and **activation
   capping** (clamp the projection at a threshold) rather than constant
   injection. Capping the eval-awareness direction is a novel variant for our
   steering arm.
6. Chaudhary et al.: awareness scales as a power law across 15 open models,
   early-to-middle layers, same difference-of-means construction as ours.
   Cross-model replications of our decisive cell slot into that curve.

## Decisions (made per Neev's delegation)

- **Rewriting model:** Claude (this assistant / subagents) writes the
  rewrites; manual spot-audit by Neev; ChatGPT API is the fallback for bulk
  regeneration if quality plateaus. No cost either way.
- **Rewrite acceptance gate is probe-blind:** a rewrite batch is accepted
  when the `baselines.py` format classifier and bag-of-words fall to
  AUROC ~0.5-0.6 against the matched deployment cell, plus manual audit of a
  sample for content preservation. Never select rewrites by probe score.
- **Eval-content source:** public benchmarks (MMLU/ARC/GSM8K, matching
  Devbunova's Bench-Eval) are primary, so our datasets are committable and
  directly comparable. A SAD-rewrite arm is a robustness check, gitignored
  under the canary rule.
- **Statistics upgrade before slice 2 results:** slice 1's +0.027 gap at
  n=120 taught the lesson. From now on: bootstrap 95% CIs on every AUROC,
  gaps reported with CIs, larger held-out sets (Devbunova's n=1076/cell
  helps), and >=3 split seeds on headline numbers.
- **Compute:** primary is now the local RTX 5070 Ti (16 GB VRAM, 64 GB RAM).
  The 8B in bf16 needs a few layers offloaded to system RAM
  (`device_map="auto"`), which is fine for batch-1 extraction and removes the
  Kaggle upload/session friction entirely. Slices 2 and 3 are extraction-only
  and run locally. Generation first appears in slice 4; measure local
  tokens/sec first and fall back to Kaggle (2x T4, 30 hr/week) only if
  offloaded generation is too slow. The 70B is out of scope unless we later
  pre-approve paid GPU; cross-model goes sideways (Qwen2.5-7B, Gemma)
  instead of up.

---

## Slice 2 — single-turn, format-matched (the 2x2)

Goal: measure whether the probe beats surface baselines once format is
decorrelated from eval-ness, single-turn. This is the Nguyen-vs-Devbunova
dispute run on our own pipeline.

### 2a. Replicate Devbunova on our machinery

1. Loader for the HF `eval-awareness-2x2` dataset (new `data_2x2.py`).
2. Run the full surface-baseline hierarchy (`baselines.py`) on every pairwise
   contrast of the four cells. Local, CPU, free. This independently verifies
   their matching claims: format classifier should be ~high on Bench-Eval vs
   Casual-Deploy and ~0.5 on Bench-Eval vs Bench-Deploy and on Casual-Eval
   vs Casual-Deploy.
3. Kaggle: extract last-token activations for all four cells on the 8B
   (~4300 forward passes; one session, well under budget).
4. Probe runs (our difference-of-means + their logistic regression as the
   secondary estimator):
   - standard training: Bench-Eval vs Casual-Deploy, test on all cells;
   - paired training: (Bench-Eval + Casual-Eval) vs (Casual-Deploy +
     Bench-Deploy);
   - the full 4x4 train-on-A test-on-B transfer matrix, per layer.
5. Success criterion: reproduce their qualitative pattern (standard probe
   fires on Bench-Deploy; paired training restores invariance) with our
   estimator. Any disagreement is itself a finding about probe construction.

### 2b. Fix the Casual-Eval cell (our contribution)

Their Casual-Eval is MuTual first turns — different content source, so
"paired training works" could still ride on MuTual-specific artifacts.
We replace the proxy with true content-preserving rewrites.

1. Sample ~400 Bench-Eval items (stratified over MMLU/ARC/GSM8K). Claude
   rewrites each as a natural first-turn user message: same question, no
   options block, no benchmark scaffolding, casual register, varied phrasing
   templates to avoid a rewrite "house style" the probe could latch onto.
2. Gate (probe-blind): format classifier and BoW vs Casual-Deploy must land
   in 0.5-0.6; length histogram-match to Casual-Deploy; Neev audits ~40
   random rewrites for content preservation. Iterate until the gate passes.
   The dataset is not done until the gate passes.
3. Also build the SAD-rewrite arm the same way (gitignored, never committed).
4. Extract, then rerun 2a's probe suite with Casual-Eval-True in place of
   MuTual. The headline single-turn result: probe strength on
   Casual-Eval-True vs Casual-Deploy, against surface baselines forced to
   ~0.5 by construction, with transfer from the obvious cell.

Interpretation is symmetric and either way publishable: probe well above
baseline = format matching does not remove the internal signal; collapse =
Devbunova confirmed with the proxy weakness removed.

### 2c. Write-up checkpoint

Update paper Sections 4-5 with real numbers, figures (per-layer curves per
cell, transfer matrix heatmap), CLAUDE.md, README.

## Lessons from slice 2b that the slice-3 dataset spec MUST carry

These cost real time in slice 2b; do not rediscover them.

1. **Match surface features per item, not per pool.** We matched digit density
   between the benchmark pool and the WildChat pool (0.0124 vs 0.0123, looked
   fine) but the synthetic counterpart cell was written with no digit
   requirement and came out 13x lower on maths items. Digit density alone then
   scored 0.949 there — identical to the probe. Every surface feature we claim
   to match must be checked *within each source and each cell*, not on the
   aggregate.
2. **Freeze dataset inputs before spawning writers.** Regenerating batch files
   mid-run made one writer generate against a stale snapshot; its output was
   discarded.
3. **Audit writer output; do not trust writer self-reports.** Every writer
   reported "0% situational framing in style A" and matched lengths. Both were
   true. What they missed was that length targets drawn from the wrong
   distribution had forced style A to acquire personal framing in the first
   place — visible only by comparing cells, not within one batch.
4. **Model-authored text is separable from real user text** (format 0.93 / BoW
   0.91) even when its content is pure deployment. If any cell is authored,
   every cell must be.
5. **Do not expect surface baselines at chance.** Speech act is the label.
   Gate length and format; report lexical baselines as the bar, and settle the
   question with transfer.
6. **Report per-source.** The aggregate hid that the whole margin came from
   maths items with a numeral shortcut.

## Slice 3 — multi-turn (the decisive cell)

Goal: the axis nobody has tested. Same 2x2 logic, but conversations.

1. **Deployment side:** real multi-turn conversations from WildChat-1M
   (gated HF, license click-through) and LMSYS-Chat-1M. Filter: 3-8 turns,
   English, no obvious benchmark content. Truncate to end on a user turn.
2. **Eval side, obvious cell:** multi-turn eval transcripts where the eval
   is overt (benchmark item embedded verbatim, or agentic eval transcripts).
3. **Eval side, disguised cell:** Claude-authored conversations that embed
   an evaluation task across turns while matched to the deployment pool on
   turn count, lengths, register, formatting. Seed each from a real
   WildChat conversation skeleton to inherit natural structure. Same
   probe-blind gate, applied to the serialized conversation text.
4. **Code:** extend `extract.py` READ_POINTS with `post_instruction_token`,
   `final_turn_mean`, and `per_turn_trajectory` (one vector per user turn).
   Unit-test read indices on hand-built conversations before any GPU run.
5. Extract on the 8B, run the probe suite per read point. Key numbers:
   - multi-turn obvious vs deploy (should be high; sanity),
   - **multi-turn disguised vs deploy (decisive cell)**, probe vs baselines,
   - single-turn -> multi-turn transfer (does the slice-2 direction carry?),
   - per-turn trajectory: where in the conversation the signal emerges.
6. Success criterion: a clean answer in the decisive cell with CIs, either
   regime, plus the trajectory profile.

## Slice 4 — silence filtering (internality)

First slice needing generation. Budget Kaggle hours before running.

1. Generate the 8B's responses to the decisive-cell conversations (one
   completion each, temperature 0).
2. Awareness judge: Claude/GPT API reads the response text only and flags
   verbalized test-recognition. Judge is text-only, so API use is fine.
3. Re-run the probe on the unaware subset. Compare probe-detectability vs
   verbalization-detectability (a text classifier on the model's outputs).
4. Success criterion: probe strength on the silent subset, with the
   verbalization comparison. Signal surviving silence-filtering is internal.

## Slice 5 — deployment-simulation reconstruction

1. Take genuine WildChat/LMSYS multi-turn conversations. Regenerate the
   final assistant turn with the 8B (on-policy resample). Optionally swap
   earlier assistant turns in from a different model (model-swap).
2. Probe: genuine vs resampled, using slice 3 read points. The model-swap vs
   eval-context ablation separates off-policy authorship from evaluation
   context.
3. Labels here are exact (we know what was resampled), so this arm has no
   dataset-construction risk; cost is generation time.

## Slice 6 — steering (last)

1. Take the best-layer eval-awareness direction from the decisive cell.
2. Inject at a middle layer at every token position, magnitude scaled to the
   average residual norm (Assistant Axis recipe), prompt held fixed.
3. Also try **activation capping**: clamp the projection onto the direction
   below the deployment-typical threshold instead of constant injection.
   Nguyen's constant injection recovered <5% of sandbagged accuracy; capping
   is the untested variant and our expectation-setting is honest about the
   base rate.
4. Measure behavior shift on a sandbagging setup (WMDP-style) and on
   refusal/honesty metrics.

## Cross-model arm (parallel, cheap)

Once slice 2 or 3 machinery is stable, rerun the headline cells on
Qwen2.5-7B-Instruct and Gemma-2-9B/Gemma-3 on Kaggle. Each is a dataset-reuse
extraction run. Note: Chaudhary et al. excluded Qwen for suspected
contamination; if Qwen looks anomalous, report and move on. These place our
multi-turn cells on their scaling curve without paid GPU.

## Order of work, concretely

1. `data_2x2.py` loader + surface baselines on Devbunova's cells (local, no
   GPU). Verify their matching with our baselines.
2. Kaggle notebook: extract 4 cells, run probe suite + transfer matrix (2a).
3. Rewrite pipeline + gate for Casual-Eval-True; audit; extract; rerun (2b).
4. Paper + docs checkpoint (2c).
5. Multi-turn read points in `extract.py` + tests.
6. Multi-turn dataset build (deploy pool, then disguised evals) + gate.
7. Decisive cell run (slice 3), checkpoint.
8. Slices 4, 5, 6 in order, each with a budget check before GPU time.
9. Cross-model arm whenever a slice is blocked on data work.

## Risks and mitigations

- **Rewrite house style:** the probe could learn "Claude-written" instead of
  "eval." Mitigations: varied templates, the probe-blind gate, and a control
  contrast of Claude-rewritten deploy prompts vs raw deploy prompts — if a
  probe separates those, authorship is leaking and rewrites must diversify.
- **Disguised-eval definition drift:** slice 3's eval side must stay an
  *evaluation in content* (a task with a right answer, posed to test the
  model), not just formal prose. Keep a written inclusion rule per scenario
  class before authoring.
- **LMSYS/WildChat gating:** both need HF license click-through; do this
  early, on the same account as the Llama grant.
- **Kaggle weekly cap:** 30 GPU-hr/week. Extraction slices fit easily;
  generation slices (4, 5) get an estimate-first rule: measure tokens/sec on
  100 samples before committing to the full run.
