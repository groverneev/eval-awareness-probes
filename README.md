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

Slices 1 and 2 are done on Llama-3.1-8B. Slice 3's matched-coherence control
and its silence-filtered follow-up are complete at n=250/cell. Slice 5's
deployment-simulation reconstruction and its model-swap/context-ablation
decomposition are complete at n=250, and now replicate on a second corpus
(LMSYS-Chat-1M). A secondary logistic-regression estimator, permutation nulls,
and a calibrated model judge for the silence filter are done and change no
conclusion. **The steering arm is complete**: the manipulation check passed at
layer 14, steering moves verbalized self-report strongly (0.139 to 0.889,
Spearman +0.991), and safety behaviour shows no large effect — the
pre-specified "manipulable and self-reportable, but behaviourally inert on
safety" outcome.

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

- The probe beats bag-of-words under transfer by **+0.071 ± 0.013**, positive
  on every seed. Every format-matched training cell gives a positive margin.
- **Training on the standard confounded contrast is the sole exception**
  (−0.087 ± 0.032, win rate 12%): on the setup the field actually uses,
  activations generalize *worse* than word counts.
- The direction travels one way. Trained on our realistic cells it scores
  0.949 on the standard benchmark cell; trained on the standard cell it scores
  0.578 on ours, barely above chance.
- Honest limit: benchmark format on *both* sides with only content differing
  (`bench_eval` vs `bench_deploy_rewritten`) defeats every direction we can
  learn — 0.31–0.55, at or below chance. That is where content-only eval
  recognition would have to appear, and it does not.

**Slice 3 — multi-turn (n=250/cell, 5 seeds).** Real WildChat conversations
of 2–4 user turns are compared with conversations of identical shape whose final
user turn poses a benchmark question. Causal read points and per-turn
trajectories make it possible to test exactly when a signal becomes available.

- The as-written `eval_disguised` cell fails as a realism test. At the end of
  the first user turn, before the benchmark question appears, the probe is
  **0.951** and bag-of-words is 0.856. The revised batches lowered the probe
  only from 0.994 to 0.962, so they did not remove this early-turn synthetic
  lead-in signal; this cell is not evidence of
  evaluation recognition.
- The constructed `eval_shared` control holds every prefix byte-identical to its
  matched deployment conversation and swaps only the final user turn. It is
  exactly chance before that turn (probe **0.497** at the early read point; an
  end-aligned trajectory is 0.498–0.501 on every preceding turn), then reaches
  **0.995** at the final turn.
- At that final turn `eval_shared` clears bag-of-words 0.769 and format 0.576,
  a **+0.226** probe–BoW margin. This is the cleanest current evidence that the
  model's final-turn representation carries more than the tested lexical and
  structural features, but it is local to a deliberately topic-shifted question
  swap rather than a realistic disguised evaluation.

The next control is therefore matched coherence: retain the identical-prefix
construction while making the final benchmark question follow naturally from
the preceding conversation.

**Matched-coherence control (n=250, 5 seeds).** We repaired the topic-shift
failure by constructing a shared everyday-topic prefix for each pair, then
attaching either the benchmark question or a genuine same-topic help request.
The shared pre-final read is exactly chance (probe 0.500). At the final token,
the probe reaches **0.951 +/- 0.011**, above bag-of-words **0.852 +/- 0.012**
and format 0.579, for a **+0.099** probe-BoW margin. Whole-conversation length
and format baselines are 0.500 and 0.520, respectively. This is the current
cleanest multi-turn result; final-turn-only lexical baselines remain high and
are reported as a stricter diagnostic.

Cross-control transfer confirms that this is not unique to one construction:
`eval_shared` to paired coherence reaches 0.858 versus BoW 0.597, and the
reverse reaches 0.788 versus BoW 0.564. The probe advantage is positive in
both directions (+0.262 and +0.224 strength, respectively).

**Slice 5 â€” deployment-simulation reconstruction pilot (n=50, 5 seeds).**
We sampled real WildChat conversations that end in an assistant reply, then
replaced only that final reply with a greedy on-policy Llama-3.1-8B generation.
This provides exact provenance labels and a byte-identical prefix, with no
hand-authored eval/deployment prompt construction. Seven generated replies hit
the 256-token cap and are excluded as indeterminate, leaving 43 pairs. At the
final assistant end-of-turn token, the held-out probe is **0.761 +/- 0.086**;
bag-of-words is 0.575 +/- 0.041 and format is 0.667 +/- 0.048. The conservative
margin over the strongest surface baseline (format) is **+0.094** in separation
strength. Mean-pooling over the final reply gives 0.737 +/- 0.078, a +0.070
margin over format. This is evidence that response provenance is internally
detectable in this pilot, not evidence by itself of evaluation awareness; the
next step is a pre-registered n=250 scale-up, then model-swap versus
eval-context ablation.

**Slice 5 scale-up (n=250, 5 seeds).** The frozen protocol replicated at
scale. Of 250 greedy replacements, 196 terminated naturally; the other 54 are
indeterminate because they reached the 256-token cap. At the final assistant
end-of-turn token, the probe is **0.792 +/- 0.024**, versus BoW **0.633 +/-
0.011** and format **0.680 +/- 0.021**, for a conservative +0.113 strength
margin over format. At the final-assistant mean, the probe reaches **0.822 +/-
0.018**, for a **+0.142** margin over format. This confirms internally
detectable response provenance under an exact-label, shared-prefix construction;
it remains a provenance result rather than a direct evaluation-awareness claim.

**Slice 5 decomposition (n=250 generated per condition; 146 shared terminated
pairs, 5 seeds).** To distinguish model authorship from conversational context,
we regenerated every final reply both with Llama-3.2-3B under the full original
prefix and with the probed Llama-3.1-8B given only the final user message. The
8B probe then read the full original conversation with each replacement inserted.
On the exact common subset, model-swap reaches **0.811 +/- 0.046** at the final
assistant token, versus format 0.651 (margin **+0.160**); the context-ablation
condition reaches **0.868 +/- 0.012**, versus format 0.702 (margin **+0.166**).
At the final-assistant mean, the corresponding probe margins are +0.138 and
+0.101. Thus context removal produces a comparably strong, and token-wise
stronger, effect than changing the reply model. The reconstruction signal cannot
be attributed to 8B-specific authorship and is still not direct evidence of
evaluation awareness.

**Silence-filtered matched coherence (n=250 generated pairs, 5 seeds).** We
greedily generated one 8B reply for each cell, then retained only pairs whose
two replies terminated naturally and did not verbalize recognition of being
tested. **192/250** pairs met that criterion; 58 pairs with a capped reply were
treated as indeterminate, not silent. A narrow explicit-awareness scan flagged
0 pairs, and an independent text-only audit of 30 random pairs likewise found
0 explicit or paraphrased recognition statements. On the retained pairs, the
final-token probe is **0.937 +/- 0.033**, versus BoW **0.823 +/- 0.026** and
format 0.567 +/- 0.016. The probe-BoW strength margin is therefore **+0.114**.
This is evidence that the result survives an observed-silence filter, not proof
that no response ever verbalized awareness: capped replies are excluded and the
awareness judge is deliberately conservative. With zero positive verbalization
labels, a verbalization-detectability classifier is not estimable in this run.

**Estimator and null-baseline checks (no GPU; `run_estimator_check.py`).** Every
headline number above came from one fitting rule and was compared against a
nominal 0.5. Both gaps are now closed on the cached activations.

- *Second estimator.* L2 logistic regression, with the layer chosen by
  cross-validation inside the training split, reproduces every result and is
  uniformly equal or stronger: matched coherence 0.951 → **0.989**, its
  silence-filtered subset 0.937 → **0.982**, slice-5 reconstruction 0.792 →
  **0.896** at the final assistant token. Under slice-2b transfer the
  probe-minus-bag-of-words margin is **+0.056** against difference-of-means'
  +0.071, with the same sign pattern and the same sole negative
  (`devb_standard`). No conclusion depends on the estimator. The lone exception
  is slice 1, where logistic regression buys nothing (0.992 → 0.993): the
  obvious cell is already fully confounded, so the separation is trivially
  linear and a better-fitted direction has nothing left to recover.
- *Significance.* A permutation test that shuffles the entire label vector and
  reruns the whole pipeline, layer selection included, puts every real contrast
  at **p = 0.002**, the floor at 500 permutations, with a null 95th percentile
  of 0.09–0.13 strength. The two identical-prefix controls correctly land at
  p = 0.99 and p = 0.94.
- *A caveat the proposal did not anticipate.* The baseline hierarchy assumed a
  random probe sits near 0.5. It does not. Because the eval/deploy axis carries
  a large share of total activation variance, an **uninformed** direction fitted
  from shuffled training labels reaches strength 0.24–0.29 on the strongest
  cells (AUROC ≈ 0.75), and a random unit direction reaches ≈ 0.11. This does
  not affect significance — the permutation null handles that — but it means
  raw |AUROC − 0.5| overstates how much work the probe is doing, and belongs in
  the limitations.

**Verbalized-awareness judge (`judge_batches.py`).** The silence filter
previously rested on a hand-written regex plus a 30-pair manual audit. It now
also rests on a model judge run over all 500 replies, blinded to cell and pair,
with a committed and hashed rubric. Because the expected answer is "nothing
here", the batches carry planted controls: the judge caught **14/14** planted
recognition statements, including all six that only paraphrase it, and raised
**0/10** false alarms on replies that discuss tests, exams and benchmarks as
subject matter. Between-judge agreement on duplicated rows was 47/47. It flagged
**0/250** real pairs, agreeing with the regex on every response. A manual audit
of all 32 evaluation-vocabulary contexts in the real replies found only
subject-matter uses. The silence result is therefore a calibrated null rather
than an unverified one.

**Steering — the causal arm (pre-specified; `datasets/steering/prespec.md`).**
The protocol, including what each possible outcome would mean, was written and
committed before any steered output existed. The intervention adds
`alpha · ‖μ_eval − μ_deploy‖ · d̂` at a single layer with the prompt held fixed,
and a manipulation check scored by an *independent* probe (fit on `eval_shared`,
while the steering direction is fit on matched coherence) gates the behavioural
measures.

- *Manipulation check: passed.* Layer 14 moves the independent readout
  monotonically across the coefficient grid (Spearman +1.00), spanning −1.14 to
  +0.87 of the eval−deploy gap. **Layer 3 is completely flat across a 16× range
  of coefficients** — a dead injection site. The original argmax layer rule
  would have chosen that early-layer region and produced a clean-looking
  behavioural null that was really an intervention doing nothing; Amendment 1
  to the pre-spec, made before any steering, is what avoided it.
- *A finding the proposal did not anticipate:* steering efficacy increases
  sharply with depth across a plateau over which cross-validated *decodability*
  is flat. The layer where a concept is most decodable is not the layer where
  injecting it does the most work.
- *Self-report: shifts, strongly.* Excluding the two degenerate coefficients,
  the rate at which the model calls the situation an evaluation moves from
  **0.139** (α = −2) to **0.889** (α = +2), Spearman **+0.991**, with
  non-overlapping 95% intervals at the ends.
- *Safety behaviour: no large effect.* The refusal rate on 48 held-fixed
  borderline requests stays within **0.062–0.125** across every non-degenerate
  coefficient — a swing of three prompts — and every interval overlaps.
- *Degeneracy veto fired exactly where predicted*, at α = ±4 (repeated-4-gram
  fractions 0.42–0.78); those coefficients are reported but excluded from any
  causal claim.

This is **row 2 of the pre-specified outcome table**: the direction is genuinely
eval-recognition and the model can report on it, but recognition alone does not
move safety behaviour. The pre-spec recorded "low confidence" for the safety
prediction in advance, and that was right. The critical caveat is power, not
interpretation: at n = 48 with a 12.5% baseline, the arm has 80% power only
against an increase to ~0.37. It rules out a *large* effect, not a modest one,
and must be reported that way.

**Silence-filtered `eval_shared` (n=250 generated pairs, 5 seeds).** The other
identical-prefix control now has the silence filter too. Explicit verbalized
awareness on the eval side: **0/250**. On the retained pairs the final-token
probe is **0.998 +/- 0.002**, versus BoW **0.758 +/- 0.029** and format
**0.624 +/- 0.022** — a **+0.240** probe-BoW margin, more than double matched
coherence's +0.114. Only **125/250** pairs had both replies terminate naturally
under the frozen 256-token cap, so a censoring sensitivity rerun at a 512-token
cap was added: it raises retention to **227/250** (91%) and moves the probe by
0.007 (to 0.991 +/- 0.003) and the margin by 0.025 (to +0.215). The result is
therefore not an artifact of censoring. Eval-side verbalized awareness is 0/250
at both caps, and the wider cap is the stronger silence claim because replies
had twice the room to verbalize. One regex hit occurred, on the *deploy* side at
both caps, and manual reading showed it to be a TV quiz show — subject matter,
not self-recognition, and conservative in direction.

**Slice 5 replicated on LMSYS-Chat-1M (n=250, 5 seeds).** The frozen WildChat
protocol was rerun byte-for-byte on a second corpus, with redacted conversations
dropped so PII placeholders cannot act as a label. 179/250 pairs terminated
naturally. At the final assistant token the probe reaches **0.837 +/- 0.026**
versus format 0.671 (margin **+0.166**); at the final-assistant mean, **0.791
+/- 0.030** (margin +0.120). Against WildChat's 0.792 and 0.822, the effect
replicates across corpora, with the token/mean ordering swapping between them.
Slice 5 therefore no longer rests on a single corpus.

Slice 1's 8B activations have also been re-extracted locally, which both extends
the estimator and null checks to slice 1 and reproduces the original Kaggle run
exactly (best layer 4, AUROC 0.9944).

Figures in `figures/`; full detail, caveats, and a handoff section for picking
the project up mid-flight are in `CLAUDE.md`.

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
