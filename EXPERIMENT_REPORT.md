# Experimental Report: Headline Results and Claim Boundaries

## Purpose

This is the authoritative concise account of the completed experiments, their
headline numerical results, and the claims they do and do not support. It is
grounded in current code, frozen protocols, and saved artifacts rather than the
historical proposal.

Read `EXPERIMENT_LEDGER.md` only when a task requires pilot history, failed
construction details, full result matrices, secondary analyses, or artifact-
level audit reasoning. Read `ARTIFACT_INDEX.md` for implementation and artifact
locations.

Status terms used here:

- **Completed:** code and corresponding output artifacts exist.
- **Pilot:** completed on a deliberately small sample for feasibility or design
  diagnosis.
- **Rejected:** completed but excluded from the evidential chain because a gate
  or manual validity check failed.
- **Engineering check:** pipeline validation, not scientific evidence.
- **Proposed only:** not supported by completed artifacts in this repository.

## Executive conclusion

Llama-3.1-8B-Instruct hidden states linearly distinguish several independently
constructed evaluation-like and deployment-like situations. In two paired
multi-turn controls, activation separation is at chance throughout a byte-
identical prefix and appears only after the class-defining final user turn.
Qwen3-8B, Gemma-3-12B-IT, and Ministral-3-8B replicate this exact temporal
localization and above-chance final-turn decodability.

The experiments do **not** isolate a single architecture-invariant variable
that can be identified uniquely as evaluation awareness. Benchmark-versus-user
contrasts and several constructed controls remain highly distinguishable from
text alone. Character n-grams tie the activation probe on the strongest matched-
coherence control and outperform activation transfer between the two multi-turn
controls. Gemma is substantially weaker than text on both paired controls.

On the Llama reference model, separation remains among terminated reply pairs
without detected awareness disclosure. A pre-specified residual intervention
also moves an independent activation readout and explicit evaluation/deployment
self-reports in the predicted direction. It does not increase safety refusals.
The defensible conclusion is therefore that the studied constructions induce
cross-family, behaviorally accessible activation directions whose strength and
meaning remain entangled with textual task structure.

## Common analysis conventions

The primary reference model is `meta-llama/Llama-3.1-8B-Instruct`. The frozen
cross-family confirmation panel contains Qwen3-8B, Gemma-3-12B-IT, and
Ministral-3-8B. The panel covers the 5--12B range and is not a scale study.

Conversations are rendered with each model's native chat template. The main
user-side readout is the hidden state at the token closing the final user turn,
before reply generation. Identical-prefix controls also read earlier user-turn
boundaries. Slice 5 uses the closing token and mean hidden state of the final
assistant reply.

The primary probe is the normalized difference between the evaluation-class and
deployment-class mean activations. A held-out activation is scored by projection
onto that direction, and AUROC measures ranking performance. Layer selection is
performed using training data only. Most reference-model results use five
deterministic 70/30 splits and report mean plus population SD across seeds.

Surface comparisons include length, 13 structural features, unigram counts, and
where noted word 1--2-gram and character-within-word 3--5-gram TF-IDF logistic
classifiers. For byte-identical-prefix pairs, additive text baselines see only
the differing causal message; the shared prefix contains no population-level
label information and otherwise adds finite-sample noise.

The grouped strong-baseline audit keeps both members of each matched pair in the
same fold. It uses pair-cluster bootstrap intervals and paired prediction
randomization to compare probe and text AUROCs directly. Full-pipeline label
permutations repeat fitting and layer selection. Five-seed SDs, bootstrap
intervals, and permutation p-values are different quantities and must not be
interchanged.

## Completed experimental chain

### 1. Benchmark evaluation versus real user prompts

Slice 1 compared 200 Situational Awareness Dataset evaluation prompts with 200
WildChat deployment prompts. The honest multi-seed direction-of-means result was
`0.99183 ± 0.00848`; a standardized logistic probe gave
`0.99333 ± 0.00772`. The structural classifier reached `0.9675` and bag of
words `0.9517` on the historical seed-0 split.

This established that the activation pipeline detects the benchmark/user
distinction. It did not establish evaluation awareness because surface form
nearly determines the labels. The historical `0.9944` headline selected its
displayed layer on the held-out curve and is not the primary inferential number.

### 2. Public 2×2 reproduction

The public `viliana-dev/eval-awareness-2x2` dataset was reproduced. Its intended
matched cells remained strongly surface-separable: for example, casual MuTual
evaluation versus casual deployment reached format AUROC `0.9988` and bag-of-
words AUROC `0.9877`. A standard activation direction behaved mainly like a
benchmark-format detector. Paired training reduced that failure but did not
remove the surface confound. This slice is diagnostic rather than clean evidence
for a format-invariant representation.

### 3. New single-turn constructions and transfer

Slice 2b constructed four 250-example cells: two casual evaluation styles, a
WildChat-derived deployment style, and a topic-matched deployment style. Length
matching passed, but the frozen surface gate failed. Format AUROCs ranged from
`0.6627` to `0.8326`, and bag-of-words AUROCs ranged from `0.9056` to `0.9783`.
The cells must not be described as surface matched.

GSM8K was excluded from the primary transfer analysis because digit density
nearly solved some contrasts. On the remaining ARC/MMLU-based transfer grid,
the main question became whether a direction fitted on one construction
generalized to another better than the same surface classifiers.

| Transfer subset | Probe strength | Bag-of-words strength | Margin, mean ± SD |
|---|---:|---:|---:|
| 20 completely disjoint source-target pairs | 0.22810 | 0.18055 | `+0.04754 ± 0.01538` |
| 10 pairs sharing one dataset cell | 0.41874 | 0.30059 | `+0.11815 ± 0.01335` |
| All 30 off-diagonal pairs | 0.29164 | 0.22057 | `+0.07108 ± 0.01341` |

The completely disjoint margin was positive in all five seeds and is the
cleanest transfer summary. A standardized logistic probe preserved the pattern,
with all-off-diagonal margin `+0.05564`. The result supports a partly shared
internal direction across the studied constructions, not removal of topic,
speech-act, or authorship information.

### 4. Multi-turn temporal localization

The first written multi-turn cells were rejected as a primary control. Their
surface gate failed, and activation separation already existed after the first
synthetic user turn, before the supposed class-defining final question.

Two repaired 250-pair controls are valid:

- **`eval_shared`:** a real-derived deployment conversation with a byte-
  identical prefix; only the final user turn is replaced by an evaluation
  question. It cleanly localizes the change but also changes topic and speech
  act.
- **`coherent_pairs`:** both alternatives are written together with a byte-
  identical, topically coherent prefix and matched final-turn style. This is a
  stronger coherence control but uses a synthetic prefix.

Reference-model train-only results were:

| Control/read point | Probe AUROC, mean ± SD | Unigram BoW | Interpretation |
|---|---:|---:|---|
| `eval_shared`, final user | `0.99168 ± 0.00330` | 0.93504 | Strong final-turn separation |
| `eval_shared`, pre-final | `0.49847 ± 0.00111` | 0.50000 | Chance throughout shared prefix |
| `coherent_pairs`, final user | `0.95132 ± 0.01147` | `0.93337 ± 0.00680` | High but largely lexical |
| `coherent_pairs`, pre-final | `0.49902 ± 0.00152` | 0.50000 | Chance throughout shared prefix |

The later grouped strong-text audit is the authoritative comparison:

| Analysis | Probe | Strongest text | Difference (95% CI) | One-sided p |
|---|---:|---:|---:|---:|
| `eval_shared`, within | 0.98781 | word TF-IDF 0.95293 | `+0.03488` `[+0.01875, +0.05330]` | 0.00450 |
| `coherent_pairs`, within | 0.94682 | character TF-IDF 0.94474 | `+0.00208` `[-0.01341, +0.01675]` | 0.43778 |
| `eval_shared` → coherent | 0.84496 | character TF-IDF 0.86614 | `-0.02118` `[-0.04887, +0.00811]` | 0.87856 |
| coherent → `eval_shared` | 0.71874 | character TF-IDF 0.84854 | `-0.12981` `[-0.16654, -0.09081]` | 1.00000 |

Thus, `eval_shared` retains an activation advantage despite its topic/speech-act
discontinuity. Matched coherence is tied by character n-grams, and cross-control
activation transfer does not beat stronger text classifiers.

### 5. Separation conditional on observed silence

Llama generated greedy replies for both members of each pair. A pair was kept
only when both replies terminated and neither regex checks, a blinded two-stage
judge procedure, nor manual audit found explicit or implicit evaluation-
awareness language. The judge validation detected all 14 planted positives,
produced 0/10 false alarms on hard negatives, and showed exact agreement on all
52 duplicate checks.

On 192 retained matched-coherence pairs, the probe was
`0.93686 ± 0.03250` and bag of words `0.93543 ± 0.02814`, a margin of only
`0.00143`. On `eval_shared`, 124 retained cap-256 pairs gave probe
`0.99752 ± 0.00205` versus bag of words `0.95047 ± 0.01956`. Repeating with a
512-token cap retained 227 pairs and gave `0.99083 ± 0.00321` versus
`0.94602 ± 0.01513`, a margin of `0.04481`.

Overt disclosure is therefore not necessary for high raw separation, but the
matched-coherence result is lexically explained. “No detected disclosure” does
not establish unexpressed or unconscious awareness, and the analysis remains
conditioned on reply termination.

### 6. Genuine versus reconstructed assistant replies

Slice 5 asks a different question: whether Llama hidden states distinguish a
genuine assistant reply from a greedy generated replacement inserted after the
same conversation prefix.

For WildChat, 196/250 replacements terminated. The direction-of-means probe
reached `0.79247 ± 0.02448` at the final assistant token and
`0.82206 ± 0.01803` over the reply mean. The strongest reply-only surface
baseline was `0.77374`, leaving margins `0.01873` and `0.04832`. On LMSYS,
179/250 replacements terminated; token and mean probes reached
`0.83700 ± 0.02556` and `0.79119 ± 0.02976`, versus a strongest baseline of
`0.73573`.

The signal survived when Llama-3.2-3B generated the replacements. It was also at
least as strong at the token readout when the 8B generator saw only the final
user message. Reply-only format features explained most of the shared-
intersection control differences. The result therefore concerns provenance,
contextual fit, or distributional mismatch—not specifically self-authorship and
not evaluation awareness.

### 7. Estimator and null robustness

A standardized logistic probe generally increased AUROC but did not change
which controls passed or failed. Every real separating contrast in the final
robustness table reached the minimum resolvable full-pipeline permutation
`p=1/501=0.001996`. The two identical-prefix pre-final controls remained null:
`p=0.99002` for `eval_shared` and `p=0.94012` for matched coherence.

Absolute separation strength `|AUROC-0.5|` can be nontrivial for shuffled or
random high-dimensional directions, so it is not a significance test. Signed
AUROC and full-pipeline permutations must accompany it.

### 8. Pre-specified activation steering

The steering direction was fitted from all 250 matched-coherence pairs. Four
candidate layers and nine intervention strengths were pre-specified. Layer 14
was selected from the manipulation check because an independent `eval_shared`
readout moved monotonically with intervention strength. The endpoint strengths
`-4` and `+4` violated the frozen degeneration rule and are excluded from
behavioral trends.

Across nondegenerate strengths `[-2,-1,-0.5,0,0.5,1,2]`, evaluation self-report
rates were `[0.1389,0.6111,0.6667,0.7778,0.8056,0.8889,0.8889]`, with Spearman
correlation `0.991`. Safety-refusal rates were
`[0.0833,0.1042,0.1042,0.1250,0.1042,0.0833,0.0625]`, with correlation
`-0.356`, opposite the predicted increase. With only 48 safety prompts and a
12.5% baseline refusal rate, the experiment rules out a large predicted effect
but is underpowered for small effects.

The intervention causally moves the independent readout and explicit
eval/deploy self-report. It is not shown to govern broad safety policy.

### 9. Authorship-factorial pilot

The v2 pilot crossed eval/deploy final-turn status with target-model, same-family
off-policy, and cross-family off-policy histories. All 150 generated histories
terminated under the amended frozen protocol, but only 36/50 scenarios survived
the combined model and human coherence gate, below the required 40/50.

The pilot therefore stopped before activation extraction. Authorship invariance
remains untested and must not be inferred from the existing results.

### 10. Cross-model confirmation

The frozen suite reran `eval_shared` and `coherent_pairs` on three revision-
pinned open-weight instruction models. Independent causal-prefix passes ensured
exactly comparable shared-prefix states. Every model had maximum positive/
negative activation difference `0.0` at byte-identical pre-final boundaries and
nonzero final-turn differences.

Grouped strong-baseline within-control results were:

| Model | `eval_shared`: probe vs text | Matched coherence: probe vs text |
|---|---:|---:|
| Llama-3.1-8B | 0.98781 vs 0.95293 | 0.94682 vs 0.94474 |
| Qwen3-8B | 0.99357 vs 0.95293 | 0.94653 vs 0.94474 |
| Gemma-3-12B-IT | 0.72667 vs 0.95293 | 0.64744 vs 0.94474 |
| Ministral-3-8B | 0.98062 vs 0.95293 | 0.92789 vs 0.94474 |

Qwen's `eval_shared` difference was `+0.04064`, 95% CI
`[+0.02364,+0.05982]`, `p=0.0050`. Ministral's was `+0.02770`, CI
`[+0.01017,+0.04720]`, `p=0.0155`. Neither model beat text on matched coherence.
Gemma was substantially weaker than text on both controls. No replication model
beat character TF-IDF in either cross-control transfer direction.

Together with Llama, this establishes cross-family temporal decodability, but
not architecture-invariant strength or a generally activation-specific signal
beyond textual task structure.

## Supported claims

The repository supports these statements when their qualifications remain
attached:

1. Llama-3.1-8B hidden states strongly distinguish multiple studied eval-like
   and deploy-like constructions.
2. Simple benchmark-versus-user separation is heavily explained by surface
   form.
3. On Slice 2b, completely disjoint math-free activation transfer exceeds the
   tested unigram baseline by `0.04754 ± 0.01538`.
4. In two paired multi-turn controls, separation is at chance throughout the
   byte-identical prefix and emerges only after the differing final turn.
5. `eval_shared` beats the strongest tested text baseline on Llama, Qwen, and
   Ministral; matched coherence does not reliably beat character n-grams.
6. Cross-control activation transfer does not beat character n-grams.
7. High raw separation remains among terminated replies with no detected
   awareness disclosure, although matched-coherence activation and lexical
   baselines tie.
8. Genuine and reconstructed replies are separable across WildChat and LMSYS,
   but controls rule out a narrow self-authorship interpretation.
9. Main activation effects survive a second estimator and full-pipeline
   permutation tests.
10. Steering moves an independent activation readout and explicit self-report,
    but not safety refusals.
11. Four model families show exact temporal localization, with substantially
    heterogeneous final-turn strength.

## Unsupported or overstated claims

The repository does not support the following:

- The proposal was executed as written.
- The final Slice 2b cells are surface matched.
- The first written multi-turn cells prove temporal awareness.
- The model “knows it is being evaluated” as a uniquely identified semantic
  interpretation of the direction.
- The matched-coherence probe meaningfully beats text in-distribution.
- Cross-control activation transfer beats the strongest text baselines.
- Observed silence proves unconscious or unverbalized awareness.
- Slice 5 demonstrates self-recognition.
- Steering improves safety behavior or establishes a broad policy mechanism.
- The effect is invariant to history authorship; the relevant pilot failed
  before activation extraction.
- The effect has architecture-invariant strength or a demonstrated scale trend.
- The project measured agentic tool use or a real positive-disclosure class.

## Audit rules for paper work

1. Use train-only layer selection; do not revive historical held-out layer
   selection.
2. Report the completely disjoint Slice 2b transfer subset as primary or show it
   separately from shared-cell and all-off-diagonal averages.
3. Keep failed gates visible. Later controls respond to those failures; they do
   not erase them.
4. Keep rejected `eval_coherent`, stale `casual_deploy_rw`, and engineering
   checks out of headline evidence.
5. Distinguish split-seed SD, bootstrap confidence intervals, paired
   randomization, and full-label permutation tests.
6. Describe silence as observed silence and preserve termination, cap, matching,
   judging, and conservative-exclusion qualifications.
7. For identical-prefix pairs, use only the differing causal segment for
   additive text baselines.
8. Prefer the shared-intersection Slice 5 model-swap/context-ablation comparison.
9. Exclude steering endpoints `-4` and `+4` from behavioral trends under the
   frozen degeneration rule.
10. Attribute awareness judging only as far as the saved artifacts permit.
11. Treat the authorship v2 pilot as a failed pre-activation gate.
12. Preserve cross-model revision, dataset, precision, prefix-extraction, and
    manifest provenance; normalize the documented CRLF/LF manifest variant when
    validating across operating systems.
