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
multi-turn controls, exact equality of pre-final hidden states verifies the
byte-identical construction and extraction pipeline; labels become decodable
after the class-defining final user turn. Qwen3-8B, Gemma-3-12B-IT, and
Ministral-3-8B pass the same leakage check and reproduce above-chance final-turn
decodability.

The experiments do **not** isolate a single architecture-invariant variable
that can be identified uniquely as evaluation awareness. Benchmark-versus-user
contrasts and several constructed controls remain highly distinguishable from
text alone. The strongest full-input E5 prefix-ending interaction matches or
exceeds every primary linear probe: all eight model-control point differences
favor text, five paired intervals exclude zero in that direction, and three
include zero. Character n-grams also outperform activation transfer between the
two multi-turn controls. A uniform readout audit does not support a simple
extraction-failure explanation: unrestricted final-user-turn mean pooling makes
Gemma appear aligned only by selecting the input-embedding average, while
excluding that layer restores strong architecture-by-readout heterogeneity.
A uniform same-layer decoder audit changes the estimator-side diagnosis.
Nested L2-logistic probes recover most of the gap between direction-of-means
and RBF; on matched coherence, five-seed mean RBF-minus-L2 is only `+0.00090`
to `+0.01036` across the four models, and every seed-0 paired interval includes
zero. At each outer fold's primary-selected layer, the nested RBF probe reaches
`0.98112`--`0.98786` on matched coherence across all four models and beats a
RBF over the same E5 interaction (`0.95659`)
with positive paired intervals. This advantage remains positive for every model
in each of five fold seeds. Gemma's weak direction-of-means result is therefore
not an absence of decodable final-token information, but the recovery is mostly
learned linear and cannot be attributed wholesale to nonlinear geometry. The
shared-prefix RBF margin remains heterogeneous, and
source-only nonlinear activation transfer does not beat the source-selected
nonlinear text pipeline or linear character TF-IDF.

A separately frozen within-family scale ladder reruns the identical pipeline on
all six original dense Qwen3 checkpoints from 0.75B to 32.76B parameters, on
one A100 environment. Relative to the same full-input E5 interaction, five of
six margins are nonpositive on each control. The probe-minus-E5 slope is `+0.0054`
AUROC per parameter decade (95% paired-bootstrap interval
`[-0.0017,+0.0129]`) on shared prefix and `+0.0068`
(`[+0.0013,+0.0132]`) on matched coherence. Because only the latter excludes
zero, the prespecified same-direction criterion is not met: the data do not
show a consistent within-family scale trend. The five-seed nonlinear result is
likewise nearly flat across size, tying contextual E5 on shared prefix and
exceeding it by about `+0.036`--`+0.038` on matched coherence at every size.

On the Llama reference model, separation remains among terminated reply pairs
without detected awareness disclosure. A pre-specified residual intervention
also moves an independent activation readout and explicit evaluation/deployment
self-reports in the predicted direction, but two of five norm-matched orthogonal
directions produce larger self-report shifts. Neither learned nor control
directions increase safety refusals. The defensible conclusion is therefore
that the studied constructions induce cross-family, decodable activation
directions whose behavioral meaning remains entangled with textual task
structure and generic intervention sensitivity.

## Common analysis conventions

The report distinguishes four claims. **Evaluation awareness** is an internal
representation of whether the current interaction is being used to evaluate the
model, rather than merely prompt features correlated with evaluation.
**Prompt-category decodability** means that a readout separates the tested
evaluation-like and deployment-like examples, potentially using genre, topic,
format, or speech act. **Evaluation self-report** is the model's report of
whether it believes it is being evaluated. **Policy use** means that perceived
evaluation status influences task-relevant behavior. Text-controlled and
cross-construction generalization test the construct interpretation;
self-report steering with unrelated-direction controls tests direction-specific
causal control of the report; and refusal steering is one narrow policy-use
test. These properties may dissociate.

The primary reference model is `meta-llama/Llama-3.1-8B-Instruct`. The frozen
cross-family confirmation panel contains Qwen3-8B, Gemma-3-12B-IT, and
Ministral-3-8B. The panel covers the 5--12B range and is not a scale study.
The distinct Qwen3 scale audit includes revision-pinned dense 0.6B, 1.7B, 4B,
8B, 14B, and 32B repositories (0.75B--32.76B exact parameters). It re-extracts
8B and every other checkpoint in unquantized bfloat16 on the same single A100;
the historical cross-family 8B result is not substituted into the ladder.

Conversations are rendered with each model's native chat template. The main
user-side readout is the hidden state at the token closing the final user turn,
before reply generation. Identical-prefix controls also read earlier user-turn
boundaries. Slice 5 uses the closing token and mean hidden state of the final
assistant reply. A post-result cross-model sensitivity audit additionally tests
the mean hidden state over the final user turn, both with and without the input-
embedding output eligible for layer selection; it does not replace the
prespecified closing-token analysis.

The primary probe is the normalized difference between the evaluation-class and
deployment-class mean activations. A held-out activation is scored by projection
onto that direction, and AUROC measures ranking performance. Layer selection is
performed using training data only. For grouped probe--text and cross-model
within-control comparisons, it uses pair-grouped cross-validation nested inside
each outer training fold. Most reference-model results use five deterministic
70/30 splits and report mean plus population SD across seeds.

For byte-identical prefixes under deterministic evaluation, exact paired
pre-final hidden-state equality is expected. It is checked before probe fitting
as a diagnostic for construction, rendering, or extraction leakage and is not
treated as inferential evidence about awareness. Scientific comparisons begin
after the texts diverge.

A post-result decoder audit holds each outer fold's primary-selected layer
fixed and fits both `StandardScaler + L2 LogisticRegression` and
`StandardScaler + RBF SVC`. Logistic `C` in `{0.001, 0.01, 0.1, 1}` and RBF
`C` and dimension-normalized kernel width are selected in the same pair-grouped
inner folds. Capacity-matched text
comparators include word/character TF-IDF RBF and frozen E5-view RBF models;
family, representation, and hyperparameters are selected within training folds.
The exact previously strongest E5 prefix-ending interaction is also audited
with nested RBF hyperparameters. The full analysis is repeated over fold seeds
0--4. Transfer selects activation layer, orientation, kernel settings, and text
settings on the source control only. This audit changes decoder capacity without
jointly searching layers for nonlinear separability.

Surface comparisons include length, 13 structural features, unigram counts, and
where noted word 1--2-gram and character-within-word 3--5-gram TF-IDF logistic
classifiers. For byte-identical-prefix pairs, additive text baselines see only
the differing causal message; the shared prefix contains no population-level
label information and otherwise adds finite-sample noise.

A later frozen-before-execution contextual audit uses revision-pinned
`intfloat/e5-base-v2` embeddings with fixed grouped-fold logistic regression.
It reports ending-only and full-conversation embeddings plus an explicit
prefix-ending interaction over all causally available text. The interaction is
the strongest tested full-input linear text view and is therefore the current
main construct-validity comparator. It was prespecified secondary within a
post-result audit, not part of the original confirmatory experiment; that status
is retained.

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
the source construction fixes which score direction means evaluation-like.
Signed target AUROC therefore tests whether that orientation survives a change
in construction; absolute distance from chance tests only target separability.

| Transfer subset | Probe signed AUROC | Unigram signed AUROC | Signed difference, mean ± SD | Majority-oriented, probe / unigram |
|---|---:|---:|---:|---:|
| 20 completely disjoint source-target pairs | 0.65209 | 0.66565 | `-0.01355 ± 0.01961` | 15/20 / 17/20 |
| 10 pairs sharing one dataset cell | 0.91874 | 0.80059 | `+0.11815 ± 0.01335` | 10/10 / 10/10 |
| All 30 off-diagonal pairs | 0.74098 | 0.71063 | `+0.03035 ± 0.01552` | 25/30 / 27/30 |

On the completely disjoint subset, the probe preserves orientation in all five
seeds for 12/20 ordered transfers, versus 15/20 for unigrams. Across the 100
disjoint transfer-by-seed results, 26 probe AUROCs and 13 unigram AUROCs reverse
below `0.5`. Four of the five per-seed signed probe-minus-unigram means are
negative.

Folding away orientation gives the earlier separation-strength summary:

| Transfer subset | Probe strength | Unigram strength | Strength difference, mean ± SD |
|---|---:|---:|---:|
| 20 completely disjoint source-target pairs | 0.22810 | 0.18055 | `+0.04754 ± 0.01538` |
| 10 pairs sharing one dataset cell | 0.41874 | 0.30059 | `+0.11815 ± 0.01335` |
| All 30 off-diagonal pairs | 0.29164 | 0.22057 | `+0.07108 ± 0.01341` |

Thus activation directions preserve substantial separability on new targets,
but the disjoint analysis does not establish a more consistently oriented
evaluation-like direction than unigram transfer. Strength remains a secondary
diagnostic because it counts a perfectly reversed classifier as successful.

### 4. Multi-turn leakage checks and post-divergence decoding

The first written multi-turn cells were rejected as a primary control. Their
surface gate failed, and activation separation already existed after the first
synthetic user turn, before the supposed class-defining final question.

Two repaired 250-pair controls are valid:

- **`eval_shared`:** a real-derived deployment conversation with a byte-
  identical prefix; only the final user turn is replaced by an evaluation
  question. It provides an exact pre-final leakage check but also changes topic
  and speech act at the final turn.
- **`coherent_pairs`:** both alternatives are written together with a byte-
  identical, topically coherent prefix and matched final-turn style. This is a
  stronger coherence control but uses a synthetic prefix.

Reference-model train-only results were:

| Control/read point | Probe AUROC, mean ± SD | Unigram BoW | Interpretation |
|---|---:|---:|---|
| `eval_shared`, final user | `0.99168 ± 0.00330` | 0.93504 | Strong final-turn separation |
| `eval_shared`, pre-final | `0.49847 ± 0.00111` | 0.50000 | Exact-equality leakage check |
| `coherent_pairs`, final user | `0.95132 ± 0.01147` | `0.93337 ± 0.00680` | High but largely lexical |
| `coherent_pairs`, pre-final | `0.49902 ± 0.00152` | 0.50000 | Exact-equality leakage check |

The strongest full-input comparison is authoritative for the current linear
construct-validity claim:

| Control | Probe | E5 prefix-ending interaction | Difference (95% CI) | One-sided p |
|---|---:|---:|---:|---:|
| `eval_shared` | 0.98779 | 0.99554 | `-0.00774` `[-0.01557,-0.00150]` | 0.90055 |
| `coherent_pairs` | 0.94682 | 0.95325 | `-0.00643` `[-0.01850,+0.00590]` | 0.68966 |

The earlier sparse-text audit remains a secondary baseline hierarchy and the
authoritative cross-control transfer analysis:

| Analysis | Probe | Strongest text | Difference (95% CI) | One-sided p |
|---|---:|---:|---:|---:|
| `eval_shared`, within | 0.98779 | word TF-IDF 0.95293 | `+0.03486` `[+0.01833, +0.05395]` | 0.00150 |
| `coherent_pairs`, within | 0.94682 | character TF-IDF 0.94474 | `+0.00208` `[-0.01341, +0.01675]` | 0.43778 |
| `eval_shared` → coherent | 0.84496 | character TF-IDF 0.86614 | `-0.02118` `[-0.04887, +0.00811]` | 0.87856 |
| coherent → `eval_shared` | 0.71874 | character TF-IDF 0.84854 | `-0.12981` `[-0.16654, -0.09081]` | 1.00000 |

Thus the retained-history conclusion reverses when moving from word TF-IDF to
the stronger full-input E5 interaction. Matched coherence is tied by both the
E5 interaction and character n-grams, and cross-control activation transfer
does not beat character TF-IDF.

The contextual view hierarchy explains which specified representation changes
the comparison:

| Control | Probe | Full context (primary) | Ending only | Prefix-ending interaction |
|---|---:|---:|---:|---:|
| `eval_shared` | 0.98779 | 0.82104 | 0.98206 | 0.99554 |
| `coherent_pairs` | 0.94682 | 0.89088 | 0.94165 | 0.95325 |

On matched coherence, probe minus interaction is `-0.00643`, 95% pair-cluster
bootstrap CI `[-0.01850,+0.00590]`; the two are statistically tied. On
`eval_shared`, probe minus interaction is `-0.00774`, CI
`[-0.01557,-0.00150]`. The frozen full-conversation embedding was the
prespecified primary representation and is weaker, showing that simply pooling
context is not sufficient. The fixed secondary interaction representation is
the stronger causal-view comparator and ties or exceeds the probe.

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
`p=1/501=0.001996`. The two identical-prefix controls pass exact pre-final
equality checks; their near-`0.5` summaries and null permutation results
(`p=0.99002` and `p=0.94012`) are secondary consequences, not awareness evidence.

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
12.5% baseline refusal rate, the design had about 80% power only for an increase
to roughly 37%. This is a sensitivity calculation, not an exclusion bound; the
result remains inconclusive for smaller effects.

A post-result mentor-requested audit preserves prompts as the inference unit.
For self-report, the paired `+2` minus `-2` risk difference is `+0.7500`, with
a 95% prompt-bootstrap interval `[+0.5833,+0.8889]` and two-sided exact McNemar
`p=1.12e-7`; the mean within-prompt slope across all seven nondegenerate
coefficients is `+0.1759` per alpha unit (`[+0.1362,+0.2130]`). For safety, the
endpoint risk difference is `-0.0208` (`[-0.0625,0.0000]`, exact `p=1.0`) and
the mean within-prompt slope is `-0.0060` (`[-0.0169,+0.0020]`). Thus the
self-report result survives prompt-level inference while the refusal result
does not show the predicted positive effect.

The same audit freezes five seeded, norm-matched random directions projected
orthogonal to the learned direction. On the independent readout, the learned
direction moves `+1.0447` eval--deploy gap units from `-2` to `+2`; the five
controls range from `-0.0944` to `+0.0842`, with mixed signs. This supports
readout-level direction specificity.

All 840 control-direction behavioral generations are complete. Every endpoint
passes the frozen output-quality screen. The five paired self-report risk
differences are `+0.4167`, `+0.9444`, `+0.3056`, `0.0000`, and `+1.0000`; their
95% prompt-bootstrap intervals are respectively `[+0.2500,+0.5833]`,
`[+0.8611,+1.0000]`, `[+0.1667,+0.4722]`, `[-0.0833,+0.0833]`, and
`[+1.0000,+1.0000]`. Two controls exceed the learned-direction effect of
`+0.7500`, so the frozen behavioral-specificity criterion fails. Safety risk
differences are `-0.0417`, `-0.0208`, `0.0000`, `-0.0417`, and `-0.0417`; none
shows the predicted positive effect.

The intervention causally and specifically moves the independent readout, but
the self-report shift is not learned-direction-specific. The experiment does
not establish causal use of the decoded feature in self-report or broad safety
policy; it instead reveals generic direction-dependent sensitivity of the
self-report endpoint to residual-stream perturbation.

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
nonzero final-turn differences. The former is the required leakage check; the
latter is the post-divergence decoding target.

A post-result audit found that the original grouped within-control runner chose
the direction-of-means layer by in-sample training AUROC. This did not consult
the outer test fold, but high-dimensional training AUROCs saturated and made
Gemma's choice unstable. The corrected analysis applies five-fold pair-grouped
layer selection inside every outer training fold, uniformly across all models
and endpoints. Transfer already selected layers by grouped source-only CV and
is unchanged.

Primary linear comparisons with the same full-input E5 interaction were:

| Model | Shared: probe / difference (95% CI) | Coherent: probe / difference (95% CI) |
|---|---:|---:|
| Llama-3.1-8B | 0.98779 / -0.00774 [-0.01557,-0.00150] | 0.94682 / -0.00643 [-0.01850,+0.00590] |
| Qwen3-8B | 0.99192 / -0.00362 [-0.00878,+0.00062] | 0.94504 / -0.00821 [-0.02277,+0.00621] |
| Gemma-3-12B-IT | 0.97296 / -0.02258 [-0.03572,-0.01134] | 0.85902 / -0.09422 [-0.12079,-0.06841] |
| Ministral-3-8B | 0.98309 / -0.01245 [-0.02088,-0.00581] | 0.92790 / -0.02534 [-0.04490,-0.00504] |

The E5 AUROCs are `0.99554` and `0.95325`, respectively. None of the
primary linear probes outperforms this specified baseline; five intervals favor
E5 and three include zero.

The uniform alternate-readout audit was:

| Model | Shared: token / mean | Coherent: token / mean (all) / mean (no embedding) |
|---|---:|---:|
| Llama-3.1-8B | 0.98779 / 0.99158 | 0.94682 / 0.93218 / 0.93251 |
| Qwen3-8B | 0.99192 / 0.97850 | 0.94504 / 0.94392 / 0.93323 |
| Gemma-3-12B-IT | 0.97296 / 0.94883 | 0.85902 / 0.93877 / 0.63800 |
| Ministral-3-8B | 0.98309 / 0.98690 | 0.92790 / 0.87750 / 0.93803 |

For matched coherence, unrestricted mean pooling increases Gemma by `+0.07974`
(95% CI `[+0.05630,+0.10372]`) but decreases Ministral by `-0.05040` (CI
`[-0.07098,-0.02880]`). Gemma and Qwen select embedding layer zero in all five
unrestricted-mean outer folds, Llama in three, and Ministral in one. Once layer
zero is uniformly excluded, Llama, Qwen, and Ministral cluster at 0.93251--
0.93803, while Gemma falls to 0.63800 with selected layers `[4,4,48,7,48]`.
Thus mean pooling is not a uniformly stronger readout, and the apparent Gemma
alignment depends on a lexical average of input embeddings rather than a
transformer-computed state. The paper retains the prespecified closing-token
readout rather than selecting a post-result readout for visual alignment.

The initial sparse-text nonlinear-capacity audit was:

| Model | Shared: linear / RBF / RBF-minus-text | Coherent: linear / RBF / RBF-minus-text |
|---|---:|---:|
| Llama-3.1-8B | 0.98779 / 0.99784 / +0.04491 | 0.94682 / 0.98698 / +0.04037 |
| Qwen3-8B | 0.99192 / 0.99920 / +0.04627 | 0.94504 / 0.98786 / +0.04125 |
| Gemma-3-12B-IT | 0.97296 / 0.98986 / +0.03693 | 0.85902 / 0.98112 / +0.03451 |
| Ministral-3-8B | 0.98309 / 0.99877 / +0.04584 | 0.92790 / 0.98712 / +0.04051 |

This first pass used linear word TF-IDF at `0.95293` for `eval_shared` and
nested nonlinear TF-IDF at `0.94661` for matched coherence. Every displayed
RBF-minus-text 95% pair-bootstrap interval excludes zero. On
matched coherence, RBF-minus-linear is `+0.04016` for Llama, `+0.04282` for
Qwen, `+0.12210` for Gemma, and `+0.05922` for Ministral, again with all four
intervals positive.

The later same-layer decoder comparison shows that these differences should
not be interpreted as wholly nonlinear. Under the identical nested grouped
pipeline at fold seed 0:

| Model | Shared: direction-of-means / L2 / RBF | Coherent: direction-of-means / L2 / RBF |
|---|---:|---:|
| Llama-3.1-8B | 0.98779 / 0.99483 / 0.99784 | 0.94682 / 0.98722 / 0.98698 |
| Qwen3-8B | 0.99192 / 0.99834 / 0.99920 | 0.94504 / 0.98627 / 0.98786 |
| Gemma-3-12B-IT | 0.97296 / 0.98755 / 0.98986 | 0.85902 / 0.97566 / 0.98112 |
| Ministral-3-8B | 0.98309 / 0.99770 / 0.99877 | 0.92790 / 0.98494 / 0.98712 |

On matched coherence, seed-0 L2-minus-direction-of-means paired-bootstrap
intervals exclude zero for all four models. RBF-minus-L2 is `-0.00024`
`[-0.00341,+0.00266]`, `+0.00158` `[-0.00094,+0.00438]`, `+0.00546`
`[-0.00040,+0.01234]`, and `+0.00218` `[-0.00110,+0.00534]` for Llama,
Qwen, Gemma, and Ministral, respectively. Across fold seeds 0--4, mean
coherence-matched L2 AUROC is `0.98746`, `0.98641`, `0.97402`, and `0.98644`;
mean RBF-minus-L2 is `+0.00090`, `+0.00225`, `+0.01036`, and `+0.00325`.
The robust conclusion is sensitivity to learned multivariate readout, with at
most a small additional RBF increment in this audit.

The completed follow-up strengthens the text side with the previously strongest
frozen E5 prefix-ending interaction. With nested RBF hyperparameters, this text
model reaches `0.99533` on shared prefix and `0.95659` on matched coherence at
fold seed 0. The activation-minus-E5 RBF results are:

| Model | Shared RBF / difference (95% CI) | Coherent RBF / difference (95% CI) |
|---|---:|---:|
| Llama-3.1-8B | 0.99784 / +0.00251 [-0.00134,+0.00639] | 0.98698 / +0.03038 [+0.01491,+0.04610] |
| Qwen3-8B | 0.99920 / +0.00387 [+0.00106,+0.00741] | 0.98786 / +0.03126 [+0.01643,+0.04650] |
| Gemma-3-12B-IT | 0.98986 / -0.00547 [-0.01322,+0.00106] | 0.98112 / +0.02453 [+0.01003,+0.03879] |
| Ministral-3-8B | 0.99877 / +0.00344 [+0.00080,+0.00682] | 0.98712 / +0.03053 [+0.01578,+0.04591] |

Thus activation RBF exceeds E5-interaction RBF under matched coherence in every model, but
shared prefix remains heterogeneous: Qwen and Ministral have small positive
intervals, Llama ties, and Gemma is numerically below E5 with an interval that
crosses zero. Across fold seeds 0--4, matched-coherence activation RBF means are
`0.98438`--`0.98970`, and activation-minus-E5 means are `+0.03476`--`+0.04008`;
every model's margin is positive in every seed. Shared-prefix mean margins are
Llama `+0.00201`, Qwen `+0.00347`, Gemma `-0.00707`, and Ministral `+0.00242`.

Source-only nonlinear cross-control transfer does not improve the specified
probe-versus-text comparison. Mean activation RBF AUROC over the five selection
seeds is `0.71455`--`0.80612` for shared-to-coherent and `0.70794`--`0.84204`
for coherent-to-shared across models. The source-selected nonlinear text
pipeline reaches `0.82683` and `0.82990`, respectively, while the existing
linear character TF-IDF reaches `0.86614` and `0.84854`. Near-ceiling
within-control nonlinear decoding therefore does not establish a
construction-invariant nonlinear signal. The paired labels also remain
semantically confounded with their differing final messages.

Against the secondary sparse comparator, Qwen and Ministral have positive
shared-prefix probe-minus-word-TF-IDF intervals, while Gemma's includes zero.
Those results do not override the stronger full-input comparison above. No
replication model beats character TF-IDF with the primary linear probe on
matched coherence or in either cross-control transfer direction.

Together with Llama, this establishes cross-family post-divergence decodability.
Primary linear probes do not beat the full-input E5 interaction; activation RBF
exceeds E5-interaction RBF only within matched coherence, while transfer does
not beat specified text baselines and semantic identity remains unestablished.

#### Within-family Qwen3 scale ladder

The scale manifest was frozen before new activations were extracted and lists
all six original dense Qwen3 checkpoints. Every checkpoint uses the same 250
pairs per control, grouped folds, last-user-token readout, bfloat16 precision,
single 80GB GPU environment, and common text comparison. The re-extracted
8B probe AUROCs differ from the historical run by `+0.00011` (shared prefix)
and `+0.00158` (matched coherence), below the prespecified `0.01` audit trigger.

Linear probe-minus-E5-interaction margins were:

| Qwen3 size (exact B) | Shared prefix difference (95% CI) | Matched coherence difference (95% CI) |
|---|---:|---:|
| 0.75 | -0.0283 [-0.0415,-0.0160] | -0.0120 [-0.0273,+0.0028] |
| 2.03 | -0.0072 [-0.0149,-0.0008] | -0.0146 [-0.0308,+0.0010] |
| 4.02 | -0.0085 [-0.0150,-0.0024] | -0.0131 [-0.0273,+0.0011] |
| 8.19 | -0.0035 [-0.0087,+0.0008] | -0.0066 [-0.0212,+0.0076] |
| 14.77 | -0.0429 [-0.0563,-0.0302] | +0.0007 [-0.0131,+0.0146] |
| 32.76 | +0.0009 [-0.0028,+0.0048] | -0.0066 [-0.0218,+0.0093] |

The paired-scenario bootstrap resamples the same scenario IDs at all six sizes
and refits the OLS slope against log10 exact parameter count. Subtracting one
common E5 prediction vector at every size leaves the slope from the frozen
sparse-baseline analysis unchanged. Shared prefix has
slope `+0.0054` with interval `[-0.0017,+0.0129]`; matched coherence has slope
`+0.0068` with interval `[+0.0013,+0.0132]`. Under the frozen rule, these do not
constitute a consistent trend because the interval excludes zero on only one
of the two primary controls. Secondary transfer is also control-dependent:
shared-to-coherent declines with size (`-0.0529`
`[-0.0664,-0.0389]`), while coherent-to-shared is flat (`-0.0002`
`[-0.0156,+0.0149]`). Six checkpoints therefore test this Qwen3 family and
pipeline, not a universal scaling law.

The five-seed nonlinear sensitivity result does not reveal a hidden size trend.
Activation RBF minus fixed contextual E5 ranges from `-0.0024` to `+0.0036` on
shared prefix and from `+0.0358` to `+0.0383` on matched coherence across all
sizes. Thus nonlinear capacity changes the control-specific comparison at every
scale, while the paper's heterogeneity and semantic non-identification claims
remain intact.

## Supported claims

The repository supports these statements when their qualifications remain
attached:

1. Llama-3.1-8B hidden states strongly distinguish multiple studied eval-like
   and deploy-like constructions.
2. Simple benchmark-versus-user separation is heavily explained by surface
   form.
3. On Slice 2b, completely disjoint math-free activation transfer preserves
   substantial target separability, but mean signed AUROC is `0.65209` for the
   probe versus `0.66565` for unigrams, and probe orientation is less stable.
4. Two paired multi-turn controls pass exact pre-final equality checks,
   verifying construction and extraction integrity; labels become decodable
   after the first differing user message.
5. None of the eight primary linear model-control comparisons beats the
   full-input E5 interaction; five paired intervals favor E5 and three include
   zero. Positive shared-prefix margins against word TF-IDF are secondary
   sparse-text comparisons.
6. Under an identical grouped nested pipeline and fold-selected layer, L2
   logistic recovers most of the RBF gain over direction-of-means; all seed-0
   RBF-minus-L2 intervals include zero, and five-seed mean differences are
   `+0.00090`--`+0.01036`. A nested RBF sensitivity probe still beats a
   RBF over the same E5 interaction on matched coherence for all four models;
   activation AUROC is 0.98112--0.98786, and the advantage is positive in every
   one of five fold seeds. Shared-prefix nonlinear margins remain heterogeneous.
7. Neither linear nor nonlinear cross-control activation transfer beats the
   strongest tested text baselines.
8. The frozen E5 prefix-ending interaction is the strongest tested full-input
   linear text view and matches or exceeds every primary linear probe.
9. High raw separation remains among terminated replies with no detected
   awareness disclosure, although matched-coherence activation and lexical
   baselines tie.
10. Genuine and reconstructed replies are separable across WildChat and LMSYS,
   but controls rule out a narrow self-authorship interpretation.
11. Main activation effects survive a second estimator and full-pipeline
   permutation tests.
12. Learned-direction steering specifically moves an independent activation
    readout, but its self-report shift is not direction-specific: two of five
    norm-matched orthogonal controls exceed it. No direction produces the
    predicted positive safety-refusal effect.
13. Four tested checkpoints pass the exact pre-final leakage check and show
    post-divergence decodability and model-by-readout sensitivity without a
    primary linear advantage over the specified E5 baseline.
14. Across all six dense Qwen3 sizes from 0.75B to 32.76B exact parameters, the
    prespecified two-control criterion finds no consistent primary scale trend;
    nonlinear activation-versus-E5 margins are also nearly flat across size.

## Unsupported or overstated claims

The repository does not support the following:

- The proposal was executed as written.
- The final Slice 2b cells are surface matched.
- Slice 2b establishes a consistently oriented evaluation-like activation
  direction that transfers better than unigram text.
- The first written multi-turn cells prove temporal awareness.
- The model “knows it is being evaluated” as a uniquely identified semantic
  interpretation of the direction.
- The primary direction-of-means matched-coherence probe meaningfully beats
  text in-distribution; only post-result learned-decoder sensitivity analyses
  establish an advantage for the tested within-control comparison.
- The RBF recovery demonstrates specifically nonlinear geometry. L2 logistic
  recovers most of the same-layer gain, and no seed-0 RBF-minus-L2 interval
  excludes zero.
- Cross-control activation transfer beats character TF-IDF or the
  source-selected nonlinear text pipeline.
- Observed silence proves unconscious or unverbalized awareness.
- Slice 5 demonstrates self-recognition.
- Steering improves safety behavior or establishes a broad policy mechanism.
- The learned direction specifically causes the self-report shift. Two of five
  norm-matched orthogonal controls produce larger effects despite the learned
  direction's specific independent-readout manipulation check.
- The effect is invariant to history authorship; the relevant pilot failed
  before activation extraction.
- The effect has architecture-invariant strength, a consistent trend across
  both Qwen3 controls, or a universal scaling law.
- The project measured agentic tool use or a real positive-disclosure class.

## Audit rules for paper work

1. Use nested pair-grouped, train-only layer selection for grouped
   within-control comparisons; do not use the superseded in-sample training
   AUROC selector or revive historical held-out layer selection.
2. Report the completely disjoint Slice 2b transfer subset separately from
   shared-cell and all-off-diagonal averages. Signed target AUROC and orientation
   consistency are primary; folded separation strength is secondary and must be
   labeled as discarding orientation.
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
13. For the Qwen3 ladder, report every frozen size and both primary controls.
    Call a scale trend consistent only if both paired-bootstrap slope intervals
    exclude zero in the same direction; six checkpoints are not a universal
    scaling law.
