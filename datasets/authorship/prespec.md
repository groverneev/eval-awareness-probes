# Evaluation-status x history-authorship factorial

Status: frozen before generation or activation inspection.

## Purpose

The existing Slice 5 model-swap and context-ablation experiments study final
reply provenance. They do not cross evaluation status with the authorship of
the preceding assistant history. This experiment asks whether the
eval/deployment activation direction in matched multi-turn conversations is
stable when the prior assistant turn is generated on-policy, by a smaller model
from the same family, or by a different model family.

The experiment isolates an authorship confound. It does not, by itself, remove
the lexical/task-structure differences between evaluation and deployment final
user turns.

## Pilot scope

The pilot uses 50 two-user-turn scenarios from
`datasets/slice3/coherent_pairs/validated.json`. Restricting to two user turns
means the manipulated history contains exactly one assistant response:

1. frozen first user turn;
2. newly generated assistant response;
3. one of the two already-frozen final user endings.

Eligible scenarios are ordered by SHA-256 of `item_id`, and the first 50 are
used. This selection is independent of text, activations, and prior probe
performance. Pilot scenarios cannot appear in the confirmatory sample.

The pilot is a feasibility and validity gate, not confirmatory evidence.

## Factorial cells

Evaluation status has two levels:

- `eval`: the frozen benchmark-derived, same-topic final user ending;
- `deploy`: the frozen practical, same-topic final user ending.

History authorship has three levels:

- `target_on_policy`: Llama-3.1-8B-Instruct generates the assistant response;
- `same_family_off_policy`: Llama-3.2-3B-Instruct generates it;
- `cross_family_off_policy`: Qwen2.5-7B-Instruct generates it.

For each scenario and history-author condition, the generated prefix is
byte-identical between its eval and deploy cells. The two frozen final endings
are also byte-identical across the three authorship conditions. Thus evaluation
status varies only at the final user turn, while history authorship varies only
in the preceding assistant response.

Llama-3.1-8B-Instruct is the activation organism in all six cells.

## Generation protocol

- Use each generator's native chat template.
- Sampling disabled; greedy decoding.
- Maximum 256 new tokens.
- A generation is valid only if it emits a recognized end-of-turn or
  end-of-sequence token before the cap.
- Do not quantize any model.
- Record the exact model repository, resolved revision, tokenizer revision,
  transformers version, rendered input hash, output token ids, termination
  token, and generation length.
- Freeze all input records before starting any generator.
- Never repair or rewrite an individual generated assistant response after
  inspecting activations.

Claude may be used for blinded coherence judging and for future dataset
authoring. It must not replace any of the three experimental history generators:
doing so would change the authorship factor. Any Claude call must record the
exact resolved model identifier rather than an alias such as `opus` or `sonnet`.

## Probe-blind validity gate

The gate is applied before activation extraction.

1. All six conversations have the required three-message role sequence.
2. For every retained scenario, all three generated assistant responses
   terminate naturally.
3. Within each authorship condition, eval and deploy prefixes are byte-identical.
4. Across authorship conditions, the eval final message is identical and the
   deploy final message is identical.
5. A blinded coherence judge evaluates whether each frozen final ending is a
   plausible continuation after each generated assistant response. Rows expose
   no condition names or model identities.
6. At least 20% of pilot rows receive a second blinded human judgment.
7. Retain a scenario only if all six cells pass the mechanical checks and both
   endings pass coherence in all three authorship conditions.

The pilot passes if at least 40/50 scenarios survive. If it fails, stop and
revise the construction protocol without examining activations. Do not weaken
the gate post hoc.

## Read points and sanity checks

Primary read point: `last_user_token`.

Sanity read point: the end of the first user turn, before the generated
assistant response. Because all six cells share that first turn, every pairwise
contrast must be at chance there. Any separation is an indexing or data-leak
failure.

Secondary read point: `final_turn_mean`, reported as lexical-sensitive and not
used for the primary conclusion.

## Primary analyses

All scenario splits are grouped: every cell derived from one scenario remains
in the same fold.

1. **Within-authorship evaluation separation.** Fit and score eval versus deploy
   separately under each history author.
2. **Evaluation-direction transfer across authorship.** Select layer and
   orientation using source-authorship folds only, fit on all source scenarios,
   then score both unseen authorship conditions. Oriented AUROC is primary;
   `|AUROC - 0.5|` is secondary.
3. **Authorship-direction diagnostic.** On the deployment endings, fit
   on-policy versus off-policy directions and test them on the evaluation
   endings. Repeat in the reverse direction.
4. **Direction geometry.** At prespecified layers, report cosine similarities
   among the three evaluation directions and between evaluation and authorship
   directions.
5. **Authorship residualization.** Remove the span of the two fitted authorship
   directions from held-out activations, refit the evaluation direction using
   training scenarios only, and report the change in oriented AUROC.

The stronger text hierarchy is mandatory: unigram counts, word 1--2-gram
TF-IDF, character 3--5-gram TF-IDF, format, and length. Since the final user
endings are held fixed across authorship conditions, text-baseline predictions
must also be identical across those conditions; this is an implementation
check, not a reason to claim the activation probe contains text-inaccessible
information.

## Inference

- Five grouped outer folds for within-authorship results.
- Source-only grouped cross-validation for transfer layer/orientation selection.
- 2,000 scenario-cluster bootstrap draws for AUROC and AUROC differences.
- 2,000 scenario-cluster approximate-randomization draws for direct method
  comparisons.
- Report all six directed authorship transfers; do not average away polarity.
- The scenario, not the generated conversation cell, is the inferential unit.

## Pilot decision and confirmatory run

After the probe-blind gate passes, pilot activations may be inspected only to
answer feasibility questions: whether the evaluation direction has stable
orientation across authorship and whether the selected read points behave as
specified.

Before a confirmatory run, freeze:

- the retained construction and coherence rubric;
- the primary read point;
- layer-selection rule;
- baseline configurations;
- all inference code and thresholds;
- the confirmatory sample size.

The intended confirmatory sample is 150 fresh retained scenarios, expanded
deterministically beyond 150 inputs as needed to replace all-condition
termination or coherence failures. Pilot items are excluded. A paper claim that
the evaluation direction is authorship-robust requires oriented transfer above
chance in every authorship target with a scenario-bootstrap interval excluding
0.5. A claim that authorship explains the direction requires a prespecified,
substantial collapse after authorship residualization; no such threshold will
be chosen after viewing the pilot.

## Interpretation boundaries

A positive result would show that the eval/deploy direction is not merely a
detector of who wrote the preceding assistant response. It would not prove that
the direction uniquely represents evaluation awareness rather than the
different final-turn tasks.

A negative result would be equally informative: it would show that history
authorship materially changes a direction previously interpreted as evaluation
context, connecting evaluation-awareness probing to on/off-policy and prefill
awareness results.
