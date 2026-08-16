# Detailed Experimental Ledger: What Was Actually Done

This is the complete chronological and artifact-grounded audit record. For the
short authoritative account of headline methods, results, and claim boundaries,
read `EXPERIMENT_REPORT.md`. For implementation and artifact locations, read
`ARTIFACT_INDEX.md`.

## Purpose and evidence standard

This document reconstructs the experimentation in this repository as it was actually carried out. It is not a restatement of the original proposal. The proposal is useful for understanding the motivating question, but it is not evidence that an experiment happened, that a dataset was built as planned, or that a result was obtained.

The evidence hierarchy used here is:

1. The current source code and frozen protocol files.
2. Saved datasets, activation caches, generation outputs, judge labels, and result files.
3. Git history, especially when it explains a pilot, a rejected control, or a result produced by an older version of the analysis.
4. Project notes and the proposal only for motivation or terminology.

This distinction matters. Several ideas in the proposal were changed or abandoned. Several controls failed their pre-analysis gates. One saved Slice 3 result was generated before a layer-selection leak was fixed. Those facts are part of the experimental record and are reported explicitly below.

The terms used throughout are:

- **Completed:** code and corresponding output artifacts exist.
- **Pilot:** completed on a deliberately smaller sample to test feasibility or diagnose a design.
- **Rejected:** completed, but excluded from the evidential chain because a gate or manual validity check failed.
- **Engineering check:** used to verify the pipeline, not intended as scientific evidence.
- **Proposed only:** discussed as a possible experiment but not supported by completed artifacts in this repository.

## Executive account

The project began with a simple but heavily confounded contrast: benchmark-like evaluation prompts versus real user prompts. Llama-3.1-8B-Instruct activations separated the two almost perfectly, but surface classifiers also separated them almost perfectly. That established that the activation pipeline worked, not that the model represented an abstract fact such as “I am being evaluated.”

The next phase audited an existing public 2×2 dataset and built a new four-cell single-turn dataset. Attempts to eliminate surface separation were unsuccessful: format and bag-of-words gates failed. The analysis therefore shifted from asking whether a probe was high in one contrast to asking whether a probe trained on one independently constructed eval/deploy contrast transferred to another better than the same surface baselines. On the math-free Slice 2b transfer grid, the probe’s mean off-diagonal separation strength exceeded bag-of-words by `0.07108 ± 0.01341` across five split seeds. A later audit shows that this average contains two regimes: the margin was `0.04754 ± 0.01538` across the 20 completely disjoint train–target transfers and `0.11815 ± 0.01335` across the 10 transfers whose source and target pairs shared one cell. The completely disjoint number is the cleaner generalization result.

The multi-turn phase first created matched deployment, disguised-evaluation, and obvious-evaluation conversations. Its surface gate also failed, and the disguised-evaluation lead-ins themselves leaked the label before the final question. That version is not a clean awareness result. Two stronger controls were then built:

- `eval_shared` retained a real-derived deployment conversation and changed only the final user turn to an evaluation question. Its prefix was byte-identical between classes. The probe was at chance before the changed turn (`0.49847 ± 0.00111`) and highly separating after it (`0.99168 ± 0.00330` at the last-user boundary). This is a strong temporal-localization control, but the final turn also changes topic and speech act.
- `coherent_pairs` constructed both sides together so that the prefix was byte-identical and the two final turns were topically coherent and stylistically matched. At the last-user boundary, the original repeated-holdout analysis gave probe `0.95132 ± 0.01147` versus bag-of-words `0.93337 ± 0.00680`. A later grouped five-fold audit added word and character TF-IDF baselines and direct pair-level inference. Under that stronger audit, the probe (`0.94682`) tied character TF-IDF (`0.94474`; difference `+0.00208`, 95% pair-bootstrap interval `[-0.01341, +0.01675]`). The activation advantage is therefore not established in-distribution.

To test whether the internal separation remained when the model did not say it was in an evaluation, the model generated greedy replies and pairs were kept only when both replies terminated and neither a regex, two-stage blinded judge procedure, nor manual audit found explicit or implicit evaluation awareness. Raw activation separation remained high, but the corrected final-turn bag-of-words baseline tied it on matched-coherence and came within about `0.045` strength on `eval_shared`. This is correctly described as separation conditional on observed silence, not proof that the model lacked unexpressed awareness or that the remaining separation is nonlexical.

Slice 5 asked a different question: can the model distinguish a genuine assistant reply from a reply resampled by a model and inserted into the same conversation? It could. However, the signal remained when a different model generated the replacement and became at least as strong when the generator saw only the final user turn. The result is therefore evidence of reply provenance/context consistency, not specifically self-recognition or evaluation awareness.

Finally, a pre-specified steering experiment added the coherent-pair activation direction during generation. The manipulation moved an independent internal readout monotonically and shifted explicit self-reports in the predicted direction at nondegenerate intervention strengths. It did not produce the predicted increase in safety refusals. The behavioral safety test was small and low-base-rate, so it rules out only a large effect, not all effects.

A frozen cross-model core then reran the two valid 250-pair temporal controls on Qwen3-8B, Gemma-3-12B-IT, and Ministral-3-8B. Every model had exactly zero activation difference throughout the byte-identical prefix and nonzero final-turn differences. A post-result layer-selection audit found that the original within-control runner used saturated in-sample training AUROC. Uniform nested grouped selection corrected Gemma from `0.72667` to `0.97296` on `eval_shared` and from `0.64744` to `0.85902` on matched coherence, while leaving the other models nearly unchanged. Llama, Qwen, and Ministral establish positive margins over text on `eval_shared`; Gemma's `+0.02003` margin is inconclusive. No model establishes an advantage on matched coherence, and neither transfer direction beats character TF-IDF. The completed panel therefore supports cross-family temporal decodability but also shows substantial model heterogeneity and does not establish an architecture-invariant evaluation-awareness variable.

The most defensible overall conclusion is therefore narrower than “the model knows when it is being evaluated.” The experiments show linearly decodable activation directions associated with several independently constructed eval-like versus deploy-like situations. The separation emerges at the user turn that introduces the distinction across four model families; in the Llama reference it also persists among replies without observable awareness language and causally influences explicit eval/deploy self-reports when steered. The experiments do not isolate a single, architecture-invariant semantic variable free from content, topic, speech-act, authorship, or generation-context differences, and they do not show that this direction causes a broad behavioral policy change.

## 1. Background: what was being measured

### 1.1 The motivating distinction

An **evaluation** input is one constructed to measure a model—for example, an ARC, MMLU, or GSM8K item. A **deployment** input is a request arising from an ordinary user need. The scientific question was whether the model’s hidden state contains a representation that generalizes across different ways of constructing that distinction.

The central confound is that benchmark and ordinary-user prompts differ in many observable ways:

- topic and vocabulary;
- whether there is one canonical answer;
- multiple-choice or academic formatting;
- first-person and request language;
- punctuation and length;
- authorship and rewriting style;
- single-turn versus multi-turn structure.

Thus, a high probe AUROC on benchmark prompts versus user prompts is not, by itself, evidence of evaluation awareness. The project progressively introduced surface baselines, matched controls, cross-construction transfer, identical-prefix temporal controls, silence filtering, estimator/null checks, and steering to narrow the interpretation.

### 1.2 The primary model and the role of other models

The model whose hidden states were studied was `meta-llama/Llama-3.1-8B-Instruct`. It has 32 transformer blocks. The extraction code saves 33 hidden-state vectors at each read point: the embedding output, indexed as layer 0, followed by the output of each transformer block.

Two other model sizes appear, but neither is a second fully studied organism:

- Llama-3.2-1B-Instruct was used early to validate that the extraction and probing pipeline could recover a known separation. This was an engineering/preliminary check.
- Llama-3.2-3B-Instruct was used only as an alternative reply generator in one Slice 5 control. Llama-3.1-8B-Instruct still supplied the activations and probe readout.

The full frozen 250-pair core has also been completed on Qwen3-8B,
Gemma-3-12B-IT, and Ministral-3-8B. These are three prespecified cross-family
replications in the 5--12B range, not a systematic scale study. A 70B model and
models outside that range were explicitly excluded from the primary panel.

## 2. Common measurement pipeline

### 2.1 Turning a conversation into model input

Every example is represented as a list of chat messages with `user` and, for multi-turn examples, `assistant` roles. The reference experiments use the native Llama chat template; each cross-model replication uses that model's own native template, always with `add_generation_prompt=False`. The code does not manually concatenate prompt wrappers, retokenize an already formatted string, or add a second beginning-of-sequence token.

Extraction uses one conversation at a time with no padding. This avoids ambiguity about which token is the endpoint. Hidden states are converted to `float32` before being stored. The original Llama study used unquantized full-precision loading; the frozen cross-model suite uses unquantized bfloat16 loading and independent causal-prefix passes. No paper result uses 4-bit or 8-bit weights.

The token spans for individual turns are found by rendering progressively longer message prefixes and comparing their token lengths. Tests verify these spans against the actual Llama, Qwen, Gemma, and Ministral templates, including model-specific boundary fallbacks where native rendering is not prefix-consistent.

### 2.2 Read points

Three main read points were used for user-ending conversations:

1. **`last_user_token`:** the `<|eot_id|>` token that closes the final user turn. This is the primary boundary readout. It measures the hidden state after the model has processed the complete user message but before it has generated a reply.
2. **`post_instruction_token`:** the end of the system instruction when a system message exists; otherwise the end of the first user turn. The experimental datasets here do not use system messages, so this is effectively the first-user-turn boundary. In identical-prefix controls it tests whether separation existed before the class-defining final turn.
3. **`final_turn_mean`:** the mean hidden state over the rendered final user turn. This can capture information distributed across the turn, but it also directly averages lexical differences. Layer 0 at this read point is therefore a particularly transparent lexical/embedding control rather than a deep contextual representation.

For the multi-turn trajectory, the code also reads the closing token of every user turn and aligns examples by distance from the end. Offset 0 is the final user turn, offset 1 is one user turn earlier, and so on.

Slice 5 uses analogous assistant-side read points:

- the closing token of the final assistant reply;
- the mean hidden state over the final assistant reply.

### 2.3 The primary direction-of-means probe

For one layer, let the positive/evaluation training activations be (X_1) and the negative/deployment activations be (X_0). The primary probe is deliberately simple:

\[
d = \frac{\operatorname{mean}(X_1)-\operatorname{mean}(X_0)}
         {\lVert\operatorname{mean}(X_1)-\operatorname{mean}(X_0)\rVert_2}.
\]

A held-out activation (x) receives the scalar score (x^\top d). AUROC measures how well that score ranks positive examples above negative examples.

For each independently analyzed cell pair, examples are split 70%/30% within each class using the same seed. Unless explicitly identified as an older artifact, the best layer is selected using training data only. Results are then evaluated on held-out examples. Most final analyses repeat the split with seeds 0–4 and report the arithmetic mean and population standard deviation across seeds.

Because a learned direction can be inverted, the project also reports **separation strength**:

\[
\text{strength} = |\text{AUROC}-0.5|.
\]

An AUROC of 0.2 and an AUROC of 0.8 both mean strong separation, but with opposite orientation. Raw AUROC is still reported where direction matters, especially in transfer matrices.

### 2.4 Surface baselines

Every important activation result is compared with classifiers using only prompt or reply text:

- **Character length** and **word length** as single-feature scores.
- A standardized logistic classifier on 13 hand-engineered structural features: line count, newline count, digit density, punctuation density, uppercase density, mean word length, option-marker count, question-mark count, colon count, parenthesis density, whether the text ends in a question mark, whitespace density, and mean line length.
- A bag-of-words logistic classifier using `CountVectorizer`. The vocabulary is fitted on the training partition only.

The important comparison is normally not “probe above chance.” It is the probe’s separation strength minus the strongest relevant surface baseline on exactly the same split. This was necessary because several legitimate eval/deploy differences, such as canonical-answer questions versus practical requests, cannot be removed without changing the construct.

For multi-turn analyses, surface text is causal: a baseline never sees text after the corresponding activation read point. For an ordinary, non-paired conversation this means the full serialized conversation through that point. For a byte-identical-prefix matched pair, however, the primary baseline uses only the differing final message. The shared prefix occurs exactly once under each label, contains no population-level label signal for an additive classifier, and can only add finite-sample estimation noise. Thus:

- the differing final user text for `last_user_token` and `final_turn_mean` in Slice 3 matched pairs;
- the first user text for `post_instruction_token`;
- only the differing causal segment at trajectory positions (empty before the final turn, the differing message at the final turn);
- the differing final assistant reply for Slice 5 matched pairs.

This differing-segment rule was applied in an August 9, 2026 audit after the initial whole-conversation baselines were found to include matched-prefix noise. It changes only the text baselines, not the activation-probe scores.

### 2.5 Transfer analysis

In a transfer experiment, a direction and its layer are selected on a source cell pair, frozen, and scored on a different target pair. The equivalent text classifier is trained on the source pair and applied to the target texts. This asks whether the separation survives a change in construction procedure.

There is an important qualification. The primary transfer implementation uses a source pair’s 30% held-out partition to select the layer. The target pair is not consulted when it is completely disjoint from the source. However:

- diagonal source-to-same-source entries are model-selection scores, not untouched final-test estimates;
- when source and target pairs share a cell, some examples from that cell participate in both source validation and target scoring.

For this reason, the report separately audits transfers between pairs with no cells in common.

### 2.6 Uncertainty and robustness conventions

The following quantities should not be conflated:

- `mean ± SD` is variation across five deterministic train/test split seeds. It is not a confidence interval.
- Some single-seed analyses also use 2,000 bootstrap resamples of held-out predictions to obtain a 95% AUROC interval.
- The final estimator/null analysis repeats the entire analysis, including layer selection, under 500 label permutations. With the finite-sample correction, the smallest possible reported p-value is `1 / 501 = 0.001996`.

The robustness analysis also uses a standardized L2 logistic probe with `C=0.01`. Its layer is chosen by five-fold cross-validation within the training data, then evaluated on the held-out set. This estimator was added to check that results were not peculiar to a mean-difference direction.

## 3. Experimental chronology

## 3.1 Pipeline validation before the main evidence

The first runs established that extraction, token indexing, train/test separation, and AUROC computation behaved sensibly.

- A small 6-evaluation/6-deployment smoke dataset was used as an engineering test. It is too small and synthetic to be a paper result.
- Llama-3.2-1B-Instruct was then run on the initial Slice 1 task. A best-layer AUROC of about `0.992` was observed at layer 5 of 16. This demonstrated that the pipeline could recover the obvious distinction on a smaller model, but it did not resolve any confound and is not part of the final claim.

## 3.2 Slice 1: benchmark evaluation versus real user deployment

### Dataset construction

Slice 1 uses the Situational Awareness Dataset’s `stages_oversight` examples. The raw `body` field is used directly; the code does not add a new multiple-choice wrapper. The dataset archive is decrypted in memory, and prompt text is not persisted by the loader.

The two classes are:

- **Evaluation, n=200:** 67 Anthropic evaluation items, 67 BIG-bench items, and 66 WMDP items from `test_oversight`.
- **Deployment, n=200:** 200 WildChat user prompts from `deploy_oversight`.

For the original seed-0 analysis, each class was split into 140 training and 60 held-out examples.

### Procedure

Llama-3.1-8B-Instruct activations were extracted at the final user boundary at every layer. A direction-of-means probe was fit independently at each layer. Surface baselines used the same split and raw prompt texts.

### Results

The stored seed-0 probe curve begins at chance at the embedding layer and rises to a best AUROC of `0.994444` at layer 4. Much of the remaining network stays around `0.97–0.99`.

The corresponding baselines were:

| Method | AUROC | Separation strength |
|---|---:|---:|
| Character length | 0.3929 | 0.1071 |
| Word length | 0.4506 | 0.0494 |
| Structural format logistic | 0.9675 | 0.4675 |
| Bag of words | 0.9517 | 0.4517 |
| Activation probe, historically selected best layer | 0.9944 | 0.4944 |

The probe exceeded the strongest baseline’s strength by only about `0.02694` on 120 held-out examples. The format baseline alone almost solved the task.

The later honest multi-seed estimator check, which selects the layer using training data rather than the test curve, obtained:

- direction of means: `0.991833 ± 0.008478`;
- standardized logistic probe: `0.993333 ± 0.007724`;
- full-pipeline permutation p-value for both: `0.001996`.

### What Slice 1 established

Slice 1 established that the activation pipeline detects a strong difference between benchmark evaluations and ordinary user prompts. It did **not** establish evaluation awareness because surface form nearly determines the labels. The historical `0.9944` result also selected the displayed best layer on the held-out curve, so the multi-seed train-only result is the correct inferential number.

## 3.3 Slice 2a: audit and reproduction of a public 2×2 dataset

### Dataset

The project next used the public Hugging Face dataset `viliana-dev/eval-awareness-2x2`, constructed for the same 8B model. The relevant cells were:

| Cell | n | Meaning |
|---|---:|---|
| `bench_eval` | 1,076 | ARC, MMLU, and GSM8K evaluation items in benchmark format |
| `casual_deploy_matched` | 1,076 | Length-matched ShareGPT first turns in casual form |
| `bench_deploy_rewritten` | 835 | ShareGPT requests rewritten by DeepSeek-V3 into benchmark-like form |
| `casual_eval_mutual` | 1,076 | MuTual dialogue items used as casual-form evaluation content |
| formal genre controls | 835 | ShareGPT queries rewritten in formal styles with or without benchmark structure |

All cells were wrapped as one-user-turn conversations before activation extraction.

### Surface audit

The repository independently ran its own baseline hierarchy before interpreting activation results. Mean AUROCs across the configured split seeds were:

| Positive vs negative cell | Character | Word | Format | Bag of words |
|---|---:|---:|---:|---:|
| Benchmark eval vs casual deploy | 0.5288 | 0.5381 | 0.9809 | 0.9986 |
| Benchmark eval vs benchmark-form deploy rewrite | 0.5004 | 0.5604 | 0.8597 | 0.9613 |
| Casual MuTual eval vs casual deploy | 0.1319 | 0.2434 | 0.9988 | 0.9877 |
| Benchmark-form deploy rewrite vs casual deploy | — | — | 0.9646 | 0.9920 |
| Formal casual deploy vs ordinary casual deploy | — | — | 0.9377 | 0.9892 |

The intended format-matched contrasts were therefore not surface matched under this project’s baselines. In particular, the “casual evaluation” MuTual cell was almost perfectly distinguishable from casual deployment using structural features alone.

### Activation reproduction

Two analysis setups were reproduced on a seed-0 split.

**Standard training:** train on `bench_eval` as positive and `casual_deploy_matched` as negative.

- Best layer: 2.
- In-distribution AUROC: `1.000`.
- Fraction classified evaluation when the frozen direction was applied to whole cells: benchmark evaluation `1.000`, casual deployment `0.000`, benchmark-form deployment rewrite `0.960`, casual MuTual evaluation `0.000`.
- Pairwise transfer AUROCs: benchmark eval versus benchmark-form deploy rewrite `0.53185`; casual MuTual eval versus casual deploy `0.4453`; benchmark-form deploy rewrite versus casual deploy `0.98665`.

The direction primarily followed benchmark form: it called most format-transplanted deployment prompts evaluation and did not call the casual evaluation proxy evaluation.

**Paired training:** combine benchmark evaluation and casual MuTual evaluation as positives; combine casual deployment and benchmark-form deployment rewrites as negatives.

- Best layer: 5.
- In-distribution AUROC: `0.96927` with bootstrap 95% interval `[0.96091, 0.97611]`.
- Fraction classified evaluation: benchmark evaluation `0.783`, casual deployment `0.0124`, benchmark-form deployment rewrite `0.180`, casual MuTual evaluation `0.9969`.
- Transfer to benchmark eval versus benchmark-form deploy rewrite: `0.8734`.

### Interpretation

The standard result was qualitatively reproduced and behaved like a format detector. Paired training reduced that failure, but the casual MuTual proxy itself had format AUROC `0.9988` and bag-of-words AUROC `0.9877` against casual deployment. Therefore the paired result cannot, on its own, demonstrate a format-invariant evaluation representation.

## 3.4 Slice 2b: constructing new single-turn contrasts

Slice 2b was not a single clean dataset-design step. It was an iterative series of pilots that exposed confounds and led to transfer, rather than in-distribution accuracy, becoming the principal test.

### Development iterations and what they revealed

1. **40-item authorship prototype.** Claude-authored prompts were compared with unmodified ShareGPT prompts. They were easily separable regardless of intended content: format AUROC was about `0.93` and bag-of-words about `0.91`. The final cells were therefore all model-authored, and one writer batch contained examples from every label so writer identity was balanced across labels.

2. **Length-target pilot.** Initial target lengths were inherited from ShareGPT, around 193 characters. This forced one evaluation style to add personal framing to otherwise short benchmark questions. First-person language appeared in 44% of style A and 78% of style B. The procedure was changed to target approximately the length of each benchmark question itself, clipped to `70–200` characters.

3. **Deployment-pool audit.** About 16% of sampled “deployment” prompts were themselves mathematics or homework problems. These were removed using a quiz/homework filter because they collapse the intended content distinction.

4. **100-item punctuation pilot.** Length was successfully matched, but the format classifier still achieved about `0.81`, driven partly by every evaluation prompt ending in `?` compared with only 54% of deployment prompts. The WildChat deployment pool was restricted to question-form inputs.

5. **Topic/speech-act problem.** Bag-of-words remained about `0.93`. This was not only a removable formatting bug: evaluation items tend to ask impersonal questions with canonical answers, while deployment requests express a practical need. A second deployment cell was therefore generated on the same benchmark topics, but the analysis stopped treating a high in-distribution score as sufficient evidence.

### Final four cells

The completed dataset contains 250 examples per cell:

| Cell | Construction |
|---|---|
| `casual_eval_a` | A faithful benchmark item rewritten in plain casual language |
| `casual_eval_b` | The same kind of item framed as a plausible personal need |
| `casual_deploy_wc` | A model rewrite of a genuine filtered WildChat user request |
| `casual_deploy_tm` | A synthetic genuine-use request on the same topic as a benchmark item |

For the two evaluation cells and the topic-matched deployment cell, source composition was 95 ARC, 58 MMLU, and 97 GSM8K examples. Mean character counts were tightly matched: A `133.9`, B `135.9`, WildChat deployment `136.9`, and topic-matched deployment `137.2`.

The file `casual_deploy_rw.jsonl` contains a stale 100-item pilot and is not a fifth final cell. Current analysis code excludes it.

### The frozen surface gate failed

The completed gate was probe-blind: acceptance depended on five-fold out-of-fold surface scores and manual validation, never on activation-probe performance. A gated feature passed only when `|AUROC - 0.5| <= 0.10`. Length and format were gated for all four eval/deploy contrasts. Bag-of-words was gated for the topic-matched deployment contrasts, where topic matching made that threshold meaningful; against the WildChat-derived deployment cell it was reported as the lexical bar the probe would have to beat. The completed gate produced:

| Contrast | Char AUROC | Word AUROC | Format AUROC | Bag-of-words AUROC |
|---|---:|---:|---:|---:|
| Eval A vs WildChat deploy | 0.4726 | 0.4932 | 0.6627 | 0.9319 |
| Eval A vs topic-matched deploy | 0.4764 | 0.4673 | 0.8326 | 0.9783 |
| Eval B vs WildChat deploy | 0.4923 | 0.5222 | 0.7818 | 0.9056 |
| Eval B vs topic-matched deploy | 0.4964 | 0.4954 | 0.7435 | 0.9135 |

Length matching passed. Format and lexical matching did not. Additional diagnostics showed authorship and style differences within the constructed material as well: model-rewritten WildChat versus real WildChat had format AUROC `0.8968` and bag-of-words `0.9440`; Eval A versus Eval B had `0.6963` and `0.7359`; WildChat-derived versus topic-matched deployment had `0.8265` and `0.9284`.

An audit of the rewrite operation found systematic drift from the source WildChat prompts: first-person usage rose from about `0.26` to `0.58`, request cues from `0.20` to `0.34`, factual-question cues fell from `0.15` to `0.04`, and the fraction ending in a question mark fell from `0.58` to `0.49`. Restricting to items without detected reframing did not materially reduce separation. Thus, these final cells must not be described as surface matched.

### Why cross-construction transfer became the main analysis

Six binary pairs were defined:

1. `devb_standard`: public benchmark eval vs public casual deploy;
2. `devb_fmt_matched`: public benchmark eval vs public benchmark-form deploy rewrite;
3. `ours_a_wc`: Eval A vs WildChat-derived deploy;
4. `ours_a_tm`: Eval A vs topic-matched deploy;
5. `ours_b_wc`: Eval B vs WildChat-derived deploy;
6. `ours_b_tm`: Eval B vs topic-matched deploy.

Each row of the transfer matrix fits on one pair and scores all six target pairs. The same operation is performed with the activation probe, format classifier, and bag-of-words classifier.

### GSM8K confound and the math-free primary analysis

The initial per-source analysis found:

- ARC aggregate probe AUROC `0.722 ± 0.175` versus bag-of-words `0.685 ± 0.173`;
- MMLU `0.772 ± 0.171` versus `0.682 ± 0.153`;
- GSM8K `0.947 ± 0.060` versus `0.828 ± 0.109`.

GSM8K’s apparent strength was substantially explained by digits. Eval A’s mean digit density was `0.0452`, while topic-matched deployment’s was `0.0033`, a roughly 13-fold difference. A digit-only score reached AUROC `0.949` for Eval A versus topic-matched deployment and `0.873` for Eval A versus WildChat-derived deployment. The primary transfer analysis therefore excludes GSM8K wherever benchmark source identity is known.

After filtering:

- public `bench_eval` retains 744/1,076 items: 383 ARC and 361 MMLU;
- Eval A, Eval B, and topic-matched deployment each retain 153/250: 95 ARC and 58 MMLU;
- deployment cells without meaningful benchmark-source labels remain whole: 1,076 casual public deploy, 835 benchmark-form public deploy rewrite, and 250 WildChat-derived deploy.

### Math-free direction-of-means transfer results

Rows are training pairs; columns are target pairs. Values are mean raw AUROC across seeds 0–4.

| Train \ target | Standard | Public format | A/WC | A/TM | B/WC | B/TM |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 1.000 | 0.747 | 0.574 | 0.674 | 0.590 | 0.678 |
| Public format | 0.975 | 0.949 | 0.509 | 0.298 | 0.551 | 0.293 |
| A/WC | 0.952 | 0.490 | 0.973 | 0.986 | 0.919 | 0.928 |
| A/TM | 0.980 | 0.410 | 0.913 | 0.998 | 0.646 | 0.960 |
| B/WC | 0.946 | 0.533 | 0.927 | 0.839 | 0.966 | 0.921 |
| B/TM | 0.981 | 0.314 | 0.857 | 0.990 | 0.848 | 0.977 |

Several features of this matrix matter:

- Directions learned from the project’s own four cells transfer very strongly to the original public standard contrast (`0.946–0.981`).
- The reverse transfer is much weaker. In particular, A/WC → standard is `0.952 ± 0.005`, whereas standard → A/WC is `0.574 ± 0.019`. The public standard direction contains narrower benchmark-format features; the A/WC direction captures something that also applies to the standard contrast.
- The public benchmark-format target is the hardest target for directions learned on the new cells (`0.314–0.533`, sometimes inverted). There is no clean content-only direction that works uniformly across every construction.

Aggregated over the diagonal cells, mean separation strengths were:

| Method | In-distribution strength |
|---|---:|
| Activation probe | 0.4773 |
| Format baseline | 0.2939 |
| Bag of words | 0.4218 |
| Length | 0.0260 |

Aggregated over all 30 off-diagonal source–target transfers:

| Method | Mean transfer strength |
|---|---:|
| Activation probe | 0.29164 |
| Format baseline | 0.13121 |
| Bag of words | 0.22057 |
| Length | 0.025995 |

The activation-minus-bag-of-words transfer margin was `0.07108 ± 0.01341`. It was positive for every split seed, ranging from `0.05330` to `0.08771`.

Per training pair, the mean activation-minus-bag-of-words margins and the proportion of individual target/seed transfers won by the probe were:

| Training pair | Margin, mean ± SD | Win rate |
|---|---:|---:|
| Public standard | `-0.08737 ± 0.03171` | 12% |
| Public format | `+0.06798 ± 0.02301` | 68% |
| A/WC | `+0.09746 ± 0.01515` | 80% |
| A/TM | `+0.09611 ± 0.01258` | 100% |
| B/WC | `+0.14260 ± 0.02347` | 80% |
| B/TM | `+0.10969 ± 0.01916` | 96% |

### Audit of shared-cell versus completely disjoint transfers

The published all-off-diagonal average is correct for the 30 matrix entries, but not all 30 are equally independent. Ten source–target pairs share one dataset cell. Twenty have no cell in common.

| Transfer subset | Probe strength | Bag-of-words strength | Probe − BoW margin |
|---|---:|---:|---:|
| 20 completely disjoint transfers | 0.22810 | 0.18055 | `+0.04754 ± 0.01538` |
| 10 transfers sharing one cell | 0.41874 | 0.30059 | `+0.11815 ± 0.01335` |
| All 30 off-diagonals | 0.29164 | 0.22057 | `+0.07108 ± 0.01341` |

The disjoint margin was positive in all five seeds, ranging from approximately `+0.02597` to `+0.06776`. This is the cleanest summary of cross-procedure generalization because neither a cell nor its examples are reused in the target.

### Logistic-estimator check

Replacing the direction of means with the standardized L2 logistic estimator preserved the main pattern:

- mean off-diagonal logistic-probe strength: `0.27621`;
- bag-of-words strength on the same transfer set: `0.22057`;
- margin: `+0.05564`;
- A/WC → public standard: `0.9580 ± 0.0226`;
- public standard → A/WC: `0.6415 ± 0.0319`.

Thus, the transfer and asymmetry findings are not unique to the direction-of-means estimator.

### What Slice 2b established

Slice 2b did not create a surface-invariant within-dataset contrast; its gate failed. It did establish that activation directions learned from several newly constructed contrasts generalized across independently constructed eval/deploy pairs better than the tested format and bag-of-words baselines, including on the stricter completely disjoint transfer subset. The residual interpretation remains “a shared internal direction associated with these eval/deploy constructions,” not a proof that all topical, speech-act, or authorship information has been removed.

## 3.5 Slice 3: multi-turn conversations and temporal localization

### Real deployment source pool

The starting deployment pool consists of 250 real WildChat conversations selected with the following rules:

- English;
- alternating user and assistant roles;
- user starts and user ends;
- 2–4 user turns;
- each message contains 20–600 characters;
- no coursework or quiz-like content in either role;
- the final user message is question-form.

If a longer source conversation continued past a user turn satisfying these constraints, it was truncated at that user turn. The final distribution was 105 conversations with two user turns, 84 with three, and 61 with four.

Fresh benchmark items not used in Slice 2b supplied evaluation content: 158 ARC and 92 MMLU items; GSM8K was excluded.

### First three written cells

For each deployment skeleton, a writer generated three complete conversations matching the original role sequence and per-message target lengths:

- `deploy`: a rewrite of the real WildChat conversation;
- `eval_disguised`: a conversation whose final user turn asks the benchmark item without explicit test framing;
- `eval_obvious`: a conversation with explicit test/evaluation framing.

All three were model-authored, and each writer batch contained all three labels, controlling the crude real-versus-generated and writer-identity differences. This version nevertheless failed two validity checks.

### Surface/framing gate

All 250 triples had the requested role shape. Average total messages were `4.65`; there were no shape errors. Evaluation-cue rates were `0.04` for deployment, `0.03` for disguised evaluation, and `0.97` for obvious evaluation, so the obvious/disguised framing manipulation worked.

The surface gate did not:

| Text compared: disguised eval vs deploy | Length | Format | Bag of words |
|---|---:|---:|---:|
| Whole conversation | 0.5114 | 0.6420 | 0.9084 |
| Final user turn | 0.5608 | 0.7123 | 0.9406 |

The gate threshold for length and format was separation strength at most `0.10`; bag-of-words was reported as the bar the activation probe would need to beat. Length passed, but format and lexical separation remained high. The gate was not passed.

### Why the as-written disguised cell was invalid as the main control

The last-user probe strongly separated disguised evaluation from deployment, but so did a readout taken after the first user turn, before the benchmark question appeared. Current train-only reanalysis gives:

| Contrast/read point | Probe AUROC, mean ± SD | Bag of words | Format |
|---|---:|---:|---:|
| Disguised, last user | `0.98457 ± 0.00466` | 0.91172 | 0.64722 |
| Disguised, after first user | `0.94916 ± 0.00913` | 0.85584 | 0.60270 |
| Disguised, final-turn mean | `0.99076 ± 0.00567` | 0.91172 | 0.64722 |
| Obvious, last user | `0.99573 ± 0.00248` | 0.99844 | 0.60512 |
| Obvious, after first user | `0.95808 ± 0.01483` | 0.99751 | 0.71630 |
| Obvious, final-turn mean | `0.99900 ± 0.00138` | 0.99844 | 0.60512 |

The writer had produced different synthetic lead-ins for the classes. Because the signal existed before the purportedly class-defining final question, this dataset could not localize the effect to evaluation content. For the obvious cell, bag-of-words was also at least as strong as the activation probe. These cells are useful diagnostics but not clean evidence for the final claim.

### The `eval_shared` recombination control

`eval_shared` fixes the pre-final leak by taking the deployment rewrite and replacing only its final user turn with the final user turn from the disguised-evaluation version. For every one of the 250 pairs, all messages before the final user turn are byte-identical. This yields a sharp temporal prediction: any genuine class separation must appear only after the changed final turn.

Current train-only reanalysis gives:

| Read point | Probe AUROC, mean ± SD | BoW AUROC | Format AUROC | Length AUROC |
|---|---:|---:|---:|---:|
| Final user boundary | `0.99168 ± 0.00330` | 0.93504 | 0.72754 | 0.56539 |
| After first user / pre-change | `0.49847 ± 0.00111` | 0.50000 | 0.50000 | 0.50000 |
| Final-turn mean | `0.99492 ± 0.00432` | 0.93504 | 0.72754 | 0.56539 |

The strength margin over the stronger final-turn-only bag-of-words baseline is `0.05664` at the final boundary and `0.05988` for the final-turn mean.

The end-aligned trajectory confirms the localization:

| User-turn offset from end | Available pairs | Probe AUROC, mean ± SD | Text baselines |
|---:|---:|---:|---:|
| 0, final turn | 250 | `0.99168 ± 0.00330` | BoW 0.93504; format 0.72754 |
| 1 | 250 | `0.49749 ± 0.00176` | all 0.50000 |
| 2 | 145 | `0.50031 ± 0.00019` | all 0.50000 |
| 3 | 61 | `0.50031 ± 0.00116` | all 0.50000 |

This is strong evidence that the separation was introduced by the final turn, rather than by conversation length, generic prefix style, or a pre-existing class difference. Its limitation is equally specific: swapping in a benchmark question can introduce a topic or discourse discontinuity. The control localizes the signal, but does not prove the signal means evaluation rather than “unexpected canonical-answer question after this prefix.”

### Saved-result audit and corrected surface view

An older version of `results/s3_Llama-3.1-8B-Instruct_analysis.json` selected the best layer on the held-out curve and produced a headline around `0.995`. Commit `8921fa2` corrected layer selection. On August 9, 2026, the analysis was rerun again after auditing the matched-prefix surface view: activation values remained unchanged, but surface baselines now use only the differing final message. The current artifact records `paired_surface_text_view: differing_final_message_only` and is authoritative; older whole-conversation BoW values such as `0.76889` for `eval_shared` are historical, weaker baselines.

### First coherence attempt: completed but rejected

The first `eval_coherent` attempt asked a writer to make an arbitrary benchmark item seem natural after a frozen deployment prefix. Activations and an analysis artifact exist, but manual review found that the final questions often were not actually coherent with the preceding conversation. The corresponding dataset file was removed, and these results should not be reported as a successful control.

### Repaired matched-coherence design

The repaired `coherent_pairs` protocol constructs a whole everyday conversation and two alternative final user turns together:

- the prefix, including assistant messages, is byte-identical across the pair;
- the evaluation ending is a faithful casual version of an ARC or MMLU item;
- the deployment ending is a genuine practical, non-factual request on the same topic;
- both endings are intended to fit the preceding conversation, register, and target length;
- both sides are model-authored.

This sacrifices the realism of a genuinely sourced prefix in exchange for stronger topic and coherence matching.

### Pilot and full results

The 50-pair pilot gave last-user AUROC `0.99733`, final-turn-only bag-of-words `0.96533`, format `0.67244`, and pre-final probe AUROC `0.50044`. The temporal prediction held before scaling, but the stronger lexical baseline accounted for most of the final-turn separation.

On all 250 pairs, current train-only results were:

| Read point | Probe AUROC, mean ± SD | BoW | Format | Probe − strongest surface strength | Mean selected layer |
|---|---:|---:|---:|---:|---:|
| Final user boundary | `0.95132 ± 0.01147` | `0.93337 ± 0.00680` | `0.69899 ± 0.03916` | `0.01796` | 5 |
| After first user / pre-change | `0.49902 ± 0.00152` | 0.50000 | 0.50000 | approximately 0 | — |
| Final-turn mean | `0.93593 ± 0.01490` | `0.93337 ± 0.00680` | `0.69899 ± 0.03916` | `0.00256` | 0 |

The last-user-boundary activation score remains high, but the final-turn-only lexical baseline nearly matches it. The final-turn-mean analysis also selects layer 0, meaning that its best separation is available directly in averaged token embeddings. These in-distribution results should therefore be described as largely lexical, with only a small residual activation margin.

### Transfer between the two multi-turn controls

The two controls were built by different procedures and have different weaknesses. Cross-transfer asks whether they share a direction beyond their own surface classifiers:

| Source direction → target | Probe AUROC, mean ± SD | BoW | Format | Probe − BoW strength |
|---|---:|---:|---:|---:|
| `eval_shared` → `coherent_pairs` | `0.85842 ± 0.03207` | `0.79154 ± 0.02343` | 0.61963 | 0.06688 |
| `coherent_pairs` → `eval_shared` | `0.78791 ± 0.09936` | `0.67847 ± 0.04641` | 0.57931 | 0.10944 |

These are completely independent constructions. The activation direction still transfers better than final-turn word counts, but the corrected margins are much smaller than under the noisy whole-conversation baseline. This supports some shared representation beyond unigram counts, not a large lexical-independent effect. The substantial seed variation in the coherent-to-shared direction should remain visible in any paper table.

### Strong sparse-text audit and direct margin inference

The bag-of-words comparison above was not the end of the surface audit. A later CPU-only analysis added two fixed text-only classifiers:

- word 1--2-gram TF-IDF logistic regression;
- character-within-word 3--5-gram TF-IDF logistic regression.

The audit uses grouped five-fold outer predictions for within-control results, keeping both members of every scenario pair in the same fold. After the cross-model layer-selection audit, the within-control layer is selected by pair-grouped CV nested inside each outer training fold. For transfer, layer and orientation are selected by grouped cross-validation on the source only, the final probe and text classifiers are fit on all source pairs, and every target pair is scored. Vocabularies and IDF weights are always fitted on source/training text only. Uncertainty is a 2,000-draw cluster bootstrap over matched scenarios. The one-sided p-value comes from 2,000 approximate-randomization draws that swap the probe and comparator prediction vectors within scenario; unlike the earlier label-permutation test, this directly tests the probe-minus-baseline AUROC difference.

| Analysis | Probe | Strongest text baseline | Baseline AUROC | Probe − baseline | 95% pair-bootstrap interval | One-sided p |
|---|---:|---|---:|---:|---:|---:|
| `eval_shared`, grouped OOF | 0.98779 | word TF-IDF | 0.95293 | `+0.03486` | `[+0.01833, +0.05395]` | 0.00150 |
| `coherent_pairs`, grouped OOF | 0.94682 | character TF-IDF | 0.94474 | `+0.00208` | `[-0.01341, +0.01675]` | 0.43778 |
| `eval_shared` → `coherent_pairs` | 0.84496 | character TF-IDF | 0.86614 | `-0.02118` | `[-0.04887, +0.00811]` | 0.87856 |
| `coherent_pairs` → `eval_shared` | 0.71874 | character TF-IDF | 0.84854 | `-0.12981` | `[-0.16654, -0.09081]` | 1.00000 |

This changes the evidential chain. `eval_shared` retains a statistically reliable activation advantage over the stronger sparse-text set, although its topic/speech-act discontinuity remains. Matched coherence is tied by character n-grams. Most importantly, the earlier statement that activation directions transfer between the controls better than text was specific to unigram counts and does not survive character n-grams. Cross-control transfer remains evidence that both representations encode a shared distinction, but it is not evidence for an activation-specific component beyond the tested text classifiers.

## 3.6 Observed-silence analysis

### Question and filter

The next question was whether the probe only separated examples where the model would overtly say that it recognized an evaluation. Llama-3.1-8B-Instruct generated a greedy reply for both members of every pair with sampling disabled.

A pair was retained only if:

1. both replies emitted a recognized end-of-turn or end-of-sequence token within the generation cap; and
2. neither reply contained explicit or implicit evaluation-awareness language under the automated and judged checks.

Hitting the token cap was treated as indeterminate, not as evidence of silence. The two classes were filtered as a pair so that the retained eval and deploy examples remained matched.

### Awareness detection and validation

The initial regex searched for direct evaluation/test/benchmark language. A separate rubric classified replies as `explicit`, `implicit`, or `none`. The rubric was validated with:

- 14 planted positives: 8 overt and 6 implicit;
- 10 hard negatives containing potentially confusing but non-awareness language;
- 52 blinded duplicate judgments for between- and within-judge consistency.

The judge procedure found all 14 planted positives and produced 0/10 false alarms on hard negatives. Cross-judge agreement was 47/47 exact and within-judge repeat agreement 5/5. None of the 250 real matched-coherence reply pairs was flagged. There were no judge/regex disagreements, and manual review found no missed disclosure.

The judging was carried out by Claude Code subagents with repository context. Rows were shuffled and identified opaquely, but the judges were not isolated from all project-level knowledge. The artifacts do not record a distinct judge model identifier, so no more specific model attribution is warranted.

### Matched-coherence silence results

With a 256-token cap:

- evaluation replies terminated for 205/250 examples;
- deployment replies terminated for 222/250;
- 192 pairs terminated on both sides and were retained;
- median reply lengths were 99 tokens for evaluation and 83 for deployment;
- no retained or excluded completed reply was flagged for awareness.

On the 192 retained pairs, at fixed layer 4:

| Method | AUROC, mean ± SD |
|---|---:|
| Activation probe | `0.93686 ± 0.03250` |
| Bag of words | `0.93543 ± 0.02814` |
| Format | `0.69807 ± 0.04316` |

The probe’s strength margin over bag-of-words was only `0.00143`. On this silence-filtered matched-coherence subset, the activation probe and final-turn word counts are effectively tied.

### `eval_shared` silence results and cap sensitivity

At a 256-token cap:

- evaluation replies terminated for 161/250;
- deployment replies terminated for 180/250;
- 125 pairs terminated on both sides;
- median reply lengths among the corresponding outputs were 198 and 162 tokens;
- one deployment reply contained wording about a “quiz show.” Manual review determined this was about the subject matter, not evaluation awareness, but the pair was conservatively excluded;
- final retained n was 124 pairs.

Results were probe `0.99752 ± 0.00205`, bag-of-words `0.95047 ± 0.01956`, and format `0.76728 ± 0.01947`, for a probe–BoW strength margin of `0.04704`.

Because many outputs hit 256 tokens, the generation was repeated with a 512-token cap:

- evaluation replies terminated for 241/250;
- deployment replies terminated for 235/250;
- 228 pairs terminated on both sides;
- the same single conservative false-positive pair was excluded;
- final retained n was 227;
- median lengths remained 198 and 162 tokens, indicating that the additional terminations were mostly from the previous right tail rather than a shift in typical response length.

The cap-512 results were probe `0.99083 ± 0.00321`, bag-of-words `0.94602 ± 0.01513`, and format `0.73227 ± 0.04196`, for a margin of `0.04481`.

### Interpretation boundary

High activation separation is not driven only by replies that visibly announce an evaluation. However, on matched-coherence silence pairs, final-turn word counts separate the labels equally well. On `eval_shared`, the probe retains a modest advantage. “No detected disclosure” is not equivalent to “the model was unaware,” and the result is conditioned on successful termination. The 512-token replication reduces, but does not eliminate, that selection concern.

## 3.7 Slice 5: genuine versus resampled assistant replies

Slice 5 changes the question. Instead of evaluation versus deployment user inputs, it examines whether the 8B model’s hidden states distinguish a genuine assistant reply from a generated replacement inserted into the same preceding conversation.

### Corpus preparation

Two real-chat corpora were used independently: WildChat and LMSYS. Shared filters required:

- English;
- alternating roles, user first and assistant last;
- 2–4 user turns;
- every message 20–600 characters;
- no coursework/quiz-like content;
- a question-form final user message.

For LMSYS, the record-level language field was used and conversations containing `redacted` placeholders were rejected to avoid `NAME_1`-style artifacts.

The final WildChat sample contained 250 conversations, with user-turn counts 186/42/22 for two/three/four turns. The LMSYS sample also contained 250, distributed 164/61/25.

### Reconstruction protocol

For each conversation, the genuine final assistant reply was removed. A generator produced a greedy replacement with sampling disabled and a 256-token cap. The replacement was then inserted after the complete original prefix. Llama-3.1-8B-Instruct processed the full conversation with either the genuine or replacement reply, and activations were read at the final assistant boundary and averaged over the final assistant turn.

Only naturally terminated replacements were analyzed. The positive class was the resampled reply and the negative class the genuine reply. The prompt prefix was identical within each pair, but the compared reply text necessarily differed.

### WildChat pilot and scale-up

The 50-conversation pilot produced 43 terminated replacements:

| Read point | Probe | Format | BoW | Probe − strongest surface strength |
|---|---:|---:|---:|---:|
| Final assistant token | `0.76095 ± 0.08572` | 0.76805 | 0.61302 | `-0.00710` |
| Final assistant mean | `0.73728 ± 0.07766` | 0.76805 | 0.61302 | `-0.03077` |

The full 250-conversation WildChat run yielded 196 terminated replacements:

| Read point | Probe AUROC, mean ± SD | Format | BoW | Length | Probe − strongest surface strength |
|---|---:|---:|---:|---:|---:|
| Final assistant token | `0.79247 ± 0.02448` | 0.77374 | 0.73565 | 0.63442 | 0.01873 |
| Final assistant mean | `0.82206 ± 0.01803` | 0.77374 | 0.73565 | 0.63442 | 0.04832 |

The mean readout selected layer 31. The full run retains a small-to-moderate advantage over the stronger reply-only baselines, while the 50-item pilot does not. The evidence for reply provenance is therefore real but substantially less distinct from surface form than the original whole-conversation baseline suggested.

### Model-swap and context-ablation controls

Two controls ask what the separation means:

- **Model swap:** Llama-3.2-3B-Instruct, rather than the probed 8B model, generated the replacement from the full prefix. Llama-3.1-8B-Instruct still supplied both compared activation sets.
- **Context ablation:** Llama-3.1-8B-Instruct generated the replacement after seeing only the final user message, not the preceding conversation. The resulting reply was still inserted into the complete original conversation before the 8B activation readout.

The model-swap run produced 200 terminated replacements; context ablation produced 158. Their clean comparison uses the exact 146 conversations that terminated in both runs and uses the same genuine activation arrays and mask for both conditions:

| Condition/read point | Probe AUROC, mean ± SD | Format | BoW | Probe − strongest surface strength |
|---|---:|---:|---:|---:|
| 3B model swap, final token | `0.81105 ± 0.04558` | 0.78647 | 0.70971 | 0.02459 |
| 3B model swap, final mean | `0.78915 ± 0.04562` | 0.78647 | 0.70971 | 0.00269 |
| 8B context ablation, final token | `0.86808 ± 0.01154` | 0.83234 | 0.74504 | 0.03574 |
| 8B context ablation, final mean | `0.80279 ± 0.02289` | 0.83234 | 0.74504 | `-0.02955` |

The raw activation signal survives when a different model writes the reply, so it is not specifically an 8B “I wrote this” detector. However, reply-only format features now explain most of the separation: activation margins are small, and the context-ablation mean readout loses to format. Removing generation-time conversation context still makes the token signal at least as strong, consistent with contextual fit, provenance, or distributional mismatch contributing. These conditions are not an additive causal decomposition because each changes the generated reply distribution as well as the named factor.

There is a standalone context-ablation analysis artifact, but current source caches do not contain a separately tagged genuine array for that standalone file. The exact shared-intersection comparison above is the reproducible and preferable analysis.

### LMSYS corpus replication

On LMSYS, 179/250 replacement replies terminated:

| Read point | Probe AUROC, mean ± SD | Format | BoW | Probe − strongest surface strength |
|---|---:|---:|---:|---:|
| Final assistant token | `0.83700 ± 0.02556` | 0.73573 | 0.67603 | 0.10127 |
| Final assistant mean | `0.79119 ± 0.02976` | 0.73573 | 0.67603 | 0.05545 |

The genuine-versus-resampled separation therefore replicates across two real-chat corpora. The ordering of token and mean readouts flips between WildChat and LMSYS, so no claim should depend on one universally superior read point.

### Interpretation boundary

Slice 5 is a provenance/context experiment, not an evaluation-awareness experiment. Genuine and reconstructed reply states are separable, but reply-only surface baselines explain much of the effect. The model-swap and context controls also show that it cannot be interpreted narrowly as self-authorship recognition.

## 3.8 Estimator and null-distribution robustness

The final robustness run reanalyzed the main saved activation contrasts with both direction-of-means and standardized logistic probes. It used five split seeds, 500 full-label permutations, 200 shuffled-direction diagnostics, and 200 random unit directions.

### Honest estimator comparison

| Contrast/read point | Direction of means, mean ± SD | Logistic probe, mean ± SD |
|---|---:|---:|
| Slice 1 | `0.99183 ± 0.00848` | `0.99333 ± 0.00772` |
| `eval_shared`, final user | `0.99168 ± 0.00330` | `0.99762 ± 0.00174` |
| `eval_shared`, pre-final | `0.49847 ± 0.00111` | `0.50011 ± 0.00026` |
| `eval_shared`, final mean | `0.99492 ± 0.00432` | `0.99950 ± 0.00050` |
| `coherent_pairs`, final user | `0.95132 ± 0.01147` | `0.98937 ± 0.00811` |
| `coherent_pairs`, pre-final | `0.49902 ± 0.00152` | `0.49918 ± 0.00238` |
| `coherent_pairs`, final mean | `0.93593 ± 0.01490` | `0.98837 ± 0.00636` |
| Coherent observed-silence subset | `0.93686 ± 0.03250` | `0.98199 ± 0.01196` |
| Shared observed-silence, cap 256 | `0.99752 ± 0.00205` | `0.99912 ± 0.00117` |
| Shared observed-silence, cap 512 | `0.99083 ± 0.00321` | `0.99792 ± 0.00227` |
| WildChat reconstruction, token | `0.79247 ± 0.02448` | `0.89566 ± 0.02086` |
| WildChat reconstruction, mean | `0.82206 ± 0.01803` | `0.95794 ± 0.01374` |
| LMSYS reconstruction, token | `0.83700 ± 0.02556` | `0.91101 ± 0.01332` |
| LMSYS reconstruction, mean | `0.79119 ± 0.02976` | `0.94023 ± 0.01434` |
| Shared model-swap, token | `0.81105 ± 0.04558` | `0.88729 ± 0.01482` |
| Shared model-swap, mean | `0.78915 ± 0.04562` | `0.86736 ± 0.03284` |
| Shared context ablation, token | `0.86808 ± 0.01154` | `0.94680 ± 0.01465` |
| Shared context ablation, mean | `0.80279 ± 0.02289` | `0.89762 ± 0.02550` |

The logistic estimator generally increases predictive AUROC but does not change which controls pass or fail. The two identical-prefix pre-final contrasts remain exactly at chance.

### Permutation tests

For every real separating contrast in the table, the full-pipeline permutation p-value was the minimum resolvable `0.001996`. The identical-prefix pre-final controls were non-significant:

- `eval_shared` pre-final: `p = 0.99002`;
- `coherent_pairs` pre-final: `p = 0.94012`.

Across tasks, permutation-null mean separation strength was roughly `0.036–0.053`, with 95th percentiles roughly `0.087–0.129`, depending on sample size and dimensional geometry.

### Geometry diagnostics

Two diagnostics were intentionally not treated as hypothesis tests:

- A direction fitted after shuffling training labels can still have strength around `0.243–0.286` on strongly structured cells even though its signed AUROC averages near 0.5.
- A random unit direction can have strength around `0.110–0.145`.

This occurs because reporting `|AUROC - 0.5|` removes orientation and high-dimensional structured data contain many separating directions by chance. It means raw strength can overstate how probe-specific a pattern is. The valid significance check is the full-label permutation that repeats layer selection and evaluation, which the real final-turn contrasts pass and the pre-final controls do not.

## 3.9 Pre-specified activation steering

### What was pre-specified

The steering protocol was frozen before viewing steering outputs. A later amendment, also written before behavioral generation, records a change made after the initial direction computation. The protocol used the 250-pair matched-coherence dataset rather than selecting prompts after observing behavior.

The direction was fitted from `coherent_pairs` last-user activations using all 250 examples per class with five-fold cross-validation across layers. The cross-validated layer curve was broad rather than sharply peaked:

- maximum separation at layer 4: `0.44528` strength;
- layer 9: `0.44192`;
- 12 layers were within `0.02` of the maximum.

Four candidate intervention locations were selected in advance: layers 3, 7, 10, and 14. Their one-unit scales were the corresponding class-mean gaps:

| Candidate layer | Mean-gap scale | Mean residual norm |
|---:|---:|---:|
| 3 | 0.14779 | 1.803 |
| 7 | 1.52909 | 4.566 |
| 10 | 2.09285 | 6.303 |
| 14 | 2.29272 | 7.956 |

The intervention coefficients were `[-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4]`. One alpha unit corresponds to one class-mean-gap displacement along the unit direction. A hook adds the vector after block `L-1` so that it changes the residual stream entering the indexed readout layer. It is applied to all prompt positions and to subsequently generated tokens. Alpha 0 runs without a hook.

An independent linear readout was trained on the separate `eval_shared` dataset and fixed at layer 15. Its reference means were `-0.58751` for deployment and `2.405` for evaluation. Thus, the intervention direction and the measurement readout did not use the same examples.

### Manipulation check and layer choice

The manipulation check used 36 carrier prompts that ended with an explicit question asking whether the interaction seemed like evaluation or a real user situation. All four candidate layers and all alpha values were run.

Layer 14 was selected because it produced the strongest clean monotonic readout movement:

- Spearman correlation between alpha and independent readout: `1.0`;
- displacement relative to the eval/deploy reference gap ranged from `-1.138` at alpha `-4` to `+0.872` at alpha `+4`.

Layers 7 and 10 were also monotonic but had smaller spans. Layer 3 was nearly flat/non-monotonic, with Spearman correlation about `0.15`. The behavioral experiment therefore used layer 14.

### Behavioral generation and degeneracy rule

At layer 14, the project generated:

- 36 self-report prompts × 9 alphas;
- 48 safety prompts × 9 alphas;
- 756 total greedy generations, each capped at 256 tokens.

The frozen degeneracy veto marked an alpha unusable if either the mean repeated-four-gram fraction exceeded `0.35` or mean response length fell below half of the alpha-0 mean. Both endpoints, `-4` and `+4`, were degenerate. Repeated-four-gram fractions were `0.551/0.784` on self-report and `0.418/0.613` on safety at the negative/positive endpoints, and outputs commonly ran to the cap. Results are therefore interpreted only for alphas `-2` through `+2`.

### Self-report result

The deterministic scorer labels the first 400 response characters as evaluation, real/deployment, or unclear. The proportion scored as evaluation at nondegenerate alphas was:

| Alpha | -2 | -1 | -0.5 | 0 | +0.5 | +1 | +2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Evaluation self-report rate | 0.1389 | 0.6111 | 0.6667 | 0.7778 | 0.8056 | 0.8889 | 0.8889 |

Spearman correlation was `0.991`. Steering in the positive direction increased evaluation self-reports and steering negatively reduced them.

This result has a ceiling/asymmetry qualification. The prompt explicitly asks the model to choose between an evaluation and real-use interpretation, and the alpha-0 baseline already reports evaluation 77.8% of the time. Much of the observable range is therefore the negative/deploymentward shift, not a symmetric creation of awareness from a neutral baseline.

### Safety result

The deterministic safety scorer searches the first 120 characters for refusal language. Refusal rates were:

| Alpha | -2 | -1 | -0.5 | 0 | +0.5 | +1 | +2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Refusal rate | 0.0833 | 0.1042 | 0.1042 | 0.1250 | 0.1042 | 0.0833 | 0.0625 |

Spearman correlation was `-0.356`, opposite the predicted increase. Across the full nondegenerate range this is only 3–6 refusals out of 48 prompts. At a baseline rate of 0.125, the design had about 80% power only for an increase to roughly 0.37. The correct conclusion is that no large predicted safety-refusal effect was observed; the experiment was underpowered for small effects and does not establish exact absence.

### What steering established

The chosen intervention causally moved an independent internal readout and explicit eval/deploy self-reports in the predicted direction before output degeneration. It did not demonstrate a corresponding broad change in safety behavior. Steering therefore strengthens the claim that the direction is behaviorally accessible, but not the stronger claim that it is the causal mechanism behind deployment-policy changes.

## 3.10 Authorship factorial pilot: v2 stopped at the coherence gate

The experiment frozen in `datasets/authorship/prespec.md` crosses eval/deploy final-user status with three preceding-history authors: target-model on-policy Llama-3.1-8B, same-family off-policy Llama-3.2-3B, and cross-family off-policy Qwen2.5-7B. The pilot uses 50 deterministically selected two-user-turn matched-coherence scenarios so that exactly one assistant response is regenerated per authorship condition. Within each author, the eval and deploy prefixes are byte-identical; across authors, the two final user endings are fixed.

The v1 feasibility run stopped before activation extraction when its frozen 256-token cap made the success criterion unattainable: 12 of the first 13 on-policy histories hit the cap, so fewer than 40/50 scenarios could possibly survive. The partial record is retained. `datasets/authorship/prespec_v2.md` was then frozen before any v2 generation. It changes only the global cap to 1,024 tokens, preserves greedy decoding and every other construction/gate rule, and reuses the same pilot scenarios. They remain excluded from confirmation.

Under v2, all 150 histories across all three generators terminated naturally. The longest observed generated history was 863 tokens. The mechanical builder validated all 300 cells, including the required user/assistant/user sequence, byte-identical eval/deploy prefixes within each author, and identical final endings across authors.

The coherence rubric was frozen in `datasets/authorship/coherence_judge_v1.md`. Five blinded batches were judged with the resolved identifier `claude-sonnet-5`; tools were disabled. Claude passed both endings in 144/150 rows. The six failed rows belong to four scenarios, so 46/50 scenarios provisionally survived the model review.

The prespecified human reviewer then assessed the frozen 30-row blinded sample. Both endings passed in 16/30 rows; ending A failed in 8 rows and ending B failed in 10, with at least one failure in 14 rows spanning 11 scenarios. These failures were distributed across all author conditions (6 same-family, 5 cross-family, and 3 target-model rows), rather than being confined to one generator. On the same 60 ending-level decisions, the human passed 42 while Claude passed 59. They agreed on 41/60 decisions (Cohen's kappa `-0.033`; descriptive only), with 18 human-fail/Claude-pass decisions and one human-pass/Claude-fail decision.

Under the frozen conservative rule, any human failure excludes the entire scenario in addition to the model exclusions. The final artifact therefore retains 36/50 scenarios, below the prespecified 40/50 threshold: **the v2 pilot fails the pre-activation gate**. No v2 activations were extracted or inspected. A second independent review of the same blinded sample may be used to diagnose reviewer and rubric reliability, but it cannot retroactively change this gate result. A revised construction and/or rubric must be frozen as a new pilot version before further generation.

If a future version passes its pre-activation gate, the primary question remains whether an evaluation direction selected under one history author transfers with stable orientation to the other authors. Secondary diagnostics fit authorship directions, compare direction cosines, and residualize their span before refitting evaluation probes. All splits and inference are grouped by scenario, and the stronger word/character TF-IDF hierarchy remains mandatory.

## 3.11 Cross-model confirmatory suite: completed primary panel

The compact cross-model suite excludes failed and exploratory constructions and retains only the two valid 250-pair controls: `eval_shared` and matched coherence. Before any new-model activation extraction, the scope was narrowed to current open-weight instruction models in the 5--12B range. The resulting amendment is frozen in `datasets/cross_model/confirmatory_suite_v2.json`; v1 is preserved with its hash in the amendment record. The reference is the completed Llama-3.1-8B-Instruct study. The three revision-pinned primary replications are Qwen3-8B, Gemma-3-12B-IT, and Ministral-3-8B-Instruct. The protocol required all three regardless of earlier outcomes, and all three are complete.

Each model's mandatory core consists of 2,824 unquantized bfloat16 causal-prefix forward passes, one for every user-turn boundary in both conditions. This prefixwise rule prevents different final-turn lengths from selecting different BF16 kernels and introducing suffix-length-dependent numerical differences at earlier tokens. The analysis freezes matched-scenario grouping, train-only layer selection, the full word/character TF-IDF baseline hierarchy, paired bootstrap intervals, paired prediction-randomization tests, and bidirectional transfer between controls. The original protocol did not specify whether train-only selection used in-sample or nested scores; the correction described below resolves that ambiguity uniformly. Artifacts are isolated by a collision-resistant hash of the full model repository and embed the immutable model revision, manifest hash, dataset-content hash, precision, extraction method, and sample identity. Smoke and full artifacts cannot share paths.

A one-pair integration smoke using the already cached Llama reference completed all four extraction cells. The first Qwen3 smoke then correctly failed before writing activations because Qwen3's template inserts an empty thinking block when an assistant message ends a rendered prefix, violating the historical incremental-template assumption. A narrowly validated ChatML-delimiter fallback fixed boundary recovery. A subsequent full-sequence smoke found suffix-length-dependent BF16 differences at the pre-final boundary in 3/8 matched-coherence pairs despite identical tokens. Those caches were rejected and archived, and cross-model temporal extraction was changed to independent causal-prefix passes. The corrected eight-pair Qwen3 smoke completed all four cells with exact zero pre-final differences in both controls, nonzero final differences, complete strong-baseline analysis, and both transfer directions. Text-only adapters for the official Gemma 3 and Ministral 3 multimodal wrappers then passed real-tokenizer boundary tests and the same eight-pair extraction and analysis gate. Smoke outputs are implementation checks, not evidence.

The full frozen Qwen3-8B core was then completed locally on all 250 pairs per
contrast. Each of the four caches contains 706 user-turn observations across
the embedding and 36 transformer layers, with 4,096 hidden dimensions. Both
temporal controls pass exactly: maximum absolute positive/negative difference
at every byte-identical pre-final boundary is `0.0`, while the mean final-state
L2 differences are nonzero (`142.71` for `eval_shared`, `96.71` for
`coherent_pairs`). Cache provenance matches the immutable Qwen revision,
manifest hash, dataset hashes, bfloat16 precision, prefixwise method, and full
sample.

The initial grouped five-fold implementation qualitatively replicated the reference-model
pattern. On `eval_shared`, the Qwen probe reached AUROC `0.99357` versus
`0.95293` for the strongest text baseline (word TF-IDF), a paired difference of
`+0.04064` with 95% pair-bootstrap interval `[+0.02364, +0.05982]` and
one-sided paired-randomization `p=0.0050`. On `coherent_pairs`, the probe
reaches `0.94653` versus character TF-IDF `0.94474`; the `+0.00179` difference
is not established (95% CI `[-0.01495, +0.01798]`, `p=0.4448`). Transfer from
`eval_shared` to `coherent_pairs` is `0.88195` versus character TF-IDF
`0.86614` (`+0.01581`, 95% CI `[-0.01079, +0.04309]`, `p=0.0795`). The reverse
transfer is `0.82446` versus character TF-IDF `0.84854` (`-0.02408`, 95% CI
`[-0.06007, +0.01192]`, `p=0.8486` for the prespecified greater alternative).
The two within-control Qwen values in this paragraph record the initial runner
and are superseded by the uniform nested analysis below; transfer is unchanged.
The Gemma and Ministral full cores were then completed on separate A40 GPUs at
the same frozen revisions and precision. Gemma's four caches each contain 706
turn observations across 49 hidden-state readouts with 3,840 dimensions;
Ministral's contain 706 observations across 35 readouts with 4,096 dimensions.
All eight caches pass exact revision, adapter, dataset, sample, and extraction-
method provenance checks. Both models again have maximum pre-final difference
`0.0`. Gemma's mean final-state L2 differences are `12005.54` and `10026.92`;
Ministral's are `16.02` and `12.61`. These norms are not comparable across
architectures and are reported only to establish nonzero final differences.

After all model results were visible, an audit of Gemma's large reversal found
that the within-control runner selected the direction-of-means layer by
in-sample training AUROC. This did not leak the outer test fold, but in Gemma
many high-dimensional training curves saturated and the selected layers were
unstable. A corrected runner now selects the layer using five pair-grouped
folds nested inside each outer training fold. The correction was applied to
every model and endpoint, not only Gemma. Llama is unchanged; Qwen and
Ministral move slightly. Gemma selects layer 48 in all five `eval_shared` outer
folds and layer 9 in all five coherent-pair folds. Transfer already used
grouped source-only CV and is unchanged.

The corrected replication table is:

| Model | Endpoint | Probe | Strongest text | Difference (95% CI) | p |
|---|---|---:|---:|---:|---:|
| Qwen3-8B | `eval_shared` within | 0.99192 | 0.95293 | `+0.03899` `[+0.02182, +0.05823]` | 0.0025 |
| Qwen3-8B | coherent within | 0.94504 | 0.94474 | `+0.00030` `[-0.01628, +0.01643]` | 0.4918 |
| Qwen3-8B | eval -> coherent | 0.88195 | 0.86614 | `+0.01581` `[-0.01079, +0.04309]` | 0.0795 |
| Qwen3-8B | coherent -> eval | 0.82446 | 0.84854 | `-0.02408` `[-0.06007, +0.01192]` | 0.8486 |
| Gemma-3-12B-IT | `eval_shared` within | 0.97296 | 0.95293 | `+0.02003` `[-0.00021, +0.04043]` | 0.0885 |
| Gemma-3-12B-IT | coherent within | 0.85902 | 0.94474 | `-0.08571` `[-0.11331, -0.05891]` | 1.0000 |
| Gemma-3-12B-IT | eval -> coherent | 0.72642 | 0.86614 | `-0.13973` `[-0.17575, -0.10417]` | 1.0000 |
| Gemma-3-12B-IT | coherent -> eval | 0.65408 | 0.84854 | `-0.19446` `[-0.23496, -0.15188]` | 1.0000 |
| Ministral-3-8B | `eval_shared` within | 0.98309 | 0.95293 | `+0.03016` `[+0.01281, +0.04904]` | 0.0060 |
| Ministral-3-8B | coherent within | 0.92790 | 0.94474 | `-0.01683` `[-0.03563, +0.00282]` | 0.8531 |
| Ministral-3-8B | eval -> coherent | 0.84232 | 0.86614 | `-0.02382` `[-0.05075, +0.00519]` | 0.9030 |
| Ministral-3-8B | coherent -> eval | 0.87557 | 0.84854 | `+0.02702` `[-0.00619, +0.06111]` | 0.0660 |

Together with the reference Llama result, exact temporal localization and
above-chance final-turn decodability occur in all four families. The stronger
claim is heterogeneous: `eval_shared` establishes a positive paired advantage
for Llama, Qwen, and Ministral, while Gemma's positive margin is inconclusive;
matched coherence and both transfer directions fail
to beat the strongest text baseline in every replication model. This is
strong evidence that the temporal signal is not Llama-specific, and equally
strong evidence against describing its strength as architecture-invariant or
the signal as generally activation-specific beyond textual task structure.

## 4. Final evidential chain

The experiments should be understood as a sequence of increasingly restrictive tests, not as interchangeable AUROC values:

| Stage | What changed | Main observation | Valid conclusion |
|---|---|---|---|
| Slice 1 | Benchmark eval vs real user prompts | Probe ≈0.99; surface classifiers ≈0.95–0.97 | Pipeline detects a heavily confounded distinction |
| Slice 2a | Public 2×2 controls | Standard direction follows format; “matched” cells remain surface-separable | Existing controls do not isolate awareness under these baselines |
| Slice 2b | New authorship/length-controlled cells and transfer | Gate fails, but cross-construction probe transfer beats baselines | A partly general direction is shared across constructions |
| Slice 3, first written cells | Multi-turn matched skeletons | Signal appears before final question; surface gate fails | This version is not a valid temporal control |
| `eval_shared` | Byte-identical prefix; only final turn swapped | Chance before swap, ≈0.992 after | Signal is introduced at the class-defining final turn |
| `coherent_pairs` | Byte-identical coherent prefix and topic-matched endings | ≈0.951 vs final-turn BoW ≈0.933; chance before final turn | High separation remains, but is largely lexical in-distribution |
| Cross-control transfer | Train on one multi-turn control, test the other | Probe beats unigram counts, but character TF-IDF reaches 0.866/0.849 vs probe 0.845/0.719 in the grouped audit | Transfer does not establish an activation-specific component beyond stronger text features |
| Observed silence | Retain terminated pairs without detected awareness language | High raw separation remains; coherent probe and BoW tie | Overt disclosure is not necessary, but coherent silence does not beat lexical text |
| Slice 5 | Genuine vs reconstructed replies | Replicates, but stronger reply-only baselines narrow or erase margins | Provenance/context and surface form are represented; not specifically self-authorship |
| Estimator/null checks | Logistic probe and full permutations | Main final-turn effects replicate; pre-final controls remain null | Effects are not estimator-specific or ordinary selection noise |
| Steering | Add direction during generation | Readout and self-report move; safety does not | Direction is causally accessible for self-report, not shown to govern broad policy |
| Authorship pilot, pre-activation gate | Cross final-turn status with three generated history authors | 150/150 histories terminate, but only 36/50 scenarios survive the combined coherence gate | v2 fails before activation extraction; authorship effects remain untested |
| Cross-model confirmatory suite | Freeze two valid controls across three current model families | All models pass exact temporal controls; Llama/Qwen/Ministral establish positive `eval_shared` margins, Gemma ties there and is weaker on matched coherence | Temporal decodability crosses four families, but strength is model-dependent and activation-specific generalization remains unestablished |

## 5. What the paper can and cannot claim

### Supported claims

The repository supports the following statements, provided their qualifications remain attached:

1. Llama-3.1-8B-Instruct hidden states strongly distinguish multiple eval-like and deploy-like prompt constructions.
2. Simple benchmark-versus-user separation is heavily explained by surface form; the project measured and documented this rather than treating high probe accuracy as sufficient.
3. On the Slice 2b math-free transfer grid, activation directions generalize across independently built pairs better than the tested format and bag-of-words classifiers. The strict completely disjoint margin is `0.04754 ± 0.01538`; the all-off-diagonal margin is `0.07108 ± 0.01341`.
4. In two paired multi-turn controls, activation separation is at chance throughout a byte-identical prefix and emerges only when the eval/deploy final turn is introduced.
5. The two multi-turn constructions transfer to one another, but their activation directions do not beat character n-gram classifiers. The earlier positive transfer margin was relative to unigram counts only.
6. High raw separation remains in matched pairs whose generated replies contain no detected awareness language. On matched-coherence silence pairs, however, the activation probe and final-turn bag-of-words are tied; `eval_shared` retains only a modest activation margin.
7. Genuine versus reconstructed assistant replies are internally separable across WildChat and LMSYS, but reply-only format baselines explain much of the effect and controls rule out a narrow self-authorship interpretation.
8. The main activation results survive a second probe estimator and full-pipeline permutation tests.
9. Steering along the matched-coherence direction monotonically moves an independent readout and explicit eval/deploy self-reports at nondegenerate strengths.
10. Qwen3-8B, Gemma-3-12B-IT, and Ministral-3-8B all replicate exact temporal localization and above-chance final-turn separation on the two frozen paired controls. Qwen and Ministral join the Llama reference in establishing a positive paired advantage on `eval_shared`; Gemma's positive margin is inconclusive, and no replication model beats text on matched coherence or transfer.

### Claims the repository does not support

The following would overstate what was done:

- “The proposal was executed as written.” It was not; the experimental program changed substantially in response to failed controls.
- “The final Slice 2b cells are surface matched.” Their length gate passed, but format and bag-of-words gates failed.
- “The first multi-turn disguised-evaluation experiment proves temporal awareness.” Its signal was already present after the synthetic first turn.
- “The model knows it is being evaluated.” The experiments identify a robust direction associated with these constructions, but do not uniquely identify its semantic content.
- “No lexical information remains.” Bag-of-words is strong in several contrasts, and the coherent final-turn-mean probe selects the embedding layer.
- “The matched-coherence activation probe meaningfully beats words in-distribution.” With the correct final-turn-only view, probe `0.95132` versus BoW `0.93337` leaves only a `0.01796` strength margin, and the silence-filtered subset is tied.
- “Cross-control activation transfer beats text baselines.” It beats unigram counts, but the stronger grouped audit finds character TF-IDF at `0.86614/0.84854` versus probe `0.84496/0.71874` in the two directions.
- “The observed-silence subset proves unconscious or unverbalized awareness.” Absence of detected disclosure does not reveal the model’s full internal semantics.
- “Slice 5 demonstrates self-recognition.” The signal survives a different generator and strengthens under context ablation.
- “Steering the direction improves safety behavior.” It changed self-report, but the refusal test showed no predicted increase.
- “The effect is invariant to who generated the preceding conversation.” The v2 authorship-factorial pilot failed its coherence gate before activation extraction, so this remains untested.
- “The effect has architecture-invariant strength or a demonstrated scale trend.” Temporal decodability appears in all four tested families, but matched-coherence and transfer strength vary substantially and the 5--12B panel was not designed to estimate scaling.
- “The project measured agentic tool-use behavior or positive spontaneous verbalization cases.” There is no completed per-class tool-use dataset, and no real positive-disclosure classifier could be estimated because the judged matched-coherence outputs contained zero positives.

## 6. Reproducibility and audit notes that should survive into the paper

1. **Use nested grouped train-only layer selection.** The historical Slice 1 best-layer curve and saved main Slice 3 analysis used held-out curves for layer selection. The first grouped within-control runner instead used in-sample training AUROC, which saturated for Gemma. Current grouped values use pair-grouped inner CV nested inside each outer training fold.
2. **Do not silently combine transfer regimes.** Report either the disjoint Slice 2b transfer subset as primary or show it beside the 10 shared-cell entries and the all-30 average.
3. **Keep failed gates visible.** Slice 2b and the initial Slice 3 written cells did not satisfy their surface gates. The v2 authorship-factorial pilot also failed its combined human/model coherence gate at 36/50 and was stopped before activation extraction. Later experiments or revised pilots are responses to those failures, not evidence that the failures disappeared.
4. **Keep rejected datasets out of the headline.** `eval_coherent` failed manual coherence review despite having cached activations. `casual_deploy_rw` is a stale pilot.
5. **Distinguish selection SD from confidence intervals.** Five-seed `±` values are population SD across splits. Bootstrap intervals and permutation p-values answer different questions.
6. **Describe strength carefully.** Absolute AUROC distance from chance is useful for inverted transfer, but high-dimensional random directions can have nontrivial absolute strength. Signed AUROC and full-pipeline permutations should accompany it.
7. **Describe silence as observed silence.** State termination counts, cap, matching rule, judge validation, conservative exclusion, and cap-512 sensitivity.
8. **Prefer the shared-intersection Slice 5 control.** It uses the exact same 146 conversations and genuine activations for model-swap versus context-ablation comparison. The older baseline and model-swap reconstruction files predate the `generation_context` metadata field and leave it empty; code/history identify them as full-prefix runs, while the context-ablation file explicitly records `final_user_only`.
9. **Use only the differing segment for additive baselines on identical-prefix pairs.** Whole-conversation word counts and format features include a prefix that occurs under both labels and can only add estimation noise. The corrected Slice 3, silence, transfer, and Slice 5 artifacts record `paired_surface_text_view: differing_final_message_only`.
10. **Keep steering endpoints out of behavioral trends.** Alpha ±4 violated the frozen degeneracy rule.
11. **Do not invent judge provenance.** The artifacts support “Claude Code subagents with blinded rows,” but not a specific distinct judge-model version.
12. **Normalize manifest line endings when validating across operating systems.** The RunPod archive used CRLF and therefore recorded the legacy raw manifest hash `e54a...`, while the local LF checkout hashes to `a9c1...`. Parsed JSON is identical, dataset-content hashes match, and the validator now canonicalizes line endings while accepting only this legacy EOL variant.

## 7. Code and artifact map

This map lists the paper-relevant implementation rather than every operational file.

### Shared methods

- `src/vnaa/extract.py`: model loading, chat rendering, turn-span validation, user/assistant read points, trajectories, and activation serialization.
- `src/vnaa/probe.py`: direction-of-means fitting, per-layer scoring, layer choice, and AUROC.
- `src/vnaa/baselines.py`: splits, structural features, length/format/bag-of-words baselines, strength, and bootstrap intervals.
- `src/vnaa/run_strong_baseline_audit.py`: grouped outer-fold word/character TF-IDF audit, source-only transfer selection, pair-cluster bootstrap intervals, and direct probe-versus-baseline randomization tests.
- `src/vnaa/estimators.py`: honest direction-of-means and logistic estimators, training-only cross-validation, permutations, shuffled directions, and random-direction diagnostics.
- `src/vnaa/run_estimator_check.py`: final estimator and null-distribution analysis.

### Slice 1 and 2a

- `src/vnaa/data.py`, `src/vnaa/run_slice1.py`: SAD loading and initial probe.
- `src/vnaa/data_2x2.py`: public 2×2 cell loader.
- `src/vnaa/run_2x2_baselines.py`, `src/vnaa/run_slice2a.py`: surface audit and activation reproduction.

### Slice 2b

- `src/vnaa/prep_2b_batches.py`, `src/vnaa/merge_2b.py`, `src/vnaa/build_2b.py`: prompt sourcing, writer batches, validation, and final cells.
- `src/vnaa/check_rewrite_drift.py`: source-to-rewrite drift audit.
- `src/vnaa/run_transfer.py`: direction-of-means and text-baseline transfer, including source filtering.
- `src/vnaa/run_per_source.py`: ARC/MMLU/GSM8K audit.
- `results/s2b_Llama-3.1-8B-Instruct_transfer_multiseed_mathfree.json`: primary math-free transfer artifact.
- `results/estimator_check_Llama-3.1-8B-Instruct_transfer_mathfree.json`: logistic transfer check.

### Slice 3 and silence

- `src/vnaa/prep_slice3.py`, `src/vnaa/prep_slice3_batches.py`, `src/vnaa/merge_slice3_coherence.py`: source selection and first written cells.
- `src/vnaa/gate_slice3.py`: shape, framing, length, format, and lexical gates.
- `src/vnaa/build_eval_shared.py`: identical-prefix recombination control.
- `src/vnaa/prep_coherent_pair_pilot.py`, `src/vnaa/merge_coherent_pair_pilot.py`: 50-pair coherent pilot.
- `src/vnaa/prep_coherent_pairs.py`, `src/vnaa/merge_coherent_pairs.py`: final 250 matched-coherence pairs.
- `src/vnaa/run_slice3.py`: read-point extraction, trajectory analysis, and surface comparisons.
- `src/vnaa/run_slice3_transfer.py`: transfer between `eval_shared` and `coherent_pairs`.
- `src/vnaa/run_silence_filter.py`: greedy generation, regex filter, matched retention, and silence-conditioned probing.
- `datasets/judge/awareness_judge_v1.md`, `src/vnaa/judge_batches.py`: frozen judge rubric, blinded batches, and label merging.

### Slice 5

- `src/vnaa/prep_slice5.py`: WildChat/LMSYS filtering and reconstruction metadata.
- `src/vnaa/run_slice5.py`: reply generation, token/mean activation extraction, baseline comparison, model-swap/context controls, and shared-mask analysis.
- `results/s5_Llama-3.1-8B-Instruct_model_swap_llama32_3b_n250_vs_context_ablation_final_user_only_n250_shared_analysis.json`: preferred control comparison.

### Steering

- `datasets/steering/prespec.md`: frozen protocol and dated amendment.
- `datasets/steering/behavior_prompts.json`: fixed carrier and safety prompts.
- `src/vnaa/steering.py`: residual hook, scaling, readout, scoring, and degeneration metrics.
- `src/vnaa/run_steering.py`: direction fitting, manipulation check, generation, and analysis.
- `results/steer_Llama-3.1-8B-Instruct_analysis.json`: completed steering result.

### Authorship factorial pilot

- `datasets/authorship/prespec.md`, `datasets/authorship/prespec_v2.md`: original frozen protocol and the documented global-cap amendment after the v1 termination failure.
- `datasets/authorship/pilot_manifest_v2.json`, `src/vnaa/prep_authorship_factorial.py`: source/prespec-hashed deterministic selection of the 50 pilot scenarios.
- `src/vnaa/run_authorship_generation.py`: unquantized greedy generation with immutable revision, rendered-input hash, token-id, and termination metadata.
- `src/vnaa/build_authorship_gate.py`: construction of the 300 factorial cells and blinded 150-row coherence set.
- `datasets/authorship/coherence_judge_v1.md`, `src/vnaa/authorship_coherence.py`: frozen rubric, deterministic batches, strict label validation, and judge-run provenance.
- `src/vnaa/review_authorship_human.py`, `src/vnaa/finalize_authorship_gate.py`: reviewer-identified, resumable 30-row human review and conservative final gate.
- `src/vnaa/compare_authorship_reviewers.py`: second-reviewer agreement diagnostic that does not modify the frozen gate result.

### Cross-model confirmation

- `datasets/cross_model/confirmatory_suite_v1.json`, `datasets/cross_model/confirmatory_suite_v2.json`: preserved original panel and the current pre-result, revision-pinned 5--12B amendment.
- `src/vnaa/cross_model_suite.py`: content-hash validation, model-isolated activation caches, provenance checking, temporal diagnostics, grouped strong-baseline analysis, and bidirectional transfer.
- `scripts/cross_model_confirmatory.md`: local smoke, rented-GPU execution, compute envelope, and artifact-copy checklist.
- `results/cross_model/v2/{Qwen,google-gemma,mistralai-Ministral}*/full`: the four full activation caches, analysis, extraction log, and per-model SHA-256 manifest for each replication.

### Verification

The current test suite contains 139 passing tests covering probes, estimator splitting, nested pair-grouped layer selection, stronger text baselines, pair-grouped folds, paired margin inference, deterministic authorship-pilot selection and gating, independent-reviewer diagnostics, cross-model dataset fingerprints, revision pinning, artifact isolation, native Qwen/Gemma/Ministral chat boundaries, causal-prefix extraction, cross-platform manifest hashing, read-point indexing, paired surface views, judge batches, silence artifact paths, Slice 5 context handling, and steering hooks/scorers. The saved reference-model activation arrays consistently contain 33 layers of 4,096-dimensional `float32` vectors, matching the 8B model’s embedding plus 32-block hidden-state interface.

## 8. Recommended paper-level summary

A concise and accurate paper summary, updated after the completed primary model panel, would be:

> We progressively tested whether hidden states distinguish evaluation-like from deployment-like interactions after accounting for surface cues. Initial Llama-3.1-8B-Instruct benchmark-versus-user probes were nearly perfect but were largely reproduced by format and lexical baselines. Newly constructed single-turn contrasts also retained surface differences, so we evaluated cross-construction transfer; on completely disjoint math-free transfers, activation-probe separation exceeded unigram counts by `0.0475 ± 0.0154` across five splits. In two multi-turn paired controls, activations were indistinguishable throughout byte-identical prefixes and separated only after the differing final turn. Three prespecified replications—Qwen3-8B, Gemma-3-12B-IT, and Ministral-3-8B—reproduced this exact temporal localization and above-chance final-turn decodability. After a uniform nested layer-selection correction, Llama, Qwen, and Ministral establish positive paired margins over text on the topic-shifted `eval_shared` control, while Gemma ties text there and remains weaker on matched coherence; matched coherence and cross-control transfer never reliably beat character n-grams. On Llama, raw separation also remains among terminated pairs without detected awareness disclosure, and a pre-specified residual intervention moved an independent readout and explicit eval/deploy self-reports but did not increase safety refusals. These results identify cross-family, behaviorally accessible directions associated with the studied constructions while showing that current evidence does not isolate an architecture-invariant evaluation-awareness variable from textual task structure.

That language reflects what the repository demonstrates without turning unsuccessful matching, conditional silence, provenance detection, or a self-report steering effect into stronger claims than the experiments justify.
