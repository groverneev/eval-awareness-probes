# Paper contract

## Research question
Do language-model activations encode an evaluation/deployment distinction that survives realistic controls, strong textual baselines, and changes in model family?

## Main answer
Evaluation-like and deployment-like situations create temporally localized,
linearly decodable activation differences across four model families. However,
strong text baselines and model heterogeneity prevent interpreting this as a
universal, content-independent evaluation-awareness variable.

## Contributions
1. We show that near-perfect evaluation-awareness probes can largely reflect
   format and lexical differences.
2. We introduce paired multi-turn controls with byte-identical prefixes and
   localize the activation difference to the class-defining final user turn.
3. We evaluate stronger text baselines, cross-model replication, and causal
   steering, establishing both the signal's robustness and its interpretive
   limits.

## Explicit non-claims
- We do not establish that models possess a universal concept meaning
  "I am being evaluated."
- We do not establish an activation-specific advantage on matched-coherence
  or cross-control transfer.
- We do not show that steering improves safety behavior.
- We do not study tool-using or autonomous agents.

## Main evidence
1. Confounded benchmark-versus-user probes are nearly perfect, but surface
   classifiers are also nearly perfect.
2. Completely disjoint Slice 2b transfer: probe-minus-unigram margin
   +0.04754 ± 0.01538 across five splits.
3. Both paired controls are at chance throughout byte-identical prefixes and
   separate only after the differing final turn.
4. Llama eval_shared: probe 0.98781 versus word TF-IDF 0.95293,
   difference +0.03488, 95% CI [+0.01875, +0.05330].
5. Matched coherence: probe 0.94682 versus character TF-IDF 0.94474,
   difference +0.00208, not significant.
6. Llama, Qwen, Gemma, and Ministral all show temporal decodability, but
   strength is heterogeneous.
7. Steering changes the independent readout and explicit self-report, but
   not safety refusals.