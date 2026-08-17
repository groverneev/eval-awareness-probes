# Paper contract

## Research question
Do language-model activations encode an evaluation/deployment distinction that survives realistic controls, strong textual baselines, and changes in model family?

## Main answer
Evaluation-like and deployment-like situations create temporally localized,
linearly and nonlinearly decodable activation differences across four model
families. The primary linear comparison is heterogeneous, but a nested RBF
sensitivity audit recovers a consistent matched-coherence advantage over a
capacity-matched contextual E5 RBF. Shared-prefix margins remain heterogeneous,
and nonlinear cross-control transfer does not beat text. These results still do not identify a
universal, content-independent evaluation-awareness variable.
Across six dense Qwen3 checkpoints from 0.75B to 32.76B exact parameters, the
paired scale-slope interval excludes zero on matched coherence but not shared
prefix, so the frozen same-direction criterion finds no consistent within-
family scale trend.

A frozen contextual text audit further narrows the result: an explicit
prefix--ending E5 interaction reaches AUROC 0.95325 on matched coherence versus
0.9468 for the reference activation probe, statistically tying it, and reaches
0.99554 on the shared-prefix control versus 0.9878 for the probe.

## Contributions
1. We show that near-perfect evaluation-awareness probes can largely reflect
   format and lexical differences.
2. We introduce paired multi-turn controls with byte-identical prefixes and
   localize the activation difference to the class-defining final user turn.
3. We evaluate stronger text baselines, cross-model replication, and causal
   steering, establishing both the signal's robustness and its interpretive
   limits.
4. We test every original dense Qwen3 size under uniform hardware and find that
   activation advantage over text is control-dependent rather than consistently
   scale-predictable.

## Explicit non-claims
- We do not establish that models possess a universal concept meaning
  "I am being evaluated."
- We do not establish an activation-specific nonlinear cross-control advantage
  or identify the within-control nonlinear signal as evaluation awareness.
- We do not show that steering improves safety behavior.
- We do not study tool-using or autonomous agents.

## Main evidence
1. Confounded benchmark-versus-user probes are nearly perfect, but surface
   classifiers are also nearly perfect.
2. Completely disjoint Slice 2b transfer: probe-minus-unigram margin
   +0.04754 ± 0.01538 across five splits.
3. Both paired controls are at chance throughout byte-identical prefixes and
   separate only after the differing final turn.
4. Llama eval_shared: probe 0.98779 versus word TF-IDF 0.95293,
   difference +0.03486, 95% CI [+0.01833, +0.05395].
5. Matched coherence: probe 0.94682 versus character TF-IDF 0.94474,
   difference +0.00208, not significant.
6. Frozen contextual interaction: matched coherence 0.95325 versus probe
   0.94682, probe-minus-baseline -0.00643, 95% CI [-0.0185, +0.0059];
   shared-prefix 0.99554 versus probe 0.98779, difference -0.00774,
   95% CI [-0.0156, -0.0015].
7. Llama, Qwen, Gemma, and Ministral all show temporal decodability. After a
   uniform nested layer-selection correction, Llama/Qwen/Ministral establish
   positive shared-prefix margins; Gemma ties there but remains weaker than
   text on matched coherence and transfer. A uniform post-result readout audit
   shows that unrestricted mean pooling aligns Gemma only through the input-
   embedding average; transformer-only means preserve strong architecture-by-
   readout heterogeneity.
8. At each primary-selected layer, a nested RBF probe reaches 0.98112--0.98786
   on matched coherence across all models and beats a nested RBF over the
   strongest contextual E5 interaction (0.95659) with positive paired intervals.
   The advantage is positive in every one of five fold seeds. Gemma reaches
   0.98112, showing that its weak primary result is decoder-geometry dependent.
9. Nonlinear cross-control activation transfer does not beat either the
   source-selected nonlinear text pipeline or the stronger linear character
   TF--IDF comparator. Shared-prefix nonlinear margins also remain
   heterogeneous, with Gemma numerically below E5.
10. The six-size Qwen3 ladder has paired-bootstrap slopes +0.0054
    [-0.0017,+0.0129] on shared prefix and +0.0068 [+0.0013,+0.0132] on matched
    coherence. Because only one excludes zero, it does not meet the frozen
    criterion for a consistent trend. Five-seed nonlinear margins are nearly
    flat across size.
11. Steering changes the independent readout and explicit self-report, but
   not safety refusals.
