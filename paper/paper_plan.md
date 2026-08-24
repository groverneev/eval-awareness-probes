# Paper contract

## Research question
Do language-model activations encode an evaluation/deployment distinction that survives realistic controls, strong textual baselines, and changes in model family?

## Construct definitions
- **Evaluation awareness:** an internal representation of whether the current
  interaction is being used to evaluate the model, rather than merely prompt
  features correlated with evaluation.
- **Prompt-category decodability:** a readout can distinguish the tested
  evaluation-like and deployment-like examples, potentially using genre, topic,
  format, or speech act.
- **Evaluation self-report:** the model reports whether it believes it is being
  evaluated.
- **Policy use:** perceived evaluation status influences task-relevant behavior.

Text-controlled and cross-construction generalization test whether a decoded
signal supports the evaluation-awareness interpretation. Self-report steering
with unrelated-direction controls tests direction-specific causal control of
the report, and refusal steering is one narrow test of policy use. These
properties may dissociate.

## Main answer
Paired controls pass an exact pre-final leakage check, and evaluation-like and
deployment-like labels become decodable after the first differing user message
across four tested checkpoints. None of the eight primary direction-of-means
comparisons outperforms the strongest full-input E5 interaction; five paired
intervals exclude zero in the text-favoring direction and three include zero.
At the same fold-selected
layers, nested L2-logistic probes recover most of the apparent RBF gain, while
RBF probes retain a consistent matched-coherence advantage over a
capacity-matched E5-interaction RBF. Shared-prefix margins remain heterogeneous,
and nonlinear cross-control transfer does not beat the source-selected text
pipeline or character TF-IDF. In the single-turn
grid, activation directions preserve target separability but reverse label
orientation more often than unigram classifiers. These results still do not
identify a universal, content-independent evaluation-awareness variable.
Across six dense Qwen3 checkpoints from 0.75B to 32.76B exact parameters, the
paired scale-slope interval excludes zero on matched coherence but not shared
prefix, so the frozen same-direction criterion finds no consistent within-
family scale trend.

On the Llama reference model, learned-direction steering specifically moves an
independent activation readout. Its paired self-report endpoint effect is
`+0.7500` (`[+0.5833,+0.8889]`), but five norm-matched orthogonal controls yield
`+0.4167`, `+0.9444`, `+0.3056`, `0.0000`, and `+1.0000`; the frozen behavioral-
specificity criterion therefore fails. Neither the learned direction nor the
controls produces the predicted positive safety-refusal effect. The experiment
supports causal accessibility of the decoded readout, not direction-specific
causal control of evaluation self-report or broad policy use.

A frozen contextual text audit supplies the main linear comparator: an explicit
prefix--ending E5 interaction reaches AUROC 0.95325 on matched coherence versus
0.9468 for the reference activation probe, statistically tying it, and 0.99554
on shared prefix versus 0.9878, significantly favoring text.

## Contributions
1. We show that near-perfect evaluation-awareness probes can largely reflect
   format and lexical differences.
2. We introduce paired multi-turn controls with byte-identical prefixes, verify
   exact pre-final equality as a deterministic leakage check, and measure
   decoding only after the first differing user message.
3. We evaluate stronger text baselines, cross-model replication, and causal
   steering with norm-matched orthogonal controls, establishing both the
   signal's robustness and its interpretive limits.
4. We test every original dense Qwen3 size under uniform hardware and find that
   probe-minus-E5 margins are not consistently scale-predictable across both
   controls.

## Explicit non-claims
- We do not establish that models possess a universal concept meaning
  "I am being evaluated."
- We do not establish an activation-specific nonlinear cross-control advantage
  or identify the within-control nonlinear signal as evaluation awareness.
- We do not establish a consistently oriented cross-construction activation
  direction that transfers better than unigram text.
- We do not establish direction-specific causal control of evaluation
  self-report; orthogonal controls match or exceed the learned effect.
- We do not show that steering improves safety behavior.
- We do not study tool-using or autonomous agents.

## Main evidence
1. Confounded benchmark-versus-user probes are nearly perfect, but surface
   classifiers are also nearly perfect.
2. Completely disjoint Slice 2b transfer: mean signed target AUROC is 0.65209
   for the probe versus 0.66565 for unigram counts. Probe orientation is
   preserved in a majority of seeds for 15/20 ordered transfers and in all
   five seeds for 12/20, versus 17/20 and 15/20 for unigrams. Folding away
   orientation gives strength 0.22810 versus 0.18055, but does not establish a
   shared evaluation-like direction.
3. Both paired controls pass exact pre-final equality checks. This verifies the
   construction and extraction pipeline; it is not evidence about awareness.
   Labels become decodable after the first differing user message.
4. Sparse-text diagnostics give probe-minus-TF-IDF differences +0.03486 on
   shared prefix and +0.00208 on matched coherence, but they are secondary to
   the stronger full-input comparison.
5. Frozen contextual interaction: matched coherence 0.95325 versus probe
   0.94682, probe-minus-baseline -0.00643, 95% CI [-0.0185, +0.0059];
   shared-prefix 0.99554 versus probe 0.98779, difference -0.00774,
   95% CI [-0.0156, -0.0015].
6. Llama, Qwen, Gemma, and Ministral all pass the leakage check and show
   post-divergence decodability. Against the same E5 interaction, all eight
   primary linear point differences are negative; five intervals favor text
   and three include zero. A uniform post-result readout audit
   shows that unrestricted mean pooling aligns Gemma only through the input-
   embedding average; transformer-only means preserve strong architecture-by-
   readout heterogeneity.
7. At each primary-selected layer, a nested RBF probe reaches 0.98112--0.98786
   on matched coherence across all models and beats a nested RBF over the
   strongest contextual E5 interaction (0.95659) with positive paired intervals.
   The advantage is positive in every one of five fold seeds. Gemma reaches
   0.98112. Under the same grouped nested pipeline, L2 logistic reaches
   0.97566 on Gemma and 0.98494--0.98722 on the other models at seed 0;
   every RBF-minus-L2 interval includes zero. Across five seeds, mean
   RBF-minus-L2 is only +0.00090--+0.01036. The weak primary results are
   therefore estimator-sensitive, and the full recovery is not specifically
   evidence of nonlinear geometry.
8. Nonlinear cross-control activation transfer does not beat either the
   source-selected nonlinear text pipeline or the stronger linear character
   TF--IDF comparator. Shared-prefix nonlinear margins also remain
   heterogeneous, with Gemma numerically below E5.
9. The E5-referenced six-size Qwen3 ladder has paired-bootstrap slopes +0.0054
    [-0.0017,+0.0129] on shared prefix and +0.0068 [+0.0013,+0.0132] on matched
    coherence. Because only one excludes zero, it does not meet the frozen
    criterion for a consistent trend. Five-seed nonlinear margins are nearly
    flat across size.
10. Learned-direction steering specifically moves the independent readout, but
    its self-report effect is not direction-specific: two of five orthogonal
    controls exceed it. Neither learned nor control directions produce the
    predicted positive safety-refusal effect.
