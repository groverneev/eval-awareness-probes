# Contextual text-baseline audit (frozen before execution)

## Motivation and status

This is a post-hoc robustness audit of the paper's two byte-identical-prefix
Slice-3 controls.  The existing sparse baselines deliberately see only the
differing final user message because the shared prefix has no additive label
signal.  A nonlinear encoder can nevertheless represent whether an ending is
semantically compatible with, or changes meaning in, its prefix.  This audit
therefore gives a text-only model the entire causally available conversation.

The analysis below was fixed before inspecting any contextual-baseline result.
It does not change the confirmatory status of the original experiment.

## Frozen encoder and text views

- Encoder: `intfloat/e5-base-v2` at immutable revision
  `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`.
- The encoder is frozen; no language-model parameters are updated.
- Every encoder input is prefixed with `query: `, as required by the E5 usage
  convention. Messages are serialized with literal `[USER]` and `[ASSISTANT]`
  role markers.
- Inputs are capped at 512 wordpiece tokens. Left truncation preserves the
  final user message and the immediately preceding context.
- The last hidden states are attention-mask mean pooled and L2 normalized.

Three fixed feature views are reported:

1. **Full context (primary):** one embedding of the complete visible
   conversation through the final user message.
2. **Ending only (secondary):** one embedding of the differing final user
   message. This separates gains from a stronger semantic encoder from gains
   due to context.
3. **Prefix-ending interaction (secondary):** concatenate the prefix
   embedding, ending embedding, their absolute difference, and their
   elementwise product.

## Estimation and outcomes

For each representation, fit an L2-regularized logistic regression with fixed
`C=1`, standardized using training-fold statistics. Use the same deterministic
five grouped folds (seed 0) as the strong-baseline audit, keeping both members
of a scenario pair in one fold. Concatenate positive examples before negative
examples, matching the activation analysis. Report pooled out-of-fold AUROC.

The primary outcome is full-context AUROC on the 250 matched-coherence pairs.
Full-context AUROC on `eval_shared`, and both secondary representations on each
control, are secondary outcomes. No representation or regularization
hyperparameter is selected using held-out outcomes.

For the reference-model activation probe, compare its grouped out-of-fold
predictions with the primary contextual baseline using a 2,000-draw
scenario-cluster bootstrap for the AUROC difference and a 2,000-draw paired
approximate-randomization test. Positive differences favor the activation
probe. Regardless of direction or significance, report the contextual result
and narrow the manuscript's claims accordingly.
