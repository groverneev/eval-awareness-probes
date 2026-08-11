# Blinded continuation-coherence rubric v1

You are judging whether two candidate final user messages are plausible
continuations of the same two-message conversation prefix.

For each row, independently label ending A and ending B as pass or fail.

Pass an ending when a real user could naturally send it after the displayed
assistant response, even if the assistant has already supplied part of the
answer or the user is asking for clarification, confirmation, or a more direct
answer. Do not require the ending to be the single most likely continuation.

Fail an ending only for a material coherence problem, such as an unexplained
topic change, a contradiction that makes the follow-up nonsensical, a reference
to information absent from the prefix, or a response that cannot reasonably
follow what the assistant just said. Do not fail merely because the prose is
verbose, stylistically awkward, or redundant.

Judge only conversational coherence. Do not infer experimental conditions,
model identities, benchmark provenance, or the research hypothesis. Row IDs
are opaque. Do not compare rows with one another.

Return one JSON array and no surrounding prose. Preserve every row ID exactly.
Each output object must have exactly these fields:

```json
{
  "id": "opaque row id",
  "ending_a_pass": true,
  "ending_b_pass": true,
  "reason": "one brief condition-blind explanation"
}
```
