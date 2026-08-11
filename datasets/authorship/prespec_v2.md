# Evaluation-status x history-authorship factorial: pilot amendment v2

Status: frozen after the v1 probe-blind termination failure and before any v2
generation or activation extraction.

This amendment incorporates every design, gate, analysis, inference, and
interpretation rule in `datasets/authorship/prespec.md`, except for the single
generation-cap change below.

## Reason for the amendment

The v1 pilot was stopped as soon as its prespecified success criterion became
unattainable. Under greedy decoding with a 256-new-token cap,
Llama-3.1-8B-Instruct naturally terminated 1 of the first 13 histories; the
other 12 reached the cap. With at least 11 scenarios already invalid, fewer
than 40 of 50 scenarios could survive the all-condition gate. No off-policy
histories were generated, and no activations were extracted or inspected.

The partial v1 generation record is retained as
`results/authorship/failed_v1/target_on_policy_partial.json`.

## Only protocol change

The maximum generation length is increased globally from 256 to 1,024 new
tokens for all three history generators. Natural termination remains required.
Greedy decoding, native chat templates, no quantization, metadata capture, and
every other rule are unchanged. The cap is not adapted by item or model.

The same 50 deterministically selected feasibility-pilot scenarios are reused.
They remain permanently excluded from the confirmatory sample. If the v2 pilot
again fails to retain at least 40/50 scenarios, stop without inspecting
activations and revise the construction protocol in another dated amendment.
