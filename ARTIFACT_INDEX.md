# Experiment Artifact and Implementation Index

## Purpose

Use this file to locate paper-relevant code, frozen protocols, datasets, and
result artifacts. Experimental conclusions and headline numbers belong in
`EXPERIMENT_REPORT.md`; detailed chronology and audit reasoning belong in
`EXPERIMENT_LEDGER.md`.

`data/` and `results/` are gitignored local artifacts. Do not delete or
regenerate them unless a task specifically requires it. SAD prompt text must
never be written to committed files.

## Shared methods

- `src/vnaa/extract.py`: model loading, native chat rendering, turn-span
  validation, user/assistant read points, trajectories, and activation
  serialization.
- `src/vnaa/probe.py`: direction-of-means fitting, per-layer scoring, layer
  choice, and AUROC.
- `src/vnaa/baselines.py`: splits, structural features, length/format/bag-of-
  words baselines, separation strength, and bootstrap intervals.
- `src/vnaa/run_strong_baseline_audit.py`: grouped outer-fold word and character
  TF-IDF, nested pair-grouped within-control layer selection, source-only
  transfer selection, pair-cluster bootstrap intervals, and direct
  probe-versus-baseline randomization tests.
- `datasets/contextual_text_baseline/prespec.md` and
  `src/vnaa/run_contextual_text_baseline.py`: revision-pinned frozen E5
  ending/full-context/prefix--ending interaction audit with grouped predictions
  and paired probe comparisons. Local output is
  `results/contextual_text_baseline/contextual_text_baseline.json`.
- `src/vnaa/estimators.py`: honest direction-of-means and logistic estimators,
  training-only cross-validation, permutations, shuffled directions, and random-
  direction diagnostics.
- `src/vnaa/run_estimator_check.py`: estimator and null-distribution analysis.

## Slice 1 and public 2×2 reproduction

- `src/vnaa/data.py`, `src/vnaa/run_slice1.py`: SAD loading and initial probe.
- `src/vnaa/data_2x2.py`: public 2×2 cell loader.
- `src/vnaa/run_2x2_baselines.py`, `src/vnaa/run_slice2a.py`: surface audit and
  activation reproduction.

## Slice 2b single-turn constructions

- `src/vnaa/prep_2b_batches.py`, `src/vnaa/merge_2b.py`,
  `src/vnaa/build_2b.py`: prompt sourcing, writer batches, validation, and final
  cells.
- `src/vnaa/check_rewrite_drift.py`: source-to-rewrite drift audit.
- `src/vnaa/run_transfer.py`: direction-of-means and text-baseline transfer,
  including source filtering.
- `src/vnaa/run_per_source.py`: ARC/MMLU/GSM8K audit.
- `results/s2b_Llama-3.1-8B-Instruct_transfer_multiseed_mathfree.json`: primary
  math-free transfer artifact.
- `results/estimator_check_Llama-3.1-8B-Instruct_transfer_mathfree.json`:
  logistic transfer check.

## Slice 3 and observed silence

- `src/vnaa/prep_slice3.py`, `src/vnaa/prep_slice3_batches.py`,
  `src/vnaa/merge_slice3_coherence.py`: source selection and first written cells.
- `src/vnaa/gate_slice3.py`: shape, framing, length, format, and lexical gates.
- `src/vnaa/build_eval_shared.py`: identical-prefix recombination control.
- `src/vnaa/prep_coherent_pair_pilot.py`,
  `src/vnaa/merge_coherent_pair_pilot.py`: 50-pair coherence pilot.
- `src/vnaa/prep_coherent_pairs.py`, `src/vnaa/merge_coherent_pairs.py`: final
  250 matched-coherence pairs.
- `src/vnaa/run_slice3.py`: read-point extraction, trajectories, and surface
  comparisons.
- `src/vnaa/run_slice3_transfer.py`: transfer between `eval_shared` and
  `coherent_pairs`.
- `src/vnaa/run_silence_filter.py`: greedy generation, regex filtering, matched
  retention, and silence-conditioned probing.
- `datasets/judge/awareness_judge_v1.md`, `src/vnaa/judge_batches.py`: frozen
  awareness rubric, blinded batches, and label merging.

## Slice 5 reply reconstruction

- `src/vnaa/prep_slice5.py`: WildChat/LMSYS filtering and reconstruction
  metadata.
- `src/vnaa/run_slice5.py`: reply generation, token/mean activation extraction,
  baseline comparison, model-swap/context controls, and shared-mask analysis.
- `results/s5_Llama-3.1-8B-Instruct_model_swap_llama32_3b_n250_vs_context_ablation_final_user_only_n250_shared_analysis.json`:
  preferred exact-intersection control comparison.

## Steering

- `datasets/steering/prespec.md`: frozen protocol and dated amendment.
- `datasets/steering/behavior_prompts.json`: fixed carrier and safety prompts.
- `src/vnaa/steering.py`: residual hook, scaling, readout, scoring, and
  degeneration metrics.
- `src/vnaa/run_steering.py`: direction fitting, manipulation check, generation,
  and analysis.
- `results/steer_Llama-3.1-8B-Instruct_analysis.json`: completed steering result.

## Authorship-factorial pilot

- `datasets/authorship/prespec.md`, `datasets/authorship/prespec_v2.md`: original
  frozen protocol and the global-cap amendment after the v1 termination failure.
- `datasets/authorship/pilot_manifest_v2.json`,
  `src/vnaa/prep_authorship_factorial.py`: source/prespec-hashed deterministic
  selection of 50 pilot scenarios.
- `src/vnaa/run_authorship_generation.py`: unquantized greedy generation with
  immutable revision, rendered-input hash, token IDs, and termination metadata.
- `src/vnaa/build_authorship_gate.py`: 300 factorial cells and the blinded
  150-row coherence set.
- `datasets/authorship/coherence_judge_v1.md`,
  `src/vnaa/authorship_coherence.py`: frozen rubric, deterministic batches,
  strict label validation, and judge-run provenance.
- `src/vnaa/review_authorship_human.py`,
  `src/vnaa/finalize_authorship_gate.py`: reviewer-identified resumable human
  review and conservative final gate.
- `src/vnaa/compare_authorship_reviewers.py`: second-reviewer diagnostic that
  does not modify the frozen gate result.

## Cross-model confirmation

- `datasets/cross_model/confirmatory_suite_v1.json`,
  `datasets/cross_model/confirmatory_suite_v2.json`: preserved original panel
  and current revision-pinned 5--12B amendment.
- `src/vnaa/cross_model_suite.py`: content-hash validation, model-isolated
  activation caches, provenance checks, temporal diagnostics, grouped strong-
  baseline analysis, and bidirectional transfer.
- `scripts/cross_model_confirmatory.md`: local smoke, rented-GPU execution,
  compute envelope, and artifact-copy checklist.
- `results/cross_model/v2/{Qwen,google-gemma,mistralai-Ministral}*/full`: full
  activation caches, analyses, extraction logs, and per-model SHA-256 manifests.

## Paper figures

- `paper/make_fig1.py` through `paper/make_fig5.py`: source for the paired-
  control schematic, anchor results, disjoint transfer, steering effects, and
  failed-control figures, respectively.
- `paper/build_figures.ps1`: regenerate all figure PDFs and ignored preview
  PNGs independently of the LaTeX build.
- `paper/build.ps1`: build the manuscript from the current figure PDFs without
  rerunning the figure scripts.

## Verification expectations

Run `.venv\Scripts\python -m pytest` when code changes warrant it. The current
suite covers probe splitting, stronger text baselines, nested pair-grouped
layer selection, pair-grouped folds,
paired margin inference, deterministic authorship selection and gating,
cross-model fingerprints and revision pinning, model-specific chat boundaries,
causal-prefix extraction, cross-platform manifest hashing, read-point indexing,
paired surface views, judge batches, silence paths, Slice 5 context handling,
and steering hooks and scorers.

Reference-model activation arrays contain 33 layers of 4,096-dimensional
`float32` vectors: the embedding output plus 32 transformer-block outputs.
Cross-model caches additionally embed immutable model revision, manifest hash,
dataset-content hash, precision, extraction method, and sample identity.
