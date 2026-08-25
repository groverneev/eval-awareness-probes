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
- `src/vnaa/run_full_context_linear_comparison.py`: CPU-only reprioritization
  audit comparing every primary cross-model and Qwen-scale linear probe with
  the cached full-input E5 prefix--ending interaction under paired inference.
  Local output is
  `results/full_context_linear_comparison/analysis.json`.
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

- `datasets/steering/prespec.md`: original frozen protocol, pre-generation
  amendment, and explicitly post-result Priority 8 robustness amendment.
- `datasets/steering/behavior_prompts.json`: fixed carrier and safety prompts.
- `src/vnaa/steering.py`: residual hook, scaling, readout, scoring, and
  degeneration metrics.
- `src/vnaa/run_steering.py`: direction fitting, manipulation check, generation,
  and analysis.
- `results/steer_Llama-3.1-8B-Instruct_analysis.json`: completed steering result.
- `src/vnaa/run_steering_priority8.py`: prompt-cluster bootstrap and exact
  paired inference, deterministic norm-matched random/orthogonal control
  construction, resumable control readout/generation, and specificity audit.
- `scripts/steering_priority8.md` and
  `scripts/run_steering_priority8_pod.sh`: 24GB+ GPU handoff used for the 840
  completed control-direction behavioral generations.
- `results/steering_priority8/`: ignored frozen control directions, completed
  independent-readout controls, prompt-level learned-direction analysis,
  behavioral-control outputs, logs, SHA-256 inventory, and final specificity
  analysis.

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
- `datasets/qwen_scale/scale_ladder_v1.json`: pre-result, revision-pinned
  six-checkpoint dense Qwen3 scale ladder. It freezes exact parameter counts,
  uniform-hardware extraction, paired scale-trend inference, and the rule that
  all checkpoints and endpoints are reported regardless of direction.
- `scripts/qwen_scale_ladder.md`: local validation and uniform 80GB-pod
  execution runbook for the frozen scale ladder.
- `scripts/run_qwen_scale_pod.sh`, `scripts/launch_qwen_scale_pod.sh`: resumable
  uniform-pod runner and detached launch wrapper used for the six checkpoints.
- `src/vnaa/run_qwen_scale_analysis.py`: reconstructs and validates the paired
  out-of-fold predictions from each scale cache, performs the prespecified
  paired-scenario bootstrap of the slope against log10 exact parameter count,
  and checks the uniform-hardware 8B rerun against the historical 8B result.
- `src/vnaa/cross_model_suite.py`: content-hash validation, model-isolated
  activation caches, provenance checks, temporal diagnostics, grouped strong-
  baseline analysis, and bidirectional transfer.
- `src/vnaa/run_readout_audit.py`: uniform final-user-turn mean-cache
  extraction and nested grouped sensitivity analysis, including a transformer-
  only layer restriction.
- `src/vnaa/run_nonlinear_probe_audit.py`: pair-grouped nested RBF-capacity
  audit at the primary-selected activation layer, plus a capacity-matched
  nonlinear TF-IDF comparator.
- `src/vnaa/run_nonlinear_followup.py`: frozen-E5 RBF comparison, source-only
  nonlinear cross-control transfer, and five-seed stability analysis. It caches
  revision-pinned E5 features under the ignored results directory.
- `src/vnaa/run_decoder_comparison.py`: CPU-only same-layer comparison of
  direction-of-means, nested L2 logistic, and nested RBF across the frozen
  four-checkpoint panel and five grouped-fold seeds.
- `scripts/cross_model_confirmatory.md`: local smoke, rented-GPU execution,
  compute envelope, and artifact-copy checklist.
- `results/cross_model/v2/{Qwen,google-gemma,mistralai-Ministral}*/full`: full
  activation caches, analyses, extraction logs, and per-model SHA-256 manifests.
- `results/readout_audit/`: post-result mean-readout caches and analyses for all
  four models; ignored because the activation arrays are large.
- `results/nonlinear_probe_audit/`: completed per-model nonlinear sensitivity
  analyses; ignored with the other reproducible result artifacts.
- `results/nonlinear_followup/`: cached E5 features, fixed-interaction
  inference, source-only nonlinear transfer, and five-seed per-model/combined
  outputs; ignored with the other reproducible result artifacts.
- `results/decoder_comparison/`: completed per-model and combined same-layer
  decoder analyses, including seed-0 paired inference; ignored with the other
  reproducible result artifacts.
- `results/qwen_scale/v1/`: all six full trajectory caches, primary analyses,
  validated paired predictions, SHA-256 inventories, and the 5,000-draw scale
  analysis; ignored because it contains large reproducible arrays.
- `results/qwen_scale/nonlinear_followup_v1/`: five-seed nonlinear RBF versus
  fixed contextual-E5 results for every scale checkpoint; ignored.

## Paper figures

- `paper/make_fig1.py` through `paper/make_fig6.py`: source for the paired-
  control schematic; separate leakage-check, text-hierarchy, and cross-model
  results figures; disjoint transfer; steering effects; failed controls; and
  the Qwen3 scale ladder.
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
alternate-readout audit validation, paired surface views, judge batches, silence
paths, nonlinear grouped-capacity validation, Slice 5 context handling, and
steering hooks and scorers.

Reference-model activation arrays contain 33 layers of 4,096-dimensional
`float32` vectors: the embedding output plus 32 transformer-block outputs.
Cross-model caches additionally embed immutable model revision, manifest hash,
dataset-content hash, precision, extraction method, and sample identity.
